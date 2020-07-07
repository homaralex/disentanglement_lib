import gin
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import init_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

_EPS = 1e-8


@gin.configurable('masked_layer', whitelist=['mask_trainable'])
class _BaseMaskedLayer:
    def __init__(self, perc_sparse=0, mask_trainable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.perc_sparse = perc_sparse
        self.mask_trainable = mask_trainable

    @property
    def mask_shape(self):
        return self.kernel.shape[-2:]

    def _init_mask(self):
        mask_val = (np.random.random(self.mask_shape) >= self.perc_sparse).astype('float')
        self.mask = self.add_weight(
            name='mask',
            shape=self.mask_shape,
            initializer=init_ops.Constant(mask_val),
            trainable=self.mask_trainable,
            dtype=self.dtype,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.built = False
        self._init_mask()
        self.built = True


class MaskedConv2d(_BaseMaskedLayer, tf.layers.Conv2D):
    def call(self, inputs):
        outputs = self._convolution_op(
            inputs,
            # that's the actual change
            self.kernel * self.mask
        )

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def masked_conv2d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        perc_sparse=0,
):
    layer = MaskedConv2d(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name,
        perc_sparse=perc_sparse,
    )
    return layer.apply(inputs)


class MaskedDense(_BaseMaskedLayer, Dense):
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(
                inputs,
                # that's the actual change
                self.kernel * self.mask,
                [[rank - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)
            outputs = gen_math_ops.mat_mul(
                inputs,
                # that's the actual change
                self.kernel * self.mask,
            )
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def masked_dense(
        inputs, units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        perc_sparse=0,
):
    layer = MaskedDense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _scope=name,
        _reuse=reuse,
        perc_sparse=perc_sparse,
    )
    return layer.apply(inputs)


# TODO other parameterization version
class _BaseVDLayer:
    def __init__(
            self,
            training_phase,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.training_phase = training_phase

    @property
    def mask_shape(self):
        return self.kernel.shape[-2:]

    def _build(self):
        self.log_sigma_2 = self.add_weight(
            name='vd_log_sigma_2',
            shape=self.mask_shape,
            initializer=init_ops.Constant(-10.),
            trainable=True,
            dtype=self.dtype,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.built = False
        self._build()
        self.built = True

    def get_log_alpha(self):
        log_alpha = tf.clip_by_value(self.log_sigma_2 - tf.log(tf.square(self.kernel) + _EPS), -8., 8.)
        return tf.identity(log_alpha, name='log_alpha')

    @property
    def vd_threshold(self):
        # TODO maybe this should be a little more elegant
        return gin.query_parameter('vd_vae.vd_threshold')

    def _get_outputs(self, inputs, layer_op):
        log_alpha = self.get_log_alpha()

        if self.training_phase:
            mu = layer_op(inputs, self.kernel)
            std = tf.sqrt(
                layer_op(
                    tf.square(inputs),
                    tf.exp(log_alpha) * tf.square(self.kernel),
                ) + _EPS,
            )
            noisy_out = mu + std * tf.random_normal(tf.shape(std))

            outputs = noisy_out
        else:
            select_mask = tf.cast(tf.less(log_alpha, self.vd_threshold), tf.float32)
            masked_out = layer_op(inputs, self.kernel * select_mask)

            outputs = masked_out

        return outputs


class VDConv2D(_BaseVDLayer, tf.layers.Conv2D):
    def call(self, inputs):
        outputs = self._get_outputs(inputs, self._convolution_op)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def vd_conv2d(
        inputs,
        filters,
        kernel_size,
        training_phase,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
):
    layer = VDConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name,
        training_phase=training_phase,
    )
    return layer.apply(inputs)


class VDDense(_BaseVDLayer, Dense):
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)

        if rank > 2:
            # Broadcasting is required for the inputs.

            def _broadcasted_tensordot(_inputs, _kernel):
                return standard_ops.tensordot(
                    _inputs,
                    _kernel,
                    [[rank - 1], [0]]
                )

            outputs = self._get_outputs(inputs, _broadcasted_tensordot)

            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)

            outputs = self._get_outputs(inputs, gen_math_ops.mat_mul)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def vd_dense(
        inputs, units,
        training_phase,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
):
    layer = VDDense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _scope=name,
        _reuse=reuse,
        training_phase=training_phase,
    )
    return layer.apply(inputs)


class _BaseSoftmaxLayer:
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temperature = temperature


class SoftmaxConv2d(_BaseSoftmaxLayer, tf.layers.Conv2D):
    def call(self, inputs):
        softmax_kernel = self.kernel * tf.nn.softmax(
            logits=tf.reshape(
                tf.reduce_max(tf.abs(self.kernel), axis=(0, 1)),
                (self.kernel.shape[2], -1),
            ) / self.temperature,
            axis=0,
        )

        outputs = self._convolution_op(
            inputs,
            # that's the actual change
            softmax_kernel,
        )

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def softmax_conv2d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        temperature=1.,
):
    layer = SoftmaxConv2d(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name,
        temperature=temperature,
    )
    return layer.apply(inputs)


class SoftmaxDense(_BaseSoftmaxLayer, Dense):
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)

        softmax_kernel = self.kernel * tf.nn.softmax(
            logits=tf.abs(self.kernel) / self.temperature,
            axis=0,
        )

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(
                inputs,
                # that's the actual change
                softmax_kernel,
                [[rank - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)
            outputs = gen_math_ops.mat_mul(
                inputs,
                # that's the actual change
                softmax_kernel,
            )
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def softmax_dense(
        inputs, units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        temperature=1.,
):
    layer = SoftmaxDense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _scope=name,
        _reuse=reuse,
        temperature=temperature,
    )
    return layer.apply(inputs)
