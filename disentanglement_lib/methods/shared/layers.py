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


class _BaseMaskedLayer:
    def __init__(self, perc_sparse=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.perc_sparse = perc_sparse

    @property
    def mask_shape(self):
        return self.kernel.shape[-2:]

    def _init_mask(self):
        mask_val = (np.random.random(self.mask_shape) >= self.perc_sparse).astype('float')
        self.mask = self.add_weight(
            name='mask',
            shape=self.mask_shape,
            initializer=init_ops.Constant(mask_val),
            trainable=False,
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
                [[rank - 1], [0]])
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
                self.kernel * self.mask
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
