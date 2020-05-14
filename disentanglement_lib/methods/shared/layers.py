import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

# TODO add mask as parameter
class MaskedConv2d(tf.layers.Conv2D):
    def __init__(self, perc_sparse=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        # just a placeholder for now
        self.mask = 1
        self.perc_sparse = perc_sparse

    def _init_mask(self):
        mask_val = (np.random.random(self.kernel.shape[-2:]) >= self.perc_sparse).astype('float')
        self.mask = tf.constant(
            value=mask_val,
            name='mask',
            dtype=self.kernel.dtype,
        )
        print('init_mask')

    # def apply(self, inputs, *args, **kwargs):
    #     ret = super().apply(inputs, *args, **kwargs)
    #
    #
    #
    #     return ret

    def call(self, inputs):
        print('call')
        self._init_mask()
        outputs = self._convolution_op(inputs, self.kernel * self.mask)

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
        perc_sparse=0.,
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