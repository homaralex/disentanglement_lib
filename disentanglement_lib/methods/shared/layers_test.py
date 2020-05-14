import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from disentanglement_lib.methods.shared.layers import MaskedConv2d, masked_conv2d


def test_conv2d():
    input_shape = (1, 3, 3, 20)

    input_tensor = tf.placeholder(tf.float32, input_shape, name='input')
    non_sparse_output_tensor = masked_conv2d(
        inputs=input_tensor,
        filters=10,
        kernel_size=3,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_initializer=init_ops.ones_initializer(),
    )
    sparse_output_tensor = masked_conv2d(
        perc_sparse=.5,
        inputs=input_tensor,
        filters=10,
        kernel_size=3,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_initializer=init_ops.ones_initializer(),
    )

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        non_sparse_output = (sess.run(
            non_sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert not (non_sparse_output != 20 * 3 * 3).any()

        sparse_output = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert (sparse_output != 20 * 3 * 3).any()
        # check if whole channels are masked (as opposed to masking on a filter-level)
        assert not (sparse_output % 3 * 3 != 0).any()

        # check if the masking is consistent for future calls
        sparse_output2 = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert np.array_equal(sparse_output, sparse_output2)


if __name__ == '__main__':
    test_conv2d()