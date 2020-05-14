import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from disentanglement_lib.methods.shared.layers import masked_conv2d, masked_dense

_CHECKPOINT_PATH = "/tmp/model.ckpt"


def test_dense():
    tf.reset_default_graph()

    def get_ops():
        input_tensor = tf.placeholder(tf.float32, input_shape, name='input')
        _non_sparse_output_tensor = masked_dense(
            inputs=input_tensor,
            units=10,
            bias_initializer=init_ops.zeros_initializer(),
            kernel_initializer=init_ops.ones_initializer(),
        )
        _sparse_output_tensor = masked_dense(
            perc_sparse=.5,
            inputs=input_tensor,
            units=10,
            bias_initializer=init_ops.zeros_initializer(),
            kernel_initializer=init_ops.ones_initializer(),
        )

        return _non_sparse_output_tensor, _sparse_output_tensor

    input_shape = (1, 20)
    non_sparse_output_tensor, sparse_output_tensor = get_ops()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        non_sparse_output = (sess.run(
            non_sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert not (non_sparse_output != 20).any()

        sparse_output = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        # note that there is a non-zero probability that this will fail anyway because masks are generated randomly -
        # try re-running the test then
        assert (sparse_output != 20).any()

        # check if the masking is consistent for future calls
        sparse_output2 = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert np.array_equal(sparse_output, sparse_output2)

        saver.save(sess, _CHECKPOINT_PATH)

    tf.reset_default_graph()
    non_sparse_output_tensor, sparse_output_tensor = get_ops()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CHECKPOINT_PATH)

        # check if the masking is consistent for loaded graphs
        sparse_output2 = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert np.array_equal(sparse_output, sparse_output2)


def test_conv2d():
    tf.reset_default_graph()

    def get_ops():
        input_tensor = tf.placeholder(tf.float32, input_shape, name='input')
        _non_sparse_output_tensor = masked_conv2d(
            inputs=input_tensor,
            filters=10,
            kernel_size=3,
            bias_initializer=init_ops.zeros_initializer(),
            kernel_initializer=init_ops.ones_initializer(),
        )
        _sparse_output_tensor = masked_conv2d(
            perc_sparse=.5,
            inputs=input_tensor,
            filters=10,
            kernel_size=3,
            bias_initializer=init_ops.zeros_initializer(),
            kernel_initializer=init_ops.ones_initializer(),
        )

        return _non_sparse_output_tensor, _sparse_output_tensor

    input_shape = (1, 3, 3, 20)
    non_sparse_output_tensor, sparse_output_tensor = get_ops()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

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
        # note that there is a non-zero probability that this will fail anyway because masks are generated randomly -
        # try re-running the test then
        assert (sparse_output != 20 * 3 * 3).any()
        # check if whole channels are masked (as opposed to masking on a filter-level)
        assert not (sparse_output % 3 * 3 != 0).any()

        # check if the masking is consistent for future calls
        sparse_output2 = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert np.array_equal(sparse_output, sparse_output2)

        saver.save(sess, _CHECKPOINT_PATH)

    tf.reset_default_graph()
    non_sparse_output_tensor, sparse_output_tensor = get_ops()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CHECKPOINT_PATH)

        # check if the masking is consistent for loaded graphs
        sparse_output2 = (sess.run(
            sparse_output_tensor, feed_dict={
                'input:0': np.ones(input_shape),
            }))
        assert np.array_equal(sparse_output, sparse_output2)


# TODO run as tf.TestCase
if __name__ == '__main__':
    test_dense()
    test_conv2d()
