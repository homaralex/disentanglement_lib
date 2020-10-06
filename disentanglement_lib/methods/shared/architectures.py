# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of commonly used architectures and reconstruction losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import gin.tf

from disentanglement_lib.methods.shared.layers import masked_conv2d, masked_dense, vd_conv2d, vd_dense, softmax_dense, \
    softmax_conv2d, CodeNorm, dropout_conv2d, dropout_dense


@gin.configurable("encoder", whitelist=["num_latent", "encoder_fn"])
def make_gaussian_encoder(input_tensor,
                          is_training=True,
                          num_latent=gin.REQUIRED,
                          encoder_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a Gaussian encoder configurable with gin.

    This is a separate function so that several different models (such as
    BetaVAE and FactorVAE) can call this function while the gin binding always
    stays 'encoder.(...)'. This makes it easier to configure models and parse
    the results files.

    Args:
      input_tensor: Tensor with image that should be encoded.
      is_training: Boolean that indicates whether we are training (usually
        required for batch normalization).
      num_latent: Integer with dimensionality of latent space.
      encoder_fn: Function that that takes the arguments (input_tensor,
        num_latent, is_training) and returns the tuple (means, log_vars) with the
        encoder means and log variances.

    Returns:
      Tuple (means, log_vars) with the encoder means and log variances.
    """
    with tf.variable_scope("encoder"):
        return encoder_fn(
            input_tensor=input_tensor,
            num_latent=num_latent,
            is_training=is_training)


@gin.configurable("decoder", whitelist=["decoder_fn"])
def make_decoder(latent_tensor,
                 output_shape,
                 is_training=True,
                 decoder_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a decoder configurable with gin.

    This is a separate function so that several different models (such as
    BetaVAE and FactorVAE) can call this function while the gin binding always
    stays 'decoder.(...)'. This makes it easier to configure models and parse
    the results files.

    Args:
      latent_tensor: Tensor latent space embeddings to decode from.
      output_shape: Tuple with the output shape of the observations to be
        generated.
      is_training: Boolean that indicates whether we are training (usually
        required for batch normalization).
      decoder_fn: Function that that takes the arguments (input_tensor,
        output_shape, is_training) and returns the decoded observations.

    Returns:
      Tensor of decoded observations.
    """
    with tf.variable_scope("decoder"):
        return decoder_fn(
            latent_tensor=latent_tensor,
            output_shape=output_shape,
            is_training=is_training)


@gin.configurable("discriminator", whitelist=["discriminator_fn"])
def make_discriminator(input_tensor,
                       is_training=False,
                       discriminator_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a discriminator configurable with gin.

    This is a separate function so that several different models (such as
    FactorVAE) can potentially call this function while the gin binding always
    stays 'discriminator.(...)'. This makes it easier to configure models and
    parse the results files.

    Args:
      input_tensor: Tensor on which the discriminator operates.
      is_training: Boolean that indicates whether we are training (usually
        required for batch normalization).
      discriminator_fn: Function that that takes the arguments
      (input_tensor, is_training) and returns tuple of (logits, clipped_probs).

    Returns:
      Tuple of (logits, clipped_probs) tensors.
    """
    with tf.variable_scope("discriminator"):
        logits, probs = discriminator_fn(input_tensor, is_training=is_training)
        clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
    return logits, clipped


@gin.configurable("fc_encoder", whitelist=[])
def fc_encoder(input_tensor, num_latent, is_training=True):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.

    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """
    del is_training

    flattened = tf.layers.flatten(input_tensor)
    e1 = tf.layers.dense(flattened, 1200, activation=tf.nn.relu, name="e1")
    e2 = tf.layers.dense(e1, 1200, activation=tf.nn.relu, name="e2")
    means = tf.layers.dense(e2, num_latent, activation=None)
    log_var = tf.layers.dense(e2, num_latent, activation=None)
    return means, log_var


@gin.configurable("conv_encoder", blacklist=['input_tensor', 'num_latent', 'is_training'])
def conv_encoder(
        input_tensor,
        num_latent,
        perc_sparse=None,
        all_layers=True,
        perc_units=1.,
        vd_layers=False,
        softmax_layers=False,
        softmax_temperature=1.,
        scale_temperature=False,
        code_normalization=False,
        perc_dropout=None,
        is_training=True,
):
    """Convolutional encoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    use_masked = perc_sparse is not None

    def conv2d(*args, **kwargs):
        conv_fn = tf.layers.conv2d

        if all_layers:
            if use_masked:
                conv_fn = masked_conv2d
                kwargs['perc_sparse'] = perc_sparse
            elif vd_layers:
                kwargs['training_phase'] = is_training
                conv_fn = vd_conv2d
            elif softmax_layers:
                kwargs['temperature'] = softmax_temperature
                kwargs['scale_temperature'] = scale_temperature
                conv_fn = softmax_conv2d
            elif perc_dropout is not None:
                kwargs['rate'] = perc_dropout
                kwargs['training_phase'] = is_training
                conv_fn = dropout_conv2d

        return conv_fn(*args, **kwargs)

    def dense(*args, **kwargs):
        dense_fn = tf.layers.dense

        if all_layers or kwargs['name'] in ('means', 'log_var'):
            if use_masked:
                dense_fn = masked_dense
                kwargs['perc_sparse'] = perc_sparse
            elif vd_layers:
                kwargs['training_phase'] = is_training
                dense_fn = vd_dense
            elif softmax_layers:
                kwargs['temperature'] = softmax_temperature
                kwargs['scale_temperature'] = scale_temperature
                dense_fn = softmax_dense
            elif perc_dropout is not None:
                kwargs['rate'] = perc_dropout
                kwargs['training_phase'] = is_training
                dense_fn = dropout_dense

        return dense_fn(*args, **kwargs)

    # never mask the first layer
    e1 = tf.layers.conv2d(
        inputs=input_tensor,
        filters=int(32 * perc_units),
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e1",
    )
    e2 = conv2d(
        inputs=e1,
        filters=int(32 * perc_units),
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e2",
    )
    e3 = conv2d(
        inputs=e2,
        filters=int(64 * perc_units),
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e3",
    )
    e4 = conv2d(
        inputs=e3,
        filters=int(64 * perc_units),
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e4",
    )
    flat_e4 = tf.layers.flatten(e4)
    e5 = dense(inputs=flat_e4, units=256, activation=tf.nn.relu, name="e5")

    if perc_dropout is not None:
        # since we are branching out at this level make sure the same dropout is applied for both branches
        e5 = tf.layers.dropout(inputs=e5, rate=perc_dropout, training=is_training)
        perc_dropout = None

    means = dense(inputs=e5, units=num_latent, activation=None, name="means")
    log_var = dense(inputs=e5, units=num_latent, activation=None, name="log_var")

    if code_normalization:
        means, log_var = CodeNorm(num_latent=num_latent, training_phase=is_training)(means, log_var)

    # TODO actual if statement
    if True:
        def mask_init():
            return [0.] * int(means.shape[1])

        # This a bit tricky since dynamic execution is a pain in tf1
        # We have two non-trainable variables - a placeholder and the actual mask.
        # The placeholder is used for communication with the outer "non-tf" world - the greedy agent can control
        # its values via feed_dict.
        # The actual mask tensor however is used for the model automatically store its last value - e.g., when loading
        # for evaluation after training is finished.
        # During training the outer non-tf agent passes the current mask value via feed_dict - if the passed mask is
        # "wider" than the last used one, values in 'greedy_mask' tensor will be updated with 'greedy_mask_placeholder'.
        mask_placeholder = tf.Variable(initial_value=mask_init(), name='greedy_mask_placeholder', trainable=False)
        mask = tf.Variable(initial_value=mask_init(), name='greedy_mask', trainable=False)

        # we compare if the placeholder has more non-zero values (and thus a greater sum of elements)
        # based on that we return either the placeholder's or the current mask value
        mask_to_assign = tf.cond(
            pred=tf.less(tf.reduce_sum(mask), tf.reduce_sum(mask_placeholder)),
            true_fn=lambda: tf.identity(mask_placeholder),
            false_fn=lambda: tf.identity(mask),
        )
        # then we update the actual mask tensor either with the placeholder's value or itself
        mask = tf.assign(ref=mask, value=mask_to_assign)

        # TODO remove print
        mask = tf.Print(mask, [mask], message='Mask: ', summarize=10)
        means, log_var = means * mask, log_var * mask

    return means, log_var


@gin.configurable("fc_decoder", whitelist=[])
def fc_decoder(latent_tensor, output_shape, is_training=True):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.

    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
      intensities.
    """
    del is_training
    d1 = tf.layers.dense(latent_tensor, 1200, activation=tf.nn.tanh)
    d2 = tf.layers.dense(d1, 1200, activation=tf.nn.tanh)
    d3 = tf.layers.dense(d2, 1200, activation=tf.nn.tanh)
    d4 = tf.layers.dense(d3, np.prod(output_shape))
    return tf.reshape(d4, shape=[-1] + output_shape)


@gin.configurable("deconv_decoder", whitelist=[])
def deconv_decoder(latent_tensor, output_shape, is_training=True):
    """Convolutional decoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
        pixel intensities.
    """
    del is_training
    d1 = tf.layers.dense(latent_tensor, 256, activation=tf.nn.relu)
    d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
    d2_reshaped = tf.reshape(d2, shape=[-1, 4, 4, 64])
    d3 = tf.layers.conv2d_transpose(
        inputs=d2_reshaped,
        filters=64,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )

    d4 = tf.layers.conv2d_transpose(
        inputs=d3,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )

    d5 = tf.layers.conv2d_transpose(
        inputs=d4,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )
    d6 = tf.layers.conv2d_transpose(
        inputs=d5,
        filters=output_shape[2],
        kernel_size=4,
        strides=2,
        padding="same",
    )
    return tf.reshape(d6, [-1] + output_shape)


@gin.configurable("fc_discriminator", whitelist=[])
def fc_discriminator(input_tensor, is_training=True):
    """Fully connected discriminator used in FactorVAE paper for all datasets.

    Based on Appendix A page 11 "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983.pdf)

    Args:
      input_tensor: Input tensor of shape (None, num_latents) to build
        discriminator on.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      logits: Output tensor of shape (batch_size, 2) with logits from
        discriminator.
      probs: Output tensor of shape (batch_size, 2) with probabilities from
        discriminator.
    """
    del is_training
    flattened = tf.layers.flatten(input_tensor)
    d1 = tf.layers.dense(flattened, 1000, activation=tf.nn.leaky_relu, name="d1")
    d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
    d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
    d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
    d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
    d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
    logits = tf.layers.dense(d6, 2, activation=None, name="logits")
    probs = tf.nn.softmax(logits)
    return logits, probs


@gin.configurable("test_encoder", whitelist=["num_latent"])
def test_encoder(input_tensor, num_latent, is_training):
    """Simple encoder for testing.

    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """
    del is_training
    flattened = tf.layers.flatten(input_tensor)
    means = tf.layers.dense(flattened, num_latent, activation=None, name="e1")
    log_var = tf.layers.dense(flattened, num_latent, activation=None, name="e2")
    return means, log_var


@gin.configurable("test_decoder", whitelist=[])
def test_decoder(latent_tensor, output_shape, is_training=False):
    """Simple decoder for testing.

    Args:
      latent_tensor: Input tensor to connect decoder to.
      output_shape: Output shape.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
        pixel intensities.
    """
    del is_training
    output = tf.layers.dense(latent_tensor, np.prod(output_shape), name="d1")
    return tf.reshape(output, shape=[-1] + output_shape)
