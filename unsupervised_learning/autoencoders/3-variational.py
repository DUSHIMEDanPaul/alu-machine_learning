#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""


import tensorflow as tf
import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """
    Samples the latent representation from the mean and log variance
    using the reparameterization trick, and adds the KL divergence loss
    """

    def call(self, inputs):
        """
        Returns z = mean + exp(log_var / 2) * epsilon
        """
        mean, log_var = inputs
        kl = -0.5 * tf.reduce_sum(
            1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl))
        epsilon = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    parameters:
        input_dims [int]:
            contains the dimensions of the model input
        hidden_layers [list of ints]:
            contains the number of nodes for each hidden layer in the encoder
                the hidden layers should be reversed for the decoder
        latent_dims [int]:
            contains the dimensions of the latent space representation

    All layers should use relu activation except for the mean and log
        variance layers in the encoder, which should use None,
        and the last layer, which should use sigmoid activation
    Autoencoder model should be compiled with Adam optimization
        and binary cross-entropy loss

    returns:
        encoder, decoder, auto
            encoder [model]: the encoder model,
                which should output the latent representation, the mean,
                and the log variance
            decoder [model]: the decoder model
            auto [model]: full autoencoder model
                compiled with adam optimization and binary cross-entropy loss
    """
    if type(input_dims) is not int:
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if type(hidden_layers) is not list:
        raise TypeError("hidden_layers must be a list of ints \
        representing number of nodes for each layer")
    for nodes in hidden_layers:
        if type(nodes) is not int:
            raise TypeError("hidden_layers must be a list of ints \
            representing number of nodes for each layer")
    if type(latent_dims) is not int:
        raise TypeError("latent_dims must be an int containing dimensions of \
        latent space representation")

    # encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoder_value = encoder_inputs
    for i in range(len(hidden_layers)):
        encoder_layer = keras.layers.Dense(units=hidden_layers[i],
                                           activation='relu')
        encoder_value = encoder_layer(encoder_value)
    mean = keras.layers.Dense(units=latent_dims,
                              activation=None)(encoder_value)
    log_var = keras.layers.Dense(units=latent_dims,
                                 activation=None)(encoder_value)
    z = Sampling()([mean, log_var])
    encoder = keras.Model(inputs=encoder_inputs,
                          outputs=[z, mean, log_var])

    # decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoder_value = decoder_inputs
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = keras.layers.Dense(units=hidden_layers[i],
                                           activation='relu')
        decoder_value = decoder_layer(decoder_value)
    decoder_output_layer = keras.layers.Dense(units=input_dims,
                                              activation='sigmoid')
    decoder_outputs = decoder_output_layer(decoder_value)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # autoencoder
    inputs = encoder_inputs
    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)[0]))

    def reconstruction_loss(y_true, y_pred):
        """
        Binary cross-entropy summed over the input dimensions
        """
        return keras.losses.binary_crossentropy(y_true, y_pred) * input_dims

    auto.compile(optimizer='adam', loss=reconstruction_loss)
    return encoder, decoder, auto
