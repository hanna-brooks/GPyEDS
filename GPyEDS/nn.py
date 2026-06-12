import typing as t

import tensorflow as tf


def create_nn_AE(
    input_dim: int,
    latent_dim: int = 2,
    hidden: list[int] | None = None,
    activation: str = "relu",
) -> tuple[
    tf.keras.Model,
    tuple[tf.keras.Sequential, tf.keras.Sequential, tf.keras.Sequential],
]:
    """Generator function for neural network autoencoder architecture.

    Args:
        input_dim (int): Input dimensions for model.
        latent_dim (int, optional): Latent space dimensions/bottleneck size. Defaults to 2.
        hidden (list, optional): Dimensions for hidden layers - note model will be symmetric. Defaults to [10].
        activation (str, optional): Activation functions to use. Defaults to "relu".

    Returns:
        model (TF model): Autoencoder model.
    """

    if hidden is None:
        hidden = [10]

    enc_list: list[tf.keras.layers.Layer] = []
    for i in range(len(hidden)):
        if i == 0:
            enc_list.append(
                tf.keras.layers.Dense(
                    hidden[i], activation=activation
                )
            )
        else:
            enc_list.append(tf.keras.layers.Dense(hidden[i], activation=activation))
        enc_list.append(tf.keras.layers.LayerNormalization())
        enc_list.append(tf.keras.layers.LeakyReLU(0.02))

    dec_list: list[tf.keras.layers.Layer] = []
    for i in range(len(hidden)):
        dec_list.append(tf.keras.layers.Dense(hidden[::-1][i], activation=activation))
        dec_list.append(tf.keras.layers.LayerNormalization())
        dec_list.append(tf.keras.layers.LeakyReLU(0.02))

    dec_list.append(tf.keras.layers.Dense(input_dim))

    encoder = tf.keras.Sequential(enc_list)
    latent = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim)])
    decoder = tf.keras.Sequential(dec_list)

    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = encoder(inputs)
    lat = latent(encoded)
    decoded = decoder(lat)

    model = tf.keras.Model(inputs=inputs, outputs=decoded)

    return model, (encoder, latent, decoder)
