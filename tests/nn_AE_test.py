from GPyEDS import nn


def test_nn_AE():
    m = nn.create_nn_AE(7, latent_dim=2, hidden=[10, 20, 5], activation="relu")
