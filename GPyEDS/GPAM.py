"""Native GPflow Deep Gaussian Process implementation for GPAM models.

This module provides a 2-layer Deep GP using pure gpflow, replacing the
previously abandoned gpflux dependency.  The variational inference follows
Doubly Stochastic Variational Inference (Salimbeni & Deisenroth, 2017),
using whitened parameterization for numerical stability.
"""

import typing as t
from pathlib import Path

import gpflow
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from gpflow.base import TensorType
from tensorflow_probability import distributions as tfd


class GPLayer(gpflow.Module):
    """Variational GP layer for Deep GPs using whitened parameterization.

    Each layer maintains its own set of inducing variables and variational
    parameters (q_mu, q_sqrt).  The whitened parameterization (whiten=True)
    is used for better numerical conditioning, matching the original gpflux
    GPLayer default.

    Args:
        kernel: GPflow kernel for this layer.
        inducing_variable: GPflow inducing variable (locations Z).
        num_data: Total number of training data points (used for KL scaling).
        num_latent_gps: Number of latent GP outputs for this layer.
        mean_function: Optional mean function.  Defaults to Zero.
        name: Optional name for this module.
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        inducing_variable: gpflow.inducing_variables.InducingPoints,
        num_data: int,
        num_latent_gps: int,
        mean_function: gpflow.mean_functions.MeanFunction | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_latent_gps = num_latent_gps
        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.mean_function = (
            mean_function if mean_function is not None else gpflow.mean_functions.Zero()
        )
        self.num_data = tf.cast(num_data, t.cast(tf.DType, gpflow.default_float()))

        num_inducing = t.cast(int, self.inducing_variable.num_inducing)
        self.q_mu = tf.Variable(
            t.cast(t.Any, np.zeros((num_inducing, self.num_latent_gps))),
            dtype=t.cast(tf.DType, gpflow.default_float()),
            name="q_mu",
        )

        q_sqrt_init: npt.NDArray[np.float64] = np.array(
            [np.eye(num_inducing) for _ in range(self.num_latent_gps)],
            dtype=gpflow.default_float(),
        )
        self.q_sqrt = gpflow.Parameter(
            q_sqrt_init, transform=gpflow.utilities.triangular()
        )

    def prior_kl(self) -> tf.Tensor:
        """KL divergence KL[q(u) || p(u)] using whitened parameterization."""
        return gpflow.kullback_leiblers.prior_kl(
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            self.q_sqrt,
            whiten=True,
        )

    def conditional(self, inputs: TensorType) -> tuple[TensorType, TensorType]:
        """Compute the conditional mean and variance at *inputs*."""
        f_mean, f_var = gpflow.conditionals.conditional(
            inputs,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            white=True,
            full_cov=False,
            full_output_cov=False,
        )
        if self.mean_function is not None:
            f_mean += self.mean_function(inputs)
        return f_mean, f_var

    def __call__(
        self,
        inputs: TensorType | tfd.Distribution,
        training: bool | None = None,
    ) -> tfd.Normal:
        """Forward pass: returns a Normal distribution over outputs.

        If *inputs* is a ``tfd.Distribution`` (from a preceding layer), a
        single reparameterized sample is drawn first.
        """
        if isinstance(inputs, tfd.Distribution):
            samples = inputs.sample()
        else:
            samples = inputs

        samples = tf.cast(samples, t.cast(tf.DType, gpflow.default_float()))
        f_mean, f_var = self.conditional(samples)
        return tfd.Normal(loc=f_mean, scale=tf.sqrt(t.cast(t.Any, f_var)))


class NativeDeepGP(tf.keras.Model):  # type: ignore[misc]
    """Keras model wrapping a stack of :class:`GPLayer` and a likelihood.

    The training objective is the evidence lower bound (ELBO):

        ELBO = E_q[log p(y|f)] - KL[q(u) || p(u)]

    The data term uses ``variational_expectations`` (not
    ``predict_log_density``) so that the bound is valid by Jensen's
    inequality.  Both terms are averaged per data point for consistent
    scaling across batch sizes.
    """

    def __init__(
        self,
        gp_layers: list[GPLayer],
        likelihood: gpflow.likelihoods.Likelihood,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.gp_layers_list = gp_layers
        self.likelihood = likelihood

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        all_vars: list[tf.Variable] = []
        for layer in self.gp_layers_list:
            all_vars.extend(layer.trainable_variables)
        all_vars.extend(self.likelihood.trainable_variables)
        return list({v.ref(): v for v in all_vars}.values())

    def call(
        self, inputs: t.Any, training: bool | None = None, mask: t.Any | None = None
    ) -> tfd.Normal:
        x: TensorType | tfd.Distribution = inputs
        for layer in self.gp_layers_list:
            x = layer(x, training=training)
        return x

    # --------------------------------------------------------------------- #
    #  Helpers for unpacking the various data formats Keras may pass us.     #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _unpack_data(
        data: t.Any,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if isinstance(data, dict):
            x = data["inputs"]
            y = data["targets"]
        elif isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            else:
                x = data[0]
                y = data[0]
        else:
            x = data
            y = data

        return (
            tf.cast(x, t.cast(tf.DType, gpflow.default_float())),
            tf.cast(y, t.cast(tf.DType, gpflow.default_float())),
        )

    def _compute_elbo(
        self, x: tf.Tensor, y: tf.Tensor, training: bool
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return ``(loss, ell, kl)`` for a batch."""
        f_dist = self(x, training=training)
        f_mean = f_dist.mean()
        f_var = f_dist.variance()

        # E_q[log p(y|f)] — correct variational expectation (ELBO data term)
        ell = tf.reduce_mean(
            self.likelihood.variational_expectations(x, f_mean, f_var, y)
        )

        # KL[q(u) || p(u)] / N  for each layer
        kl_loss = tf.reduce_sum(
            [layer.prior_kl() / layer.num_data for layer in self.gp_layers_list]
        )

        loss = -ell + kl_loss
        return loss, ell, kl_loss

    def train_step(self, data: t.Any) -> dict[str, tf.Tensor]:
        x, y = self._unpack_data(data)

        with tf.GradientTape() as tape:
            loss, ell, kl_loss = self._compute_elbo(x, y, training=True)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        if self.optimizer is not None:
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {"loss": loss, "ell": ell, "kl": kl_loss}

    def test_step(self, data: t.Any) -> dict[str, t.Any]:
        x, y = self._unpack_data(data)
        loss, ell, kl_loss = self._compute_elbo(x, y, training=False)
        return {"loss": loss, "ell": ell, "kl": kl_loss}


# ------------------------------------------------------------------ #
#  Factory helpers                                                    #
# ------------------------------------------------------------------ #


def create_two_layer_GPAM_from_data(
    input_data: npt.NDArray[np.float64],
    num_inducing: int = 50,
    return_layers: bool = False,
    n_latent: int = 2,
) -> t.Any:
    """Create a 2-layer Deep GP from an existing dataset.

    Args:
        input_data: Dataset array of shape ``(N, D)``.
        num_inducing: Number of inducing points.  Clamped to at most ``N``
            to prevent duplicate selections.  Defaults to 50.
        return_layers: If ``True``, also return the individual GP layers.
        n_latent: Dimension of the latent space.  Defaults to 2.

    Returns:
        A compiled :class:`NativeDeepGP` model, or
        ``(model, gp_layer1, gp_layer2)`` when *return_layers* is ``True``.
    """
    num_data = input_data.shape[0]
    num_inducing = min(num_inducing, num_data)

    z_indices = np.random.choice(num_data, size=num_inducing, replace=False)
    inducing_z = input_data[z_indices].copy()

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1] * input_data.shape[1])
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(inducing_z)
    gp_layer1 = GPLayer(
        kernel1,
        inducing_variable1,
        num_data=num_data,
        num_latent_gps=n_latent,
        mean_function=gpflow.mean_functions.Zero(),
    )

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1] * n_latent)
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(
        np.random.rand(num_inducing, n_latent)
    )
    gp_layer2 = GPLayer(
        kernel2,
        inducing_variable2,
        num_data=num_data,
        num_latent_gps=input_data.shape[1],
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflow.likelihoods.Gaussian(0.1)
    model = NativeDeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model.compile(optimizer="adam")

    if return_layers:
        return model, gp_layer1, gp_layer2
    return model


def create_two_layer_GPAM_from_scratch(
    num_input: int,
    num_data: int = 1,
    Z: npt.NDArray[np.float64] | None = None,
    num_inducing: int = 50,
    return_layers: bool = False,
    n_latent: int = 2,
) -> t.Any:
    """Create a 2-layer Deep GP from scratch (without existing data).

    Args:
        num_input: Number of input dimensions.
        num_data: Number of training data points (used for ELBO scaling).
        Z: Optional array of inducing-point locations.  Generated at random
            if ``None``.
        num_inducing: Number of inducing points.  Defaults to 50.
        return_layers: If ``True``, also return the individual GP layers.
        n_latent: Dimension of the latent space.  Defaults to 2.

    Returns:
        A compiled :class:`NativeDeepGP` model, or
        ``(model, gp_layer1, gp_layer2)`` when *return_layers* is ``True``.
    """
    if Z is None:
        Z = np.random.rand(num_inducing, num_input)

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1] * num_input)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer1 = GPLayer(
        kernel1,
        inducing_variable1,
        num_data=num_data,
        num_latent_gps=n_latent,
        mean_function=gpflow.mean_functions.Zero(),
    )

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1] * n_latent)
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(
        np.random.rand(num_inducing, n_latent)
    )
    gp_layer2 = GPLayer(
        kernel2,
        inducing_variable2,
        num_data=num_data,
        num_latent_gps=num_input,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflow.likelihoods.Gaussian(0.1)
    model = NativeDeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model.compile(optimizer="adam")

    if return_layers:
        return model, gp_layer1, gp_layer2
    return model


def model_inference(
    data: npt.NDArray[np.float64],
    encoder: t.Any,
    batch_size: int = 20000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Batched model inference to reduce memory usage.

    Args:
        data: Data array of shape ``(N, D)``.
        encoder: A :class:`GPLayer` (or callable returning a ``tfd.Distribution``).
        batch_size: Batch size for inference.  Defaults to 20000.

    Returns:
        ``(means, variances)`` — each of shape ``(N, L)``.
    """

    means = []
    variances = []
    # Loop over the data in chunks of batch_size, handling perfect division cleanly without an empty slice
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        res = encoder(data[i : i + batch_size])
        means.append(res.mean())
        variances.append(res.variance())

    return np.concatenate(means, axis=0), np.concatenate(variances, axis=0)
