import typing as t

import gpflow
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class GPLayer(gpflow.Module):
    """Native GPflow implementation of a Variational GP Layer for Deep GPs."""
    def __init__(self, kernel, inducing_variable, num_data, num_latent_gps, mean_function=None, name=None):
        super().__init__(name=name)
        self.num_latent_gps = num_latent_gps
        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.mean_function = mean_function if mean_function is not None else gpflow.mean_functions.Zero()
        self.num_data = tf.cast(num_data, gpflow.default_float())

        num_inducing = self.inducing_variable.num_inducing
        self.q_mu = tf.Variable(np.zeros((num_inducing, self.num_latent_gps)), dtype=gpflow.default_float(), name="q_mu")
        
        q_sqrt_init = np.array([np.eye(num_inducing) for _ in range(self.num_latent_gps)], dtype=gpflow.default_float())
        self.q_sqrt = gpflow.Parameter(q_sqrt_init, transform=gpflow.utilities.triangular())

    def prior_kl(self):
        return gpflow.kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=False
        )

    def conditional(self, inputs):
        f_mean, f_var = gpflow.conditionals.conditional(
            inputs,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            white=False,
            full_cov=False,
            full_output_cov=False
        )
        if self.mean_function is not None:
            f_mean += self.mean_function(inputs)
        return f_mean, f_var

    def __call__(self, inputs, training=None):
        if isinstance(inputs, tfd.Distribution):
            samples = inputs.sample()
        else:
            samples = inputs
            
        samples = tf.cast(samples, gpflow.default_float())
        f_mean, f_var = self.conditional(samples)
        return tfd.Normal(loc=f_mean, scale=tf.sqrt(f_var))

class NativeDeepGP(tf.keras.Model):
    """A Keras model wrapper for a sequence of GPLayers and a Gaussian Likelihood."""
    def __init__(self, gp_layers, likelihood, **kwargs):
        super().__init__(**kwargs)
        self.gp_layers_list = gp_layers
        self.likelihood = likelihood

    @property
    def trainable_variables(self):
        vars = []
        for l in self.gp_layers_list:
            vars.extend(l.trainable_variables)
        vars.extend(self.likelihood.trainable_variables)
        return list({v.ref(): v for v in vars}.values())

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.gp_layers_list:
            x = layer(x, training=training)
        return x

    def train_step(self, data):
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

        x = tf.cast(x, gpflow.default_float())
        y = tf.cast(y, gpflow.default_float())

        with tf.GradientTape() as tape:
            f_dist = self(x, training=True)
            f_mean = f_dist.mean()
            f_var = f_dist.variance()
            
            ell = tf.reduce_sum(self.likelihood.predict_log_density(x, f_mean, f_var, y))
            
            kl_loss = tf.reduce_sum([l.prior_kl() / l.num_data for l in self.gp_layers_list])
            
            loss = -ell + kl_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "ell": ell, "kl": kl_loss}

def create_two_layer_GPAM_from_data(
    input_data: npt.NDArray[np.float64],
    num_inducing: int = 50,
    return_layers: bool = False,
    n_latent: int = 2,
) -> t.Any:
    """Generator function to create 2 layer GP given a dataset and its dimensions etc.

    Args:
        input_data (ndarray): dataset to be used to train model - this is where we get parameters off of.
        num_inducing (int, optional): Number of inducing points to use. Defaults to 50.
        return_layers (bool, optional): Set to true if individual layers are to be returned alongside model. Defaults to False.
        n_latent (int, optional): Dimension of latent space. Defaults to 2.

    Returns:
        model (NativeDeepGP): final model object.
    """

    num_data = input_data.shape[0]

    Z = input_data[np.random.choice(input_data.shape[0], size=num_inducing)]
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1] * input_data.shape[1])
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
        num_latent_gps=input_data.shape[1],
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflow.likelihoods.Gaussian(0.1)
    model = NativeDeepGP([gp_layer1, gp_layer2], likelihood_layer)
    model.compile(optimizer="adam")

    if return_layers:
        return model, gp_layer1, gp_layer2
    else:
        return model

def create_two_layer_GPAM_from_scratch(
    num_input: int,
    num_data: int = 1,
    Z: npt.NDArray[np.float64] | None = None,
    num_inducing: int = 50,
    return_layers: bool = False,
    n_latent: int = 2,
) -> t.Any:
    """Generator function to create two layer GPAM model natively.
    Args:
        num_input (int): Number of input dimensions.
        num_data (int, optional): Number of data points used for training, important to calculate loss properly. Defaults to 1.
        Z (ndarray, optional): Array of inducing locations. Defaults to None - will be generated at random.
        num_inducing (int, optional): Number of inducing points. Defaults to 50.
        return_layers (bool, optional): Set to true if individual layers are to be returned alongside model. Defaults to False.
        n_latent (int, optional): Dimension of latent space. Defaults to 2.

    Returns:
        model (NativeDeepGP): final model object.
    """

    if Z is not None:
        pass
    else:
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
    else:
        return model

def model_inference(
    data: npt.NDArray[np.float64], encoder: t.Any, batch_size: int = 20000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Utility function for batched model inference to reduce memory usage.

    Args:
        data (ndarray): Data to be used for inference.
        encoder (model layer): Encoding layer(s) from GPAM model to use for inference.
        batch_size (int, optional): Size of batch to use - this depends on memory to be used. Defaults to 20000.

    Returns:
        latents (tuple of 2 ndarrays): Mean and variance of latent distributions for every data point.
    """
    import tqdm

    max_iter = len(data) / batch_size
    means = []
    vars = []
    for i in tqdm.tqdm(range(int(max_iter) + 1)):
        if max_iter - i < 0:
            res = encoder(data[batch_size * i :])
        else:
            res = encoder(data[batch_size * i : batch_size * (i + 1)])

        means.append(res.mean())
        vars.append(res.variance())

    return np.concatenate(means, axis=0), np.concatenate(vars, axis=0)

