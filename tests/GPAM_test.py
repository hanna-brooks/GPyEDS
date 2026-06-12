import numpy as np
import tensorflow as tf

from GPyEDS import GPAM


def test_GPAM() -> None:
    dummy = np.ones((100, 7))

    m1 = GPAM.create_two_layer_GPAM_from_data(dummy)
    m2 = GPAM.create_two_layer_GPAM_from_scratch(7, 100)

    _ = GPAM.model_inference(dummy, m1.gp_layers_list[0])


def test_elbo_improves_over_training() -> None:
    """Train a Deep GP on non-degenerate data and verify the ELBO improves."""
    rng = np.random.default_rng(42)
    n, d = 60, 3
    data = rng.standard_normal((n, d)).astype(np.float64)

    model = GPAM.create_two_layer_GPAM_from_data(data, num_inducing=20)

    # Evaluate initial loss
    initial_metrics = model.evaluate(
        {"inputs": data, "targets": data},
        batch_size=n,
        verbose=0,
        return_dict=True,
    )
    initial_loss = float(initial_metrics["loss"])

    # Train for a few epochs
    model.fit(
        {"inputs": data, "targets": data},
        epochs=20,
        batch_size=20,
        verbose=0,
    )

    # Evaluate final loss
    final_metrics = model.evaluate(
        {"inputs": data, "targets": data},
        batch_size=n,
        verbose=0,
        return_dict=True,
    )
    final_loss = float(final_metrics["loss"])

    # ELBO should improve (loss should decrease)
    assert final_loss < initial_loss, (
        f"ELBO did not improve: initial_loss={initial_loss:.4f}, "
        f"final_loss={final_loss:.4f}"
    )


def test_model_output_shapes() -> None:
    """Verify the output distribution has expected shapes."""
    rng = np.random.default_rng(123)
    n, d = 40, 5
    data = rng.standard_normal((n, d)).astype(np.float64)

    model = GPAM.create_two_layer_GPAM_from_data(data, num_inducing=15)

    x = tf.cast(data, tf.float64)
    f_dist = model(x, training=False)

    assert f_dist.mean().shape == (n, d)
    assert f_dist.variance().shape == (n, d)
