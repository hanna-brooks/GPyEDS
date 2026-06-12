import numpy as np

from GPyEDS import GPAM


def test_GPAM() -> None:
    dummy = np.ones((100, 7))

    m1 = GPAM.create_two_layer_GPAM_from_data(dummy)
    m2 = GPAM.create_two_layer_GPAM_from_scratch(7, 100)

    _ = GPAM.model_inference(dummy, m1.gp_layers_list[0])
