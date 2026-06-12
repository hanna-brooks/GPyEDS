import numpy as np

import GPyEDS


def test_mean_centre_norm():
    dummy = np.random.rand(100, 7)

    norm = GPyEDS.mean_centre(dummy)
