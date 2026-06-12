import typing as t

import numpy as np
import numpy.typing as npt

from GPyEDS import spatial_filters


def create_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    cmap = np.zeros((10, 10), dtype="float64")

    for i in range(10):
        cmap[:, i] = 8 - i - 0.5
        cmap[i, :] -= i / 4

    cmap[cmap > 3.5] = 3.5
    cmap[cmap < 0.51] = 0.5
    map = cmap.copy()
    cmap += np.random.randn(10, 10) / 5
    mask = map > 0.5
    return cmap, mask


def test_median() -> None:
    cmap, mask = create_data()
    res = spatial_filters.median_filter(cmap, mask, 1)


def test_mean() -> None:
    cmap, mask = create_data()
    res = spatial_filters.linear_filter(cmap, mask, 1)


def test_gaussian() -> None:
    cmap, mask = create_data()
    res = spatial_filters.linear_filter(cmap, mask, 1, type="gaussian")
