import typing as t

import numpy as np
import numpy.typing as npt
import scipy.signal


def gkern(l: int = 5, sig: float = 1.0) -> npt.NDArray[np.float64]:
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)  # type: ignore


def linear_filter(
    map: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    range_: int,
    type: str = "mean",
    sigma: float | None = None,
) -> npt.NDArray[np.float64]:
    if type == "mean":
        n = (2 * range_ + 1) ** 2
        filter: t.Any = [
            [1 / n for _ in range(2 * range_ + 1)] for k in range(2 * range_ + 1)
        ]

    elif type == "gaussian":
        n = 2 * range_ + 1
        if sigma is None:
            sigma = range_
        filter = gkern(n, sigma)
    else:
        raise ValueError("Cannot identify method.")

    pmap = np.pad(map, (range_, range_), mode="constant")
    pmask = np.pad(mask, (range_, range_), mode="constant")

    meanres = scipy.signal.convolve2d(
        np.multiply(pmap, pmask), np.asarray(filter), boundary="symm", mode="same"
    )
    maskres = scipy.signal.convolve2d(
        pmask, np.asarray(filter), boundary="symm", mode="same"
    )

    numerator = np.multiply(meanres, pmask)
    r = np.divide(numerator, maskres, out=np.zeros_like(numerator), where=maskres != 0)
    r[~pmask.astype("bool")] = np.nan
    return r[range_:-range_, range_:-range_]


def median_filter(
    map: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_], range_: int
) -> npt.NDArray[np.float64]:
    pmap = np.pad(map, (range_, range_), mode="constant")
    pmask = np.pad(mask, (range_, range_), mode="constant")

    medianres = np.zeros_like(map)
    k = 2 * range_ + 1
    n, m = map.shape
    for i in range(n):
        for j in range(m):
            region = pmap[i : i + k, j : j + k][
                pmask[i : i + k, j : j + k].astype("bool")
            ]
            if len(region > 0):
                medianres[i, j] += np.median(region)

    medianres[~mask] = np.nan

    return medianres
