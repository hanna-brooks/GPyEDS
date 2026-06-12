import numpy as np
import pandas as pd

from GPyEDS import utils


def test_split() -> None:
    word = "norm"
    r1, r2 = utils.split_at(word, "o", 0)


def test_get_img() -> None:
    mask = np.ones((10, 10))
    rand = np.random.rand(100)

    img = utils.get_img(rand, mask)


def test_stacking() -> None:
    l = [np.ones((10, 10)), np.ones((10, 10))]

    s = utils.list2stack(l)
    l2 = utils.stack2list(s)


def test_gauss_filter() -> None:
    mask = np.ones((10, 10))
    rand = np.random.rand(10, 10)

    f = utils.gaussian_filter(rand, mask)


def test_feature_norm() -> None:
    dummy = np.random.rand(100, 7)

    norm, params = utils.feature_normalisation(dummy, True)
    norm1, params1 = utils.feature_normalisation(dummy[:, 0], True)


def test_get_masks() -> None:
    dummy = np.random.randint(5, size=(100, 100))

    masks = utils.get_masks(dummy)


def test_build_conc() -> None:
    x = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, x)
    r = np.random.rand(100)
    df = pd.DataFrame(
        data={
            "X": xx.ravel().astype("int64"),
            "Y": yy.ravel().astype("int64"),
            "val": r,
        }
    )

    _ = utils.build_conc_map(df)


def test_decomp() -> None:
    data = np.random.randn(100, 4)
    r = utils.decompose(data)
    return None


def test_plot_decomp() -> None:
    data = np.random.randn(100, 4)
    r = utils.decompose(data, plot=True)
    return None
