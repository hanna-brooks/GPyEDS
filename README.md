# GPyEDS
[![codecov](https://codecov.io/gh/norberttoth398/GPyEDS/graph/badge.svg?token=I4Ukc39QBJ)](https://codecov.io/gh/norberttoth398/GPyEDS) [![Documentation Status](https://readthedocs.org/projects/gpyeds/badge/?version=latest)](https://gpyeds.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13837097.svg)](https://doi.org/10.5281/zenodo.13837097)
GPyEDS is a Python toolbox for the analysis of Energy Dispersive X-ray Spectroscopy (EDS) or other hyperspectral data sets. It leverages Gaussian Processes and Neural Network Autoencoders to provide robust spatial-spectral segmentation, preprocessing, and calibration workflows.

## Installation

GPyEDS requires Python 3.12. 

> [!NOTE]
> **Python Version Restriction:** We are currently capped at Python 3.12 because `gpflow` inherently requires `tensorflow-macos` on Apple Silicon. Google stopped publishing `tensorflow-macos` after version 2.16 (which maxes out at Python 3.12) since Apple Silicon support was merged directly into the mainline `tensorflow` package in version 2.17. Until `gpflow` drops this legacy requirement, we must stay on Python 3.12 for universal cross-platform compatibility.

We strongly recommend using [uv](https://github.com/astral-sh/uv) for the fastest installation and environment management.

### Using uv (Recommended)

To install GPyEDS and create a virtual environment simultaneously:
```bash
uv pip install git+https://github.com/norberttoth398/GPyEDS.git
```

If you are developing or contributing to GPyEDS, you can clone the repository and sync the environment:
```bash
git clone https://github.com/norberttoth398/GPyEDS.git
cd GPyEDS
uv sync --all-extras
```

### Using pip

If you are not using `uv`, you can install GPyEDS using standard `pip` (ensure you are inside a Python 3.12 virtual environment):
```bash
pip install git+https://github.com/norberttoth398/GPyEDS.git
```
