# Installation

GPyEDS requires Python 3.12. We strongly recommend using [uv](https://github.com/astral-sh/uv) for the fastest installation and environment management.

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
