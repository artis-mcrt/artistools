# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

[![DOI](https://zenodo.org/badge/53433932.svg)](https://zenodo.org/badge/latestdoi/53433932)
[![Installation and pytest](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml/badge.svg)](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/artis-mcrt/artistools/branch/main/graph/badge.svg?token=XFlarJqeZd)](https://codecov.io/gh/artis-mcrt/artistools)
![PyPI - Version](https://img.shields.io/pypi/v/artistools)

## Installation
Requires Python >= 3.10

The artistools command be invoked with uvx artistools or pipx artistools. For development (editable install), you will need [the rust compiler](https://www.rust-lang.org/tools/install). First, clone the repository:
```sh
git clone https://github.com/artis-mcrt/artistools.git
cd artistools
```

To use a [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) virtual environment with locked dependency versions run:
```sh
uv sync --frozen
uv pip install --editable .[dev]
pre-commit install
```

The artistools command will be available after activating the project environment (source artistools/.venv/bin/activate) or can made globally available by adding the following alias to your startup script:
```sh
alias artistools="uv run --frozen --project ~/PATH/TO/artistools -- artistools"
```

Alternatively, to avoid using uv and install into the system environment with pip:
```sh
python3 -m pip install --editable .[dev]
pre-commit install
```

## Usage
Type "artistools" at the command-line to get a full list of commands. The most frequently used commands are:
- artistools plotspectra
- artistools plotlightcurve
- artistools plotestimators
- artistools plotnltepops
- artistools describeinputmodel

Use the -h option to get a list of command-line arguments for each command. Most of these commands would usually be run from within an ARTIS simulation folder.

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)


## Citing Artistools

If you artistools for a paper or presentation, please cite it. For details, see [https://zenodo.org/badge/latestdoi/53433932](https://zenodo.org/badge/latestdoi/53433932).
