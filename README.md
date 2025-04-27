# artistools

[![DOI](https://zenodo.org/badge/53433932.svg)](https://zenodo.org/badge/latestdoi/53433932)
[![PyPI - Version](https://img.shields.io/pypi/v/artistools)](https://pypi.org/project/artistools)
[![License](https://img.shields.io/github/license/artis-mcrt/artistools)](https://github.com/artis-mcrt/artistools/blob/main/LICENSE)

[![Supported Python versions](https://img.shields.io/pypi/pyversions/artistools)](https://pypi.org/project/artistools/)
[![Installation and pytest](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml/badge.svg)](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/artis-mcrt/artistools/branch/main/graph/badge.svg?token=XFlarJqeZd)](https://codecov.io/gh/artis-mcrt/artistools)

Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

## Installation
Requires Python >= 3.11

The artistools command be invoked with uvx artistools or pipx artistools.

For development, you will need [the rust compiler](https://www.rust-lang.org/tools/install) and a clone of the repository:
```sh
git clone https://github.com/artis-mcrt/artistools.git
cd artistools
```

To create a [uv](https://docs.astral.sh/uv/getting-started/installation/) virtual environment with locked dependency versions, run:
```sh
uv sync --frozen
uv pip install --group dev --editable .[extras]
pre-commit install
```

The artistools command from the project environment can be available by activating the environment (source artistools/.venv/bin/activate) or by adding the following alias to your startup script:
```sh
alias artistools="~/PATH/TO/artistools/.venv/bin/artistools"
```

Alternatively, to avoid uv and install into the system environment with pip:
```sh
pip install --group dev --editable .[extras]
pre-commit install
```

To learn how to enable command-line autocompletions, run:
```sh
artistools completions
```

## Citing artistools

If you artistools for a paper or presentation, please cite it. For details, see [https://zenodo.org/badge/latestdoi/53433932](https://zenodo.org/badge/latestdoi/53433932).

## Usage
Run "artistools" at the command-line to get a full list of subcommands. Some common commands are:
- artistools plotspectra
- artistools plotlightcurve
- artistools plotestimators
- artistools plotnltepops
- artistools describeinputmodel

Use the -h option to get a list of command-line arguments for each subcommand. Most of these commands should be run either within an ARTIS simulation folder or by passing the folder path as the last argument.

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)
