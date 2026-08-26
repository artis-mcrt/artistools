# artistools

[![DOI](https://zenodo.org/badge/53433932.svg)](https://zenodo.org/badge/latestdoi/53433932)
[![PyPI - Version](https://img.shields.io/pypi/v/artistools)](https://pypi.org/project/artistools)
[![License](https://img.shields.io/github/license/artis-mcrt/artistools)](https://github.com/artis-mcrt/artistools/blob/main/LICENSE)

[![Supported Python versions](https://img.shields.io/pypi/pyversions/artistools)](https://pypi.org/project/artistools/)
[![Installation and pytest](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml/badge.svg)](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/artis-mcrt/artistools/branch/main/graph/badge.svg?token=XFlarJqeZd)](https://codecov.io/gh/artis-mcrt/artistools)

Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

## Installation
Requires Python >= 3.13

The artistools command can be invoked with `uvx artistools` (after installing [uv](https://docs.astral.sh/uv/getting-started/installation/)), installed globally with `uv tool install artistools`, or installed into your environment with `pip install artistools`.

## Development (editable installation)
For development, you will need [a rust compiler](https://www.rust-lang.org/tools/install) and a clone of the repository:
```sh
git clone https://github.com/artis-mcrt/artistools.git
cd artistools
```

To make the artistools command available using an isolated [uv](https://docs.astral.sh/uv/getting-started/installation/) virtual environment, run:
```sh
uv tool install --editable .[extras]
prek install
```

Alternatively, to avoid uv and install into the system environment with pip:
```sh
pip install --group dev --editable .[extras]
prek install
```

To learn how to enable command-line autocompletions, run:
```sh
artistools completions
```

## Citing artistools

If you artistools for a paper or presentation, please cite it. For details, see [https://zenodo.org/badge/latestdoi/53433932](https://zenodo.org/badge/latestdoi/53433932).

## Usage
Run "artistools" (or the short alias "at") at the command-line to get a full list of subcommands, and "artistools --version" to check the installed version. Some common commands are:
- artistools plotspectra (alias: at spec)
- artistools plotlightcurve (alias: at lc)
- artistools plotestimators (alias: at estimators)
- artistools plotnltepops
- artistools inputmodel describe

Use the -h option to get a list of command-line arguments for each subcommand. Set `ARTISTOOLS_TRACEBACK=1` to get the full traceback of an error. Most of these commands should be run either within an ARTIS simulation folder or by passing the folder path as the last argument.

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)
