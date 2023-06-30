# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

[![DOI](https://zenodo.org/badge/53433932.svg)](https://zenodo.org/badge/latestdoi/53433932)
![GitHub Build and test status](https://github.com/artis-mcrt/artistools/workflows/Build%20and%20test/badge.svg)
[![codecov](https://codecov.io/gh/artis-mcrt/artistools/branch/main/graph/badge.svg?token=XFlarJqeZd)](https://codecov.io/gh/artis-mcrt/artistools)

## Installation
Requires Python >= 3.9

artistools can be installed from PyPI using `pip install artistools'. For development, clone the repository and make an editable install:
```sh
git clone https://github.com/artis-mcrt/artistools.git
cd artistools
python3 -m pip install -e .
pre-commit install
```

## Usage
Type "artistools" at the command-line to get a full list of commands. The most frequently used commands are:
- plotartisestimators
- plotartislightcurve
- plotartisnltepops
- plotartisnonthermal
- plotartisradfield
- plotartisspectrum

Use the -h option to get a list of command-line arguments for each command. Most of these commands would usually be run from within an ARTIS simulation folder.

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)


## Citing Artistools

If you make use of artistools please cite it. For details, see [https://zenodo.org/badge/latestdoi/53433932](https://zenodo.org/badge/latestdoi/53433932).