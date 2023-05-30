# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

![GitHub Build and test status](https://github.com/artis-mcrt/artistools/workflows/Build%20and%20test/badge.svg)
[![codecov](https://codecov.io/gh/artis-mcrt/artistools/branch/main/graph/badge.svg?token=XFlarJqeZd)](https://codecov.io/gh/artis-mcrt/artistools)
[![CodeFactor](https://www.codefactor.io/repository/github/artis-mcrt/artistools/badge)](https://www.codefactor.io/repository/github/artis-mcrt/artistools)

## Installation
Requires Python >= 3.9

First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artistools.git
```
Then from within the repository directory run:
```sh
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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7942849.svg)](https://doi.org/10.5281/zenodo.7942849)

There is a Zenodo entry for Artistools. If you make use of Artistools please cite this record.

Bibtex:
```bibtex

@software{luke_shingles_2023_7942849,
  author       = {Luke Shingles and
                  Christine E. Collins and
                  Alexander Holas and
                  Fionntan Callan and
                  Stuart Sim},
  title        = {artis-mcrt/artistools},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v2023.5.16.3},
  doi          = {10.5281/zenodo.6418832},
  url          = {https://doi.org/10.5281/zenodo.7942849}
}
```