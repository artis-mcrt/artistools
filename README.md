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

## Use from Python

artistools is mainly a set of commands, but a script or a notebook can call the same functions. `import artistools as at` gives a small set of names at the top level. Each package holds the other names, e.g. `at.spectra` and `at.inputmodel`.

```python
from pathlib import Path

import artistools as at
import polars as pl

modelpath = Path("mymodel")
dfmodel, modelmeta = at.get_modeldata(modelpath)
times_days = at.get_timestep_times(modelpath, loc="mid")
dftemperature = at.scan_estimators(modelpath).filter(pl.col("timestep") == 40).select("modelgridindex", "Te").collect()
dfspectrum = at.spectra.get_spectra(modelpath, timestepmin=40, timestepmax=45)[-1].collect()
```

A function with the prefix `scan_` returns a polars LazyFrame, or one LazyFrame for each direction bin, and reads nothing until `.collect()`. A function with the prefix `read_` reads at once. The direction bin -1 is the average over all directions.

### Names at the top level

| Name | Purpose |
| --- | --- |
| `at.get_modeldata` | Read `model.txt`. Return the cells as a LazyFrame, and the model parameters as a dictionary. |
| `at.add_derived_cols_to_modeldata` | Add columns that follow from the model file, e.g. the volume and the mass of each cell. |
| `at.get_inputparams` | Return the parameters of `input.txt`. |
| `at.get_model_name` | Return the name of the model that holds a path. |
| `at.get_nprocs` | Return the number of MPI processes of the run. |
| `at.get_timestep_times` | Return the time in days of each timestep (`loc` is `"start"`, `"mid"`, `"end"`, or `"delta"`). |
| `at.get_timestep_of_timedays` | Return the timestep that holds a time in days. |
| `at.scan_estimators` | Read the estimators of a full run as a LazyFrame, with a row for each timestep and cell. |
| `at.read_estimators` | Read the estimators of a few cells into a dictionary. This is slow for many cells. |
| `at.get_deposition` | Return `deposition.out` as a LazyFrame. |
| `at.get_ionstring` | Return a text such as `Fe II` for an atomic number and an ion stage. |
| `at.get_ion_tuple` | Return `(26, 2)` for a text such as `FeII`, `Fe II`, or `26_2`. |
| `at.get_elsymbol` | Return the element symbol of an atomic number. |
| `at.get_atomic_number` | Return the atomic number of an element symbol. |
| `at.get_z_a_nucname` | Return the atomic number and the mass number of a text such as `Pb208`. |
| `at.decode_roman_numeral` | Return the integer of a Roman numeral. |
| `at.get_path` | Return a known path by name, e.g. the package folder. |
| `at.firstexisting` | Return the first file of a list that exists, with a compressed copy as an alternative. |
| `at.zopen` | Open a file, or its `.zst`, `.gz`, or `.xz` copy. |
| `at.set_mpl_style` | Apply the matplotlib style of artistools. |

### Main readers in the packages

| Name | Purpose |
| --- | --- |
| `at.spectra.get_spectra` | Return the spectrum of each direction bin, as an average over a range of timesteps. |
| `at.spectra.get_from_packets` | Return a spectrum from the packets files, for a range of arrival times. |
| `at.lightcurve.scan_lightcurve` | Read a light curve file. Return one LazyFrame for each direction bin. |
| `at.lightcurve.get_from_packets` | Return the luminosity against time from the packets files. |
| `at.packets.get_packets` | Return the number of ranks and a LazyFrame of the packets of a run. |
| `at.nltepops.read_nltepops` | Read the NLTE populations of a timestep and of one or more cells. |
| `at.atomic.get_levels` | Return the energy levels of each ion, with the transitions as an option. |
| `at.inputmodel.save_modeldata` | Write `model.txt` from a dataframe of the cells. |

### The command of a package

A package that has a plot command gives it as `plot`, e.g. `at.spectra.plot`, `at.lightcurve.plot`, `at.estimators.plot`, `at.nltepops.plot`, `at.packets.plot`, `at.nonthermal.plot`, and `at.gsinetwork.plot`. This function is the command itself. Each keyword is one command-line argument:

```python
at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", outputfile="spectrum.pdf")
```

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)
