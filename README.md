# artistools

[![DOI](https://zenodo.org/badge/53433932.svg)](https://zenodo.org/badge/latestdoi/53433932)
[![PyPI - Version](https://img.shields.io/pypi/v/artistools)](https://pypi.org/project/artistools)
[![License](https://img.shields.io/github/license/artis-mcrt/artistools)](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt)

[![Supported Python versions](https://img.shields.io/pypi/pyversions/artistools)](https://pypi.org/project/artistools/)
[![Installation and pytest](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml/badge.svg)](https://github.com/artis-mcrt/artistools/actions/workflows/pytest.yml)

Artistools is collection of plotting, analysis, and file format conversion tools for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code.

## Installation
Requires Python >= 3.13. A free-threaded Python requires version 3.15 or later, e.g. Python 3.15t.

The artistools command can be invoked with `uvx artistools` (after installing [uv](https://docs.astral.sh/uv/getting-started/installation/)), installed globally with `uv tool install artistools`, or installed into your environment with `pip install artistools`.

## Development (editable installation)
For development, you will need [a rust compiler](https://www.rust-lang.org/tools/install) and a clone of the repository:
```sh
git clone https://github.com/artis-mcrt/artistools.git
cd artistools
```

To make a [uv](https://docs.astral.sh/uv/getting-started/installation/) virtual environment in the clone, with the development tools, run:
```sh
uv sync --all-extras
uv run -- prek install
```

`uv run -- artistools` then runs the command of the clone. To make the artistools command available outside the clone, also run `uv tool install --editable .[extras]`.

Alternatively, to avoid uv and install into the system environment with pip:
```sh
pip install --group dev --editable .[extras]
prek install
```

To enable tab completion, run `artistools completions`. The command detects your shell and gives the
steps for it.

## Citing artistools

If you use artistools for a paper or a presentation, please cite it. For details, see [https://zenodo.org/badge/latestdoi/53433932](https://zenodo.org/badge/latestdoi/53433932).

## Usage
Run "artistools" (or the short alias "at") at the command-line to get a full list of subcommands, and "artistools --version" to check the installed version. Some common commands are:
- artistools plotspectra (alias: at spec)
- artistools plotlightcurves (alias: at lc)
- artistools plotestimators (alias: at estimators)
- artistools plotnltepops
- artistools describe (a summary of a model: its cells, timesteps, output files, and ions)
- artistools inputmodel describe

Use the -h option to get a list of command-line arguments for each subcommand. Set `ARTISTOOLS_TRACEBACK=1` to get the full traceback of an error. Run a command in a model folder, or give the path of the model folder. A command that reads a model takes the model path with `--modelpath`, e.g. `artistools timesteps --modelpath mymodel`. Most of these commands also take the model path as the last positional argument, e.g. `artistools plotspectra -t 300 mymodel`.

### The plotted data as text

Each plot command gives the plotted data as a table with a header row and a space between the values. `--print-data` prints the table to the standard output. The progress messages go to the standard error, thus a pipe or a redirect gets the table alone. `--output-format json` or `--output-format csv` gives the table in that format. `--write-data` writes the same table to a .txt file beside the figure. A value that is not available is `nan`, and a value or a column name that contains a space has quotes. A written file starts with comment lines that give the artistools version, the time, the working folder, and the command. Thus the file tells how the command made each value. Read it with `pl.read_csv(path, separator=" ", comment_prefix="#", null_values="nan")`, or with `comment="#"` in pandas. For example, this command writes the electron temperature against the velocity at 300 days to a file:

```sh
artistools plotestimators Te -t 300 mymodel --print-data > te.txt
```

A plot can give more than one table, because a series with its own grid needs its own table. Examples are a reference spectrum in `<figure>_references.txt`, a model with a different wavelength grid in `<figure>_grid2.txt`, and the statistics and the curves of a residual panel in `<figure>_residuals.txt` and `<figure>_residualcurves.txt`. On the standard output, an empty line separates two tables. With `--output-format json`, the standard output holds one JSON object, with the rows of each table under its name, e.g. `data` and `residuals`. With `--write-data`, `--output-format` also gives the format and the suffix of each file, e.g. `<figure>.json`.

### A model on a different host

These commands accept a model path of the form `host:path`, where `host` is a name that ssh knows:

- plotestimators and its `--interactive` viewer;
- plotlightcurves and its `--interactive` viewer;
- plotspectra and its `--interactive` viewer;
- `at.scan_estimators` in a Python script.

```sh
artistools plotspectra -t 300 vae26:~/scratch/mymodel
```

The rule of rsync decides whether a path is remote: a colon before the first slash makes a remote path. A local folder with such a colon in its name needs "./" at the start, e.g. `./model:2`.

artistools starts the server on that host through ssh, and it prints the command and the reason for it. A release of artistools starts `uvx artistools@<version> server`, with the versions of the local artistools and polars. Thus the host needs only [uv](https://docs.astral.sh/uv/).

The server and the client must have the same server protocol, which is a number. It changes when a reader changes its arguments or its results, or when the format of a message changes. A clone with the server protocol of the release also starts the release. A clone with a different server protocol starts its git commit with uvx. The host then also needs git and Rust. The host cannot get a commit that is only in the clone, thus such a commit must have the server protocol of the release.

The server protocol does not change when the code of a reader changes. Thus a release on the host can give a different result from the clone. If the package files of the clone differ from the release, artistools prints a warning with the number of these files and the name of one file. The warning also gives the command that starts the git commit on the host.

The command sets `POLARS_MAX_THREADS=16`, because a login node can have hundreds of cores, and other users share them. The server reads the files and makes the data of each plot, e.g. it puts the packets into bins. Only the data to draw comes back, and the parquet caches stay beside the data on the host. A script sends the filter and the columns of each query to the host, and only the rows of the query come back.

An option that reads its files on the local host stops with an error. To start the server with a different command on the host, e.g. an artistools in a clone, set `ARTISTOOLS_REMOTE_COMMAND`:

```sh
export ARTISTOOLS_REMOTE_COMMAND='POLARS_MAX_THREADS=16 ~/artistools/.venv/bin/artistools server'
```

The environment of such a command must have the same version of polars as the client. The filter of a query goes to the host as a polars expression, and the format of an expression changes between two versions. If the versions differ, the client stops with an error that gives the uvx option.

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

A function with the prefix `scan_` returns a polars LazyFrame, or one LazyFrame for each direction bin, and reads nothing until `.collect()`. A function with the prefix `read_` reads at once. A function with the prefix `get_` can do either, and the tables below say which one returns a LazyFrame. The direction bin -1 is the average over all directions.

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
| `at.get_deposition` | Return `deposition.out` as a LazyFrame. |
| `at.get_composition_data` | Return the elements and the ion stages of `compositiondata.txt` as a DataFrame. |
| `at.get_ionstring` | Return a text such as `Fe II` for an atomic number and an ion stage. |
| `at.get_ion_tuple` | Return `(26, 2)` for a text such as `FeII`, `Fe II`, or `26_2`. |
| `at.get_elsymbol` | Return the element symbol of an atomic number. |
| `at.get_atomic_number` | Return the atomic number of an element symbol. |
| `at.get_z_a_nucname` | Return the atomic number and the mass number of a text such as `Pb208`. |
| `at.decode_roman_numeral` | Return the integer of a Roman numeral. |
| `at.get_path` | Return a known path by name, e.g. the package folder. |
| `at.firstexisting` | Return the first file of a list that exists, with a compressed copy as an alternative. |
| `at.zopen` | Open a file, or its `.zst`, `.gz`, or `.xz` copy. |

### Main readers in the packages

| Name | Purpose |
| --- | --- |
| `at.estimators.read_estimators` | Read the estimators of a few cells into a dictionary. This is slow for many cells, thus prefer `at.scan_estimators`. |
| `at.spectra.get_spectra` | Return a dictionary with a LazyFrame of the spectrum of each direction bin, as an average over a range of timesteps. |
| `at.spectra.get_from_packets` | Return a dictionary with a LazyFrame of the spectrum of each direction bin from the packets files, for a range of arrival times. |
| `at.lightcurve.scan_lightcurve` | Read a light curve file. Return one LazyFrame for each direction bin. |
| `at.lightcurve.get_from_packets` | Return a dictionary with a LazyFrame of the luminosity against time of each direction bin, from the packets files. |
| `at.packets.get_packets` | Return the number of ranks and a LazyFrame of the packets of a run. |
| `at.nltepops.read_nltepops` | Read the NLTE populations of a timestep and of one or more cells. |
| `at.atomic.get_levels` | Return the energy levels of each ion, with the transitions as an option. |
| `at.inputmodel.save_modeldata` | Write `model.txt.zst` from a dataframe of the cells. |
| `at.plottools.set_mpl_style` | Apply the matplotlib style of artistools. |

### The command of a package

A package that has a plot command gives it as `plot`, e.g. `at.spectra.plot`, `at.lightcurve.plot`, `at.estimators.plot`, `at.nltepops.plot`, `at.packets.plot`, `at.nonthermal.plot`, and `at.gsinetwork.plot`. This function is the command itself. Each keyword is one command-line argument, and `argsraw` gives more arguments as text, e.g. `argsraw=["-t", "300"]`:

```python
at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", outputfile="spectrum.pdf")
```

## Use with an LLM agent

artistools contains a skill for a large language model (LLM) agent, e.g. Claude Code. The skill tells the agent how to use artistools to answer a question about a model. An example is "What is the range of Te at 10 days?". This command installs the skill for Claude Code:

```sh
artistools installskill
```

The command makes a link in `~/.claude/skills` to the skill in the installed package. Thus an update of artistools also updates the skill. If the environment variable `CLAUDE_CONFIG_DIR` has a value, the command uses `$CLAUDE_CONFIG_DIR/skills`. If artistools runs from the uv cache, e.g. with `uvx`, the command stops and makes no link, because uv can delete its cache. In that case, install artistools with `uv tool install artistools`, or give `--copy`.

For Codex, Cursor, or OpenCode, give `--layout agents`. The command then makes the link in `~/.agents/skills`. For a different agent, give its skills folder with `--skillsdir`.

Start a new session of the agent after the installation. Then ask a question about a model. In Claude Code, you can also type `/artis-data`. In Codex, you can also type `$artis-data`.

## Example output

![Emission plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-emission.png)
![NLTE plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-nlte-Ni.png)
![Estimator plot](https://github.com/artis-mcrt/artistools/raw/main/images/fig-estimators.png)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/artis-mcrt/artistools/blob/main/LICENSE.txt) for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)
