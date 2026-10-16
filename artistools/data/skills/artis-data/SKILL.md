---
name: artis-data
description: Answer a question about the data of an ARTIS model. Examples are the range of the electron temperature (Te) at 10 days and the density of a cell. Other examples are the deposition rate, the peak of the light curve, and the flux of a spectrum. Use it for the estimators, the cells, the deposition, the light curves, the spectra, or the packets of a model. It reads the data with Python calls into artistools.
---

# Answer a question about the data of an ARTIS model

Read the data only through artistools. Do not open the parquet files of a model folder directly. They are caches,
and artistools checks their version and their age against the text files. A direct read can give old data with no
error.

## Run Python with artistools

Use the Python of the environment that holds the `artistools` command:

```sh
ATPY="$(dirname "$(command -v artistools)")/python"
"$ATPY" - <<'EOF'
import artistools as at
...
EOF
```

If no `artistools` command exists, ask the user for the environment. Run the script in the model folder, or give
the model path in the script. A remote model path has the form `host:path`, and artistools reads it on that host.
Call `help()` on a function to get all its arguments.

## Terms

- A **model** is one ARTIS simulation. A **model path** names its model folder.
- A **run** is one job of ARTIS. A run on a cluster writes its files into a run folder, e.g.
  `12411848.slurm/`. artistools finds and joins the run folders. Give the model path. Do not give a run folder.
- A **cell** has the index `modelgridindex`.
- A **time step** is an index. `at.get_timestep_times(modelpath, loc="mid")` gives the mid time of each time step
  in days. `loc` can also be `"start"`, `"end"`, or `"delta"`. `at.get_timestep_of_timedays(modelpath, 10)` gives
  the time step that holds 10 days.
- A **direction bin** is a range of viewing directions. A reader that gives a dict of LazyFrames uses the direction bin
  as the key. The key -1 gives the average over all directions.

## Rules for all data

- Some readers take a time in days, and other readers take a time step. Give the time step and its mid time with
  the answer. The mid time can be different from the time that the user gave.
- Most columns use cgs units (centimetre, gram, and second). Some columns give the unit at the end of the name, e.g.
  `_Lsun` (solar luminosity), `_days`, and `_angstroms`. Use the constants in `artistools.constants` to change a unit.
- Call `.collect()` one time at the end of a query. Do not collect inside a loop over the cells or the time steps.
- Use polars expressions. Give each result of an aggregation its own name with `.alias()`.
- Print the column names of a LazyFrame with `.collect_schema().names()` before you select a column. The columns can
  change with the version of ARTIS.

## Estimators: the values of each cell at each time step

`at.scan_estimators(modelpath, timestep=..., modelgridindex=..., join_modeldata=False)` gives a LazyFrame with one row
for each time step and cell. If you give `timestep` or `modelgridindex`, artistools reads less data and is faster. The
main columns are:

- `Te` (electron temperature, K), `TR` (radiation temperature, K), `TJ` (J temperature, K), and `W` (dilution factor);
- `nne` (free electron density, cm^-3);
- `nnelement_<El>`, `nnion_<El>_<ion stage>`, and `nniso_<nuclide>` (number densities, cm^-3);
- `heating_<process>`, `cooling_<process>`, and `deposition_<particle>` (rates, erg/s/cm^3).

To list the columns of a model and their units, run `artistools plotestimators --listvariables <modelpath>`.
`join_modeldata=True` adds these columns:

- `tmid_days` (the mid time of the time step);
- `rho` (density at that time, g/cm^3) and `volume` (cm^3);
- `mass_g` (the mass of the cell);
- `init_<column>` (a value of `model.txt` at the start time of the model).

Possible errors:

- The estimators hold only the cells that contain matter. Thus the number of rows can be less than the number of
  cells in `model.txt`. Give the number of cells that you used.
- ARTIS limits Te and the other temperatures to the range from `MINTEMP` to `MAXTEMP`. A temperature at a limit is
  not a physical value. Get the limits with `at.misc.get_artis_option(modelpath, "MINTEMP")`. If that gives `None`,
  read `artisoptions.h` in the model folder. Give the number of cells at each limit.
- If a `heating_`, `cooling_`, or `deposition_` column has no value, the value is 0. It does not mean "no data".
- A statistic without a weight gives the same weight to a small outer cell and to a large inner cell. If the user
  asks for a typical value, also give a mean with the weight `mass_g`.

Example: the range of a column at a time.

```python
import artistools as at
import polars as pl

modelpath = "."
column = "Te"
timestep = at.get_timestep_of_timedays(modelpath, 10)

dfcells = (
    at
    .scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
    .select("modelgridindex", "tmid_days", "mass_g", column)
    .collect()
)
print(
    dfcells.select(
        pl.len().alias("cellcount"),
        pl.col("tmid_days").first(),
        pl.col(column).min().alias("min"),
        pl.col(column).quantile(0.05).alias("q05"),
        pl.col(column).median().alias("median"),
        pl.col(column).quantile(0.95).alias("q95"),
        pl.col(column).max().alias("max"),
        ((pl.col(column) * pl.col("mass_g")).sum() / pl.col("mass_g").sum()).alias("massweightedmean"),
    )
)
for limitname in ("MINTEMP", "MAXTEMP"):
    limit = at.misc.get_artis_option(modelpath, limitname)
    if limit is None:
        print(f"artis/artisoptions.h gives no {limitname}. Read artisoptions.h in the model folder.")
    else:
        atlimitcount = dfcells.filter(pl.col(column) == float(limit)).height
        print(f"{limitname} = {limit}: {atlimitcount} cells")
```

## Model data: the cells of `model.txt`

`at.get_modeldata(modelpath, get_elemabundances=False)` gives a tuple. The first item is a LazyFrame with one row for
each cell. The second item is a dict of the model parameters. Each value of `model.txt` is at the start time of the
model, `t_model_init_days`. The columns are:

- `inputcellid` and `modelgridindex` (the index of the cell in the estimators);
- `rho` (3D and 2D, g/cm^3) or `logrho` (1D, the log10 of the density);
- `pos_x_min`, `pos_y_min`, and `pos_z_min` (3D, the corner of the cell, cm), or `vel_r_max_kmps` (1D, km/s);
- `X_<nuclide>` (the mass fractions of the radioactive nuclides, e.g. `X_Ni56`) and `X_Fegroup`;
- `X_<El>` (the mass fractions of the elements). Give `get_elemabundances=True` to read these columns. They come from
  `abundances.txt`.

`at.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)` calculates more columns from these values. Examples are
`rho` and `logrho` for each number of dimensions, `mass_g`, `volume`, and the velocities, e.g. `vel_r_mid` (cm/s) and
`vel_r_mid_on_c`. Call it before you filter the rows. A 1D cell takes its inner velocity from the row before it.

The main parameters are `t_model_init_days`, `vmax_cmps` (cm/s), `dimensions`, and `npts_model` (the number of cells).

Possible errors:

- A cell with `rho` = 0 is empty. It has no estimators. Do not include it in a statistic of the cells that contain
  matter.
- The density at a time t is `rho * (t_model_init_days / t) ** 3`, because the ejecta expand homologously.
  `scan_estimators` with `join_modeldata=True` gives this density as `rho` at each time step.

Example: the total mass and the mass of Ni56.

```python
import artistools as at
import polars as pl
from artistools.constants import Msun_to_g

modelpath = "."
dfmodel, modelmeta = at.get_modeldata(modelpath)
dfmodel = at.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
print(
    dfmodel.select(
        (pl.col("rho") > 0).sum().alias("cellswithmatter"),
        (pl.col("mass_g").sum() / Msun_to_g).alias("mass_msun"),
        ((pl.col("mass_g") * pl.col("X_Ni56")).sum() / Msun_to_g).alias("mass_Ni56_msun"),
    ).collect()
)
print("t_model_init_days:", modelmeta["t_model_init_days"], "dimensions:", modelmeta["dimensions"])
```

## Deposition: the energy that the ejecta absorb

`at.get_deposition(modelpath)` reads `deposition.out`. It gives a LazyFrame with one row for each time step. The
LazyFrame always has the columns `timestep` and `tmid_days`. The other columns come from the header of the file, and
they change with the version of ARTIS. Examples are:

- `total_dep_Lsun`, `gammadep_Lsun`, `positrondep_Lsun`, `elecdep_Lsun`, and `alphadep_Lsun` (the deposition rate of
  each type of particle, in solar luminosities);
- `eps_gamma_Lsun`, `eps_elec_Lsun`, and `eps_alpha_Lsun` (the rate at which the decays release
  energy, in solar luminosities). A name with `_ana` gives the analytic value;
- `eps_erg/s/g` and `Qdot_ana_erg/s/g` (the specific rates, erg/s/g).

Possible errors:

- The deposition rates are in solar luminosities and not in erg/s. Multiply by `Lsun_to_erg_per_s` from
  `artistools.constants`.
- `deposition.out` gives the sum over the ejecta. For the deposition rate of each cell, use the `deposition_`
  columns of the estimators.

Example: the deposition at a time, and the fraction of the gamma-ray energy that the ejecta absorb.

```python
import artistools as at
import polars as pl
from artistools.constants import Lsun_to_erg_per_s

modelpath = "."
timestep = at.get_timestep_of_timedays(modelpath, 10)
dfdep = at.get_deposition(modelpath).filter(pl.col("timestep") == timestep).collect()
print(
    dfdep.select(
        "timestep",
        "tmid_days",
        (pl.col("total_dep_Lsun") * Lsun_to_erg_per_s).alias("total_dep_erg/s"),
        (pl.col("gammadep_Lsun") / pl.col("eps_gamma_Lsun")).alias("gamma_deposition_fraction"),
    )
)
```

## Light curves

`at.lightcurve.find_lightcurve_file(modelpath)` gives the path of `light_curve.out`. Give `gamma=True` for
`gamma_light_curve.out`, or `directionresolved=True` for `light_curve_res.out`. `at.lightcurve.scan_lightcurve(path)`
reads the file. Give the same `directionresolved` value to both functions. The result is a dict with one LazyFrame for
each direction bin, with these columns:

- `time_days` (the mid time of the time step);
- `luminosity_erg/s` and `luminosity_Lsun` (the bolometric luminosity in the observer frame);
- `luminosity_cmf_erg/s` and `luminosity_cmf_Lsun` (the luminosity in the comoving frame);
- `mag` (the absolute bolometric magnitude).

`at.lightcurve.get_from_packets(modelpath, escape_type="TYPE_RPKT", timedaysmin=..., timedaysmax=...)` makes the light
curve from the packets files. Use it for a time range or a type of packet that the light curve files do not give. It is
slower than `scan_lightcurve`.

Possible errors:

- The luminosity is 0 at the early time steps, before packets escape. The magnitude of such a row is `inf`.
- The light curve of a direction bin has more Monte Carlo noise than the average over all directions.

Example: the time and the luminosity of the peak.

```python
import artistools as at
import polars as pl

modelpath = "."
lcpath = at.lightcurve.find_lightcurve_file(modelpath)
dflc = at.lightcurve.scan_lightcurve(lcpath)[-1].collect()
print(lcpath)
print(dflc.sort("luminosity_erg/s", descending=True).head(1).select("time_days", "luminosity_erg/s", "mag"))
```

## Spectra

`at.spectra.get_spectra(Path(modelpath), timestepmin, timestepmax, directionbins=[-1])` reads `spec.out`. It gives a
dict with one LazyFrame for each direction bin. Each spectrum is the mean over the time steps from `timestepmin` to
`timestepmax`, with the width of each time step as the weight. The columns are:

- `lambda_angstroms` and `f_lambda` (erg/s/cm^2/Å);
- `nu` (Hz) and `f_nu` (erg/s/cm^2/Hz).

The flux is at a distance of 1 Mpc. Give `gamma=True` for the gamma-ray spectrum. Give only the direction bins that you
need. For a bin of 0 or more, artistools reads the large file `spec_res.out`.

`at.spectra.get_from_packets(modelpath, timelowdays, timehighdays)` makes the spectrum from the packets files for a
time range in days. It is slower than `get_spectra`.

Possible errors:

- `get_spectra` takes time steps and not days. Change the days to time steps with `at.get_timestep_of_timedays`.
- Multiply the flux by `4 * math.pi * megaparsec_to_cm**2` to get a luminosity density. Integrate it over the
  wavelength or the frequency to get a luminosity. `megaparsec_to_cm` is in `artistools.constants`.

Example: the mean spectrum from 10 to 12 days, and the wavelength of its maximum.

```python
import math
from pathlib import Path

import artistools as at
import polars as pl
from artistools.constants import megaparsec_to_cm

modelpath = "."
timestepmin = at.get_timestep_of_timedays(modelpath, 10)
timestepmax = at.get_timestep_of_timedays(modelpath, 12)
dfspec = at.spectra.get_spectra(Path(modelpath), timestepmin, timestepmax, directionbins=[-1])[-1].collect()
print(f"time steps {timestepmin} to {timestepmax}")
print(
    dfspec
    .filter(pl.col("lambda_angstroms").is_between(3000, 10000))
    .sort("f_lambda", descending=True)
    .head(1)
    .select("lambda_angstroms", "f_lambda")
)
luminosity = (
    dfspec.sort("nu").select(pl.col("f_nu") * pl.col("nu").diff()).sum().item() * 4 * math.pi * megaparsec_to_cm**2
)
print(f"luminosity in the spectrum: {luminosity:.3e} erg/s")
```

## Other data

- `at.packets.get_packets(modelpath)` gives `(number of ranks, LazyFrame of the packets)`. The packets files are large.
  Select the columns and filter the rows before `.collect()`.
- `at.get_inputparams(modelpath)` gives a dict of the parameters of `input.txt`.
- For a different reader, read the section "Use from Python" of the artistools `README.md`.

## Report

Give these items with the answer:

- the model path;
- the time step or the range of time steps, and the mid time of each time step in days;
- the reader and the columns that you used;
- the units of each value;
- for the cells, the number that you used and the cells that you did not use, with a reason for each;
- each assumption that you made, e.g. the choice of a time step, a direction bin, or a weight.
