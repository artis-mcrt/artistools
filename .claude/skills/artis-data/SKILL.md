---
name: artis-data
description: Answer a question about the data of an ARTIS model. Examples are the range of the electron temperature (Te) at 10 days, the ion densities of a cell, and the luminosity at a time. Use it for the estimators, the cells, the time steps, the packets, the light curve, or the spectra of a model. It reads the data with Python calls into artistools.
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

## Terms

- A **model** is one ARTIS simulation. A **model path** names its model folder.
- A **run** is one job of ARTIS. A run on a cluster writes its files into a run folder, e.g.
  `12411848.slurm/`. artistools finds and joins the run folders. Give the model path. Do not give a run folder.
- A **time step** is an index. `at.get_timestep_times(modelpath, loc="mid")` gives the mid time of each time step
  in days. `at.get_timestep_of_timedays(modelpath, 10)` gives the time step that holds 10 days.
- A **cell** has the index `modelgridindex`.

## Readers

| Name | Result |
| --- | --- |
| `at.scan_estimators(modelpath, timestep=..., modelgridindex=..., join_modeldata=False)` | LazyFrame, one row for each time step and cell |
| `at.get_modeldata(modelpath)` | `(LazyFrame of the cells of model.txt, dict of model parameters)` |
| `at.get_timestep_times(modelpath, loc="mid")` | list of times in days (`loc` is `"start"`, `"mid"`, `"end"`, or `"delta"`) |
| `at.get_timestep_of_timedays(modelpath, timedays)` | time step index |
| `at.get_inputparams(modelpath)` | dict of the parameters of `input.txt` |
| `at.get_deposition(modelpath)` | LazyFrame of `deposition.out` |
| `at.lightcurve.scan_lightcurve(filepath)` | dict of a LazyFrame for each direction bin (-1 is the angle average) |
| `at.spectra.get_spectra(modelpath, timestepmin, timestepmax)` | dict of a LazyFrame for each direction bin |
| `at.packets.get_packets(modelpath)` | `(number of ranks, LazyFrame of the packets)` |
| `at.misc.get_artis_option(modelpath, name)` | the value of an option in `artis/artisoptions.h` as text, or `None` |

For a different reader, read the section "Use from Python" of the artistools `README.md`. Call `help()` on a
function to get its arguments.

## Estimator columns

The columns use cgs units (centimetre, gram, and second). The main columns are:

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

## Possible errors

- The estimators hold only the cells that contain matter. Thus the number of rows can be less than the number of
  cells in `model.txt`. Give the number of cells that you used.
- ARTIS limits Te and the other temperatures to the range from `MINTEMP` to `MAXTEMP`. A temperature at a limit is
  not a physical value. Get the limits with `at.misc.get_artis_option(modelpath, "MINTEMP")`. If that gives `None`,
  read `artisoptions.h` in the model folder. Give the number of cells at each limit.
- If a `heating_`, `cooling_`, or `deposition_` column has no value, the value is 0. It does not mean "no data".
- A statistic without a weight gives the same weight to a small outer cell and to a large inner cell. If the user
  asks for a typical value, also give a mean with the weight `mass_g`.
- A time step holds a range of times. Thus its mid time can be different from the time that the user gave. Give the
  time step and its mid time.
- Call `.collect()` one time at the end of a query. Do not collect inside a loop over the cells or the time steps.
- Use polars expressions. Give each result of an aggregation its own name with `.alias()`.

## Example: the range of a column at a time

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

## Report

Give these items with the answer:

- the model path;
- the time step and its mid time in days;
- the number of cells that you used, and the cells that you did not use, with a reason for each;
- the units of each value;
- each assumption that you made, e.g. the choice of a time step or a weight.
