---
name: artis-data
description: Answer a question about the data of an ARTIS model. Examples are the range of the electron temperature (Te) at 10 days and the density of a cell. Other examples are the deposition rate, the peak of the light curve, and the flux of a spectrum. Use it for the estimators, the cells, the ionisation, the deposition, the light curves, the spectra, and the packets. Also use it for the radiation field, the heating and the cooling processes, or another internal quantity of a model. It reads the data with Python calls into artistools.
---

# Answer a question about the data of an ARTIS model

Read the data only through artistools. Do not open the parquet files of a model folder directly. They are caches,
and artistools checks their version and their age against the text files. A direct read can give old data with no
error.

This file gives the rules and the main readers. The file `quantities.md` in the folder of this skill lists each file of
an ARTIS model, what it holds, and its reader. Read it to explain the behaviour of a model. Examples are:

- the radiation field;
- the heating and the cooling processes;
- the emission of each ion in a spectrum;
- the last interaction of each packet;
- the atomic data.

The file `physics.md` in the same folder explains the physics of ARTIS and gives the papers that describe it. Read it
to explain how ARTIS calculates a value. Examples are:

- the temperatures Te and TR, and the dilution factor W;
- the ionisation;
- the population of a superlevel.

The file `recipes.md` gives tested examples for questions about why a model behaves as it does. Examples are:

- the energy budget at a time;
- the temperature and the cooling of each region;
- the balance of ionisation and recombination;
- the region where a spectral feature forms.

## Run Python with artistools

Use the Python of the environment that holds the `artistools` command. `artistools python` gives its path:

```sh
ATPY="$(artistools python --quiet)"
"$ATPY" -P - <<'EOF'
import artistools as at
...
EOF
```

Always give `-P`. Without it, Python imports a file of the working folder before the installed package. A model
folder from a different person can hold a file such as `artistools.py`, and Python then runs that file.

If no `artistools` command exists, ask the user for the environment. Run the script in the model folder, or give
the model path in the script. Call `help()` on a function to get all its arguments.

Run `artistools describe <modelpath>` first. It gives these items:

- the cells, the time steps, and the output files, with the command that reads each file;
- the plot selections, e.g. the direction bins for `-plotviewingangle` and the virtual packet spectra for
  `-plotvspecpol`;
- the ions and the ARTIS version;
- the physics options of the run, e.g. NLTE, non-thermal ionisation, and thermalisation, and an `effect:` line that
  says what each option means for the outputs.

The physics options change the meaning of many outputs. More commands help before an answer:

- `artistools check <modelpath>` gives one row for each check of trust, e.g. the cells at a temperature limit, the
  complete time steps, a restart with a different build, and the Katz energy test. `--frompackets` adds the packet
  noise. Read its warnings before you trust a number of the model;
- `artistools catalog <folder>` gives one row for each model below a folder, e.g. the physics options and the peak;
- `artistools report <modelpath>` makes an HTML page with the check, the summary, the logs, and standard figures.
- `artistools values Te nne -t 300 -modelgridindex 12 <modelpath>` prints the raw estimators of cells and timesteps,
  with no average and the unit in each column name. Use it for the value of one cell, in place of Python.
- `-dirbin 90` selects the cells whose centre is in the cos θ range of a direction bin, e.g. near the pole. `values`,
  `plotestimators`, `plotopacity`, and `plotlinecontributions` take it. Use it to compare two directions, e.g.
  `artistools plotestimators init_X_lanthanides Ye -t 10 -dirbin 50 <modelpath> --print_data --quiet`. The summary
  table gives the mean with the mass of each cell as the weight for a mass fraction and for Ye.

Each plot command prints the plotted data as a table with `--print_data --quiet`, e.g. `artistools plotestimators Te
-t 300 <modelpath> --print_data --quiet`. Use it to check a result, or for a quick answer with no Python. A value or a
column name that contains a space has quotes. `--write_data` writes the table to a file, and the file starts with "# "
lines that give the command. Read such a file with `pl.read_csv(path, separator=" ", comment_prefix="#",
null_values="nan")`.

`--explain` prints the time steps, the direction bins, and the output file that the arguments select. Then it stops
before it reads the data. A time or a wavelength can have a unit, e.g. `-t 12h` or `-lambdamin 600nm`. A value with no unit
is in days or in Angstroms.

Without `--quiet`, plotestimators, plotlightcurves, and plotspectra also log one summary line for each drawn series.
Read these lines before you calculate a value yourself. `--print_data` and `--write_data` also give the same values
as a table named _summary, with one row for each series and the unit in each column name. Read that table in place of the
line when you need a number. Examples of the lines are:

- `Te: subplot 0, 540 points, x 5009 to 2.878e+04 km/s, min 3500 K at x = 1.711e+04, ..., mean 8.394e+04 K, weight of
  the mean: cell volume times time step duration`. A point is a cell at a time step, and the line names the weight;
- `B mymodel: 54 points from 250.4 to 299.3 d, peak -13.12 mag at 267.9 d (brightest point in the time range), Δm15 0.248 mag
  (linear interpolation of the magnitude at the peak time + 15 d)`;
- `mymodel: 518 points from 2500 to 1.9e+04 angstrom, largest drawn flux 5.121e-16 at 4648 angstrom, luminosity
  2.763e+40 erg/s = 4π (1 Mpc)² ∫ f_λ dλ from 2500 to 19000 Å (trapezoidal rule, ...)`. The luminosity is for the
  drawn wavelengths alone, and not for the full spectrum;
- with `--frompackets`, the line also gives the packet count and the relative Monte Carlo noise, e.g. `noise of the
  peak point 17.41% = sqrt(Σ e²) / Σ e of its 33 packets`. A difference smaller than the noise is not significant;
- with `--showemission` or `--showabsorption`, a heading with the wavelength range and the integration rule, one line
  for each drawn series with its flux at 1 Mpc and its share, e.g. `Co II      emission 3.48e-08 (68.3%)  absorption
  1.50e-08 (96.7%)`, and one line for the series in "Other".

A line that starts with `WARNING:` gives a problem of the plot. Examples are:

- cells at `MINTEMP` or `MAXTEMP`;
- a peak at an edge of the time range;
- values that the y range or a log axis does not show;
- x bins that have no value.

Give each such warning with the answer.

A remote model path has the form `host:path`. Give it as `Path("host:path")`. Some readers read a `str` path on this
computer and not on the host. These readers read a remote model on its host:

- `at.scan_estimators`, `at.get_timestep_times`, `at.get_timestep_of_timedays`, and `at.misc.get_time_range`;
- `at.get_modeldata`, `at.get_inputparams`, `at.get_composition_data`, and `at.get_deposition`;
- `at.nltepops.read_nltepops`, which needs a `timestep` or a `modelgridindex` for a remote model;
- `at.spectra.get_spectra` and `at.spectra.get_from_packets`;
- `at.lightcurve.find_lightcurve_file`, `at.lightcurve.scan_lightcurve`, and `at.lightcurve.get_from_packets`;
- `at.packets.get_packets_column_names`, `at.misc.get_artis_options`, and `at.misc.get_artis_option`.

The other readers stop with an error. For these readers, run the script on the host.

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
- ARTIS calculates its values with Monte Carlo packets, thus each value has noise. A value has more noise if the model
  has few packets, if the cell is small, or if the value is for one direction bin. Before you report a change as real,
  compare the value with the values of the time steps and the cells near it.
- To compare two processes, e.g. two heating processes or two ionisation processes, change each one to a rate per unit
  volume. Multiply that rate by `volume`, then add the values of all the cells. A rate per ion or per atom does not show
  which process is more important.

## Estimators: the values of each cell at each time step

`at.scan_estimators(modelpath, timestep=..., modelgridindex=..., join_modeldata=False)` gives a LazyFrame with one row
for each time step and cell. If you give `timestep` or `modelgridindex`, artistools reads less data and is faster. The
main columns are:

- `Te` (electron temperature, K), `TR` (radiation temperature, K), `TJ` (J temperature, K), and `W` (dilution factor);
- `nne` (free electron density, cm^-3);
- `nnelement_<El>`, `nnion_<El>_<ion stage>`, and `nniso_<nuclide>` (number densities, cm^-3);
- `heating_<process>`, `cooling_<process>`, and `deposition_<particle>` (rates, erg/s/cm^3). A column with `/` in its
  name, e.g. `heating_heating_dep/total_dep`, is a ratio and not a rate.

To get a table of each column of a model with its units and a description, run
`artistools plotestimators --listvariables --print_data --quiet <modelpath>`.
`join_modeldata=True` adds these columns:

- `tmid_days` (the mid time of the time step);
- `rho` (density at that time, g/cm^3), `volume` (cm^3), and `volume_prevtimestep` (the volume at the time step
  before, cm^3);
- `mass_g` (the mass of the cell);
- `init_<column>` (a value of `model.txt` at the start time of the model).

Possible errors:

- The estimators hold only the cells that contain matter. Thus the number of rows can be less than the number of
  cells in `model.txt`. Give the number of cells that you used.
- ARTIS limits Te and the other temperatures to the range from `MINTEMP` to `MAXTEMP`. A temperature at a limit is
  not a physical value. Get the limits with `at.misc.get_artis_option(modelpath, "MINTEMP")`. It reads `artisoptions.h`
  in the model folder. If that file does not exist, it reads `artis/artisoptions.h`. Give the number of cells at each
  limit.
- If a rate column of `heating_`, `cooling_`, or `deposition_` has no value, the rate is 0. ARTIS writes no value
  for a rate of 0. Do not include a ratio column in a sum of the rates.
- ARTIS writes the Monte Carlo deposition rate of time step n into the row of time step n + 1. It divides the rate by
  the volume of time step n. Thus the row of n + 1 gives the rate of time step n in erg/s/cm^3. For the rate of the
  cell in erg/s, multiply that value by `volume_prevtimestep`. The last time step has no Monte Carlo rate.
- `deposition_gamma` always holds a Monte Carlo rate, and `deposition_spfission` always holds an analytic rate. An
  analytic rate is for the time step of its own row. The positrons, the electrons, and the alpha particles have an
  analytic rate if `PARTICLE_THERMALISATION_SCHEME` is `INSTANTFULLDEPOSITION`, or if `INSTANT_PARTICLE_DEPOSITION`
  is true in an older ARTIS. `at.estimators.deposition.get_configured_analytic_channels(modelpath)` gives these
  channels.
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
        print(f"The artisoptions.h of the model gives no {limitname}.")
    else:
        atlimitcount = dfcells.filter(pl.col(column) == float(limit)).height
        print(f"{limitname} = {limit}: {atlimitcount} cells")
```

### Ionisation

The estimators give the ionisation state of each element in each cell:

- `nnion_<El>_<ion stage>` is the number density of an ion (cm^-3). The ion stage is a Roman numeral, e.g. `IV` in
  `nnion_Fe_IV`. The ion charge is the ion stage minus 1. `at.decode_roman_numeral("IV")` gives 4.
- The ion fraction is `nnion_<El>_<ion stage> / nnelement_<El>`.
- The mean charge of an element is the sum of the charge times the ion fraction, over the ions of the element.
- `gamma_R_<ion>` (photoionisation, s^-1) and `gamma_NT_<ion>` (non-thermal ionisation, s^-1) are the rates of
  ionisation from that ion. Thus the top ion has no such column. `Alpha_R_<ion>` (cm^3/s) is the coefficient of
  recombination from that ion to the ion below it. `Alpha_R*nne_<ion>` (s^-1) is that coefficient times `nne`.

Possible errors:

- ARTIS includes only the ions that `compositiondata.txt` gives. `at.get_composition_data(modelpath)` gives the columns
  `Z`, `lowermost_ion_stage`, and `uppermost_ion_stage`. ARTIS cannot ionise an element above its top ion. Thus a large
  fraction in the top ion shows that the true ionisation can be higher. A mean charge equal to the charge of the top ion
  is a lower limit and not a physical value. Give the number of cells where the top ion holds most of the element.
- To get the ion fraction of a group of cells, divide the number of ions by the number of atoms of the element. Use the
  weight `volume` from `join_modeldata=True`. A mean of the ion fractions of the cells gives the same weight to a cell
  with few atoms and to a cell with many atoms.
- An element with `nnelement_<El>` = 0 in a cell has no ion fraction there. Do not include such a cell.

Example: the ion fractions of Fe in all the cells, the mean charge of each cell, and the cells at the top ion.

```python
import artistools as at
import polars as pl

modelpath = "."
element = "Fe"
timestep = at.get_timestep_of_timedays(modelpath, 10)

elementcol = f"nnelement_{element}"
lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
ioncols = [col for col in lfcells.collect_schema().names() if col.startswith(f"nnion_{element}_")]
ioncharges = {col: at.decode_roman_numeral(col.removeprefix(f"nnion_{element}_")) - 1 for col in ioncols}
topioncol = max(ioncharges, key=ioncharges.__getitem__)
dfcells = (
    lfcells
    .filter(pl.col(elementcol) > 0)
    .select("modelgridindex", "tmid_days", "volume", elementcol, *ioncols)
    .with_columns(
        meancharge=pl.sum_horizontal(charge * pl.col(col) for col, charge in ioncharges.items()) / pl.col(elementcol)
    )
    .collect()
)
print(
    dfcells.select([
        ((pl.col(col) * pl.col("volume")).sum() / (pl.col(elementcol) * pl.col("volume")).sum()).alias(
            col.removeprefix("nnion_")
        )
        for col in ioncols
    ])
)
print(
    dfcells.select(
        pl.len().alias("cellcount"),
        pl.col("tmid_days").first(),
        pl.col("meancharge").min().alias("min"),
        pl.col("meancharge").median().alias("median"),
        pl.col("meancharge").max().alias("max"),
    )
)

dfcomposition = at.get_composition_data(modelpath).filter(pl.col("Z") == at.get_atomic_number(element))
topstage = dfcomposition["uppermost_ion_stage"].item()
attopcount = dfcells.filter(pl.col(topioncol) > 0.5 * pl.col(elementcol)).height
topion = at.get_ionstring(at.get_atomic_number(element), topstage)
print(f"top ion: {topion}. It holds most of {element} in {attopcount} cells.")
```

`artistools plotestimators -plot "Fe IV" "Fe V" ionpoptype=elpop -plot averageionisation Fe -t 10 <modelpath>` plots
the ion fractions and the mean charge against the velocity. Use it to check a result.

### Ionisation processes

The estimators give two rates of ionisation for each ion except the top ion:

- `gamma_R_<ion>` (s^-1) is the photoionisation rate of each ion of `<ion>`. ARTIS adds the rates of the levels that it
  can ionise, with a correction for stimulated recombination. The rate does not include collisional ionisation.
- `gamma_NT_<ion>` (s^-1) is the non-thermal ionisation rate of each ion, from the fast electrons of the radioactive
  decays. ARTIS calculates it from the Spencer-Fano solution or from the work-function approximation. One non-thermal
  ionisation can remove more than one electron.

Possible errors:

- ARTIS writes `gamma_NT_` only if non-thermal ionisation is on. Without it, the model has no non-thermal ionisation.
- The more important process can be different for each ion and for each region. Give the answer for each ion that
  holds much of the element, e.g. Fe II and Fe III. For a question about a region, select its cells, e.g. with
  `vel_r_mid` from `join_modeldata=True`.

Example: the ionisations per second by each process for each ion at a time.

```python
import artistools as at
import polars as pl

modelpath = "."
timestep = at.get_timestep_of_timedays(modelpath, 50)
lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
columns = lfcells.collect_schema().names()
# the top ion has no ionisation rate, thus each rate column names the lower ion of the process
ions = sorted(
    {col.removeprefix("gamma_R_") for col in columns if col.startswith("gamma_R_")}
    & {col.removeprefix("gamma_NT_") for col in columns if col.startswith("gamma_NT_")}
)
if not ions:
    print("This model has no gamma_R or no gamma_NT columns. ARTIS writes gamma_NT only with non-thermal ionisation.")
else:
    # the ionisations per second in all the cells: the rate per ion times the ion density times the volume
    dftotals = lfcells.select(
        *(
            (pl.col(f"{process}_{ion}") * pl.col(f"nnion_{ion}") * pl.col("volume")).sum().alias(f"{process}_{ion}")
            for process in ("gamma_R", "gamma_NT")
            for ion in ions
        ),
        pl.col("tmid_days").first(),
    ).collect()
    dfions = pl.DataFrame({
        "ion": ions,
        "photoionisations_per_s": [dftotals[f"gamma_R_{ion}"].item() for ion in ions],
        "nonthermal_ionisations_per_s": [dftotals[f"gamma_NT_{ion}"].item() for ion in ions],
    }).with_columns(
        nonthermal_fraction=pl.col("nonthermal_ionisations_per_s")
        / (pl.col("photoionisations_per_s") + pl.col("nonthermal_ionisations_per_s"))
    )
    print(f"time step {timestep}, mid time {dftotals['tmid_days'].item():.1f} days")
    with pl.Config(tbl_rows=-1):
        print(dfions)
```

### NLTE level populations

`at.nltepops.read_nltepops(modelpath, timestep=..., modelgridindex=...)` gives the level populations that ARTIS
calculates out of local thermodynamic equilibrium (non-LTE, or NLTE). It gives a DataFrame with the columns `timestep`,
`modelgridindex`, `Z`, `ion_stage`, `level`, `n_LTE`, and `n_NLTE`. `n_NLTE` is the number density of the level
(cm^-3). `n_LTE` is a Boltzmann population relative to the NLTE population of the ground level. Thus `n_NLTE / n_LTE`
is 1 at the ground level, and it gives the departure from a Boltzmann distribution inside the ion.

Possible errors:

- Only a model that wrote the files `nlte_*.out` has these data. artistools needs minutes to read a large model. Thus
  give a time step and the cells.
- A row with `level` = -1 is a superlevel. It holds the levels above the highest NLTE level of its ion, up to the last
  level that ARTIS keeps. The top ion has only its ground level.
- The sum of `n_NLTE` over the levels of an ion can be different from `nnion_`. Use `nnion_` for the ion fraction.

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
  columns of the estimators. A `deposition_` column with a Monte Carlo rate holds the rate of the time step before its
  row. See "Estimators".

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
- The light curve has values only for the time steps of the model. The true peak can be outside the time range of
  the model in these cases:
  - the maximum of the observer-frame luminosity is in the first two or the last two time steps;
  - the first time step is almost as bright as the maximum;
  - the maximum of the comoving-frame luminosity is in the last two time steps, thus the light curve still rises at
    the end.

  Give this warning with the answer.
- In the last time steps, the luminosity in the observer frame falls, e.g. to 0, but the luminosity in the comoving
  frame does not. The observer-frame time of a packet includes the travel time of the light across the ejecta. Thus
  the last time steps do not get the packets that escape after the last time step of the model. Do not report this
  fall as the decline of the light curve.

Example: the time and the luminosity of the peak, the rows near it, and a test of the edges of the time range.

```python
import polars as pl

import artistools as at

modelpath = "."
lcpath = at.lightcurve.find_lightcurve_file(modelpath)
dflc = at.lightcurve.scan_lightcurve(lcpath)[-1].collect().with_row_index("row")
print(lcpath)
# a NaN or an infinite value is not a luminosity, thus each peak takes only the finite rows. "row" keeps the row number
dffinite = dflc.filter(pl.col("time_days").is_finite() & pl.col("luminosity_erg/s").is_finite())
dfcmffinite = dflc.filter(pl.col("time_days").is_finite() & pl.col("luminosity_cmf_erg/s").is_finite())
print(f"{dffinite.height} of {dflc.height} rows have a finite time and luminosity")
peakrow = dffinite["row"][dffinite["luminosity_erg/s"].arg_max()]
cmfpeakrow = dfcmffinite["row"][dfcmffinite["luminosity_cmf_erg/s"].arg_max()]
print(dflc.slice(max(peakrow - 2, 0), 5).select("row", "time_days", "luminosity_erg/s", "luminosity_cmf_erg/s", "mag"))
peakluminosity = dflc["luminosity_erg/s"][peakrow]
firstluminosity = dffinite["luminosity_erg/s"][0]
if peakrow < dffinite["row"][0] + 2 or peakrow > dffinite["row"][-1] - 2:
    print("The maximum is at an edge of the time range, thus the true peak can be outside the range.")
# the observer-frame luminosity falls at the end, thus only the comoving frame shows a curve that still rises
elif cmfpeakrow > dfcmffinite["row"][-1] - 2:
    print("The comoving-frame luminosity has its maximum at the end, thus the true peak can be after the time range.")
elif firstluminosity > 0.9 * peakluminosity:
    print(
        f"The first luminosity is {firstluminosity / peakluminosity:.0%} of the maximum. The true peak can be earlier."
    )
```

## Spectra

`at.spectra.get_spectra(Path(modelpath), timestepmin, timestepmax, directionbins=[-1])` reads `spec.out`. It gives a
dict with one LazyFrame for each direction bin. Each spectrum is the mean over the time steps from `timestepmin` to
`timestepmax`, with the width of each time step as the weight. The columns are:

- `lambda_angstroms` (Å), the wavelength of the bin;
- `f_lambda` (erg/s/cm^2/Å), the flux per unit wavelength at a distance of 1 Mpc;
- `nu` (Hz), the frequency of the bin;
- `f_nu` (erg/s/cm^2/Hz), the flux per unit frequency at a distance of 1 Mpc.

The flux is at a distance of 1 Mpc. Give `gamma=True` for the gamma-ray spectrum. Give only the direction bins that you
need. For a bin of 0 or more, artistools reads the large file `spec_res.out`.

`at.spectra.get_from_packets(modelpath, timelowdays, timehighdays, directionbins=[-1])` makes the spectrum from the
packets files for a time range in days. It is slower than `get_spectra`.

Possible errors:

- `get_spectra` takes time steps and not days. Change a range of days to time steps with
  `at.misc.get_time_range(modelpath, timemin=..., timemax=...)`. It keeps the time steps whose mid time is in the range,
  as `artistools plotspectra -timedays` does. For a range after the end of the model, it gives the time step -1.
- A maximum in a range of wavelengths can be at an edge of that range. Compare it with the maximum of the full spectrum.
- Multiply the flux by `4 * math.pi * megaparsec_to_cm**2` to get a luminosity density. Integrate it over the
  wavelength or the frequency to get a luminosity. `megaparsec_to_cm` is in `artistools.constants`.
- Integrate with the trapezoidal rule, e.g. `np.trapezoid`, as artistools does. A sum of the flux times the width of
  each interval takes only one end of each interval.

Example: the mean spectrum from 10 to 12 days, the wavelength of its maximum in the range from 3000 to 10000 Å, and
its luminosity.

```python
import math
from pathlib import Path

import artistools as at
import numpy as np
import polars as pl
from artistools.constants import megaparsec_to_cm

modelpath = "."
timestepmin, timestepmax, daysmin, daysmax = at.misc.get_time_range(modelpath, timemin=10, timemax=12)
dfspec = at.spectra.get_spectra(Path(modelpath), timestepmin, timestepmax, directionbins=[-1])[-1].collect()
print(f"time steps {timestepmin} to {timestepmax}, {daysmin:.2f} to {daysmax:.2f} days")
print(dfspec.filter(pl.col("lambda_angstroms").is_between(3000, 10000)).top_k(1, by="f_lambda"))
lambdapeak = dfspec.top_k(1, by="f_lambda")["lambda_angstroms"].item()
if not 3000 <= lambdapeak <= 10000:
    print(f"The maximum of the full spectrum is at {lambdapeak:.0f} Å, outside the range from 3000 to 10000 Å.")
dfbyfrequency = dfspec.sort("nu")
flux = float(np.trapezoid(dfbyfrequency["f_nu"].to_numpy(), x=dfbyfrequency["nu"].to_numpy()))
luminosity = flux * 4 * math.pi * megaparsec_to_cm**2
print(f"luminosity in the spectrum: {luminosity:.3e} erg/s")
```

## Other data

- `at.packets.get_packets(modelpath)` gives `(number of ranks that it read, LazyFrame of the packets)`. artistools
  divides a sum of the packet energies by that number, e.g. for a luminosity. Do the same. Before it returns, the first
  call writes a parquet cache for each batch of 100 packets files. On a large model this takes many minutes. To prevent
  a slow step:
  - get the columns with `at.packets.get_packets_column_names(modelpath)`, which writes no cache;
  - give `maxpacketfiles=100` to read only the packets files of the first 100 ranks;
  - select the columns and filter the rows before `.collect()`.

  `artistools cache make <modelpath>` writes all the caches of a model at one time, and `artistools cache list` gives
  them.

  `get_packets_column_names` gives the columns of the packets files. The LazyFrame does not have some of these columns,
  and it adds other columns, e.g. `t_arrive_d` and `dirbin`.
- `at.get_inputparams(modelpath)` gives a dict of the parameters of `input.txt`.
- For a different reader, read the section "Use from Python" of the artistools README:
  https://github.com/artis-mcrt/artistools#use-from-python

## Report

Give these items with the answer:

- the model path;
- the time step or the range of time steps, and the mid time of each time step in days;
- the reader and the columns that you used;
- the units of each value;
- for the cells, the number that you used and the cells that you did not use, with a reason for each;
- each assumption that you made, e.g. the choice of a time step, a direction bin, or a weight.
