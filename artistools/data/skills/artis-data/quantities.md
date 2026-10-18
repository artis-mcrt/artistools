# The internal quantities of an ARTIS model and their readers

This file lists each file of an ARTIS model, what it holds, and the artistools function that reads it. Read it to
explain the behaviour of a model, e.g. which process heats, ionises, or cools the ejecta. SKILL.md gives the rules for
all the data, and the examples.

A rank file is one file for each MPI rank, e.g. `estimators_0012.out`, or one `<name>_allranks.out` for each run folder.
`at.misc.read_rank_outputfiles(modelpath, "<name>_{mpirank:04d}.out", timestep=..., modelgridindex=...)` reads any rank
file into a DataFrame. It finds the run folders, and it drops the time step that a restarted run writes again.

The estimators, the NLTE populations, and the radiation field of row n are the state that ARTIS uses in time step n.
ARTIS calculates this state at the start of time step n from the Monte Carlo estimators of time step n - 1.

## Thermodynamics, ionisation, heating, and cooling: `estimators_*.out`

`at.scan_estimators(modelpath, timestep=..., modelgridindex=..., join_modeldata=True)` gives a LazyFrame with one row
for each time step and cell. SKILL.md gives the main columns. The other column families are:

- `heating_<process>` of each cell (erg/s/cm^3): `ff` (free-free), `bf` (bound-free), `coll` (collisional), and `dep`
  (deposition);
- `cooling_<process>` of each cell (erg/s/cm^3): `ff`, `fb` (free-bound), `coll`, and `adiabatic`;
- `heating_<process>_<ion>` and `cooling_<process>_<ion>` (erg/s/cm^3): the part of each ion. ARTIS writes them only
  with the option `WRITE_ION_HEATING_COOLING_RATES`. A bound-free or free-bound column has no value for the top ion;
- `emission_ana_<particle>` and `deposition_<particle>` (erg/s/cm^3): the analytic energy release and the deposition of
  each type of particle. See SKILL.md for the time step of a Monte Carlo deposition;
- `gamma_R_<ion>`, `gamma_NT_<ion>`, and `gamma_R_integral_<ion>` (s^-1): the ionisation rates of each ion. ARTIS
  writes `gamma_R_integral_` only with `DETAILED_BF_ESTIMATORS_ON`;
- `corrphotoionrenorm_<ion>` and `gammaestimator_<ion>`: the correction of the photoionisation rate of the ground
  level, and the ionisation rate per ground-level population. For an element with no NLTE levels, `gammaestimator_`
  holds the analytic rate of radiative and collisional ionisation. ARTIS writes them only with `USE_LUT_PHOTOION`;
- `Alpha_R_<ion>` (cm^3/s): the recombination coefficient, in an output of an older ARTIS.

`artistools plotestimators --listvariables --print_data --quiet <modelpath>` gives a table of each column of a model,
with its units and a description.

To find the process that heats or cools the ejecta most, multiply each rate by `volume`, then add the values of all the
cells. Do this for each process at the same time step.

## Radiation field: `radfield_*.out` (rank files)

`at.plotradfield.read_radfield(modelpath, modelgridindex=...)` gives a DataFrame with these columns:

- `timestep`, `modelgridindex`, and `bin_num`;
- `nu_lower` and `nu_upper` (Hz), the limits of the frequency bin;
- `J` (erg/s/cm^2/sr), `J_nu_avg` (erg/s/cm^2/sr/Hz), and `nuJ`;
- `T_R` (K) and `W`, the temperature and the dilution factor of the fit of the bin.

A row with `bin_num` = -1 holds the fit of the full spectrum. A row with `bin_num` below -1 holds a line estimator.
`nu_lower` and `nu_upper` both hold the frequency of the line, `J_nu_avg` holds the estimator, and the other values are
-1. ARTIS writes this file only with `MULTIBIN_RADFIELD_MODEL_ON`. The
command `artistools plotradfield` plots the field of a cell.

## Non-thermal electrons

- The estimators give the non-thermal ionisation rates (`gamma_NT_`) and the deposition (`deposition_`).
- `artistools spencerfano` solves the Spencer-Fano equation again for one cell. It gives the fractions of the
  deposited energy that go into heating, ionisation, and excitation. It needs the package `pynonthermal`.
- An older ARTIS wrote `nonthermalspec_*.out`, the spectrum of the non-thermal electrons. Read it with
  `at.misc.read_rank_outputfiles(modelpath, "nonthermalspec_{mpirank:04d}.out")`.

## Populations

- `at.nltepops.read_nltepops(modelpath, timestep=..., modelgridindex=...)` reads `nlte_*.out`. SKILL.md gives the
  columns. `artistools plotnltepops` plots the populations of the levels of an ion.
- The estimators give the ion densities (`nnion_`), the element densities (`nnelement_`), and the nuclide densities
  (`nniso_`).
- `macroatom_*.out` holds the transitions of the macro atom. Read it with
  `at.misc.read_rank_outputfiles(modelpath, "macroatom_{mpirank:04d}.out")`. ARTIS writes it only if `LOG_MACROATOM`
  in `macroatom.cc` is true.

## Atomic data

- `at.atomic.get_levels(modelpath, get_transitions=True)` reads `adata.txt` and `transitiondata.txt`. It gives one row
  for each ion, with a DataFrame of the levels (`energy_ev`, `g`) and a LazyFrame of the transitions (`A` in s^-1).
- `at.get_composition_data(modelpath)` reads `compositiondata.txt`: the ions of each element.
- `at.atomic.get_linelist_pldf(modelpath)` reads `linelist.out`, the list of the lines. The row number of a line is the
  code of that line in the packets.
- `at.atomic.get_bflist(modelpath)` reads `bflist.out`, the list of the bound-free continua.
- `at.atomic.get_nuclides(modelpath)` reads `nuclides.out`, the radioactive nuclides of the pellets.

## Packets: the last interaction of each packet

`at.packets.get_packets(modelpath, escape_type="TYPE_RPKT")` gives `(number of ranks that it read, LazyFrame)`. The
main columns of an escaped packet are:

- `e_rf` (erg), `nu_rf` (Hz), and `t_arrive_d` (days), the energy, the frequency, and the arrival time at the observer;
- `emissiontype`, the process of the last emission: a value of 0 or more is the row of the line in `linelist.out`,
  -9999999 is free-free, and another negative value v is the bound-free continuum -1 - v of `bflist.out`;
- `trueemissiontype`, the last emission from the thermal pool, with the same codes;
- `absorption_type`, the process of the last absorption: a value of 0 or more is a line, -1 is free-free, and -2 is
  bound-free. -11 is a line in a binned opacity. A different negative value shows that the packet had no r-packet
  absorption, e.g. -10 for a pellet decay;
- `absorption_freq` (Hz), the frequency of the packet at its last absorption;
- `em_posx`, `em_posy`, `em_posz` (cm), and `em_time` (s), the place and the time of the last emission or scattering.

`artistools plotspectra --frompackets -groupby ion` and `artistools plotlastpacketinteraction` use these columns.
`at.packets.get_virtual_packets(modelpath)` reads the virtual packets of `vpackets/`.

## Spectra by the ion that emits or absorbs

- `artistools plotspectra --showemission --print_data --quiet -t <days> <modelpath>` prints the spectrum of the
  emission of each ion and process, e.g. `emission_flambda.Fe II bound-free`. `--showabsorption` gives the absorption.
- `at.spectra.get_flux_contributions(modelpath, timestepmin, timestepmax)` reads `emission.out` and `absorption.out`.
  `use_lastemissiontype=False` reads `emissiontrue.out`, the thermal emission.
- `at.spectra.get_flux_contributions_from_packets(modelpath, ...)` makes the same from the packets, e.g. by line,
  by element, or by velocity.
- `artistools plotlinefluxes --print_data` gives the ratio of the luminosities of two emission features against time.
- `artistools linecontributions -t <days> -lambdamin <Å> -lambdamax <Å> <modelpath>` gives each line in the range
  with its maximum Sobolev optical depth in the cells and its emission n_u A h nu beta, summed over the cells. `--percell` gives
  one row for each cell and line, with the level populations and the method of each one.
  `at.linecontributions.get_line_contributions(modelpath, timestep, lambdamin, lambdamax)` gives the same table.
  `artistools plotlinecontributions` plots the emission and the tau_sobolev of each line against velocity.
  `-costhetamin` and `-costhetamax` select the cells of one direction, and `-troughdirbin` adds a Sobolev estimate of
  the absorption trough for one viewing direction.
- `artistools plotlastinteractionvelocity <modelpath> -lambdamin <Å> -lambdamax <Å>` plots the median and the
  quartiles of the velocity of the last interaction of the escaped packets against time. In a range with no strong
  line, the median is an estimate of the velocity of the photosphere.

## Light curves and energy

SKILL.md gives the light curves and `deposition.out`. Other readers:

- `at.misc.get_timesteps(modelpath)` gives the start, the mid time, the end, and the width of each time step.
- `at.lightcurve.get_from_packets(modelpath, pellet_nucname=...)` gives the light curve of the energy of one nuclide.

## Model input

- `at.get_modeldata(modelpath)` reads `model.txt`. Give `get_elemabundances=True` to also read `abundances.txt`, which
  gives the mass fraction of each element, e.g. `X_Fe`. `at.add_derived_cols_to_modeldata(...)` adds derived columns,
  e.g. the mass of each cell.
- `at.get_inputparams(modelpath)` reads `input.txt`: e.g. `tmin`, `tmax` (days), and `ntstep`.
- `at.misc.get_vpkt_config(modelpath)` reads `vpkt.txt`, the observers of the virtual packets.
- `at.misc.get_grid_mapping(modelpath)` reads `grid.out`, the cells of the propagation grid and of the model.

## Logs and options

These options and commands describe a run:

- `MINTEMP` and `MAXTEMP` (K), the lowest and the highest Te of `artisoptions.h`. ARTIS sets Te to a limit when no Te
  between them gives the balance of heating and cooling, thus a Te at a limit is not a physical value. `artistools
  check <modelpath>` counts the cells at each limit.
- `artistools describe <modelpath>` gives the ARTIS version of each run folder and the physics options by topic.
  `--alloptions` gives each option of `artisoptions.h`. `at.misc.get_artis_options(modelpath)` gives them as a dict.
- `at.misc.get_artis_option(modelpath, name)` reads an option of `artisoptions.h` in the model folder. If that file
  does not exist, it reads `artis/artisoptions.h`. ARTIS renamed some options on 2026-10-16, e.g.
  `MULTIBIN_RADFIELD_FIRST_TIMESTEP` was `FIRST_BINNED_RADFIELD_TIMESTEP` and, before that,
  `FIRST_NLTE_RADFIELD_TIMESTEP`. For an older model, also try each older name.
- `at.plotlogfiles.read_time_taken(at.plotlogfiles.read_logfiles(modelpath))` gives the time of each part of each time
  step, from the logs `output_*.txt`.
- `artistools logsummary <modelpath>` counts the warnings and the errors of the logs by kind, with their time steps and
  the number of cells. Each rank logs the cells that it updates, thus `-ranks all` or `-modelgridindex <cell>` reads
  more logs. Use it to explain an odd value, e.g. a cell at `MAXTEMP`.
- The log `output_0-0.txt` holds the messages of ARTIS for each time step. Search it for a message, e.g. a warning.
