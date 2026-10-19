# Recipes for questions about why a model behaves as it does

Each recipe answers one type of question with a tested example. SKILL.md gives the rules for all the data, and
`physics.md` explains the physics. Give the time step, its mid time, and the number of cells with each answer.

## The energy budget at a time

Question: where does the deposited energy go, and does the luminosity agree with the deposition?

- In each cell with a thermal balance, ARTIS finds the Te at which the heating equals the cooling. Thus the heating
  and the cooling of such a cell agree. The recipe adds the rates of these cells only.
- The cells of these types have no thermal balance. The recipe gives the number of cells of each type:
  - a cell in LTE (see `physics.md`);
  - a cell at `MINTEMP` or `MAXTEMP`;
  - a cell with heating and cooling that differ by more than 1 %. One step of the solver changes Te by a factor of 2
    or less, thus such a cell often has a Te with few significant digits, e.g. 7000 K. With
    `NLTE_TIME_DEPENDENT_FIRST_TIMESTEP`, the change of the thermal energy also enters the balance, and the output
    does not give it.
- With `WRITE_ION_HEATING_COOLING_RATES`, ARTIS also writes the part of each ion, e.g. `heating_coll_Fe_II`. The
  total, e.g. `heating_coll`, already includes these parts, thus the recipe adds only the totals.
- `heating_dep` is the part of the deposition that heats the electrons. The other part ionises and excites the ions.
- A cell in LTE gives no heating and cooling rates. In an early time step, these cells can hold most of the deposition.
  Thus the sum of `heating_dep` can be much less than the deposition in `deposition.out`.
- `deposition.out` gives the deposition of all the ejecta. The comoving-frame luminosity of the light curve gives the
  energy that escapes. Before the peak of the light curve, the ejecta keep a part of the deposited energy as
  radiation, thus the luminosity is less than the deposition. After the peak, the luminosity can be more than the
  deposition.

```python
import re

import artistools as at
import polars as pl
from artistools.constants import Lsun_to_erg_per_s

modelpath = "."
timestep = at.get_timestep_of_timedays(modelpath, 10)
tmid_days = at.get_timestep_times(modelpath, loc="mid")[timestep]
lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
columns = lfcells.collect_schema().names()
# a ratio, e.g. heating_heating_dep/total_dep, and the part of each ion, e.g. heating_coll_Fe_II, are not totals
heatingcols = [col for col in columns if re.fullmatch(r"heating_[a-z]+", col)]
coolingcols = [col for col in columns if re.fullmatch(r"cooling_[a-z]+", col)]
temperaturelimits = [
    float(value) for name in ("MINTEMP", "MAXTEMP") if (value := at.misc.get_artis_option(modelpath, name)) is not None
]
heating = pl.sum_horizontal(heatingcols)
cooling = pl.sum_horizontal(coolingcols)
cellstate = (
    pl
    .when((pl.col("W") == 1.0) & (pl.col("Te") == pl.col("TR")) & (pl.col("Te") == pl.col("TJ")))
    .then(pl.lit("LTE"))
    .when(pl.col("Te").is_in(temperaturelimits))
    .then(pl.lit("at a temperature limit"))
    .when(((heating / cooling) - 1.0).abs() > 0.01)
    .then(pl.lit("not balanced"))
    .otherwise(pl.lit("balanced"))
)
lfcells = lfcells.with_columns(cellstate=cellstate)
print(f"time step {timestep}, mid time {tmid_days:.2f} days")
print(
    lfcells
    .group_by("cellstate")
    .agg(
        pl.len().alias("cellcount"),
        (heating * pl.col("volume")).sum().alias("heating_erg/s"),
        (cooling * pl.col("volume")).sum().alias("cooling_erg/s"),
    )
    .sort("cellstate")
    .collect()
)
print(
    lfcells
    .filter(pl.col("cellstate") == "balanced")
    .select(*((pl.col(col).fill_null(0.0) * pl.col("volume")).sum().alias(col) for col in heatingcols + coolingcols))
    .collect()
    .unpivot(variable_name="process", value_name="rate_erg/s")
    .with_columns(kind=pl.col("process").str.split("_").list.first())
    .with_columns(fraction=pl.col("rate_erg/s") / pl.col("rate_erg/s").sum().over("kind"))
    .sort(["kind", "rate_erg/s"], descending=[False, True])
)

dfdep = at.get_deposition(modelpath).filter(pl.col("timestep") == timestep).collect()
dflc = at.lightcurve.scan_lightcurve(at.lightcurve.find_lightcurve_file(modelpath))[-1].collect()
lcrow = dflc.with_columns(dt=(pl.col("time_days") - tmid_days).abs()).sort("dt").head(1)
print(f"deposition {dfdep['total_dep_Lsun'].item() * Lsun_to_erg_per_s:.4g} erg/s")
luminosity_cmf, lctime_days = lcrow.select("luminosity_cmf_erg/s", "time_days").row(0)
print(f"comoving-frame luminosity {luminosity_cmf:.4g} erg/s at {lctime_days:.2f} days")
```

## The temperature and the cooling of the regions of the ejecta

Question: which process cools each region of the ejecta, and why is Te high or low there?

- The `cooling_` columns give the rates of the processes that remove energy from the electrons. ARTIS also uses
  them to select the channel of each k-packet. Thus the main cooling process shows how the thermal energy becomes
  radiation, e.g. `cooling_coll` for the collisional excitation of lines.
- Use the velocity of each cell to make the regions. The ejecta expand homologously, thus the velocity of a cell
  gives its position.
- A cell with no thermal balance has rates that do not explain its Te. The recipe does not include these cells:
  a cell in LTE, a cell at a temperature limit, and a cell with heating and cooling that differ by more than 1 %.
  See "The energy budget at a time".
- A region can have a high Te because its cooling is weak, e.g. because of a low density or few ions with strong
  lines. Compare the cooling of a region with its heating.

```python
import re

import artistools as at
import numpy as np
import polars as pl
from artistools.constants import km_to_cm

modelpath = "."
timestep = at.get_timestep_of_timedays(modelpath, 10)
nbins = 5
lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
columns = lfcells.collect_schema().names()
# the part of each ion, e.g. cooling_coll_Fe_II, is also in the total of its process, e.g. cooling_coll
heatingcols = [col for col in columns if re.fullmatch(r"heating_[a-z]+", col)]
coolingcols = [col for col in columns if re.fullmatch(r"cooling_[a-z]+", col)]
temperaturelimits = [
    float(value) for name in ("MINTEMP", "MAXTEMP") if (value := at.misc.get_artis_option(modelpath, name)) is not None
]
islte = (pl.col("W") == 1.0) & (pl.col("Te") == pl.col("TR")) & (pl.col("Te") == pl.col("TJ"))
isunbalanced = ((pl.sum_horizontal(heatingcols) / pl.sum_horizontal(coolingcols)) - 1.0).abs() > 0.01
isleftout = islte | pl.col("Te").is_in(temperaturelimits) | isunbalanced
leftoutcount = lfcells.select(isleftout.sum()).collect().item()
print(f"removed {leftoutcount} cells in LTE, at a temperature limit, or with no thermal balance")
vmax = lfcells.select(pl.col("vel_r_mid").max()).collect().item()
dfregions = (
    lfcells
    .filter(~isleftout)
    .with_columns(velbin=(pl.col("vel_r_mid") / vmax * nbins).floor().clip(0, nbins - 1).cast(pl.Int32))
    .group_by("velbin")
    .agg(
        pl.len().alias("cellcount"),
        (pl.col("vel_r_mid").min() / km_to_cm).alias("vmin_km/s"),
        (pl.col("vel_r_mid").max() / km_to_cm).alias("vmax_km/s"),
        ((pl.col("Te") * pl.col("mass_g")).sum() / pl.col("mass_g").sum()).alias("Te_massweighted"),
        *((pl.col(col).fill_null(0.0) * pl.col("volume")).sum().alias(col) for col in coolingcols),
    )
    .sort("velbin")
    .collect()
)
coolingrates = dfregions.select(coolingcols).to_numpy()
totalcooling = coolingrates.sum(axis=1)
dfregions = dfregions.with_columns(
    dominant=pl.Series([coolingcols[index] for index in np.argmax(coolingrates, axis=1)], dtype=pl.String),
    dominant_fraction=pl.Series(
        np.divide(coolingrates.max(axis=1), totalcooling, out=np.zeros(len(totalcooling)), where=totalcooling > 0)
    ),
)
print(dfregions.drop(coolingcols))
```

## The balance of ionisation and recombination

Question: which process sets the ionisation of an ion, and is the ionisation in balance?

- The ionisations per second from ion i are (`gamma_R_<i>` + `gamma_NT_<i>`) × `nnion_<i>` × `volume`. The
  recombinations per second to ion i are `Alpha_R*nne_<i+1>` × `nnion_<i+1>` × `volume`.
- ARTIS wrote this rate as `Alpha_R*nne_` before August 2019, and as `AlphaR*nne_` until June 2022. The output of a
  later version of ARTIS has no recombination rate. For such a model, compare only the ionisation processes (see
  "Ionisation processes" in SKILL.md).
- A ratio of the ionisations to the recombinations that is not near 1 can have these causes:
  - the collisional ionisation and recombination, which these columns do not include;
  - the charge transfer;
  - the time-dependent term of `NLTE_TIME_DEPENDENT_FIRST_TIMESTEP`;
  - the Monte Carlo noise of `gamma_R_`.

```python
import artistools as at
import polars as pl

modelpath = "."
timestep = at.get_timestep_of_timedays(modelpath, 50)
lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
columns = lfcells.collect_schema().names()
recombinationprefix = next(
    (prefix for prefix in ("Alpha_R*nne_", "AlphaR*nne_") if any(col.startswith(prefix) for col in columns)), ""
)
pairs = []
for col in columns:
    if col.startswith("gamma_R_") and not col.startswith("gamma_R_integral"):
        ion = col.removeprefix("gamma_R_")
        atomic_number, ion_stage = at.get_ion_tuple(ion)
        upperion = at.get_ionstring(atomic_number, ion_stage + 1, sep="_")
        if recombinationprefix and f"{recombinationprefix}{upperion}" in columns:
            pairs.append((ion, upperion))
if not pairs:
    print("This model has no recombination rates. ARTIS wrote them only until June 2022.")
else:

    def get_ionisation_expr(ion: str) -> pl.Expr:
        nonthermal = pl.col(f"gamma_NT_{ion}").fill_null(0.0) if f"gamma_NT_{ion}" in columns else pl.lit(0.0)
        return (pl.col(f"gamma_R_{ion}").fill_null(0.0) + nonthermal) * pl.col(f"nnion_{ion}") * pl.col("volume")

    dftotals = lfcells.select(
        *(get_ionisation_expr(ion).sum().alias(f"ionisations_{ion}") for ion, _ in pairs),
        *(
            (pl.col(f"{recombinationprefix}{upperion}") * pl.col(f"nnion_{upperion}") * pl.col("volume"))
            .sum()
            .alias(f"recombinations_{ion}")
            for ion, upperion in pairs
        ),
    ).collect()
    dfbalance = pl.DataFrame({
        "ion": [ion for ion, _ in pairs],
        "ionisations_per_s": [dftotals[f"ionisations_{ion}"].item() for ion, _ in pairs],
        "recombinations_per_s": [dftotals[f"recombinations_{ion}"].item() for ion, _ in pairs],
    }).with_columns(ratio=pl.col("ionisations_per_s") / pl.col("recombinations_per_s"))
    print(dfbalance)
```

## The region where a spectral feature forms

Question: at which velocities do the packets of a spectral feature last interact, and which lines emit them?

- `artistools plotlastpacketinteraction -modelpath <modelpath> -t <days> -groupby line -wavelen <Å> -binwidth <Å>`
  plots the luminosity of each line of the last emission against the radial velocity of the last interaction.
  `-groupby ion` and `-groupby element` give larger groups. `--absorption` selects the packets by the wavelength of
  their last absorption. It also groups them by the element, the ion, or the line of that absorption.
  `--print_data` gives the values and the number of packets in each bin. Its `_summary` table gives one row for each
  group, with the luminosity, the number of packets, and the quartiles of the velocity.
- `artistools plotlinecontributions -t <days> -lambdamin <Å> -lambdamax <Å> <modelpath>` plots the Sobolev emission
  and the median tau_sobolev of the strongest lines against velocity, from the level populations. `-rankby tau`
  selects the lines with the largest tau_sobolev, e.g. for an absorption trough. The `_summary` table of
  `--print_data` gives the velocity range where the median tau_sobolev is above 1, and the quartiles of the velocity
  of the emission.
- `em_posx`, `em_posy`, `em_posz`, and `em_time` give the place and the time of the last emission or scattering of a
  packet. `at.packets.get_emission_velocity_expr("em")` gives its radial velocity.
- An `emissiontype` of 0 or more is the row of the line in `linelist.out`. Thus the packets of a line feature show
  the lines that emit in the wavelength range.
- The packets give the region where the radiation last interacts. This region can be different from the region where
  the line absorbs, e.g. for a P Cygni absorption trough.
- If the range has few packets, the answer has much Monte Carlo noise. Give the number of packets with each answer.
- The packets give the last interaction alone. `artistools linecontributions` gives the lines from the level
  populations of each cell. It does not count packets, thus a range with few packets gives no additional noise. Its
  tau_sobolev shows where a line can absorb, e.g. for a P Cygni absorption trough. Its emission is the energy that
  leaves each cell in the line. It is not the flux that escapes the ejecta. Compare it with the packets before you
  give an answer.

```python
import artistools as at
import polars as pl
from artistools.constants import c_ang_per_s
from artistools.constants import km_to_cm

modelpath = "."
lambdamin, lambdamax = 3500.0, 4500.0
timestepmin, timestepmax, daysmin, daysmax = at.misc.get_time_range(modelpath, timemin=10, timemax=12)
_, lfpackets = at.packets.get_packets(modelpath, escape_type="TYPE_RPKT")
dfpackets = (
    lfpackets
    .filter(pl.col("t_arrive_d").is_between(daysmin, daysmax, closed="left"))
    .with_columns(lambda_angstroms=c_ang_per_s / pl.col("nu_rf"))
    .filter(pl.col("lambda_angstroms").is_between(lambdamin, lambdamax))
    .with_columns(emission_velocity_km_s=at.packets.get_emission_velocity_expr("em") / km_to_cm)
    .select("e_rf", "emissiontype", "emission_velocity_km_s")
    .collect()
)
print(f"{dfpackets.height} packets from {lambdamin} to {lambdamax} Å, from {daysmin:.2f} to {daysmax:.2f} days")
if dfpackets.height > 0:
    print(
        dfpackets.select(
            pl.col("emission_velocity_km_s").quantile(quantile).alias(f"q{quantile:.2f}")
            for quantile in (0.1, 0.5, 0.9)
        )
    )
    dflines = at.atomic.get_linelist_pldf(modelpath).collect()
    print(
        dfpackets
        .filter(pl.col("emissiontype") >= 0)
        .join(dflines, left_on="emissiontype", right_on="lineindex", how="left")
        .group_by("atomic_number", "ion_stage", "lambda_angstroms")
        .agg(pl.col("e_rf").sum().alias("energy_erg"), pl.len().alias("packetcount"))
        .with_columns(fraction=pl.col("energy_erg") / pl.col("energy_erg").sum())
        .sort("energy_erg", descending=True)
        .head(10)
    )
```

## A spectral feature in one direction and not in a different direction

Question: why does the spectrum of one viewing direction have a feature, e.g. a P Cygni profile of Sr II, that the
spectrum of a different direction does not have?

Do these steps in this sequence, for each direction bin:

1. Compare the spectra: `artistools plotspectra <modelpath> --frompackets -t <days> -dirbin 50 90
   --average_over_phi_angle`. Give the wavelength of the trough and the velocity that it gives for each line.
2. Find the velocity where the continuum forms: `artistools plotlastinteractionvelocity <modelpath> -lambdamin <Å>
   -lambdamax <Å> -dirbin 50 90`. Give the model path first. `-dirbin` takes a list, thus it takes a model path after
   it as a direction bin. Use a wavelength range with no strong line, e.g. a range at a shorter wavelength than the
   trough. The median is an estimate of the velocity of the photosphere in that direction. The quartiles show the
   width of the region.
3. Compare the optical depth of the lines in the cells of each direction: `artistools plotlinecontributions <modelpath>
   -t <days> -lambdamin <Å> -lambdamax <Å> -costhetamin 0.8`. These are the cells of `-dirbin 90`. For the cells of
   `-dirbin 50`, give `-costhetamin 0 -costhetamax 0.2`. A line makes an absorption trough only with tau_sobolev above
   1 in front of the photosphere, i.e. at a velocity above the median of step 2.
4. Estimate the trough: `artistools plotlinecontributions ... -troughdirbin 90 -photospherevelocity <v>`, with the
   median of step 2. The estimate has no emission of the lines, no limb darkening, and only the ranked lines. If the
   estimate gives a deep trough and the spectrum does not, the emission of the lines can fill the trough. As an
   alternative, lines of different ions can change the feature.
5. Compare the line blanketing of each direction. Run `artistools plotopacity -modelpath <modelpath> -t <days> -vmin
   <v> -costhetamin 0.8 -cellaverage median --write_data`, with the median of step 2 as `-vmin`. Then run the same
   command with `-costhetamin 0 -costhetamax 0.2`. `-vmin` needs a unit, e.g. `0.157c` or `47000km/s`. Use
   `-cellaverage median`. A few dense cells can make the means of the two directions almost equal. The expansion
   opacity sums all the lines of all the ions. A larger opacity at the wavelengths of the trough and of the continuum
   on its blue side can hide the feature. Compare the opacity at these wavelengths with the opacity at the wavelength
   of the line.
6. Compare the composition of the cells of each direction with `at.scan_estimators(..., join_modeldata=True)` and
   `at.get_modeldata(..., get_elemabundances=True)`. Select the cells by cos θ = v_z / |v| of the cell centre.
   Examples of the values are:
   - Ye;
   - the mass fraction of the element;
   - the lanthanide mass fraction;
   - the ion fraction.

Possible errors:

- `--absorption` gives the last absorption of the packets that escape toward the observer. A trough is the radiation
  that the line removes from the line of sight. The two are different, thus similar fractions from `--absorption` in
  two directions do not show that the troughs are similar.
- A few dense cells can make the mean tau_sobolev of a bin much larger than its median. Use the median and the volume
  fraction with tau_sobolev above 1, which the bin table of `plotlinecontributions` gives. The `_summary` table gives
  the velocity range where the median is above 1.
- The velocity of the last interaction is a radial velocity, and not a velocity along the line of sight. An electron
  scattering also sets it. In a kilonova, a range with no strong line still holds many weak lines.
- The trough estimate takes a sharp photosphere and pure absorption. If many lines are optically thick in front of the
  photosphere (step 5), the light scatters many times, and the estimate gives a trough that is too deep.
- `plotlinecontributions` and `linecontributions` calculate each pair of a cell and a line. For a wide range with all
  the ions of a kilonova, give `-ions` or `-taumin 1`. `-taumin` leaves out each line that cannot reach that
  tau_sobolev in any cell. In `plotlinecontributions`, use `-taumin` with `-rankby tau`. An optically thin line can
  emit much, and the ranking by emission and the emission fractions take only the lines that `-taumin` keeps.
- `-dirbin 90` holds the directions with 0.8 ≤ cos θ ≤ 1, i.e. 0° to 37°, as a mean over phi. `-costhetamin 0.8`
  selects the cells of the same cone. A 1D model has no direction.
