"""Give the deposition rate of a model per unit volume, per ion, and per unit mass."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import EV_to_erg
from artistools.estimators.core import scan_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import get_artis_options
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import parse_float_range
from artistools.misc import print_detail
from artistools.misc import print_product
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import resolve_table_outputfile
from artistools.misc.cliutils import CommaJoinAction
from artistools.misc.cliutils import get_provenance_lines

DEPOSITIONPREFIX = "deposition_"
DEFAULTOUTPUTNAME = "deposition.txt"


def get_deposition_channels(colnames: Sequence[str]) -> list[str]:
    """Return the deposition channels that the estimators of a model give, in alphabetical order.

    The reader of the estimator files keeps the columns in a hash map, thus the order of the columns
    changes between runs. The sort gives one order to the listing and to each message.
    """
    return sorted(name.removeprefix(DEPOSITIONPREFIX) for name in colnames if name.startswith(DEPOSITIONPREFIX))


def select_deposition_channels(colnames: Sequence[str], channels: Sequence[str] | None) -> list[str]:
    """Return the deposition channels to add, e.g. gamma and alpha, or stop if the model has no rate of a channel.

    The command needs the rate of each cell, which the deposition_ columns hold. The estimators of a model from an
    older ARTIS version hold total_dep alone, and artistools derives that column from the heating rate. On the
    classic-mode test model that total is 1400 times below the tally of deposition.out, thus such a model gets an
    error and no number.
    """
    available = get_deposition_channels(colnames)
    if not available:
        msg = (
            "The estimators of this model hold no deposition_ column, thus they give no deposition "
            "rate of a cell. The file deposition.out holds the rate of the whole model, which "
            "artistools plotlightcurves -deposition total draws"
        )
        raise ValueError(msg)

    # a channel that the command line gives two times must count one time in the sum
    channels = (
        list(dict.fromkeys(channel.removeprefix(DEPOSITIONPREFIX) for channel in channels)) if channels else available
    )
    if unknown := [channel for channel in channels if channel not in available]:
        msg = f"This model gives no deposition rate of {' and '.join(unknown)}. It holds {', '.join(available)}"
        raise ValueError(msg)
    return channels


def get_channel_sum_expression(channels: Sequence[str]) -> pl.Expr:
    """Return the expression that gives the sum of the deposition rates of the channels of one cell in erg/s/cm3."""
    if not channels:
        return pl.lit(0.0)
    # a cell that reports no channel deposited nothing in it, thus the sum reads a null as zero
    return pl.sum_horizontal(pl.col(f"{DEPOSITIONPREFIX}{channel}") for channel in channels)


def get_deposition_expression(colnames: Sequence[str], channels: Sequence[str] | None) -> pl.Expr:
    """Return the expression that gives the deposition rate of one cell in erg/s/cm3, for the channels or for all."""
    return get_channel_sum_expression(select_deposition_channels(colnames, channels))


def get_configured_analytic_channels(modelpath: Path | str) -> list[str] | None:
    """Return the deposition channels that hold an analytic rate, from the artisoptions.h of the model, or None.

    ARTIS always gives the analytic rate for spontaneous fission. It also gives it for the positrons, the electrons,
    and the alpha particles if PARTICLE_THERMALISATION_SCHEME is INSTANTFULLDEPOSITION. An ARTIS from before June
    2024 names this option INSTANT_PARTICLE_DEPOSITION, and an ARTIS from before May 2022 has neither option.
    """
    options = get_artis_options(modelpath)
    if "PARTICLE_THERMALISATION_SCHEME" in options:
        instantdeposition = "INSTANTFULLDEPOSITION" in options["PARTICLE_THERMALISATION_SCHEME"]
    elif "INSTANT_PARTICLE_DEPOSITION" in options:
        instantdeposition = options["INSTANT_PARTICLE_DEPOSITION"] == "true"
    else:
        return None
    return ["spfission", *(("positron", "electron", "alpha") if instantdeposition else ())]


def get_analytic_deposition_channels(dfestim: pl.LazyFrame, channels: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return the channels that hold an analytic rate, and the channels with no evidence of their scheme.

    ARTIS gives the analytic rate for spontaneous fission. It also gives it for the positrons, the electrons, and the
    alpha particles if the model sets PARTICLE_THERMALISATION_SCHEME to INSTANTFULLDEPOSITION. Then the channel equals
    its emission_ana_ column in each row. An analytic rate is the rate of the timestep of its own row. A Monte Carlo
    channel, e.g. gamma, holds the rate of the timestep before its row. A channel that is zero in each row equals its
    emission_ana_ column with either scheme, thus that agreement is no evidence.
    """
    colnames = dfestim.collect_schema().names()
    candidates = [channel for channel in channels if channel != "gamma" and f"emission_ana_{channel}" in colnames]
    if not candidates:
        return [], []
    dfevidence = dfestim.select(
        *(equals_analytic_rate(channel).all().alias(f"equal_{channel}") for channel in candidates),
        *((pl.col(f"emission_ana_{channel}") != 0.0).any().alias(f"nonzero_{channel}") for channel in candidates),
    ).collect()
    agrees = [channel for channel in candidates if dfevidence[f"equal_{channel}"].item()]
    analytic = [channel for channel in agrees if dfevidence[f"nonzero_{channel}"].item()]
    return analytic, [channel for channel in agrees if channel not in analytic]


def equals_analytic_rate(channel: str) -> pl.Expr:
    """Return an expression that is true where a channel equals its emission_ana_ column.

    ARTIS writes both columns with the format 11.5e, thus an analytic channel agrees to 6 significant digits.
    """
    return (pl.col(f"{DEPOSITIONPREFIX}{channel}") - pl.col(f"emission_ana_{channel}")).abs() <= 1e-5 * pl.col(
        f"emission_ana_{channel}"
    ).abs()


def check_channels_of_each_run(
    dfestim: pl.LazyFrame, analyticchannels: Sequence[str], montecarlochannels: Sequence[str]
) -> None:
    """Stop if the rows of a timestep do not follow the deposition scheme of artisoptions.h for a channel.

    The estimators join the rows of all the run folders, and artisoptions.h gives the options of one build. A restart
    with a different PARTICLE_THERMALISATION_SCHEME changes the rule of its rows in either direction:

    - an analytic channel differs from its emission_ana_ rate in a row of a Monte Carlo run;
    - a Monte Carlo channel equals its emission_ana_ rate in each row of a timestep of an analytic run. Monte Carlo
      noise makes that agreement impossible, unless the rates are zero.

    A Monte Carlo rate belongs to the timestep before its row, thus one rule for all the rows gives an incorrect rate.
    """
    colnames = dfestim.collect_schema().names()
    checkedanalytic = [channel for channel in analyticchannels if f"emission_ana_{channel}" in colnames]
    checkedmontecarlo = [channel for channel in montecarlochannels if f"emission_ana_{channel}" in colnames]
    if not checkedanalytic and not checkedmontecarlo:
        return
    dfanalytic = dfestim.select(
        pl.col("timestep").filter(~equals_analytic_rate(channel)).unique().sort().implode().alias(channel)
        for channel in checkedanalytic
    ).collect()
    dfmontecarlo = (
        dfestim
        .group_by("timestep")
        .agg(
            (equals_analytic_rate(channel).all() & (pl.col(f"emission_ana_{channel}") > 0.0).any()).alias(channel)
            for channel in checkedmontecarlo
        )
        .collect()
    )
    for scheme, channel, timesteps in [
        *(("an analytic", channel, dfanalytic[channel].item().to_list()) for channel in checkedanalytic),
        *(
            ("a Monte Carlo", channel, dfmontecarlo.filter(pl.col(channel))["timestep"].sort().to_list())
            for channel in checkedmontecarlo
        ),
    ]:
        if timesteps:
            msg = (
                f"artisoptions.h gives {scheme} rate for the deposition channel {channel}, but the rows of the"
                f" timesteps {timesteps[:10]} do not follow that rule. A run of a different ARTIS build can have a"
                " different PARTICLE_THERMALISATION_SCHEME, and one rule for the rows of all the runs gives an"
                " incorrect rate. Select the timesteps of one run with -timestep"
            )
            raise ValueError(msg)


def parse_ye_range(text: str) -> tuple[float, float]:
    """Return the lower end and the upper end of an init_Ye range such as 0-0.2.

    An init_Ye is the number of electrons per nucleon, thus each end must be 0 to 1. That comparison
    also refuses a nan and an infinity, because each comparison with a nan gives false.
    """
    yemin, yemax = sorted(parse_float_range(text, "an init_Ye range, such as 0-0.2"))
    if not 0.0 <= yemin <= yemax <= 1.0:
        msg = f"{text!r} names an init_Ye outside 0 to 1"
        raise ValueError(msg)

    return yemin, yemax


def check_ye_range(modelpath: Path | str, yerange: tuple[float, float], verbose: bool = False) -> tuple[int, int]:
    """Return how many cells an init_Ye range selects and how many hold matter, and print the counts.

    A range that selects no cell gives an error.

    A cell with no matter gives no deposition rate, and artistools writes an init_Ye of 0 for such a
    cell. Thus the count covers the cells that hold matter, and a range of a low init_Ye gets the
    number of cells that the table sums.

    The query selects only the density and Ye, and it reads no abundance. Thus a range that selects no
    cell stops the command before the much slower read of the estimator files, and the check itself
    costs little on a model of many cells.
    """
    modelname = get_model_name(modelpath)
    dfmodel, modelmeta = get_modeldata(modelpath, printwarningsonly=not verbose)
    dfmodel = add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
    if "Ye" not in dfmodel.collect_schema().names():
        msg = f"{modelname} gives no init_Ye of a cell. The model file must hold a Ye column"
        raise ValueError(msg)

    yemin, yemax = yerange
    modelyemin, modelyemax, cellsinrange, cellswithmatter = (
        dfmodel
        .filter(pl.col("rho") > 0)
        .select(
            modelyemin=pl.col("Ye").min(),
            modelyemax=pl.col("Ye").max(),
            cellsinrange=pl.col("Ye").is_between(yemin, yemax).sum(),
            cellswithmatter=pl.len(),
        )
        .collect()
        .row(0)
    )
    if not cellsinrange:
        # a model with no cell of matter, and a model of null Ye values, both give no minimum
        yeofmodel = (
            "It gives no init_Ye of a cell that holds matter"
            if modelyemin is None
            else f"Its cells with matter have an init_Ye of {modelyemin:g} to {modelyemax:g}"
        )
        msg = f"{modelname} has no cell with matter and an init_Ye of {yemin:g} to {yemax:g}. {yeofmodel}"
        raise ValueError(msg)

    print(
        f"{modelname}: {cellsinrange} of {cellswithmatter} cells with matter "
        f"({cellsinrange / cellswithmatter:.1%}) have an init_Ye of {yemin:g} to {yemax:g}"
    )

    return cellsinrange, cellswithmatter


def aggregate_deposition_rates(
    dfestim: pl.LazyFrame,
    timesteps: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    yerange: tuple[float, float] | None = None,
    configuredanalyticchannels: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return the deposition rate per unit volume, per ion, and per unit mass of each timestep.

    ARTIS writes the Monte Carlo deposition rate of a timestep into the estimators of the next timestep. It divides that
    rate by the volume of the timestep that took the energy. Thus the rate of timestep n comes from the rows of timestep
    n + 1 and from the column volume_prevtimestep. If the selection contains a Monte Carlo channel, the last timestep
    gets no row, because only a later timestep holds that rate. An analytic channel, see
    get_analytic_deposition_channels, holds the rate of its own row, and it takes the volume of that row.

    The join on the cell gives one row of one timestep to each cell. Thus the rate, the ion count,
    the volume, and the mass of a row cover one set of cells.

    An init_Ye range keeps the cells whose init_Ye is inside it. Both ends of the range count.

    configuredanalyticchannels gives the analytic channels from the options of the run, see
    get_configured_analytic_channels. Without it, a channel that equals its emission_ana_ column in each row is
    analytic. A Monte Carlo channel can also agree, e.g. when both are zero in the selected rows.
    """
    colnames = dfestim.collect_schema().names()
    if "nntot" not in colnames:
        msg = (
            "The estimators of this model hold no number density of the elements, "
            "thus the command cannot count the ions"
        )
        raise ValueError(msg)

    if yerange is not None:
        # join_cell_modeldata puts an init_ prefix on each column of the model snapshot
        if "init_Ye" not in colnames:
            msg = "This model gives no init_Ye of a cell, thus an init_Ye range selects nothing"
            raise ValueError(msg)

        dfestim = dfestim.filter(pl.col("init_Ye").is_between(yerange[0], yerange[1]))

    selectedchannels = select_deposition_channels(colnames, channels)
    # the rates, the cells, and the test of the channels read the same columns. On a 3D model of 80 timesteps, one
    # .collect() of these columns took 1.90 GB, and a lazy query that read them for each branch took 2.29 GB
    neededcolumns = dict.fromkeys([
        *("timestep", "modelgridindex", "tmid_days", "mass_g", "volume", "volume_prevtimestep", "nntot"),
        *(f"{DEPOSITIONPREFIX}{channel}" for channel in selectedchannels),
        *(f"emission_ana_{channel}" for channel in selectedchannels if f"emission_ana_{channel}" in colnames),
    ])
    dfestim = dfestim.select(list(neededcolumns)).collect().lazy()
    if configuredanalyticchannels is not None:
        analyticchannels = [channel for channel in selectedchannels if channel in configuredanalyticchannels]
        # an analytic rate comes from the row of its timestep, and a Monte Carlo rate from the row of the next
        # timestep, thus each check reads only the rows that give its rates
        check_channels_of_each_run(
            dfestim if timesteps is None else dfestim.filter(pl.col("timestep").is_in(list(timesteps))),
            analyticchannels,
            [],
        )
        check_channels_of_each_run(
            dfestim if timesteps is None else dfestim.filter(pl.col("timestep").is_in([ts + 1 for ts in timesteps])),
            [],
            [channel for channel in selectedchannels if channel not in analyticchannels],
        )
        analyticsource = "The options of the run give an analytic rate for these deposition channels"
    else:
        # ARTIS gives an analytic rate for spontaneous fission with each scheme, thus that channel needs no evidence
        inferredchannels, undecidedchannels = get_analytic_deposition_channels(
            dfestim, [channel for channel in selectedchannels if channel != "spfission"]
        )
        analyticchannels = [*(["spfission"] if "spfission" in selectedchannels else []), *inferredchannels]
        analyticsource = (
            "artisoptions.h gives no particle thermalisation scheme. ARTIS gives an analytic rate for spfission, and"
            " the other channels here equal their analytic emission, which is not zero in each row"
        )
        if undecidedchannels:
            print_warning(
                "artisoptions.h gives no particle thermalisation scheme, and the deposition channels"
                f" {', '.join(undecidedchannels)} are zero in each selected row. Thus the scheme is not known, and"
                " each one takes the Monte Carlo scheme, which gives no rate for the last timestep"
            )
    montecarlochannels = [channel for channel in selectedchannels if channel not in analyticchannels]
    if analyticchannels:
        print_detail(
            f"{analyticsource}, thus each one gives the rate of the timestep of its own row:"
            f" {', '.join(analyticchannels)}"
        )

    dfrates_analytic = dfestim.select(
        "timestep", "modelgridindex", dep_analytic=get_channel_sum_expression(analyticchannels) * pl.col("volume")
    )
    if montecarlochannels:
        # each timestep needs the Monte Carlo rate from the row of the next timestep, thus the last timestep has none
        dfrates = (
            dfestim
            .select(
                timestep=pl.col("timestep") - 1,
                modelgridindex=pl.col("modelgridindex"),
                dep_erg_per_s=get_channel_sum_expression(montecarlochannels) * pl.col("volume_prevtimestep"),
            )
            .join(dfrates_analytic, on=("timestep", "modelgridindex"), how="left")
            .select(
                "timestep",
                "modelgridindex",
                dep_erg_per_s=pl.col("dep_erg_per_s") + pl.col("dep_analytic").fill_null(0.0),
            )
        )
    else:
        dfrates = dfrates_analytic.rename({"dep_analytic": "dep_erg_per_s"})

    # only a cell that holds matter has an ion. The filter also names is_finite, because a NaN
    # number density passes the comparison and gives no count
    dfcells = dfestim.filter(pl.col("nntot") > 0, pl.col("nntot").is_finite()).select(
        "timestep", "modelgridindex", "tmid_days", "mass_g", "volume", ioncount=pl.col("nntot") * pl.col("volume")
    )
    if timesteps is not None:
        dfcells = dfcells.filter(pl.col("timestep").is_in(list(timesteps)))

    # a cell can have no row at the next timestep, e.g. after a bad restart. The left join keeps the cell in the
    # denominators, and its rate is not known. A timestep with no rate row at all, e.g. the last one, gives no row
    return (
        dfcells
        .join(dfrates.with_columns(hasraterow=pl.lit(value=True)), on=("timestep", "modelgridindex"), how="left")
        .group_by("timestep")
        .agg(
            tmid_days=pl.col("tmid_days").first(),
            # a missing rate is not a zero, thus it makes the sum a NaN, as a NaN rate does
            dep_erg_per_s=pl.col("dep_erg_per_s").fill_null(float("nan")).sum(),
            ioncount=pl.col("ioncount").sum(),
            volume=pl.col("volume").sum(),
            mass_g=pl.col("mass_g").sum(),
            # the sum of a NaN is a NaN, thus the count of the cells with no rate goes beside it
            cellswithnorate=pl.col("dep_erg_per_s").is_finite().fill_null(value=False).not_().sum(),
            raterowcount=pl.col("hasraterow").fill_null(value=False).sum(),
        )
        .filter(pl.col("raterowcount") > 0)
        .with_columns(
            # a rate in erg divided by erg per eV gives the rate in eV
            dep_per_volume=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("volume"),
            dep_per_ion=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("ioncount"),
            dep_per_mass=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("mass_g"),
        )
        .select(
            "timestep",
            # the table prints a number, thus a null arrives there as a NaN
            *(
                pl.col(name).fill_null(float("nan"))
                for name in ("tmid_days", "dep_per_volume", "dep_per_ion", "dep_per_mass")
            ),
            "cellswithnorate",
        )
        .sort("timestep")
        .collect()
    )


def selects_only_analytic_channels(
    colnames: Sequence[str], channels: Sequence[str] | None, configuredanalyticchannels: Sequence[str] | None
) -> bool:
    """Return whether each selected channel holds an analytic rate, which is the rate of the timestep of its own row.

    With no options of the run, only spfission is analytic for certain, because ARTIS gives it with each scheme.
    """
    analyticchannels = configuredanalyticchannels if configuredanalyticchannels is not None else ["spfission"]
    return all(channel in analyticchannels for channel in select_deposition_channels(colnames, channels))


def get_deposition_rates(
    modelpath: Path | str,
    timesteps: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    yerange: tuple[float, float] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """Read the estimators of a model, and return the deposition rate of each timestep.

    A value of None for timesteps reads every timestep of the model. A value of None for yerange
    keeps every cell. Call check_ye_range in front of this function, which gives a message for a
    range that selects no cell.
    """
    readtimesteps = None if timesteps is None else tuple({*timesteps} | {timestep + 1 for timestep in timesteps})
    dfestim = scan_estimators(modelpath, timestep=readtimesteps, join_modeldata=True, verbose=verbose)
    configuredanalyticchannels = get_configured_analytic_channels(modelpath)
    # an analytic channel gives the rate of its own row, thus a later row of a restart with another scheme must not
    # stop the command when no selected channel needs it
    if timesteps is not None and selects_only_analytic_channels(
        dfestim.collect_schema().names(), channels, configuredanalyticchannels
    ):
        dfestim = dfestim.filter(pl.col("timestep").is_in(list(timesteps)))

    return aggregate_deposition_rates(
        dfestim, timesteps, channels, yerange, configuredanalyticchannels=configuredanalyticchannels
    )


def get_selected_timesteps(modelpath: Path | str, timedays: str | None, timestep: str | None) -> list[int] | None:
    """Return the timesteps that -timedays or -timestep selects, or None if the caller gives no time.

    Each argument also takes a list, e.g. 1,5,30. get_time_range reads one item of that list. Thus a
    time, a range of times, a timestep, a range of timesteps, and "last" mean here what they mean on
    every other command.
    """
    if timedays is timestep is None:
        return None

    selected: set[int] = set()
    for token in str(timestep if timedays is None else timedays).split(","):
        timestepmin, timestepmax, _, _ = (
            get_time_range(modelpath, timestep_range_str=token)
            if timedays is None
            else get_time_range(modelpath, timedays_range_str=token)
        )
        selected.update(range(timestepmin, timestepmax + 1))

    return sorted(selected)


def format_channel_list(modelpath: Path | str, verbose: bool = False) -> str:
    """Return the deposition channels that the estimators of a model hold.

    The scan reads the files of one cell, thus the listing costs much less than the table.
    """
    channels = get_deposition_channels(
        scan_estimators(modelpath, modelgridindex=0, verbose=verbose).collect_schema().names()
    )

    return f"{get_model_name(modelpath)} holds {', '.join(channels) if channels else 'no deposition channel'}"


def format_deposition_table(
    modelpath: Path | str,
    dftable: pl.DataFrame,
    channels: Sequence[str] | None,
    yerange: tuple[float, float] | None = None,
    yecells: tuple[int, int] | None = None,
) -> str:
    """Return the deposition rates of one model as a table of one row for each timestep.

    The counts that check_ye_range gives go in the title. Thus the file that -o writes says how many
    cells the init_Ye range selected, and --quiet takes no part of that away.
    """
    channeltext = ", ".join(channels) if channels else "every channel"
    yetext = (
        f" in {yecells[0]} of {yecells[1]} cells with matter, with an init_Ye of {yerange[0]:g} to {yerange[1]:g}"
        if yerange is not None and yecells is not None
        else ""
    )
    lines = [
        f"{get_model_name(modelpath)}: deposition of {channeltext}{yetext}",
        f"{'timestep':>8s} {'t_days':>10s} {'dep_per_volume':>15s} {'dep_per_ion':>15s} {'dep_per_mass':>15s}",
        f"{'':>8s} {'[d]':>10s} {'[eV/s/cm^3]':>15s} {'[eV/s]':>15s} {'[eV/s/g]':>15s}",
    ]
    lines.extend(
        f"{timestep:8d} {tmid_days:10.3f} {dep_per_volume:15.3e} {dep_per_ion:15.3e} {dep_per_mass:15.3e}"
        for timestep, tmid_days, dep_per_volume, dep_per_ion, dep_per_mass in dftable.select(
            "timestep", "tmid_days", "dep_per_volume", "dep_per_ion", "dep_per_mass"
        ).iter_rows()
    )

    return "\n".join(lines)


def warn_about_gaps(
    modelpath: Path | str,
    dftable: pl.DataFrame,
    timesteps: Sequence[int] | None,
    lasttimestep: int,
    *,
    onlyanalytic: bool = False,
) -> None:
    """Print a warning for each timestep that got no row, and for the cells that gave no rate.

    A Monte Carlo rate arrives in the row of the next timestep, thus a model gives no such rate for its last timestep.
    A selection of every timestep then asks for the timesteps in front of that one alone. An analytic channel gives
    the rate of its own row, thus a selection of analytic channels alone also asks for the last timestep. A complete
    model then gives no warning, and a model that stopped early still gives one.
    """
    modelname = get_model_logname(modelpath)
    gotrows = set(dftable["timestep"].to_list())
    wanted = set(timesteps) if timesteps is not None else set(range(lasttimestep + 1 if onlyanalytic else lasttimestep))
    if missing := sorted(wanted - gotrows):
        # a long list of timesteps hides the message, thus many of them give the count and the ends
        missingtext = (
            f"timestep {', '.join(str(timestep) for timestep in missing)}"
            if len(missing) <= 6
            else f"{len(missing)} timesteps, from {missing[0]} to {missing[-1]}"
        )
        # that rule explains the last timestep of a Monte Carlo rate alone. An analytic channel gives the rate of its
        # own row, thus a gap of an analytic selection or of a different timestep gets no reason
        reason = (
            " The rate of a timestep arrives in the file of the next timestep, "
            "thus the last timestep of a model gives none."
            if lasttimestep in missing and not onlyanalytic
            else ""
        )
        print_warning(f"{modelname}: no deposition rate for {missingtext}.{reason}")

    if cellswithnorate := int(dftable["cellswithnorate"].sum()):
        print_warning(f"{modelname}: {cellswithnorate} cells gave no finite deposition rate, thus a row holds a NaN")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths to ARTIS model folders")
    addarg_timedays(
        parser, kind="str", helptext="Times in days, e.g. 30, a range 10-30, or a list 1,5,30. Default: every timestep"
    )
    addarg_timestep(parser, helptext="Timesteps, e.g. 40, a range 45-65, a list 4,9, or last")
    parser.add_argument(
        "-channels",
        "-channel",
        dest="channels",
        action=CommaJoinAction,
        help=(
            "Deposition channels to add, e.g. gamma,electron,alpha. "
            "Default: every channel of the model. Give --listchannels to show the channels"
        ),
    )
    # --listchannels stops before it reads a cell, thus an init_Ye range with it selects nothing
    listgroup = parser.add_mutually_exclusive_group()
    listgroup.add_argument(
        "-ye",
        help=(
            "Select the cells whose init_Ye is in a range, e.g. 0-0.2. Both ends of the range count. "
            "init_Ye is the electrons per nucleon of the Ye column of the model file, thus the model "
            "file must hold that column. The estimator variable Ye is different, because the decay "
            "chains change the electrons per nucleon of a cell. Default: every cell"
        ),
    )
    listgroup.add_argument(
        "--listchannels", action="store_true", help="Show the deposition channels of the model. The command then stops"
    )
    addarg_output(parser, kind="file", helptext="Path/filename for the output text file")
    addarg_force(parser)
    addarg_verbose(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give the deposition rate of a model per unit volume, per ion, and per unit mass."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    channels = str(args.channels).split(",") if args.channels else None
    yerange = parse_ye_range(str(args.ye)) if args.ye is not None else None
    modelpaths = normalize_path_list(args.modelpath)
    outputfile = resolve_table_outputfile(args, DEFAULTOUTPUTNAME)

    # each model gets its check in front of the first slow read, thus a bad path wastes no work
    yecellcounts = [
        check_ye_range(modelpath, yerange, verbose=args.verbose) if yerange is not None else None
        for modelpath in modelpaths
    ]

    tables: list[str] = []
    for modelpath, yecells in zip(modelpaths, yecellcounts, strict=True):
        if args.listchannels:
            tables.append(format_channel_list(modelpath, verbose=args.verbose))
        else:
            timesteps = get_selected_timesteps(modelpath, args.timedays, args.timestep)
            dftable = get_deposition_rates(modelpath, timesteps, channels, yerange, verbose=args.verbose)
            onlyanalytic = selects_only_analytic_channels(
                scan_estimators(modelpath, verbose=args.verbose).collect_schema().names(),
                channels,
                get_configured_analytic_channels(modelpath),
            )
            warn_about_gaps(
                modelpath,
                dftable,
                timesteps,
                len(get_timestep_times(modelpath, loc="mid")) - 1,
                onlyanalytic=onlyanalytic,
            )
            tables.append(format_deposition_table(modelpath, dftable, channels, yerange, yecells))

        if outputfile is None:
            print_product(args, tables[-1])

    if outputfile is not None:
        provenancetext = "".join(f"# {line}\n" for line in get_provenance_lines(args))
        outputfile.write_text(provenancetext + "\n\n".join(tables) + "\n", encoding="utf-8")
        print_saved(outputfile)
