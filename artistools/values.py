"""Print the values of estimator variables for each cell and timestep, with no plot and no average.

Each row gives the value that ARTIS wrote for one cell at one timestep. Use this command to answer a question such
as "what is Te in cell 12 at 300 days?". The --print-data table of plotestimators gives the plotted line, which is a
weighted mean of the cells of each bin.
"""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import km_to_cm
from artistools.estimators import get_units
from artistools.estimators import scan_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_costheta_selection
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_costhetarange
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_positional_items
from artistools.misc import addarg_table_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import artis_subfolders
from artistools.misc import exit_with_error
from artistools.misc import folder_is_artis_model
from artistools.misc import format_costheta_range
from artistools.misc import get_costheta_range
from artistools.misc import get_model_folder
from artistools.misc import get_model_name
from artistools.misc import get_time_range
from artistools.misc import get_time_range_text
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import path_is_codecomparison
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_or_write_table
from artistools.misc import print_warning
from artistools.misc import resolve_positional_modelpath
from artistools.misc import resolve_table_outputfile
from artistools.misc import suggest_names
from artistools.misc.cliutils import format_range_list

# the columns of each row before the variables. The variables cannot repeat them
INDEXCOLUMNS = ("timestep", "time_days", "modelgridindex", "vel_r_mid_kmps")

# join_cell_modeldata calculates these columns from model.txt for homologous expansion to the mid time of the timestep
EXPANDEDMODELCOLUMNS = ("rho", "volume", "volume_prevtimestep")

# a selection with more timesteps than this gives only the range of the times, and not one line for each timestep
MAXLISTEDTIMESTEPS = 10

# a list of the cells that is longer than this gives only the first and the last cell
MAXCELLSTEXTLENGTH = 60

LISTVARIABLESHINT = "Run `artistools plotestimators --listvariables <folder>` to see the variables of the model"


def get_value_column_name(variable: str) -> str:
    """Return the name of the column of a variable with its units in square brackets, e.g. Te[K].

    A name with a space needs quotes in the table, thus an underscore replaces each space of the units.
    """
    units = get_units(variable, latex=False)
    return f"{variable}[{units.replace(' ', '_')}]" if units else variable


def get_values_table(
    modelpath: Path | str,
    variables: Sequence[str],
    timesteps: Sequence[int],
    modelgridindex: Sequence[int] | None = None,
    costhetarange: tuple[float, float] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """Return one row for each cell and timestep, with the value of each variable and no average.

    The cells are those of modelgridindex, or every cell with estimators when it is None. costhetarange then keeps the
    cells whose centre has cos θ = v_z / |v| in the range. ARTIS writes estimators only for a cell with matter. A
    variable that the model does not hold stops the command with a list of near names.
    """
    estimators = scan_estimators(
        modelpath=modelpath,
        modelgridindex=modelgridindex,
        timestep=tuple(timesteps),
        join_modeldata=True,
        verbose=verbose,
    )

    columns = estimators.collect_schema().names()
    for variable in variables:
        if variable not in columns:
            exit_with_error(
                f"'{variable}' is not a variable of the estimators or of the model file",
                f"{suggest_names(variable, columns)} {LISTVARIABLESHINT}".strip(),
            )
    if costhetarange is not None:
        estimators = estimators.filter(get_costheta_selection(costhetarange, columns))

    return (
        estimators
        .select(
            pl.col("timestep"),
            pl.col("tmid_days").alias("time_days"),
            pl.col("modelgridindex"),
            (pl.col("vel_r_mid") / km_to_cm).alias("vel_r_mid_kmps"),
            *(pl.col(variable).alias(get_value_column_name(variable)) for variable in variables),
        )
        .sort(["timestep", "modelgridindex"])
        .collect()
    )


def resolve_timesteps(args: argparse.Namespace, modelpath: Path) -> list[int]:
    """Return the timesteps of the time arguments, or every timestep when the user gave no time."""
    if all(value is None for value in (args.timedays, args.timemin, args.timemax, args.timestep)):
        args.timestep = f"0-{len(get_timestep_times(modelpath)) - 1}"
        print_detail("No time was given, thus the table holds every timestep")

    timestepmin, timestepmax, args.timemin, args.timemax = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    if timestepmin == timestepmax == -1:
        exit_with_error("The time range is outside the timesteps of the model", "Give a time inside that range")

    return list(range(timestepmin, timestepmax + 1))


def get_method_lines(
    modelpath: Path,
    variables: Sequence[str],
    timesteps: Sequence[int],
    dfvalues: pl.DataFrame,
    costhetarange: tuple[float, float] | None = None,
) -> list[str]:
    """Return the lines that tell which timesteps and cells the table holds, and where each column comes from."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tmids = get_timestep_times(modelpath, loc="mid")
    tends = get_timestep_times(modelpath, loc="end")

    # the line of a single timestep gives its range too, thus the range line would repeat it
    lines = (
        [get_time_range_text(timesteps[0], timesteps[-1], tstarts[timesteps[0]], tends[timesteps[-1]])]
        if len(timesteps) > 1
        else []
    )
    if len(timesteps) <= MAXLISTEDTIMESTEPS:
        lines.extend(
            f"timestep {timestep}: {tstarts[timestep]:.2f} to {tends[timestep]:.2f} d, mid time {tmids[timestep]:.2f} d"
            for timestep in timesteps
        )

    cells = dfvalues["modelgridindex"].unique().sort().to_list()
    cellstext = format_range_list(cells)
    if len(cellstext) > MAXCELLSTEXTLENGTH:
        cellstext = f"{cells[0]} to {cells[-1]}"
    if costhetarange is not None:
        lines.append(f"the cells have a centre with {format_costheta_range(costhetarange)}, where cos θ = v_z / |v|")
    lines.extend([
        f"{len(cells)} cells with estimators: modelgridindex {cellstext}",
        "each row gives the values of one cell at one timestep, with no average over cells or timesteps",
        "time_days is the mid time of the timestep",
        "vel_r_mid_kmps is the radial velocity at the middle of the cell, from the model file",
        "a column name gives the units in square brackets, e.g. Te[K]",
    ])

    if expanded := [variable for variable in variables if variable in EXPANDEDMODELCOLUMNS]:
        lines.append(
            f"the values of {', '.join(expanded)} come from model.txt with homologous expansion, where t_model is the"
            " time of model.txt"
        )
    if "rho" in variables:
        lines.append("rho = init_rho (t_model / time_days)^3, at the mid time of the timestep")
    if "volume" in variables:
        lines.append("volume = init_volume (time_days / t_model)^3, at the mid time of the timestep")
    if "volume_prevtimestep" in variables:
        lines.append(
            "volume_prevtimestep = init_volume (t_prev / t_model)^3, where t_prev is the mid time of the timestep"
            " before, and not time_days. The first timestep of the model has no timestep before, thus its value is nan"
        )
    if fromfile := [variable for variable in variables if variable.startswith("init_")]:
        lines.append(
            f"the values of {', '.join(fromfile)} come from model.txt at the time of the model file, with no expansion"
        )

    return lines


def warn_missing_rows(dfvalues: pl.DataFrame, timesteps: Sequence[int], modelgridindex: Sequence[int] | None) -> None:
    """Warn about each selected pair of cell and timestep that has no row, because a reader can expect one for each.

    With no --modelgridindex, the selected cells are the cells that have estimators at one or more selected timesteps.
    """
    cells = dfvalues["modelgridindex"].unique().to_list() if modelgridindex is None else list(modelgridindex)
    pairschema = dfvalues.select("modelgridindex", "timestep").schema
    dfmissing = (
        pl
        .DataFrame({"modelgridindex": cells})
        .join(pl.DataFrame({"timestep": list(timesteps)}), how="cross")
        .cast(pairschema)
        .join(dfvalues.select("modelgridindex", "timestep"), on=["modelgridindex", "timestep"], how="anti")
        .group_by("modelgridindex")
        .agg(pl.col("timestep").sort())
        .group_by("timestep")
        .agg(pl.col("modelgridindex").sort())
        .sort(pl.col("modelgridindex").list.first())
    )
    for missingtimesteps, missingcells in dfmissing.select("timestep", "modelgridindex").iter_rows():
        if len(missingtimesteps) == len(timesteps):
            print_warning(
                f"modelgridindex {format_range_list(missingcells)} has no estimators at the selected timesteps."
                " ARTIS writes no estimators for a cell with no matter"
            )
        elif len(missingcells) == len(cells):
            print_warning(f"Timestep {format_range_list(missingtimesteps)} has no estimators for the selected cells")
        else:
            print_warning(
                f"modelgridindex {format_range_list(missingcells)} has no estimators at timestep"
                f" {format_range_list(missingtimesteps)}, but has estimators at the other selected timesteps"
            )


def get_cells_in_costheta_range(modelpath: Path, cells: Sequence[int], costhetarange: tuple[float, float]) -> list[int]:
    """Return the cells of the list whose centre has cos θ = v_z / |v| in the range.

    A cell of the list outside the range has no row, and it can hold estimators. Thus warn_missing_rows must not
    name it.
    """
    lzmodel, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    dfcells = add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
    return (
        dfcells
        .filter(
            pl.col("modelgridindex").is_in(cells),
            get_costheta_selection(costhetarange, dfcells.collect_schema().names()),
        )
        .select("modelgridindex")
        .collect()["modelgridindex"]
        .to_list()
    )


def require_model_folder(modelpath: Path) -> None:
    """Stop with an error message when the path does not name a model folder."""
    if path_is_codecomparison(modelpath) or folder_is_artis_model(modelpath):
        return

    nearby = artis_subfolders(modelpath)
    helptext = (
        f"Name one of these subfolders after the variables: {', '.join(nearby)}"
        if nearby
        else "A model folder holds input.txt. Name one after the variables, e.g. `artistools values Te -t 300 mymodel`"
    )
    exit_with_error(f"'{modelpath.resolve()}' is not an ARTIS model folder", helptext)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_positional_items(
        parser,
        dest="variables",
        metavar="variable",
        helptext=(
            "The estimator variables, e.g. `artistools values Te TR nne -t 300`. The model folder comes after them,"
            " e.g. `artistools values Te mymodel`. The working folder is the default"
        ),
    )

    # the default is None and not Path(), thus resolve_positional_modelpath sees an explicit value
    addarg_modelpath(parser)

    addarg_modelgridindex(parser, helptext="Model grid cell or cells, e.g. 12, 3-7, or 1,4,9. The default is all cells")

    addarg_costhetarange(parser)

    addarg_timestep(parser, helptext="Timestep or range of timesteps, e.g. 40, 45-65, or last")

    addarg_timedays(parser, helptext="Time in days, or a range of times, e.g. 300 or 290-320")

    addarg_timeminmax(parser)

    addarg_table_output(parser)

    addarg_verbose(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Print the estimator values of each cell and timestep as a table, with no plot and no average."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    variables = [
        variable
        for variable in dict.fromkeys(resolve_positional_modelpath(args, "variables"))
        if variable not in INDEXCOLUMNS
    ]
    if not variables:
        exit_with_error(
            "No variable was given", f"Name the variables before the folder, e.g. Te TR -t 300. {LISTVARIABLESHINT}"
        )

    # the refusal must come before the work starts, e.g. before a remote model starts its server
    outputfile = resolve_table_outputfile(args, "values.txt")

    # a model path can name a file of the model, e.g. mymodel/light_curve.out
    modelpath = get_model_folder(args.modelpath)
    require_model_folder(modelpath)

    print(f"'{get_model_name(modelpath)}': values of {', '.join(variables)}")
    print_modelpath(modelpath)
    timesteps = resolve_timesteps(args, modelpath)
    costhetarange = get_costheta_range(args)

    dfvalues = get_values_table(
        modelpath, variables, timesteps, args.modelgridindex, costhetarange=costhetarange, verbose=args.verbose
    )
    if dfvalues.is_empty():
        exit_with_error("The selected cells hold no estimators at the selected timesteps", "Select different cells")
    expectedcells = (
        get_cells_in_costheta_range(modelpath, args.modelgridindex, costhetarange)
        if args.modelgridindex is not None and costhetarange is not None
        else args.modelgridindex
    )
    warn_missing_rows(dfvalues, timesteps, expectedcells)

    methodlines = get_method_lines(modelpath, variables, timesteps, dfvalues, costhetarange)
    for line in methodlines:
        print_detail(line)

    print_or_write_table(args, dfvalues, outputfile, methodlines)
