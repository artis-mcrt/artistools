"""Script for computing binned expansion opacities and Planck-mean opacities in postprocessing."""

import argparse
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.lines.core import addarg_excitationtemperature
from artistools.lines.core import get_cell_batches
from artistools.lines.core import get_cell_estimators
from artistools.lines.core import get_excitation_temperature_column
from artistools.lines.core import get_expansion_opacities
from artistools.lines.core import get_lambda_bin_edges
from artistools.lines.core import get_opacity_atomic_data
from artistools.lines.core import get_opacity_lines
from artistools.lines.core import get_planck_mean_opacities
from artistools.lines.core import print_planck_mean_method
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import get_model_logname
from artistools.misc import get_selected_timestep
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args
from artistools.misc import print_modelpath
from artistools.misc import print_product
from artistools.misc import wavelength_angstroms_arg
from artistools.misc import WAVELENGTH_UNIT_HELP


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_timestep(parser, helptext="Timestep number to select")
    addarg_timedays(parser, kind="str", helptext="Time in days to select")
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of the ARTIS model")
    parser.add_argument(
        "--show_binned_opacities",
        action="store_true",
        help="Show the binned opacities for each cell (can be very large)",
    )
    addarg_modelgridindex(parser, helptext="Model grid cell to select. If not specified, all cells are processed")

    # every command that reads a range of wavelengths takes both spellings, thus -xmin reaches this
    # command as well. This one bins the opacities over that range, and it draws no plot
    parser.add_argument(
        "-xmin",
        "-lambdamin",
        type=wavelength_angstroms_arg,
        default=20.0,
        help=f"Minimum wavelength of the binned opacities. {WAVELENGTH_UNIT_HELP}",
    )
    parser.add_argument(
        "-xmax",
        "-lambdamax",
        type=wavelength_angstroms_arg,
        default=50000.0,
        help=f"Maximum wavelength of the binned opacities. {WAVELENGTH_UNIT_HELP}",
    )
    parser.set_defaults(xrangeisselection=True)
    parser.add_argument(
        "-deltalambda", type=float, default=10.0, help="Wavelength bin width in Angstroms for binned opacities"
    )
    addarg_excitationtemperature(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compute binned expansion opacities and Planck-mean opacities in postprocessing."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    time_days = get_timestep_time(args.modelpath, timestep)
    print(f"Opacities of {get_model_logname(args.modelpath)} at {time_days:.2f}d (timestep {timestep})")
    print_modelpath(args.modelpath)
    temperaturecolumn = get_excitation_temperature_column(args.modelpath, args.exctemperature)
    print_planck_mean_method(args.xmin, args.xmax)
    dfestimators = get_cell_estimators(
        args.modelpath, timestep, get_single_modelgridindex(args.modelgridindex), temperaturecolumn
    )

    lambda_bin_edges = get_lambda_bin_edges(args.xmin, args.xmax, args.deltalambda)
    opacitylines = get_opacity_lines(
        get_opacity_atomic_data(args.modelpath), dfestimators.columns, lambda_bin_edges, time_days
    )

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(5000)
    cellcount = dfestimators.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()
    planckmeanopacity_times_mass = 0.0
    mass_g_sum = 0.0
    for dfcellbatch in get_cell_batches(dfestimators, len(lambda_bin_edges) - 1):
        dfbinnedopacities = get_expansion_opacities(opacitylines, dfcellbatch, lambda_bin_edges, time_days)
        if args.show_binned_opacities:
            print_product(args, dfbinnedopacities)

        dfplanckmean = get_planck_mean_opacities(dfbinnedopacities)

        print_product(args, dfplanckmean)
        planckmeanopacity_times_mass += (dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g")))).item()
        mass_g_sum += dfplanckmean.select(pl.col("mass_g").sum()).item()

        cells_processed += dfcellbatch.select(pl.len()).item()
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f" Average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}."
            f" time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    print()
    globalplanckmeanopacity = planckmeanopacity_times_mass / mass_g_sum
    print_product(args, f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")
