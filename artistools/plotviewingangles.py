"""Draw a 3D visualisation of an ARTIS model, with the direction bins as vectors around it."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import scan_modeldata
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import get_dirbin_centre_direction
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import import_optional
from artistools.misc import open_file
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.plottools import TRANSPARENT_DARKMODE_FORMATS


def gen_viewing_angle_df(length: int) -> pl.DataFrame:
    """Return the Cartesian endpoint of a vector of the given length that points into each direction bin."""
    viewing_angles: dict[str, list[float | str]] = {"Angle-bin": [], "x_coord": [], "y_coord": [], "z_coord": []}

    for i in range(get_viewingdirectionbincount()):
        x_c, y_c, z_c = (float(coordinate) for coordinate in length * get_dirbin_centre_direction(i))

        # 0 point
        viewing_angles["Angle-bin"].append(f"{i:02d}")
        viewing_angles["x_coord"].append(0.0)
        viewing_angles["y_coord"].append(0.0)
        viewing_angles["z_coord"].append(0.0)

        # end point
        viewing_angles["Angle-bin"].append(f"{i:02d}")
        viewing_angles["x_coord"].append(x_c)
        viewing_angles["y_coord"].append(y_c)
        viewing_angles["z_coord"].append(z_c)

    return pl.DataFrame(viewing_angles)


def get_plotted_tables(
    cellpositions: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    rho: npt.NDArray[np.float64],
    va: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Return the density of each cell in the volume, and the points of the direction bin lines.

    The lines have a different set of points from the volume, thus they go in the second table. Each line goes from
    the origin to its end point, thus the second table has two rows for each direction bin.
    """
    x, y, z = cellpositions
    return {
        "": pl.DataFrame({"pos_x_mid_cm": x, "pos_y_mid_cm": y, "pos_z_mid_cm": z, "rho_g_per_cm3": rho}),
        "_directions": va.select(
            pl.col("Angle-bin").cast(pl.Int64).alias("dirbin"),
            pl.col("x_coord").alias("x_cm"),
            pl.col("y_coord").alias("y_cm"),
            pl.col("z_coord").alias("z_cm"),
        ),
    }


def viewing_angles_visualisation(args: argparse.Namespace) -> None:
    """Draw a 3D visualisation of an ARTIS model, with an animation of the direction bins.

    The minimum and the maximum density of the model are the limits of the colour scale that -isomin and -isomax do
    not give. --print-data and --write-data give the plotted data as text.
    """
    outfile = str(args.outputfile)

    px = import_optional("plotly.express")
    go = import_optional("plotly.graph_objects")

    # the volume holds the density of each cell, thus the model gives the positions and the densities
    lzmodel, modelmeta = scan_modeldata(args.modelpath)
    dfmodel = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .select("pos_x_mid", "pos_y_mid", "pos_z_mid", "rho")
        .collect()
    )
    x, y, z = (dfmodel[f"pos_{ax}_mid"].cast(pl.Float64).to_numpy() for ax in ("x", "y", "z"))
    rho = dfmodel["rho"].cast(pl.Float64).to_numpy()

    isomin = min(rho.flatten()) if args.isomin is None else args.isomin
    isomax = max(rho.flatten()) if args.isomax is None else args.isomax
    assert isomin < isomax, "isomin must be smaller than isomax"

    # the vectors reach the edge of the box at a line length of one
    length = max(x.flatten()) * args.linelength
    va = gen_viewing_angle_df(length)
    if args.print_data or args.write_data:
        output_data(args, get_plotted_tables((x, y, z), rho, va), outfile)

    fig = px.line_3d(
        va,
        x="x_coord",
        y="y_coord",
        z="z_coord",
        color="Angle-bin",
        animation_frame="Angle-bin",
        hover_name="Angle-bin",
    )
    fig.update_traces(line={"width": args.linewidth})
    fig.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})

    fig = fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=rho.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=args.opacity,  # a small value makes every surface visible through the ones in front of it
            surface_count=args.surface_count,  # a large number gives a smooth volume
            colorbar={"title": "Density (g/cm³)"},
        )
    )
    fig.update_layout(
        scene_xaxis_showticklabels=False, scene_yaxis_showticklabels=False, scene_zaxis_showticklabels=False
    )

    if args.darkmode:
        istransparent = Path(outfile).suffix.removeprefix(".") in TRANSPARENT_DARKMODE_FORMATS
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)" if istransparent else "black")

    if outfile.endswith("html"):
        fig.write_html(outfile, auto_play=False)
    else:
        fig.write_image(outfile)
    print_saved(outfile)

    if args.show:
        fig.show()
    if args.open:
        open_file(outfile)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of the ARTIS model")
    addarg_output(
        parser,
        kind="file",
        defaultname="plotviewingangles.html",
        helptext="The name of the output file. A name that holds 'html' gives an html file with the animation",
    )
    parser.add_argument("-isomin", type=float, help="The smallest density of the colour scale")
    parser.add_argument("-isomax", type=float, help="The largest density of the colour scale")
    parser.add_argument("-opacity", type=float, default=0.25, help="The opacity of the isosurfaces")
    parser.add_argument("-surface_count", "-s", type=int, default=20, help="The number of isosurfaces")
    parser.add_argument("-linewidth", type=float, default=2.5, help="The width of the direction bin lines")
    parser.add_argument(
        "-linelength",
        type=float,
        default=1.0,
        help="The length of the direction bin lines, in units of the size of the box",
    )
    addarg_show(parser)
    # the earlier name of --show
    parser.add_argument("--show_plot", dest="show", action="store_true", help=argparse.SUPPRESS)
    addarg_darkmode(parser)
    addarg_data_output(parser)

    # the older spellings with two dashes. The parser gives --isomin and the others to the flags above
    parser.add_argument("--outfile", dest="outputfile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--surface_count", dest="surface_count", type=int, help=argparse.SUPPRESS)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Draw a 3D visualisation of an ARTIS model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    viewing_angles_visualisation(args)
