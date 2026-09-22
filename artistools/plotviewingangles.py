"""Draw a 3D visualisation of an ARTIS model, with the direction bins as vectors around it."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_output
from artistools.misc import get_costheta_bins
from artistools.misc import get_phi_bins
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import resolve_outputfile


def get_theta_phi(anglebin: int) -> tuple[float, float]:
    """Return the central theta and phi angles of the given direction bin.

    The bin boundaries come from the shared definitions in artistools.misc.dirbins, so the arrows
    point where every other artistools plot puts the same bin.
    """
    costhetabin, phibin = divmod(anglebin, get_viewingdirection_phibincount())
    costheta_lower, costheta_upper, _ = get_costheta_bins(usedegrees=False)
    phi_lower, phi_upper, _ = get_phi_bins(usedegrees=False)
    theta = float(np.arccos((costheta_lower[costhetabin] + costheta_upper[costhetabin]) / 2))
    phi = float((phi_lower[phibin] + phi_upper[phibin]) / 2)
    return theta, phi


def gen_viewing_angle_df(length: int) -> pl.DataFrame:
    """Return the Cartesian endpoint of a vector of the given length that points into each direction bin."""
    viewing_angles: dict[str, list[float | str]] = {"Angle-bin": [], "x_coord": [], "y_coord": [], "z_coord": []}

    for i in range(get_viewingdirectionbincount()):
        theta, phi = get_theta_phi(i)
        x_c = length * np.sin(theta) * np.cos(phi)
        y_c = length * np.sin(theta) * np.sin(phi)
        z_c = length * np.cos(theta)

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


def viewing_angles_visualisation(
    modelfile: str,
    outfile: str | None = None,
    isomin: float | None = None,
    isomax: float | None = None,
    opacity: float = 2.5,
    surface_count: int = 20,
    linewidth: float = 2.5,
    linelength: float = 1.0,
    show_plot: bool = False,
) -> tuple[float, float]:
    """Draw a 3D visualisation of an ARTIS model, with an animation of the direction bins.

    The function returns the density limits of the colour scale, which it calculates when the caller
    gives none.

    Parameters
    ----------
    modelfile : str
        The path of the ARTIS model.
    outfile : str
        The name of the output file. A name that holds "html" gives an html file with the animation.
    isomin : float
        The smallest density of the colour scale.
    isomax : float
        The largest density of the colour scale.
    opacity : float
        The opacity of the isosurfaces.
    surface_count : int
        The number of isosurfaces.
    linewidth : float
        The width of the direction bin lines.
    linelength : float
        The length of the direction bin lines, in units of the size of the box.
    show_plot : bool
        True shows the plot after the function saves it.

    Returns
    -------
    isomin, isomax : float, float

    """
    px = import_optional("plotly.express")
    go = import_optional("plotly.graph_objects")

    # the volume holds the density of each cell, thus the model gives the positions and the densities
    lzmodel, modelmeta = get_modeldata(modelfile)
    dfmodel = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .select("pos_x_mid", "pos_y_mid", "pos_z_mid", "rho")
        .collect()
    )
    x, y, z = (dfmodel[f"pos_{ax}_mid"].cast(pl.Float64).to_numpy() for ax in ("x", "y", "z"))
    rho = dfmodel["rho"].cast(pl.Float64).to_numpy()

    if isomin is None:
        isomin = min(rho.flatten())
    if isomax is None:
        isomax = max(rho.flatten())
    assert isomin is not None
    assert isomax is not None
    assert isomin < isomax, "isomin must be smaller than isomax"

    # the vectors reach the edge of the box at a line length of one
    length = max(x.flatten()) * linelength
    va = gen_viewing_angle_df(length)

    fig = px.line_3d(
        va,
        x="x_coord",
        y="y_coord",
        z="z_coord",
        color="Angle-bin",
        animation_frame="Angle-bin",
        hover_name="Angle-bin",
    )
    fig.update_traces(line={"width": linewidth})
    fig.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})

    fig = fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=rho.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,  # a small value makes every surface visible through the ones in front of it
            surface_count=surface_count,  # a large number gives a smooth volume
            colorbar={"title": "Density (g/cm³)"},
        )
    )
    fig.update_layout(
        scene_xaxis_showticklabels=False, scene_yaxis_showticklabels=False, scene_zaxis_showticklabels=False
    )

    if outfile:
        if outfile.endswith("html"):
            fig.write_html(outfile, auto_play=False)
        else:
            fig.write_image(outfile)
        print(f"Figure saved as {outfile}")

    if show_plot:
        fig.show()

    return (isomin, isomax)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("modelfile", help="The path of the ARTIS model")
    addarg_output(
        parser,
        kind="file",
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
    parser.add_argument("--show_plot", action="store_true", help="Show the plot after the command saves it")

    # deprecated double-dash spellings kept as hidden aliases
    parser.add_argument("--outfile", dest="outputfile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--isomin", dest="isomin", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--isomax", dest="isomax", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--opacity", dest="opacity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--surface_count", dest="surface_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--linewidth", dest="linewidth", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--linelength", dest="linelength", type=float, help=argparse.SUPPRESS)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Draw a 3D visualisation of an ARTIS model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    viewing_angles_visualisation(
        modelfile=args.modelfile,
        # -o promises that a path with no file extension names a folder, which the command makes
        outfile=str(resolve_outputfile(args.outputfile, "plotviewingangles.html")) if args.outputfile else None,
        isomin=args.isomin,
        isomax=args.isomax,
        opacity=args.opacity,
        surface_count=args.surface_count,
        linewidth=args.linewidth,
        linelength=args.linelength,
        show_plot=args.show_plot,
    )
