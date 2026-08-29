"""Animate the ARTIS viewing angle bins as vectors around a 3D model."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import artistools as at
from artistools.misc import addarg_output
from artistools.misc import resolve_outputfile


def get_theta_phi(anglebin: int) -> tuple[float, float]:
    """Return the central theta and phi angles of the given viewing angle bin.

    The bin boundaries come from the shared definitions in artistools.misc.dirbins, so the arrows
    point where every other artistools plot puts the same bin.
    """
    costhetabin, phibin = divmod(anglebin, at.get_viewingdirection_phibincount())
    costheta_lower, costheta_upper, _ = at.get_costheta_bins(usedegrees=False)
    phi_lower, phi_upper, _ = at.get_phi_bins(usedegrees=False)
    theta = float(np.arccos((costheta_lower[costhetabin] + costheta_upper[costhetabin]) / 2))
    phi = float((phi_lower[phibin] + phi_upper[phibin]) / 2)
    return theta, phi


def gen_viewing_angle_df(length: int) -> pl.DataFrame:
    """Return the Cartesian endpoint of a vector of the given length pointing into each viewing angle bin."""
    viewing_angles: dict[str, list[float | str]] = {"Angle-bin": [], "x_coord": [], "y_coord": [], "z_coord": []}

    for i in range(at.get_viewingdirectionbincount()):
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
    """Tool to generate a 3D visualization of an ARTIS model. Viewing angle bins will get overplotted with an animation.

    Parameters
    ----------
    modelfile : str
        File where ARTIS  model is stored.
    outfile : str
        Name of the output file. If name contains 'html',
        figure will be stored as html file including
        the animation
    isomin : float
        Minimum density value for the color coding
    isomax : float
        Maximum density value for the color coding
    opacity : float
        Opacity value
    surface_count : int
        Number of isosurfaces plotted
    linewidth : float
        Width of the viewing angle lines
    linelength : float
        Length of the viewing angle lines in units
        of the boxsize
    show_plot : bool
        If True, plot will be shown after saving

    Returns
    -------
    isomin, isomax : float | int, float

    """
    px = at.import_optional("plotly.express")
    go = at.import_optional("plotly.graph_objects")

    # Load model contents
    # get_modeldata takes the name of each column, and "pos_mid" names none of them, thus the read
    # gave no position column and the command stopped at the first one that it reads
    dfmodel = at.get_modeldata(modelfile, derived_cols=["pos_x_mid", "pos_y_mid", "pos_z_mid"])[0].collect()
    x, y, z = (dfmodel[f"pos_{ax}_mid"].cast(pl.Float64).to_numpy() for ax in ("x", "y", "z"))
    rho = dfmodel["rho"].cast(pl.Float64).to_numpy()

    if isomin is None:
        isomin = min(rho.flatten())
    if isomax is None:
        isomax = max(rho.flatten())
    assert isomin is not None
    assert isomax is not None
    assert isomin < isomax, "isomin must be smaller than isomax"

    # Generate viewing angle vectory
    length = max(x.flatten()) * linelength
    va = gen_viewing_angle_df(length)

    # Create plot
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
            opacity=opacity,  # needs to be small to see through all surfaces
            surface_count=surface_count,  # needs to be a large number for good volume rendering
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
    parser.add_argument("modelfile", help="Path to the ARTIS model")
    addarg_output(
        parser,
        kind="file",
        helptext="Name of the output file. If it contains 'html', figure will be stored as html including the animation",
    )
    parser.add_argument("-isomin", type=float, help="Minimum density for color coding")
    parser.add_argument("-isomax", type=float, help="Maximum density for color coding")
    parser.add_argument("-opacity", type=float, default=0.25, help="Opacity value")
    parser.add_argument("-surface_count", "-s", type=int, default=20, help="Number of isosurfaces plotted")
    parser.add_argument("-linewidth", type=float, default=2.5, help="Width of the viewing angle lines")
    parser.add_argument(
        "-linelength", type=float, default=1.0, help="Length of the viewing angle lines in units of the boxsize"
    )
    parser.add_argument("--show_plot", action="store_true", help="If flag is given, plot will be shown after saving")

    # deprecated double-dash spellings kept as hidden aliases
    parser.add_argument("--outfile", dest="outputfile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--isomin", dest="isomin", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--isomax", dest="isomax", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--opacity", dest="opacity", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--surface_count", dest="surface_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--linewidth", dest="linewidth", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--linelength", dest="linelength", type=float, help=argparse.SUPPRESS)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: Any) -> None:
    """Tool to generate a 3D visualization of an ARTIS model."""
    args = at.parse_cli_args(addargs, "Generate a 3D visualization of an ARTIS model.", args, argsraw, kwargs)

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


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
