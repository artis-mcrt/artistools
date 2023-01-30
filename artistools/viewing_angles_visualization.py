#!/usr/bin/env python3
import sys
import argparse

import numpy as np
import pandas as pd


def load_artis_model(modelfile):
    with open(modelfile, "r") as f:
        n = f.readline()
        res = int(np.cbrt(int(n)))
        rho = np.zeros([res, res, res])
        x = np.zeros([res, res, res])
        y = np.zeros([res, res, res])
        z = np.zeros([res, res, res])

        t_model = f.readline()
        _ = f.readline()

        for l in range(0, res):
            for k in range(0, res):
                for i in range(0, res):
                    _, dum2, dum3, dum4, rhoread = f.readline().split()
                    _, _, _, _, _ = f.readline().split()
                    rho[i, k, l] = rhoread
                    x[i, k, l] = dum2
                    y[i, k, l] = dum3
                    z[i, k, l] = dum4

    return x, y, z, rho, t_model, res


def get_theta_phi(anglebin):
    assert isinstance(anglebin, int), "Anglebin has to be int"
    cos_theta = [
        -0.9,
        -0.7,
        -0.5,
        -0.3,
        -0.1,
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ]
    theta = np.arccos(cos_theta)
    phi = (
        np.array(
            [
                0.1,
                0.3,
                0.5,
                0.7,
                0.9,
                1.9,
                1.7,
                1.5,
                1.3,
                1.1,
            ]
        )
        * np.pi
    )

    anglenumber = 0
    for t in theta:
        for p in phi:
            if int(anglenumber) == anglebin:
                return t, p

            anglenumber += 1

    return None, None


def gen_viewing_angle_df(l):
    # Build viewing angle vector DataFrame
    viewing_angles = {
        "Angle-bin": [],
        "x_coord": [],
        "y_coord": [],
        "z_coord": [],
    }

    for i in range(100):
        theta, phi = get_theta_phi(i)
        assert theta is not None
        assert phi is not None
        x_c = l * np.sin(theta) * np.cos(phi)
        y_c = l * np.sin(theta) * np.sin(phi)
        z_c = l * np.cos(theta)

        # 0 point
        viewing_angles["Angle-bin"].append("%02d" % i)
        viewing_angles["x_coord"].append(0.0)
        viewing_angles["y_coord"].append(0.0)
        viewing_angles["z_coord"].append(0.0)

        # end point
        viewing_angles["Angle-bin"].append("%02d" % i)
        viewing_angles["x_coord"].append(x_c)
        viewing_angles["y_coord"].append(y_c)
        viewing_angles["z_coord"].append(z_c)

    return pd.DataFrame(viewing_angles)


def main(
    modelfile,
    outfile=None,
    isomin=None,
    isomax=None,
    opacity=2.5,
    surface_count=20,
    linewidth=2.5,
    linelength=1.0,
    show_plot=False,
):
    """
    Tool to generate a 3D visualization of an ARTIS model.
    Viewing angle bins will get overplottet with an animation.

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
    isomin, isomax : float, float
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ModuleNotFoundError:
        print("Cannot run visualization without plotly...")
        sys.exit()

    # Load model contents
    x, y, z, rho, _, res = load_artis_model(modelfile)
    print("Found model with %d cubed resolution" % res)

    if isomin is None:
        isomin = min(rho.flatten())
    if isomax is None:
        isomax = max(rho.flatten())

    # Generate viewing angle vectory
    l = max(x.flatten()) * linelength
    va = gen_viewing_angle_df(l)

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
    fig.update_traces(line=dict(width=linewidth))
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )

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
        print("Figure saved as %s" % outfile)

    if show_plot:
        fig.show()

    return (isomin, isomax)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "modelfile",
        help="Path to the ARTIS model.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Name of the output file. If it contains 'html', figure will be stored as html including the animation.",
    )
    parser.add_argument(
        "--isomin",
        type=float,
        help="Minimum density for color coding.",
    )
    parser.add_argument(
        "--isomax",
        type=float,
        help="Maximum density for color coding.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        help="Opacity value. Default: 0.25",
        default=0.25,
    )
    parser.add_argument(
        "-s",
        "--surface_count",
        type=int,
        help="Number of isosurfaces plotted. Default: 20",
        default=20,
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        help="Width of the viewing angle lines. Default: 2.5",
        default=2.5,
    )
    parser.add_argument(
        "--linelength",
        type=float,
        help="Length of the viewing angle lines in units of the boxsize. Default: 1.0",
        default=1.0,
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="If flag is given, plot will be shown after saving.",
    )

    args = parser.parse_args()

    main(
        modelfile=args.modelfile,
        outfile=args.outfile,
        isomin=args.isomin,
        isomax=args.isomax,
        opacity=args.opacity,
        surface_count=args.surface_count,
        linewidth=args.linewidth,
        linelength=args.linelength,
        show_plot=args.show_plot,
    )

if __name__ == "__main__":
    cli()