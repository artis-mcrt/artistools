"""Convert ARTIS output to the file formats used by the HESMA model archive."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import artistools as at


def plot_hesma_spectrum(timeavg: float, axes: Sequence[mplax.Axes], hesmafile: Path | str) -> None:
    """Plot a HESMA reference spectrum at the time closest to timeavg onto each of axes."""
    hesma_spec = pd.read_csv(hesmafile, comment="#", sep=r"\s+", dtype=float)

    searchtimes = [float(x) for x in hesma_spec.keys()[1:]]

    closest_time = f"{at.match_closest_time(timeavg, searchtimes):.2f}"
    print(closest_time)

    # Scale distance to 1 Mpc
    dist_mpc = 1e-5  # HESMA specta at 10 pc
    hesma_spec[closest_time] *= dist_mpc**2  # refspecditance Mpc / 1 Mpc ** 2

    for ax in axes:
        ax.plot(hesma_spec["0.00"], hesma_spec[closest_time], label="HESMA model")


def plothesmaresspec(fig: mplfig.Figure, ax: mplax.Axes, specfiles: Sequence[Path | str]) -> None:
    """Plot the first five direction bins of each HESMA direction-resolved spectrum file."""
    for specfilename in specfiles:
        specdata = pl.from_pandas(pd.read_csv(specfilename, sep=r"\s+", header=None, dtype=float))

        res_specdata = {
            dirbin: pldf.collect().to_pandas(use_pyarrow_extension_array=True)
            for dirbin, pldf in at.split_multitable_dataframe(specdata).items()
        }

        new_column_names = res_specdata[0].iloc[0]
        new_column_names[0] = "lambda"
        print(new_column_names)

        for i in range(len(res_specdata)):
            res_specdata[i] = (
                res_specdata[i]
                .rename(columns=dict(zip(res_specdata[i].columns, new_column_names, strict=True)))
                .drop(res_specdata[i].index[0])
            )

        # 11.7935 d is the epoch these HESMA files were tabulated at, and 1e-5 rescales 10 pc to 1 Mpc
        for dirbin in range(5):
            ax.plot(
                res_specdata[dirbin]["lambda"], res_specdata[dirbin][11.7935] * (1e-5) ** 2, label=f"hesma {dirbin}"
            )

    fig.legend()


def make_hesma_vspecfiles(modelpath: Path, outpath: Path | None = None) -> None:
    """Write the virtual packet spectra of the first five direction bins in HESMA format."""
    if not outpath:
        outpath = modelpath
    modelname = at.get_model_name(modelpath)
    angles = [0, 1, 2, 3, 4]
    vpkt_config = at.get_vpkt_config(modelpath)
    angle_names = []

    for dirbin in angles:
        angle_names.append(rf"cos(theta) = {vpkt_config['cos_theta'][dirbin]}")
        print(rf"cos(theta) = {vpkt_config['cos_theta'][dirbin]}")
        vspecdata = (
            at.spectra
            .get_specpol_data(dirbin=dirbin, modelpath=modelpath)["I"]
            .collect()
            .to_pandas(use_pyarrow_extension_array=True)
        )

        timearray = vspecdata.columns.to_numpy()[1:]
        vspecdata = vspecdata.sort_values(by="nu", ascending=False)
        vspecdata["lambda_angstroms"] = at.constants.c_ang_per_s / vspecdata["nu"]
        for time in timearray:
            vspecdata[time] = vspecdata[time] * vspecdata["nu"] / vspecdata["lambda_angstroms"]
            vspecdata[time] *= 100000.0**2  # Scale to 10 pc (1 Mpc/10 pc) ** 2

        vspecdata = vspecdata.set_index("lambda_angstroms").reset_index()
        vspecdata = vspecdata.drop(["nu"], axis=1)

        vspecdata = vspecdata.rename(columns={"lambda_angstroms": "0"})

        outfilename = f"{modelname}_vspec_res.dat"
        if dirbin == 0:
            vspecdata.to_csv(outpath / outfilename, sep=" ", index=False)  # create file
        else:
            # append to file
            vspecdata.to_csv(outpath / outfilename, mode="a", sep=" ", index=False)

    with (outpath / outfilename).open("r+") as f:  # add comment to start of file
        content = f.read()
        f.seek(0, 0)
        f.write(
            f"# File contains spectra at observer angles {angle_names} for Model {modelname}.\n# A header line"
            " containing spectra time is repeated at the beginning of each observer angle. Column 0 gives wavelength."
            " \n# Spectra are at a distance of 10 pc."
            "\n" + content
        )


def make_hesma_bol_lightcurve(modelpath: Path, outpath: Path, timemin: float, timemax: float) -> None:  # ruff:ignore[unused-function-argument]
    """UVOIR bolometric light curve (angle-averaged)."""
    lightcurvedataframe = at.lightcurve.get_bol_lc_from_lightcurveout(modelpath)
    print(lightcurvedataframe)
    lightcurvedataframe = lightcurvedataframe.query("time > @timemin and time < @timemax")

    modelname = at.get_model_name(modelpath)
    outfilename = f"doubledet_2021_{modelname}.dat"

    lightcurvedataframe.to_csv(outpath / outfilename, sep=" ", index=False, header=False)


def make_hesma_peakmag_dm15_dm40(
    band: str, pathtofiles: Path, modelname: str, outpath: Path, dm40: bool = False
) -> None:
    dm15filename = f"{band}band_{modelname}_viewing_angle_data.txt"
    dm15data = pd.read_csv(
        pathtofiles / dm15filename, sep=r"\s+", header=None, names=["peakmag", "risetime", "dm15"], skiprows=1
    )

    if dm40:
        dm40filename = f"{band}band_{modelname}_viewing_angle_data_deltam40.txt"
        dm40data = pd.read_csv(
            pathtofiles / dm40filename, sep=r"\s+", header=None, names=["peakmag", "risetime", "dm40"], skiprows=1
        )

    outdata = {
        "peakmag": dm15data["peakmag"],  # dm15 peak mag probably more accurate - shorter time window
        "dm15": dm15data["dm15"],
        "angle_bin": np.arange(0, 100),
    }
    if dm40:
        outdata["dm40"] = dm40data["dm40"]

    outdataframe = pd.DataFrame(outdata).round(decimals=4)
    outdataframe.to_csv(outpath / f"{modelname}_width-luminosity.dat", sep=" ", index=False, header=True)


def save_or_show(fig: mplfig.Figure, outputfile: Path | str | None) -> None:
    """Save the figure when an output file was given, otherwise show it."""
    if outputfile:
        fig.savefig(outputfile)
        at.print_saved(outputfile)
    else:
        plt.show()
    plt.close(fig)


def plot_hesma_peakmag_dm15_dm40(pathtofiles: Path | str, outputfile: Path | str | None = None) -> None:
    """Plot peak magnitude against dm15 for every width-luminosity file in a folder."""
    fig, axis = plt.subplots()
    for filepath in sorted(Path(pathtofiles).iterdir()):
        print(f"Reading {filepath}")
        dfwidthlum = pd.read_csv(filepath, sep=r"\s+")
        axis.scatter(dfwidthlum["dm15"], dfwidthlum["peakmag"], label=filepath.stem)

    axis.invert_yaxis()
    axis.set_xlabel(r"$\Delta m_{15}$")
    axis.set_ylabel("Peak magnitude")
    axis.legend()

    save_or_show(fig, outputfile)


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        # optional so that main(argsraw=[], action=...) works, since parse_cli_args ignores
        # argsraw as soon as any keyword argument is given
        nargs="?",
        default=None,
        choices=[
            "vspecfiles",
            "bollightcurve",
            "widthluminosity",
            "plotwidthluminosity",
            "plotspectrum",
            "plotresspec",
        ],
        help=(
            "vspecfiles: write virtual packet spectra in HESMA format."
            " bollightcurve: write the angle-averaged bolometric light curve."
            " widthluminosity: build a peak magnitude/dm15 file from viewing angle data."
            " plotwidthluminosity: plot peak magnitude against dm15 for a folder of those files."
            " plotspectrum/plotresspec: plot a HESMA reference spectrum file."
        ),
    )
    at.add_modelpath_arg(parser, helptext="Path to ARTIS folder (vspecfiles, bollightcurve)")
    at.add_outputpath_arg(parser, astype=Path, helptext="Folder for the written HESMA files")
    # not -outputfile/-o, because add_outputpath_arg above already claims -o for the folder
    parser.add_argument("-plotfile", type=Path, help="Path for the plot, or omit to show it interactively")
    at.add_timeminmax_args(parser)
    parser.add_argument(
        "-hesmafile", type=Path, nargs="+", help="HESMA spectrum file(s) to plot (plotspectrum, plotresspec)"
    )
    parser.add_argument("-timedays", "-time", "-t", type=float, help="Time in days to plot (plotspectrum)")
    parser.add_argument("-band", default="B", help="Filter band of the viewing angle data (widthluminosity)")
    parser.add_argument("-modelname", help="Model name in the viewing angle filenames (widthluminosity)")
    parser.add_argument(
        "-pathtofiles", type=Path, help="Folder of viewing angle data (widthluminosity, plotwidthluminosity)"
    )
    parser.add_argument("--dm40", action="store_true", help="Also read the deltam40 file (widthluminosity)")


def require(value: t.Any, argname: str, action: str) -> t.Any:
    """Return value, or exit with a message naming the argument the action needs."""
    if value is None:
        print(f"ERROR: {action} requires {argname}")
        raise SystemExit(1)

    return value


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert ARTIS output to the file formats used by the HESMA model archive."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.action is None:
        print("ERROR: no action given. Run with --help to see the available actions.")
        raise SystemExit(1)

    outputpath = Path(args.outputpath)

    if args.action == "vspecfiles":
        make_hesma_vspecfiles(require(args.modelpath, "-modelpath", args.action), outputpath)

    elif args.action == "bollightcurve":
        make_hesma_bol_lightcurve(
            require(args.modelpath, "-modelpath", args.action),
            outputpath,
            require(args.timemin, "-timemin", args.action),
            require(args.timemax, "-timemax", args.action),
        )

    elif args.action == "widthluminosity":
        make_hesma_peakmag_dm15_dm40(
            args.band,
            require(args.pathtofiles, "-pathtofiles", args.action),
            require(args.modelname, "-modelname", args.action),
            outputpath,
            dm40=args.dm40,
        )

    elif args.action == "plotwidthluminosity":
        plot_hesma_peakmag_dm15_dm40(require(args.pathtofiles, "-pathtofiles", args.action), args.plotfile)

    elif args.action == "plotspectrum":
        fig, axis = plt.subplots()
        plot_hesma_spectrum(
            require(args.timedays, "-timedays", args.action),
            [axis],
            require(args.hesmafile, "-hesmafile", args.action)[0],
        )
        save_or_show(fig, args.plotfile)

    else:
        fig, axis = plt.subplots()
        plothesmaresspec(fig, axis, require(args.hesmafile, "-hesmafile", args.action))
        save_or_show(fig, args.plotfile)


if __name__ == "__main__":
    main()
