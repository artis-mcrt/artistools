#!/usr/bin/env python3
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import artistools as at

defaultoutputfile = "plotmacroatom_cell{0:03d}_{1:03d}-{2:03d}.pdf"


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modelpath", nargs="?", default="", help="Path to ARTIS folder")
    parser.add_argument("-timestep", type=int, default=10, help="Timestep number to plot, or -1 for last")
    parser.add_argument("-timestepmax", type=int, default=-1, help="Make plots for all timesteps up to this timestep")
    parser.add_argument("-modelgridindex", "-cell", type=int, default=0, help="Modelgridindex to plot")
    parser.add_argument("element", nargs="?", default="Fe", help="Plotted element")
    parser.add_argument("-xmin", type=int, default=1000, help="Plot range: minimum wavelength in Angstroms")
    parser.add_argument("-xmax", type=int, default=15000, help="Plot range: maximum wavelength in Angstroms")
    parser.add_argument(
        "-o", action="store", dest="outputfile", default=defaultoutputfile, help="Filename for PDF file"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the macroatom transitions."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter, description="Plot ARTIS macroatom transitions."
        )
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    if Path(args.outputfile).is_dir():
        args.outputfile = str(Path(args.outputfile, defaultoutputfile))

    atomic_number = at.get_atomic_number(args.element.lower())
    if atomic_number < 1:
        print(f"Could not find element '{args.element}'")
        raise AssertionError

    timestepmin = args.timestep

    timestepmax = timestepmin if not args.timestepmax or args.timestepmax < 0 else args.timestepmax

    input_files = list(Path(args.modelpath).glob("**/macroatom_????.out*"))

    if not input_files:
        print("No macroatom files found")
        raise FileNotFoundError

    dfall = read_files(input_files, args.modelgridindex, timestepmin, timestepmax, atomic_number)

    specfilename = Path(args.modelpath, "spec.out")

    if not specfilename.is_file():
        print(f"Could not find {specfilename}")
        raise FileNotFoundError

    outputfile = args.outputfile.format(args.modelgridindex, timestepmin, timestepmax)
    make_plot(
        dfall,
        args.modelpath,
        timestepmin,
        timestepmax,
        outputfile,
        xmin=args.xmin,
        xmax=args.xmax,
        modelgridindex=args.modelgridindex,
    )


def make_plot(
    dfmacroatom: pd.DataFrame,
    modelpath: str,
    timestepmin: int,
    timestepmax: int,
    outputfile: str,
    xmin: int,
    xmax: int,
    modelgridindex: int,
) -> None:
    time_days_min = at.get_timestep_time(modelpath, timestepmin)
    time_days_max = at.get_timestep_time(modelpath, timestepmax)

    print(f"Plotting {len(dfmacroatom)} transitions")

    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(6, 6), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )

    axis.annotate(
        f"Timestep {timestepmin:d} to {timestepmax:d} (t={time_days_min} to {time_days_max})\nCell {modelgridindex:d}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=8,
    )

    with np.errstate(divide="ignore"):
        lambda_cmf_in = 2.99792458e18 / dfmacroatom["nu_cmf_in"].to_numpy()
        lambda_cmf_out = 2.99792458e18 / dfmacroatom["nu_cmf_out"].to_numpy()
    # axis.scatter(lambda_cmf_in, lambda_cmf_out, s=1, alpha=0.5, edgecolor='none')
    axis.plot(
        lambda_cmf_in,
        lambda_cmf_out,
        linestyle="none",
        marker="o",  # alpha=0.5,
        markersize=2,
        markerfacecolor="red",
        markeredgewidth=0,
    )
    axis.set_xlabel(r"Wavelength in ($\AA$)")
    axis.set_ylabel(r"Wavelength out ($\AA$)")
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(xmin, xmax)

    # axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})

    print(f"Saving to {outputfile:s}")
    fig.savefig(outputfile, format="pdf")
    plt.close()


def read_files(
    files: Sequence[Path | str],
    modelgridindex: int | None = None,
    timestepmin: int | None = None,
    timestepmax: int | None = None,
    atomic_number: int | None = None,
) -> pd.DataFrame:
    dfall = None
    if not files:
        print("No files")
    else:
        for filepath in files:
            print(f"Reading {filepath}...")

            df_thisfile = pd.read_csv(filepath, sep=r"\s+")
            # df_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)
            if modelgridindex:
                df_thisfile = df_thisfile[df_thisfile["modelgridindex"] == modelgridindex]
            if timestepmin is not None:
                df_thisfile = df_thisfile[df_thisfile["timestep"] >= timestepmin]
            if timestepmax:
                df_thisfile = df_thisfile[df_thisfile["timestep"] <= timestepmax]
            if atomic_number:
                df_thisfile = df_thisfile[df_thisfile["Z"] == atomic_number]

            if not df_thisfile.empty:
                if dfall is None:
                    dfall = df_thisfile.copy()
                else:
                    dfall = pd.concat([dfall, df_thisfile.copy()], ignore_index=True)
                assert isinstance(dfall, pd.DataFrame)

    if dfall is None or len(dfall) == 0:
        msg = "No data found"
        raise AssertionError(msg)

    return dfall


if __name__ == "__main__":
    main()
