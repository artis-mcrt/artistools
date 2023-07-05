#!/usr/bin/env python3
import argparse
import glob
import os.path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def main(args=None, argsraw=None, **kwargs):
    """Plot the macroatom transitions."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter, description="Plot ARTIS macroatom transitions."
        )
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if Path(args.outputfile).is_dir():
        args.outputfile = str(Path(args.outputfile, defaultoutputfile))

    atomic_number = at.get_atomic_number(args.element.lower())
    if atomic_number < 1:
        print(f"Could not find element '{args.element}'")
        raise AssertionError

    timestepmin = args.timestep

    timestepmax = timestepmin if not args.timestepmax or args.timestepmax < 0 else args.timestepmax

    input_files = glob.glob(os.path.join(args.modelpath, "macroatom_????.out*"), recursive=True) + glob.glob(
        os.path.join(args.modelpath, "*/macroatom_????.out*"), recursive=True
    )

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
        str(specfilename),
        timestepmin,
        timestepmax,
        outputfile,
        xmin=args.xmin,
        xmax=args.xmax,
        modelgridindex=args.modelgridindex,
    )

    return 0


def make_plot(
    dfmacroatom,
    modelpath,
    specfilename,
    timestepmin,
    timestepmax,
    outputfile,
    xmin,
    xmax,
    modelgridindex,
    nospec=False,
    normalised=False,
):
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


def read_files(files, modelgridindex=None, timestepmin=None, timestepmax=None, atomic_number=None):
    dfall = None
    if not files:
        print("No files")
    else:
        for _, filepath in enumerate(files):
            print(f"Loading {filepath}...")

            df_thisfile = pd.read_csv(filepath, delim_whitespace=True)
            # df_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)
            if modelgridindex:
                df_thisfile = df_thisfile.query("modelgridindex==@modelgridindex")
            if timestepmin is not None:
                df_thisfile = df_thisfile.query("timestep>=@timestepmin")
            if timestepmax:
                df_thisfile = df_thisfile.query("timestep<=@timestepmax")
            if atomic_number:
                df_thisfile = df_thisfile.query("Z==@atomic_number")

            if df_thisfile is not None and len(df_thisfile) > 0:
                dfall = df_thisfile.copy() if dfall is None else dfall.append(df_thisfile.copy(), ignore_index=True)

        if dfall is None or len(dfall) == 0:
            print("No data found")

    return dfall


if __name__ == "__main__":
    sys.exit(main())
