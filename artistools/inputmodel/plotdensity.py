#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import polars as pl

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "modelpath",
        default=[],
        nargs="*",
        action=at.AppendPath,
        help="Path(s) to model.txt file(s) or folders containing model.txt)",
    )
    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Plot the radial density profile of an ARTIS model."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 6),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )

    if not args.modelpath:
        args.modelpath = ["."]

    for modelpath in args.modelpath:
        dfmodel, modelmeta = at.get_modeldata_polars(
            modelpath, derived_cols=["vel_r_min", "vel_r_mid", "vel_r_max", "cellmass_grams"]
        )
        label = at.get_model_name(modelpath)
        enclosed_xvals = []
        enclosed_yvals = []
        binned_xvals: list[float] = []
        binned_yvals: list[float] = []
        mass_cumulative = 0.0
        enclosed_xvals = [0.0]
        enclosed_yvals = [0.0]

        # total_mass = dfmodel.cellmass_grams.sum() / 1.989e33
        dfmodel = dfmodel.with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))
        dfmodel = dfmodel.sort(by="vel_r_mid")

        dfmodelcollect = dfmodel.select(
            ["modelgridindex", "vel_r_min", "vel_r_mid", "vel_r_max", "cellmass_grams"]
        ).collect()
        if "vel_r_max_kmps" in dfmodel.columns:
            for cell in dfmodelcollect.iter_rows(named=True):
                vlower = cell["vel_r_min"]
                vupper = cell["vel_r_max"]

                binned_xvals.extend((vlower / 29979245800, vupper / 29979245800))
                delta_beta = (vupper - vlower) / 29979245800
                yval = cell["cellmass_grams"] / 1.989e33 / delta_beta
                binned_yvals.extend((yval, yval))
                mass_cumulative += cell["cellmass_grams"] / 1.989e33
                enclosed_xvals.append(vupper / 29979245800)
                enclosed_yvals.append(mass_cumulative)
        else:
            ncoarsevelbins = int(
                (modelmeta["ncoordgridrcyl"] if "ncoordgridrcyl" in modelmeta else modelmeta["ncoordgridx"]) / 2.0
            )
            vlowerscoarse = [modelmeta["vmax_cmps"] / ncoarsevelbins * i for i in range(ncoarsevelbins)]
            vupperscoarse = [modelmeta["vmax_cmps"] / ncoarsevelbins * (i + 1) for i in range(ncoarsevelbins)]

            for vlower, vupper in zip(vlowerscoarse, vupperscoarse):
                velbinmass = dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))[
                    "cellmass_grams"
                ].sum()

                binned_xvals.extend((vlower / 29979245800, vupper / 29979245800))
                delta_beta = (vupper - vlower) / 29979245800
                yval = velbinmass / 1.989e33 / delta_beta
                binned_yvals.extend((yval, yval))

            vuppers = dfmodelcollect["vel_r_max"].unique().sort()
            vlowers = [0.0, *vuppers[:-1].to_list()]

            for vlower, vupper in zip(vlowers, vuppers):
                velbinmass = dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))[
                    "cellmass_grams"
                ].sum()
                mass_cumulative += velbinmass / 1.989e33
                enclosed_xvals.append(vupper / 29979245800)
                enclosed_yvals.append(mass_cumulative)

        axes[0].plot(binned_xvals, binned_yvals, label=label)
        axes[1].plot(enclosed_xvals, enclosed_yvals, label=label)

    axes[-1].set_xlabel("velocity [v/c]")
    axes[0].set_ylabel(r"$\Delta$M [M$_\odot$] / $\Delta$v/c")
    axes[1].set_ylabel(r"enclosed mass [M$_\odot$]")
    axes[0].legend()

    axes[-1].set_xlim(left=0.0)
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    outfilepath = Path(args.outputpath)
    if outfilepath.is_dir():
        outfilepath = outfilepath / "densityprofile.pdf"

    plt.savefig(outfilepath)
    print(f"Saved {outfilepath}")


if __name__ == "__main__":
    main()
