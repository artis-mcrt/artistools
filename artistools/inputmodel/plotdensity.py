#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt

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


def main(args=None, argsraw=None, **kwargs) -> None:
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
        dfmodel, _, _ = at.get_modeldata_tuple(modelpath)
        label = at.get_model_name(modelpath)
        enclosed_xvals = []
        enclosed_yvals = []
        binned_xvals: list[float] = []
        binned_yvals: list[float] = []
        mass_cumulative = 0.0
        enclosed_xvals = [0.0]
        enclosed_yvals = [0.0]

        # total_mass = dfmodel.cellmass_grams.sum() / 1.989e33
        for cell in dfmodel.itertuples(index=False):
            binned_xvals.extend((cell.velocity_inner / 299792.458, cell.velocity_outer / 299792.458))
            delta_beta = (cell.velocity_outer - cell.velocity_inner) / 299792.458

            yval = cell.cellmass_grams / 1.989e33 / delta_beta
            binned_yvals.extend((yval, yval))
            enclosed_xvals.append(cell.velocity_outer / 299792.458)
            mass_cumulative += cell.cellmass_grams / 1.989e33
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
