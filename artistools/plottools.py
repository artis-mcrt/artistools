"""Matplotlib-related plotting functions."""

import argparse
import itertools
import sys
import typing as t
from collections.abc import Iterable

import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import ticker

from artistools.configuration import get_config


def set_mpl_style() -> None:
    plt.style.use("file://" + str(get_config()["path_artistools_dir"] / "matplotlibrc"))


class ExponentLabelFormatter(ticker.ScalarFormatter):
    """Formatter to move the 'x10^x' offset text into the axis label."""

    _useMathText: bool
    _usetex: bool

    def __init__(self, labeltemplate: str, useMathText: bool = True) -> None:
        self.set_labeltemplate(labeltemplate)

        super().__init__(useOffset=False, useMathText=useMathText)
        # ticker.ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)
        self.set_scientific(True)
        self.set_powerlimits((0, 0))  # always use scientific notation

    def _set_formatted_label_text(self) -> None:
        # or use self.orderOfMagnitude
        stroffset = self.get_offset()
        if stroffset:
            stroffset = stroffset.replace(r"$\times", "$") + " "
        strnewlabel = self.labeltemplate.format(stroffset)
        assert self.axis is not None
        self.axis.set_label_text(strnewlabel)  # type: ignore[union-attr] # pyright: ignore[reportAttributeAccessIssue]
        assert self.offset == 0
        self.axis.offsetText.set_visible(False)  # type: ignore[union-attr] # pyright: ignore[reportAttributeAccessIssue]

    def set_labeltemplate(self, labeltemplate: str) -> None:
        assert "{" in labeltemplate
        self.labeltemplate = labeltemplate

    def set_locs(self, locs) -> None:  # noqa: ANN001
        super().set_locs(locs)
        self._set_formatted_label_text()

    def set_axis(self, axis) -> None:  # noqa: ANN001
        super().set_axis(axis)
        self._set_formatted_label_text()


def set_axis_properties(ax: Iterable[mplax.Axes] | mplax.Axes, args: argparse.Namespace) -> t.Any:
    if "subplots" not in args:
        args.subplots = False
    if "labelfontsize" not in args:
        args.labelfontsize = 18

    if isinstance(ax, Iterable):
        for axis in ax:
            assert isinstance(axis, mplax.Axes)
            axis.minorticks_on()
            axis.tick_params(
                axis="both",
                which="minor",
                top=True,
                right=True,
                length=5,
                width=2,
                labelsize=args.labelfontsize,
                direction="in",
            )
            axis.tick_params(
                axis="both",
                which="major",
                top=True,
                right=True,
                length=8,
                width=2,
                labelsize=args.labelfontsize,
                direction="in",
            )

    else:
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="minor",
            top=True,
            right=True,
            length=5,
            width=2,
            labelsize=args.labelfontsize,
            direction="in",
        )
        ax.tick_params(
            axis="both",
            which="major",
            top=True,
            right=True,
            length=8,
            width=2,
            labelsize=args.labelfontsize,
            direction="in",
        )

    if "ymin" in args or "ymax" in args:
        plt.ylim(args.ymin, args.ymax)
    if "xmin" in args or "xmax" in args:
        plt.xlim(args.xmin, args.xmax)

    if getattr(args, "logscalex", False):
        plt.xscale("log")
    if getattr(args, "logscaley", False):
        plt.yscale("log")

    plt.minorticks_on()
    return ax


def set_axis_labels(
    fig: mplfig.Figure,
    ax: mplax.Axes | npt.ArrayLike,
    xlabel: str,
    ylabel: str,
    labelfontsize: int | None,
    args: argparse.Namespace,
) -> None:
    if args.subplots:
        fig.text(0.5, 0.02, xlabel, ha="center", va="center")
        fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation="vertical")
    else:
        assert isinstance(ax, mplax.Axes)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)


def imshow_init_for_artis_grid(
    ngrid: int, vmax: float, plot_variable_3d_array: npt.NDArray[t.Any], plot_axes: str = "xy"
) -> tuple[npt.NDArray[t.Any], tuple[float, float, float, float]]:
    # ngrid = round(len(model['inputcellid']) ** (1./3.))
    extentdict = {"left": -vmax, "right": vmax, "bottom": vmax, "top": -vmax}
    extent = extentdict["left"], extentdict["right"], extentdict["bottom"], extentdict["top"]
    data = np.zeros((ngrid, ngrid))

    plot_axes_choices = ["xy", "xz"]
    if plot_axes not in plot_axes_choices:
        print(f"Choose plot axes from {plot_axes_choices}")
        sys.exit(1)

    for z, y, x in itertools.product(range(ngrid), range(ngrid), range(ngrid)):
        if plot_axes == "xy":
            if z == round(ngrid / 2) - 1:
                data[y, x] = plot_variable_3d_array[x, y, z]
        elif plot_axes == "xz" and y == round(ngrid / 2) - 1:
            data[z, x] = plot_variable_3d_array[x, y, z]

    return data, extent
