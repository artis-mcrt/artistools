"""Matplotlib-related plotting functions."""

import argparse
import typing as t
from collections.abc import Iterable

import matplotlib.axes as mplax
import matplotlib.axis as mplaxis
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy.typing as npt

from artistools.commands import get_path
from artistools.misc import print_saved

if t.TYPE_CHECKING:
    from pathlib import Path

# colorcet.glasbey_category20
glasbey_category20 = [
    (0.121569, 0.466667, 0.705882),
    (1.0, 0.498039, 0.054902),
    (0.172549, 0.627451, 0.172549),
    (0.839216, 0.152941, 0.156863),
    (0.580392, 0.403922, 0.741176),
    (0.54902, 0.337255, 0.294118),
    (0.890196, 0.466667, 0.760784),
    (0.498039, 0.498039, 0.498039),
    (0.737255, 0.741176, 0.133333),
    (0.090196, 0.745098, 0.811765),
    (0.227451, 0.003922, 0.513725),
    (0.0, 0.262745, 0.003922),
    (0.058824, 1.0, 0.662745),
    (0.368627, 0.0, 0.25098),
    (0.737255, 0.737255, 1.0),
    (0.847059, 0.686275, 0.635294),
    (0.721569, 0.0, 0.501961),
    (0.0, 0.305882, 0.32549),
    (0.419608, 0.396078, 0.0),
    (0.490196, 0.007843, 0.0),
    (0.380392, 0.14902, 1.0),
    (1.0, 1.0, 0.603922),
    (0.341176, 0.286275, 0.392157),
    (0.54902, 0.721569, 0.580392),
    (0.580392, 0.988235, 1.0),
    (0.007843, 0.509804, 0.407843),
    (0.568627, 1.0, 0.0),
    (0.513725, 0.0, 0.627451),
    (0.678431, 0.537255, 0.266667),
    (0.356863, 0.203922, 0.0),
    (1.0, 0.752941, 0.952941),
    (1.0, 0.435294, 0.462745),
    (0.47451, 0.54902, 1.0),
    (0.866667, 0.0, 1.0),
    (0.317647, 0.337255, 0.27451),
    (0.0, 0.270588, 0.541176),
    (1.0, 0.74902, 0.376471),
    (1.0, 0.003922, 0.552941),
    (0.745098, 0.788235, 0.811765),
    (0.686275, 0.596078, 0.709804),
    (0.717647, 0.341176, 0.0),
    (0.007843, 0.439216, 0.0),
    (0.803922, 0.533333, 1.0),
    (0.113725, 0.839216, 0.27451),
    (0.752941, 0.92549, 0.768627),
    (0.478431, 0.596078, 0.709804),
    (0.647059, 0.376471, 0.537255),
    (0.435294, 0.537255, 0.341176),
    (0.741176, 0.490196, 0.462745),
    (0.545098, 0.160784, 0.270588),
    (0.0, 0.678431, 1.0),
    (0.560784, 0.831373, 1.0),
    (0.294118, 0.427451, 0.466667),
    (0.0, 0.831373, 0.694118),
    (0.576471, 0.0, 0.952941),
    (0.545098, 0.584314, 0.0),
    (0.364706, 0.360784, 0.623529),
    (0.996078, 0.87451, 0.733333),
    (0.0, 0.576471, 0.623529),
    (1.0, 0.862745, 0.0),
    (0.0, 0.670588, 0.47451),
    (0.321569, 0.0, 0.407843),
    (0.0, 0.0, 0.572549),
    (0.043137, 0.364706, 0.243137),
    (0.65098, 0.890196, 0.462745),
    (0.384314, 0.231373, 0.254902),
    (0.776471, 0.780392, 0.541176),
    (1.0, 0.619608, 0.713725),
    (0.807843, 0.313725, 0.423529),
    (1.0, 0.027451, 0.839216),
    (0.545098, 0.227451, 0.019608),
    (0.498039, 0.243137, 0.443137),
    (1.0, 0.286275, 0.007843),
    (0.376471, 0.172549, 0.65098),
    (0.109804, 0.0, 1.0),
    (0.905882, 0.87451, 1.0),
    (0.666667, 0.231373, 0.686275),
    (0.85098, 0.611765, 0.0),
    (0.639216, 0.639216, 0.623529),
    (0.247059, 0.415686, 1.0),
    (0.27451, 0.286275, 0.05098),
    (0.482353, 0.411765, 0.521569),
    (0.423529, 0.596078, 0.552941),
    (1.0, 0.603922, 0.458824),
    (0.517647, 0.360784, 1.0),
    (0.486275, 0.423529, 0.27451),
    (0.505882, 0.717647, 0.333333),
    (0.741176, 0.0, 0.290196),
    (0.992157, 0.580392, 1.0),
    (0.364706, 0.0, 0.094118),
    (0.541176, 0.819608, 0.819608),
    (0.615686, 0.552941, 0.827451),
    (0.854902, 0.427451, 0.262745),
    (0.545098, 0.345098, 0.0),
    (0.235294, 0.317647, 0.415686),
    (0.298039, 0.423529, 0.231373),
    (0.933333, 0.815686, 0.847059),
    (0.811765, 0.933333, 1.0),
    (0.666667, 0.082353, 0.0),
    (0.87451, 1.0, 0.309804),
    (1.0, 0.168627, 0.341176),
    (0.819608, 0.286275, 0.619608),
    (0.439216, 0.490196, 0.721569),
    (0.352941, 0.501961, 0.0),
    (0.0, 0.898039, 0.992157),
    (0.466667, 0.294118, 0.584314),
    (0.407843, 0.835294, 0.54902),
    (0.243137, 0.227451, 0.447059),
    (0.67451, 0.254902, 0.247059),
    (0.839216, 0.635294, 0.4),
    (0.756863, 0.411765, 0.807843),
    (0.415686, 0.34902, 0.368627),
    (0.533333, 0.67451, 0.929412),
    (0.627451, 0.65098, 0.415686),
    (0.823529, 0.666667, 0.901961),
    (0.533333, 0.0, 0.388235),
    (0.0, 0.992157, 0.858824),
    (0.407843, 0.156863, 0.098039),
    (0.705882, 0.258824, 1.0),
    (0.054902, 0.34902, 0.772549),
    (0.090196, 0.529412, 0.262745),
    (0.568627, 0.827451, 0.0),
    (0.807843, 0.47451, 0.0),
    (0.976471, 0.352941, 1.0),
    (0.356863, 0.454902, 0.4),
    (0.556863, 0.682353, 0.701961),
    (0.611765, 0.490196, 0.54902),
    (0.27451, 0.0, 0.776471),
    (0.423529, 0.305882, 0.180392),
    (0.65098, 0.427451, 0.27451),
    (0.619608, 0.537255, 0.45098),
    (0.658824, 0.686275, 0.792157),
    (0.807843, 0.552941, 0.654902),
    (0.0, 0.996078, 0.392157),
    (0.572549, 0.47451, 0.0),
    (1.0, 0.388235, 0.631373),
    (0.960784, 1.0, 0.847059),
    (0.003922, 0.54902, 0.945098),
    (0.078431, 0.67451, 0.627451),
    (0.356863, 0.180392, 0.352941),
    (0.537255, 0.52549, 0.619608),
    (0.815686, 0.8, 0.733333),
    (0.831373, 0.686275, 0.772549),
    (0.858824, 0.866667, 0.427451),
    (0.815686, 1.0, 0.956863),
    (0.0, 0.396078, 0.52549),
    (0.0, 0.411765, 0.388235),
    (0.658824, 0.254902, 0.407843),
    (0.176471, 0.592157, 0.772549),
    (0.662745, 0.454902, 1.0),
    (0.152941, 0.733333, 0.368627),
    (0.345098, 0.717647, 0.0),
    (0.796078, 1.0, 0.654902),
    (0.643137, 0.478431, 0.670588),
    (1.0, 0.741176, 0.580392),
    (0.537255, 0.886275, 0.756863),
    (0.058824, 0.788235, 1.0),
    (0.835294, 0.0, 0.772549),
    (0.388235, 0.427451, 0.541176),
    (0.411765, 0.521569, 0.560784),
    (0.294118, 0.305882, 0.32549),
    (0.670588, 0.376471, 0.407843),
    (0.478431, 0.713725, 0.835294),
    (0.172549, 0.352941, 0.090196),
    (0.603922, 0.0, 0.145098),
    (0.745098, 0.819608, 0.952941),
    (0.541176, 0.435294, 0.407843),
    (0.415686, 0.647059, 0.419608),
    (0.52549, 0.329412, 0.407843),
    (0.682353, 0.803922, 0.729412),
    (0.533333, 0.6, 0.498039),
    (0.796078, 0.862745, 0.0),
    (0.607843, 0.015686, 0.568627),
    (0.921569, 0.737255, 0.105882),
    (0.921569, 0.611765, 0.823529),
    (0.439216, 0.0, 0.435294),
    (0.694118, 0.631373, 0.196078),
    (0.792157, 0.423529, 0.576471),
    (0.254902, 0.27451, 0.643137),
    (0.898039, 0.54902, 0.541176),
    (0.835294, 0.270588, 0.0),
    (0.780392, 0.545098, 0.796078),
    (0.717647, 0.588235, 0.592157),
    (0.831373, 0.12549, 0.462745),
    (0.447059, 0.294118, 0.8),
    (0.407843, 0.305882, 0.0),
    (0.407843, 0.133333, 0.219608),
    (0.219608, 0.337255, 0.309804),
    (0.435294, 0.733333, 0.670588),
    (0.52549, 0.227451, 0.192157),
    (0.647059, 0.827451, 0.596078),
    (0.72549, 0.686275, 0.560784),
    (0.847059, 0.894118, 0.87451),
    (0.670588, 0.0, 0.878431),
    (0.796078, 0.756863, 0.858824),
    (1.0, 0.87451, 0.54902),
    (0.890196, 0.32549, 0.301961),
    (0.4, 0.411765, 0.435294),
    (1.0, 0.0, 0.109804),
    (0.32549, 0.176471, 0.45098),
    (0.305882, 0.568627, 0.423529),
    (0.658824, 0.427451, 0.066667),
    (1.0, 0.623529, 0.14902),
    (0.372549, 0.639216, 0.690196),
    (0.784314, 0.521569, 0.341176),
    (0.572549, 0.34902, 0.596078),
    (0.639216, 0.631373, 1.0),
    (0.996078, 0.729412, 0.729412),
    (0.145098, 0.164706, 0.533333),
    (0.858824, 0.901961, 0.658824),
    (0.592157, 0.94902, 0.654902),
    (0.403922, 0.580392, 0.839216),
    (0.729412, 0.356863, 0.25098),
    (0.227451, 0.364706, 0.572549),
    (0.211765, 0.309804, 0.184314),
    (0.152941, 0.486275, 0.588235),
    (0.541176, 0.584314, 0.607843),
    (0.815686, 0.705882, 0.341176),
    (0.0, 0.278431, 0.392157),
    (0.372549, 0.364706, 0.184314),
    (0.556863, 0.556863, 0.254902),
    (0.678431, 0.247059, 0.07451),
    (0.415686, 0.588235, 0.235294),
    (0.631373, 0.239216, 0.521569),
    (0.74902, 0.717647, 0.729412),
    (0.67451, 0.776471, 0.403922),
    (0.396078, 0.411765, 0.811765),
    (0.572549, 0.690196, 0.0),
    (0.172549, 0.890196, 0.854902),
    (0.003922, 0.435294, 0.211765),
    (1.0, 0.47451, 0.32549),
    (0.258824, 0.505882, 0.498039),
    (0.309804, 0.913725, 0.0),
    (0.6, 0.329412, 0.156863),
    (0.364706, 0.039216, 0.0),
    (0.639216, 0.0, 0.345098),
    (0.047059, 0.533333, 0.0),
    (0.352941, 0.513725, 0.654902),
    (1.0, 0.92549, 0.984314),
    (0.294118, 0.411765, 0.003922),
    (0.533333, 0.462745, 0.831373),
    (0.901961, 0.780392, 1.0),
    (0.647059, 1.0, 0.854902),
    (0.847059, 0.435294, 0.470588),
    (0.87451, 0.007843, 0.294118),
    (0.415686, 0.407843, 0.360784),
    (0.470588, 0.419608, 0.635294),
    (0.494118, 0.501961, 0.403922),
    (0.352941, 0.278431, 0.52549),
    (0.0, 0.0, 0.792157),
    (0.486275, 0.0, 0.168627),
    (0.592157, 1.0, 0.447059),
    (0.713725, 0.890196, 0.882353),
    (0.862745, 0.32549, 0.788235),
    (0.466667, 0.470588, 0.203922),
    (0.345098, 0.745098, 0.556863),
]

glasbey_category20_nogreys = [
    color for color in glasbey_category20 if color[0] != color[1] or color[1] != color[2] or color[0] != color[2]
]


def set_mpl_style() -> None:
    """Apply the bundled artistools matplotlibrc style."""
    plt.style.use("file://" + str(get_path("artistools_dir") / "matplotlibrc"))


def save_figure(fig: mplfig.Figure, outpath: "Path | str", **savefig_kwargs: t.Any) -> None:
    """Save the figure to outpath, report the path, and close the figure."""
    fig.savefig(outpath, **savefig_kwargs)
    print_saved(outpath)
    plt.close(fig)


def set_plot_title(ax: mplax.Axes, title: str, args: argparse.Namespace) -> None:
    """Set the plot title, unless -notitle was given, placing it inside the axes for -inset_title."""
    if getattr(args, "notitle", False) or not title:
        return

    if getattr(args, "inset_title", False):
        ax.annotate(
            title,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="large",
        )
    else:
        ax.set_title(title, fontsize=11)


def get_next_color(ax: mplax.Axes) -> str:
    """Take the next colour from the Axes property cycle, advancing it.

    Call this to keep colours consistent with the automatic ones, or to skip a colour that would otherwise
    be reused. matplotlib exposes no public accessor, hence the private attribute.
    """
    nextcolor: str = ax._get_lines.get_next_color()  # type: ignore[attr-defined] # ruff:ignore[private-member-access] # pyright: ignore[reportAttributeAccessIssue]  # ty:ignore[unresolved-attribute]
    return nextcolor


class ExponentLabelFormatter(mplticker.ScalarFormatter):
    """Formatter to move the 'x10^x' offset text into the axis label."""

    labeltemplate: str

    def __init__(self, labeltemplate: str) -> None:
        """Store the axis label template, which must contain a placeholder for the exponent."""
        assert "{" in labeltemplate
        self.labeltemplate = labeltemplate

        super().__init__(useOffset=False, useMathText=True)
        self.set_powerlimits((0, 0))  # always use scientific notation

    @t.override
    def get_offset(self) -> str:
        # the offset is moved into the axis label, so the axis offsetText must stay
        # empty, otherwise matplotlib >= 3.11 will use its bounding box when
        # positioning the title, leading to non-finite axes positions
        return ""

    def _set_formatted_label_text(self) -> None:
        stroffset = super().get_offset()
        if stroffset:
            stroffset = stroffset.replace(r"$\times", "$") + " "
        strnewlabel = self.labeltemplate.format(stroffset)
        # a dummy axis (used when a formatter is called standalone) has no label to set
        assert self.axis is not None
        if isinstance(self.axis, mplaxis.Axis):
            self.axis.set_label_text(strnewlabel)

    @t.override
    def set_locs(self, locs: t.Any) -> None:
        super().set_locs(locs)
        if self._format is not None:
            # ScalarFormatter otherwise drops the decimal point for integer-spaced ticks.
            self._format = self._format.replace("%1.0f", "%1.1f")
            self._set_formatted_label_text()

    @t.override
    def set_axis(self, axis: t.Any) -> None:
        super().set_axis(axis)
        self._set_formatted_label_text()


def set_axis_properties(ax: Iterable[mplax.Axes] | mplax.Axes, args: argparse.Namespace) -> t.Any:
    """Apply the standard tick, minor tick, and font size settings to one or more axes."""
    if "subplots" not in args:
        args.subplots = False
    if "labelfontsize" not in args:
        args.labelfontsize = 18

    for axis in ax if isinstance(ax, Iterable) else [ax]:
        assert isinstance(axis, mplax.Axes)
        axis.minorticks_on()
        for which, ticklength in (("minor", 5), ("major", 8)):
            axis.tick_params(
                axis="both",
                which=which,
                top=True,
                right=True,
                length=ticklength,
                width=2,
                labelsize=args.labelfontsize,
                direction="in",
            )

        if "ymin" in args or "ymax" in args:
            axis.set_ylim(args.ymin, args.ymax)
        if "xmin" in args or "xmax" in args:
            axis.set_xlim(args.xmin, args.xmax)

        if getattr(args, "logscalex", False):
            axis.set_xscale("log")
        if getattr(args, "logscaley", False):
            axis.set_yscale("log")

    return ax


def set_axis_labels(
    fig: mplfig.Figure,
    ax: mplax.Axes | npt.ArrayLike,
    xlabel: str,
    ylabel: str,
    labelfontsize: int | None,
    args: argparse.Namespace,
) -> None:
    """Set the x and y axis labels, placing them on the figure rather than the axes when there are subplots."""
    if args.subplots:
        fig.text(0.5, 0.02, xlabel, ha="center", va="center")
        fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation="vertical")
    else:
        assert isinstance(ax, mplax.Axes)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
