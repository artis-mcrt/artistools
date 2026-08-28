"""Matplotlib-related plotting functions."""

import argparse
import math
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from functools import cache

import matplotlib.axes as mplax
import matplotlib.axis as mplaxis
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker

from artistools.commands import get_path
from artistools.misc import print_saved
from artistools.misc import print_warning

# the ratio across the middle half of the values, above which a log scale wins. A kilonova light
# curve gives 23 and its spectrum 7. Wider percentiles read the tails, not the data: the 5th and
# 95th give 456 and 182, which no threshold parts.
LOGSCALE_MINRATIO: t.Final[float] = 15.0

# a log axis hides a value of zero or below. A few such values are the end of a decay, thus the axis
# still shows the data. This fraction of the values is the most that a log axis may hide.
LOGSCALE_MAXHIDDEN: t.Final[float] = 0.1


def get_drawn_yvalues(ax: "AxesTree") -> "npt.NDArray[np.float64]":
    """Return the y values of every line that the axes holds."""
    import numpy as np

    columns = [np.asarray(line.get_ydata(), dtype=np.float64).ravel() for axis in iter_axes(ax) for line in axis.lines]
    return np.concatenate(columns) if columns else np.empty(0, dtype=np.float64)


def get_quartile_ratio(values: "npt.NDArray[np.float64]") -> float:
    """Return the ratio that the middle half of a series of values above zero covers.

    The quartiles hold the range of the data itself. A point of noise near zero changes nothing, and
    neither does the end of a spectrum, where the flux falls away over a few points. A series that
    decays over decades puts its quartiles decades apart, which is the difference that the scale of
    the axis must answer.
    """
    import numpy as np

    quartile1, quartile3 = np.percentile(values, [25.0, 75.0])
    return float(quartile3 / quartile1)


def wants_log_scale(values: "npt.NDArray[np.float64]") -> bool:
    """Return True when a log scale shows the values better than a linear scale.

    A linear axis draws the middle half of the values in a small part of itself when that half
    covers more than the ratio, thus such a series needs a log axis. A value of zero or below has no
    place on a log axis, thus many of them keep the linear one.
    """
    import numpy as np

    # a mask costs one byte per value, thus this counts them without a copy
    isfinite = np.isfinite(values)
    ispositive = isfinite & (values > 0.0)
    countfinite = int(isfinite.sum())
    countpositive = int(ispositive.sum())
    if countpositive < 4 or (countfinite - countpositive) > LOGSCALE_MAXHIDDEN * countfinite:
        return False

    return get_quartile_ratio(values[ispositive]) > LOGSCALE_MINRATIO


def set_auto_yscale(ax: "AxesTree", args: argparse.Namespace) -> None:
    """Set the scale of the vertical axis from the drawn values, when -yscale asks the command to choose.

    Call this after the command draws the data. It sets args.logscaley for the code that reads it,
    and it puts the log scale on the axes, because a command can set the scale before it draws.
    -yscale log and -yscale linear each give an answer already, thus this changes nothing for them.
    """
    if getattr(args, "yscale", "auto") != "auto" or getattr(args, "logscaley", False):
        return

    args.logscaley = wants_log_scale(get_drawn_yvalues(ax))
    if args.logscaley:
        for axis in iter_axes(ax):
            axis.set_yscale("log")


# subplots() gives a single axes, a 1D array, or a 2D array, thus the type nests to any depth
type AxesTree = mplax.Axes | Iterable[AxesTree]

if t.TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.legend as mpllegend
    import matplotlib.typing as mplt
    import numpy as np
    import numpy.typing as npt

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


def remove_greys(palette: Sequence["mplt.ColorType"]) -> list["mplt.ColorType"]:
    """Return the colours of palette that are not grey, i.e. those whose red, green and blue differ."""
    return [color for color in palette if len(set(mplcolors.to_rgb(color))) > 1]


glasbey_category20_nogreys = remove_greys(glasbey_category20)

# the plot colours of the reference data series, in the order that the code uses them
refseries_colors = ("0.0", "0.4", "0.6", "0.7")


def get_assigned_colors(seriescolors: Sequence[str | None]) -> set[str]:
    """Return the hex values of the colours that series were given.

    Comparing by value is the single rule for "a series already has this colour", so that the name "C0",
    the alias "tab:blue" and the value "#1F77B4" all match the first colour of the cycle. A transparent
    series holds no colour, and to_hex would report it as black.
    """
    return {mplcolors.to_hex(color) for color in seriescolors if color and mplcolors.to_rgba(color)[3] > 0.0}


def get_series_colors(isreference: Sequence[bool], usercolors: Sequence[str | None] = ()) -> list[str]:
    """Return the plot colour of each data series.

    A colour in usercolors has priority. The first reference data series get black and then lighter greys.
    The other series, and the reference series after the greys, get the colours of the matplotlib cycle.
    The code steps over a colour that the user asked for, thus two series do not get one colour.
    """
    askedfor = get_assigned_colors(usercolors)
    cyclecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # the colours of the cycle that no series has, by value rather than by spelling
    freecycleindices = [i for i, color in enumerate(cyclecolors) if mplcolors.to_hex(color) not in askedfor] or list(
        range(max(len(cyclecolors), 1))
    )

    colors: list[str] = []
    refindex = 0
    cycleindex = 0
    for seriesindex, isref in enumerate(isreference):
        usercolor = usercolors[seriesindex] if seriesindex < len(usercolors) else None
        if usercolor:
            colors.append(usercolor)
            continue

        # step over a colour that the user asked for, thus two series do not get one colour
        while isref and refindex < len(refseries_colors) and mplcolors.to_hex(refseries_colors[refindex]) in askedfor:
            refindex += 1

        if isref and refindex < len(refseries_colors):
            colors.append(refseries_colors[refindex])
            refindex += 1
        else:
            colors.append(f"C{freecycleindices[cycleindex % len(freecycleindices)]}")
            cycleindex += 1

    return colors


# the size in inches of one subplot frame at a figure scale of 1, measured from a plot of one row
# and one column. A file of this width fills the text block of a two-column page, which measures
# 180 mm in MNRAS and in A&A
FRAMEWIDTH_INCHES: t.Final[float] = 6.47
FRAMEHEIGHT_INCHES: t.Final[float] = 4.08

# the frame of a plot that fills one column of such a page, which measures 84 mm and 88 mm. A plot
# that draws few series needs no more, and a paper then sets it beside the text
COLUMNFRAMEWIDTH_INCHES: t.Final[float] = 2.99

# the y label and its tick numbers to the left of each frame, and the x label below the lowest one.
# Each column pays the width, which a log tick number such as 10^-12 drives to 0.63 inches
LABELWIDTH_INCHES: t.Final[float] = 0.78
LABELHEIGHT_INCHES: t.Final[float] = 0.47


# a saved file crops to about 7.0 inches, which is the full text width of a two-column page:
# 180 mm in MNRAS and in A&A. One column of such a page is less than half of that
FIGWIDTH_INCHES: t.Final[float] = FRAMEWIDTH_INCHES + LABELWIDTH_INCHES


# the margin to the right of the widest frame, which holds the last tick number of the x axis
RIGHTMARGIN_INCHES: t.Final[float] = 0.10

# the margin above the highest frame. It holds the last tick number of the y axis and the text of
# the exponent that a formatter puts above that axis, which reaches further
TOPMARGIN_INCHES: t.Final[float] = 0.28


def make_frame_figure(
    args: argparse.Namespace | None = None,
    *,
    rows: int = 1,
    cols: int = 1,
    aspect: float = FRAMEHEIGHT_INCHES / FRAMEWIDTH_INCHES,
    sharex: bool = True,
    sharey: bool = False,
    fullwidth: bool = True,
) -> "tuple[mplfig.Figure, npt.NDArray[t.Any]]":
    """Return a figure whose frames each hold exactly the same size, and the axes of that figure.

    A paper puts several of these files in a grid that the author builds by hand. Each file goes in
    at one width, thus every file must draw a frame of the same size, whatever its labels take.

    Divider places each frame at a size in inches, thus the size of a frame does not follow the
    length of a tick number, the number of rows, or a label that a command hides. The figure holds
    the margins that the labels need, and no layout engine takes that space back.

    save_figure crops the part of a margin that no label fills. The crop moves no artist, thus a
    file that hides its x labels keeps its width and loses only the height of those labels.

    The axes come back in a 2D array of [row][column], with row 0 at the top, as plt.subplots gives.
    A caller that has no parsed arguments gives no args, and the figure then takes a scale of 1.
    A plot that draws few series gives fullwidth=False, and its frame then fills one column of the
    page in place of the whole text block. aspect stays the height of a frame as a part of its width.
    """
    from mpl_toolkits.axes_grid1 import Divider
    from mpl_toolkits.axes_grid1 import Size

    set_mpl_style()

    # a helper that draws a figure without parsing arguments passes no args, thus it takes scale 1
    figscale = getattr(args, "figscale", 1.0)
    basewidth = FRAMEWIDTH_INCHES if fullwidth else COLUMNFRAMEWIDTH_INCHES
    framewidth = basewidth * getattr(args, "figwidthscale", 1.0) * figscale
    frameheight = basewidth * aspect * figscale

    # a column that shows its own y label needs the width of one beside it, and likewise a row
    colgap = RIGHTMARGIN_INCHES if sharey else LABELWIDTH_INCHES
    rowgap = TOPMARGIN_INCHES if sharex else LABELHEIGHT_INCHES

    horizontal = [Size.Fixed(LABELWIDTH_INCHES)]
    for column in range(cols):
        horizontal += [Size.Fixed(colgap)] if column else []
        horizontal += [Size.Fixed(framewidth)]
    horizontal += [Size.Fixed(RIGHTMARGIN_INCHES)]

    # Divider counts the vertical sizes from the bottom, thus the lowest row comes first
    vertical = [Size.Fixed(LABELHEIGHT_INCHES)]
    for row in range(rows):
        vertical += [Size.Fixed(rowgap)] if row else []
        vertical += [Size.Fixed(frameheight)]
    vertical += [Size.Fixed(TOPMARGIN_INCHES)]

    figwidth = sum(size.fixed_size for size in horizontal)
    figheight = sum(size.fixed_size for size in vertical)
    fig = plt.figure(figsize=(figwidth, figheight))
    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), horizontal, vertical, aspect=False)

    import numpy as np

    axes = np.empty((rows, cols), dtype=object)
    for row in range(rows):
        for col in range(cols):
            # the index of this frame among the sizes, which hold a margin or a gap between them
            nx = 1 + col * 2
            ny = 1 + (rows - 1 - row) * 2
            first = axes[0][0] if (row or col) else None
            axis = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=nx, ny=ny),
                sharex=first if sharex else None,
                sharey=first if sharey else None,
            )
            axes[row][col] = axis

            # the labels of an axis that another one carries would draw twice
            if sharex and row < rows - 1:
                axis.tick_params(axis="x", which="both", labelbottom=False)
            if sharey and col > 0:
                axis.tick_params(axis="y", which="both", labelleft=False)

    return fig, axes


def set_legend(ax: mplax.Axes, args: argparse.Namespace, **legendkwargs: t.Any) -> "mpllegend.Legend | None":
    """Draw the legend of the axes and return it. Return None when -nolegend was given."""
    if getattr(args, "nolegend", False):
        return None

    return ax.legend(**legendkwargs)


def get_unused_colors(
    palette: Sequence["mplt.ColorType"], seriescolors: Sequence[str | None]
) -> list["mplt.ColorType"]:
    """Return the colours of palette that no series in seriescolors was given.

    A series that takes its colour from the palette, e.g. an extra direction bin, then does not
    repeat the colour of another series. Colours are compared by value, so "C0" matches "#1f77b4".
    """
    assignedcolors = get_assigned_colors(seriescolors)

    # a palette can hold one colour twice, e.g. a tab10 colour and the rounded glasbey copy of it, and
    # handing the same colour to two series is the thing this function exists to prevent
    unused: list[mplt.ColorType] = []
    seen: set[str] = set()
    for color in palette:
        hexcolor = mplcolors.to_hex(color)
        if hexcolor not in assignedcolors and hexcolor not in seen:
            seen.add(hexcolor)
            unused.append(color)

    return unused


def set_prop_cycle_unusedcolors(axes: Iterable[mplax.Axes], seriescolors: Sequence[str | None]) -> None:
    """Remove the colours of seriescolors from the colour cycle of each axis."""
    colors = get_unused_colors(plt.rcParams["axes.prop_cycle"].by_key()["color"], seriescolors)
    if colors:
        for axis in axes:
            axis.set_prop_cycle(color=colors)


@cache
def set_mpl_style() -> None:
    """Apply the bundled artistools matplotlibrc style.

    The style holds global state, thus one call serves the process. make_frame_figure calls this,
    so that every figure draws the same fonts and ticks whether or not its command asks.
    """
    plt.style.use("file://" + str(get_path("artistools_dir") / "matplotlibrc"))


def save_figure(
    fig: mplfig.Figure,
    outpath: "Path | str",
    *,
    args: argparse.Namespace | None = None,
    show: bool = False,
    openfile: bool = False,
    isframe: bool = False,
    **savefig_kwargs: t.Any,
) -> None:
    """Save the figure to outpath, report the path, and close the figure.

    A caller passes args, and the --show and --open flags of the command then apply. With show, the
    figure opens in a window first, thus a resize there reaches the saved file. With openfile, the
    saved file opens in its default application, thus --open needs no copied command.

    isframe says that this figure is one part of a product that combine_frames makes, e.g. a frame of a
    gif. Such a figure does not open on its own, because the product opens in its place, and it takes no
    line of its own, because a merge takes the frames away and that line would name a file that went.
    --show still opens each figure, because the user asked to see them.
    """
    if args is not None:
        show = show or getattr(args, "show", False)
        openfile = openfile or getattr(args, "open", False)

    if isframe:
        openfile = False

    if show:
        plt.show()

    # a crop moves no artist, thus a fixed frame keeps its size and a file that hides its x labels
    # keeps its width. The pad keeps a stroke on the boundary whole.
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("pad_inches", 0.02)

    fig.savefig(outpath, **savefig_kwargs)
    if not isframe:
        print_saved(outpath)
    plt.close(fig)

    if openfile:
        from artistools.misc.fileio import open_file

        open_file(outpath)


def save_or_show(fig: mplfig.Figure, outputfile: "Path | str | None") -> None:
    """Save the figure when an output file was given, otherwise show it. Close the figure either way."""
    if outputfile:
        save_figure(fig, outputfile)
    else:
        plt.show()
        plt.close(fig)


def set_plot_title(ax: mplax.Axes, title: str | None, args: argparse.Namespace) -> None:
    """Set the plot title, unless -notitle was given, placing it inside the axes for -inset_title."""
    # a command that defines neither flag still gets a title, thus a caller does not have to declare a
    # flag just to use this helper (set_axis_properties and set_legend default the same way)
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
        ax.set_title(title)


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
        assert "{}" in labeltemplate
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


class PrunedLogLocator(mplticker.LogLocator):
    """Place the major ticks of a log axis, but leave out the ones against either end.

    Stacked subplots put the lowest label of one axes beside the highest label of the axes below, and the
    outer labels touch the title and the axis label of the figure. MaxNLocator takes a prune argument and
    LogLocator takes none, thus this drops the locations within a fraction of the axis length of each end.

    It prunes each time that matplotlib draws, thus the ticks follow the view. A FixedLocator of one
    view would keep those ticks through a zoom. save_figure shows the figure before it writes the
    file, thus a zoom in that window would reach the file with the wrong ticks.
    """

    def __init__(self, *args: t.Any, fraction: float = 0.04, minticks: int = 3, **kwargs: t.Any) -> None:
        """Take the margin as a fraction of the axis length, and the fewest ticks to leave."""
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.minticks = minticks

    @t.override
    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        """Return the tick locations of the view, without the ones against either end."""
        ticks = super().tick_values(vmin, vmax)
        low, high = min(vmin, vmax), max(vmin, vmax)
        if low <= 0.0 or high <= low:
            return ticks

        loglow, loghigh = math.log10(low), math.log10(high)
        margin = self.fraction * (loghigh - loglow)
        inview = [loc for loc in ticks if loc > 0.0 and loglow <= math.log10(loc) <= loghigh]
        keep = [loc for loc in inview if loglow + margin < math.log10(loc) < loghigh - margin]

        # a log axis of many decades carries few major ticks, thus keep them all rather than leave too few
        return keep if self.minticks <= len(keep) < len(inview) else ticks


def plain_label(label: str) -> str:
    r"""Return a plot label as plain text, for a log line that a terminal shows.

    A label carries LaTeX for the figure, e.g. "$\\pm$". A terminal shows those marks as they are,
    thus this gives the symbol that they stand for.
    """
    # a terminal of any encoding shows these, thus the plain form stays in ASCII
    replacements = {r"$\pm$": "+/-", r"$\times$": "x", r"\odot": "sun", "$": "", "{": "", "}": ""}
    for latex, plain in replacements.items():
        label = label.replace(latex, plain)

    return label


def prune_log_ticks(axis: mplaxis.Axis) -> None:
    """Give a log axis a locator that leaves out the ticks against either end."""
    if axis.get_scale() == "log":
        axis.set_major_locator(PrunedLogLocator())


def set_exponent_label(axis: mplax.Axes) -> None:
    """Move the power-of-ten offset of the y axis into the axis label, when the label has a place for it.

    The label carries a "{}" placeholder that ExponentLabelFormatter fills. A label without one keeps the
    offset text that matplotlib draws above the axis.
    """
    if "{}" not in axis.get_ylabel():
        return

    axis.yaxis.set_major_formatter(ExponentLabelFormatter(axis.get_ylabel()))
    axis.yaxis.set_major_locator(
        mplticker.MaxNLocator(nbins="auto", steps=[1, 2, 4, 5, 8, 10], integer=True, prune="both")
    )
    axis.yaxis.set_minor_locator(mplticker.AutoMinorLocator())


def iter_axes(ax: AxesTree) -> list[mplax.Axes]:
    """Return a flat list of the axes, whether the figure has a single axes or a grid of them.

    Iterating a 2D array of axes yields its rows, so the rows are flattened rather than returned as they are.
    """
    if isinstance(ax, mplax.Axes):
        return [ax]

    return [axis for item in ax for axis in iter_axes(item)]


def log_axis_limit(limit: float | None, *, logscale: bool, argname: str) -> float | None:
    """Return a plot range limit, or None when a log axis cannot show it.

    matplotlib ignores a non-positive limit on a log scale, but warns in terms of neither the axis nor the
    argument that asked for it.
    """
    if limit is not None and logscale and limit <= 0.0:
        print_warning(f"ignoring {argname} {limit}, which a log axis cannot show")
        return None

    return limit


def set_axis_properties(
    ax: AxesTree, args: argparse.Namespace, xlimits: tuple[float | None, float | None, str] | None = None
) -> AxesTree:
    """Apply the standard tick, minor tick, and font size settings to one or more axes.

    A command whose x range has its own argument name, e.g. the -timemin/-timemax of the light curve
    commands, passes it as xlimits=(min, max, "-timemin") rather than copying the values onto args.xmin:
    a copied value would also reach every other reader of args.xmin, and a warning about it would name an
    argument that the user did not give.
    """
    if "subplots" not in args:
        args.subplots = False
    # a Namespace membership test matches the name, thus a parser default of None reached tick_params
    # as labelsize=None, which is a silent no-op that left the rcParams size in place
    labelfontsize = getattr(args, "labelfontsize", None)

    if xlimits is None:
        xlimits = (getattr(args, "xmin", None), getattr(args, "xmax", None), "-xmin")

    logscalex, logscaley = getattr(args, "logscalex", False), getattr(args, "logscaley", False)
    ymin = log_axis_limit(getattr(args, "ymin", None), logscale=logscaley, argname="-ymin")
    ymax = log_axis_limit(getattr(args, "ymax", None), logscale=logscaley, argname="-ymax")
    xargname = xlimits[2]
    xmin = log_axis_limit(xlimits[0], logscale=logscalex, argname=xargname)
    xmax = log_axis_limit(xlimits[1], logscale=logscalex, argname=xargname.replace("min", "max"))

    axeslist = iter_axes(ax)
    for axis in axeslist:
        # the tick direction, the top and right ticks, the minor ticks and the default label size all
        # come from the artistools matplotlibrc, thus only a command that asks for a different label
        # size sets one here

        if labelfontsize is not None:
            axis.tick_params(axis="both", which="both", labelsize=labelfontsize)

        # scale first: a limit turns autoscaling off, so setting one before the scale keeps the linear
        # padding on a log axis. A limit of None on both sides is left alone for the same reason
        if logscalex:
            axis.set_xscale("log")
        if logscaley:
            axis.set_yscale("log")

        # the lowest label of one axes meets the highest label of the axes below, thus a stack needs its
        # end ticks pruned. A single axes keeps them, because they mark the ends of the data. set_yscale
        # installs the default locators of the scale, thus this must follow it
        if len(axeslist) > 1:
            prune_log_ticks(axis.yaxis)

        if ymin is not None or ymax is not None:
            axis.set_ylim(ymin, ymax)

        if xmin is not None or xmax is not None:
            axis.set_xlim(xmin, xmax)

    return ax


def set_axis_labels(
    fig: mplfig.Figure, ax: AxesTree, xlabel: str, ylabel: str, labelfontsize: int | None, args: argparse.Namespace
) -> None:
    """Set the x and y axis labels, placing them on the figure rather than the axes when there are subplots."""
    if args.subplots:
        fig.text(0.5, 0.02, xlabel, ha="center", va="center")
        fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation="vertical")
    else:
        assert isinstance(ax, mplax.Axes)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
