"""Read the style of each data series of a plot from the command line, e.g. the label, the colour, and the dashes."""

import argparse
import math
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from types import MappingProxyType

from artistools.misc.argparser import arggroup

# an entry of a list of series styles that gives no value to its series, e.g. -label default Model2. A missing entry at
# the end of a list has the same effect. The token lets a list give a value to a later series only
SERIES_DEFAULT: t.Final = "default"


# the help text of each list of series styles gives the token
SERIES_DEFAULT_HELP: t.Final = f"An entry {SERIES_DEFAULT} keeps the default of its series"


# the line styles of matplotlib that a series style and the line properties popup name
LINESTYLE_NAMES: t.Final = ("solid", "dashed", "dotted", "dashdot")


# the short names of the line styles of matplotlib, which a command can also give
LINESTYLE_ALIASES: t.Final = MappingProxyType({"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"})


# the keys of the style of one series, which also name the options of addarg_seriesstyle, e.g. -color. A series style of
# plotestimators gives the label last, because the label takes the rest of the text, e.g. a comma
SERIES_STYLE_KEYS: t.Final = ("color", "linestyle", "dashes", "linewidth", "linealpha", "label")


def series_value_arg[T](convert: Callable[[str], T]) -> Callable[[str], T | None]:
    """Return an argparse type that reads SERIES_DEFAULT as None, and each other value with convert."""

    def convert_or_default(value: str) -> T | None:
        return None if value == SERIES_DEFAULT else convert(value)

    # argparse names the type in the message of a bad value, e.g. "invalid float value"
    convert_or_default.__name__ = getattr(convert, "__name__", "value")
    return convert_or_default


class KeepDefaultColors(argparse.Action):
    """Store the colours of --color, with the default colour of its series for each entry SERIES_DEFAULT.

    A command can give a default colour to each series, e.g. C0 to C9. A --color list replaces the whole default list,
    thus an entry SERIES_DEFAULT takes the default colour of its place in the list.
    """

    def with_default_colours(self, colours: "Sequence[t.Any]") -> list[t.Any]:
        """Return the colours with the default colour of its series for each entry SERIES_DEFAULT."""
        defaults: Sequence[str] = self.default or []
        return [
            defaults[index] if colour is None and index < len(defaults) else colour
            for index, colour in enumerate(colours)
        ]

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
    ) -> None:
        """Set the colours, and give each entry SERIES_DEFAULT the default colour of its series."""
        colours: list[t.Any] = [] if values is None else [values] if isinstance(values, str) else list(values)
        setattr(namespace, self.dest, self.with_default_colours(colours))


def color_arg(value: str) -> str:
    """Return a colour the user asked for, rejecting one matplotlib cannot parse.

    The colours are compared and resolved long before anything is drawn, so a typo caught here names the
    argument that caused it instead of surfacing from inside a plotting helper.
    """
    # the misc modules load no matplotlib module, thus a command that draws nothing starts more quickly
    import matplotlib.colors as mplcolors

    if not mplcolors.is_color_like(value):
        msg = f"Not a matplotlib colour: {value}"
        raise argparse.ArgumentTypeError(msg)

    return value


def dashes_arg(value: str) -> tuple[float, ...]:
    """Return the dash pattern of one line, which matplotlib reads as a sequence of lengths.

    The user writes the lengths of the dash and of the gap with a comma between them, e.g. 5,2.
    matplotlib refuses the text, thus this function converts the numbers. The error message then
    names --dashes.

    A pattern holds a pair of lengths for each dash. matplotlib refuses an odd number of lengths at
    the time of the plot, and it gives no line for an empty pattern, thus this function refuses both.
    """
    try:
        lengths = tuple(float(part) for part in value.replace(" ", ",").split(",") if part)
    except ValueError as exc:
        msg = f"The value {value} is not a dash pattern such as 5,2"
        raise argparse.ArgumentTypeError(msg) from exc

    if not lengths or len(lengths) % 2 != 0:
        msg = f"The dash pattern {value} must hold the length of a dash and the length of a gap, e.g. 5,2"
        raise argparse.ArgumentTypeError(msg)

    # matplotlib refuses a negative length and a pattern with no length above 0 at the time of the plot
    if not all(math.isfinite(length) and length >= 0.0 for length in lengths) or not any(lengths):
        msg = f"The dash pattern {value} must hold finite lengths of 0 or more, and at least one length above 0"
        raise argparse.ArgumentTypeError(msg)

    return lengths


def addarg_seriesstyle(
    parser: argparse.ArgumentParser,
    *,
    colordefault: Sequence[str] | None = None,
    include_linestyles: bool = True,
    include_linealpha: bool = False,
    include_dashes: bool = True,
) -> None:
    """Add the per-series style list arguments shared by the multi-series plotting commands."""
    group = arggroup(parser, "Appearance")
    group.add_argument(
        "-label",
        type=series_value_arg(str),
        default=[],
        nargs="*",
        help=f"List of series label overrides. {SERIES_DEFAULT_HELP}",
    )
    group.add_argument(
        "-color",
        "-colors",
        dest="color",
        type=series_value_arg(color_arg),
        default=list(colordefault) if colordefault else [],
        action=KeepDefaultColors,
        nargs="*",
        help=f"List of line colors. {SERIES_DEFAULT_HELP}",
    )
    if include_linestyles:
        group.add_argument(
            "-linestyle",
            type=series_value_arg(str),
            default=[],
            nargs="*",
            help=f"List of line styles. {SERIES_DEFAULT_HELP}",
        )
        group.add_argument(
            "-linewidth",
            type=series_value_arg(float),
            default=[],
            nargs="*",
            help=(
                "A list of line widths. For a reference series the value gives the size of the marker."
                f" {SERIES_DEFAULT_HELP}"
            ),
        )
    if include_linealpha:
        group.add_argument(
            "-linealpha",
            type=series_value_arg(float),
            default=[],
            nargs="*",
            help=f"List of line alphas (opacities). {SERIES_DEFAULT_HELP}",
        )
    if include_dashes:
        group.add_argument(
            "-dashes",
            type=series_value_arg(dashes_arg),
            default=[],
            nargs="*",
            help=f"List of dash patterns of lines, each one a list such as 5,2. {SERIES_DEFAULT_HELP}",
        )


def get_series_label(labels: Sequence[str | None], index: int, fallback: str) -> str:
    """Return the --label value for one series, or fallback when the user gave none for it.

    trim_or_pad pads the list with None, so an entry can be missing either as a None or, when the series
    count is not the model path count, by running off the end. An empty label is not a missing one:
    matplotlib gives no legend entry to a series labelled "", which is how one is left out of the legend.
    """
    label = labels[index] if 0 <= index < len(labels) else None

    return fallback if label is None else label
