"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import itertools
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

from artistools.commands import CustomArgHelpFormatter

if t.TYPE_CHECKING:
    from collections.abc import Collection

    import numpy as np
    import numpy.typing as npt

# a path argument arrives as a scalar, as a sequence, or as a nested sequence from repeated -modelpath
type PathArg = Path | str | Sequence[PathArg] | None


def addarg_viewingangle(parser: argparse.ArgumentParser, allow_select_all: bool = False) -> None:
    """Add the viewing direction selection and averaging arguments shared by the plotting commands."""
    parser.add_argument(
        "-plotvspecpol",
        type=int,
        metavar="n",
        nargs="+",
        help="Plot viewing angles from vspecpol virtual packets. Expects int for angle = spec number in vspecpol files",
    )

    parser.add_argument(
        "-plotviewingangle",
        "-dirbin",
        type=int,
        metavar="n",
        nargs="+",
        help=(
            "Plot viewing directions. Expects int for direction bin in specpol_res.out"
            + (". Use -2 to select all viewing angles" if allow_select_all else "")
        ),
    )

    parser.add_argument(
        "--usedegrees",
        action="store_true",
        help="Use degrees instead of radians for direction angles. Only works with -plotviewingangle",
    )

    # averaging over one angle leaves one bin per index of the other, so the two cannot be combined. argparse
    # enforces this once here for every command that takes these flags
    averagegroup = parser.add_mutually_exclusive_group()

    averagegroup.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # deprecated alias for --average_over_phi_angle kept for backwards compatibility
    averagegroup.add_argument("--average_every_tenth_viewing_angle", action="store_true", help=argparse.SUPPRESS)

    averagegroup.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
    )


def addarg_modelpath(
    parser: argparse.ArgumentParser,
    *,
    positional: bool = False,
    multiplepaths: bool = False,
    required: bool = False,
    default: t.Any = None,
    helptext: str = "Path to ARTIS folder",
) -> None:
    """Add the ARTIS model path argument (-modelpath option, or a positional path when positional=True)."""
    kwargs: dict[str, t.Any] = {"type": Path, "default": default, "help": helptext}
    if multiplepaths:
        kwargs["nargs"] = "*"
    if positional:
        parser.add_argument("modelpath", **kwargs)
    else:
        if required:
            kwargs["required"] = True
        parser.add_argument("-modelpath", **kwargs)


def addarg_outputfile(
    parser: argparse.ArgumentParser,
    *,
    default: t.Any = None,
    astype: type[Path] | type[str] | None = Path,
    extraflags: Sequence[str] = (),
    helptext: str = "Path/filename for the output file",
) -> None:
    """Add the -outputfile/-o argument naming a single output file."""
    kwargs: dict[str, t.Any] = {"dest": "outputfile", "default": default, "help": helptext}
    if astype is not None:
        kwargs["type"] = astype
    parser.add_argument("-outputfile", *extraflags, "-o", **kwargs)


def addarg_outputpath(
    parser: argparse.ArgumentParser,
    *,
    default: t.Any = ".",
    astype: type[Path] | None = None,
    helptext: str = "Path for output files",
) -> None:
    """Add the -outputpath/-o argument naming a directory for output files."""
    kwargs: dict[str, t.Any] = {"default": default, "help": helptext}
    if astype is not None:
        kwargs["type"] = astype
    parser.add_argument("-outputpath", "-o", **kwargs)


def addarg_modelgridindex(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "int", "append", "list"] = "int",
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the -modelgridindex/-cell/-mgi argument that selects the model grid cell or cells.

    A command that plots one cell takes kind="int". A command that reads several cells takes a range
    string such as 3-7, which parse_range_list expands. Every command offers the same three flags.
    """
    flags = ("-modelgridindex", "-cell", "-mgi")
    helptext = helptext or "Model grid cell to plot"
    if kind == "int":
        parser.add_argument(*flags, type=int, default=default, help=helptext)
    elif kind == "append":
        parser.add_argument(*flags, action="append", default=default, help=helptext)
    elif kind == "list":
        parser.add_argument(*flags, nargs="*", default=default, help=helptext)
    else:
        parser.add_argument(*flags, default=default, help=helptext)


def addarg_timestep(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "int", "strappend"] = "rangestr",
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the -timestep/-ts argument: a range string like 45-65, a single int, or an appendable list."""
    flags = ("-timestep", "-ts")
    if kind == "rangestr":
        parser.add_argument(
            *flags, dest="timestep", nargs="?", default=default, help=helptext or "First timestep or a range e.g. 45-65"
        )
    elif kind == "int":
        parser.add_argument(*flags, type=int, default=default, help=helptext or "Timestep number to plot")
    else:
        parser.add_argument(*flags, action="append", default=default, help=helptext or "Timestep number to plot")


def addarg_timedays(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "str", "float"] = "rangestr",
    helptext: str | None = None,
    short_alias: bool = True,
) -> None:
    """Add the -timedays/-time/-t argument, either as a range string like 50-100 or a single value.

    A command that takes no -timestep passes short_alias=False. argparse joins a value to a single-dash
    flag, thus "-timestep 30" on such a parser reads as "-t imestep" and reports an invalid value for an
    argument that the user never named. A command that declares -timestep matches it exactly, thus "-t"
    is safe there. test_shared_cli_args_consistent holds this rule.
    """
    flags = ("-timedays", "-time", "-t") if short_alias else ("-timedays", "-time")
    if kind == "rangestr":
        parser.add_argument(
            *flags, dest="timedays", nargs="?", help=helptext or "Range of times in days to plot (e.g. 50-100)"
        )
    elif kind == "float":
        parser.add_argument(*flags, type=float, help=helptext or "Time in days to plot")
    else:
        parser.add_argument(*flags, help=helptext or "Time in days to plot")


def addarg_timeminmax(
    parser: argparse.ArgumentParser,
    *,
    helptext_min: str = "Lower time in days",
    helptext_max: str = "Upper time in days",
) -> None:
    """Add the -timemin and -timemax arguments bounding a time range in days."""
    parser.add_argument("-timemin", type=float, help=helptext_min)
    parser.add_argument("-timemax", type=float, help=helptext_max)


def addarg_axislimits(
    parser: argparse.ArgumentParser,
    *,
    xlimtype: type[int] | type[float] = float,
    xmindefault: float | None = None,
    xmaxdefault: float | None = None,
    xminhelp: str = "Plot range: minimum x value",
    xmaxhelp: str = "Plot range: maximum x value",
    include_x: bool = True,
    include_y: bool = True,
    wavelength_aliases: bool = False,
) -> None:
    """Add the -xmin/-xmax and -ymin/-ymax plot range arguments.

    A command whose x axis is a wavelength in Angstroms takes wavelength_aliases, which adds the
    -lambdamin and -lambdamax spellings of the same arguments.
    """
    if include_x:
        xminflags = ("-xmin", "-lambdamin") if wavelength_aliases else ("-xmin",)
        xmaxflags = ("-xmax", "-lambdamax") if wavelength_aliases else ("-xmax",)
        parser.add_argument(*xminflags, dest="xmin", type=xlimtype, default=xmindefault, help=xminhelp)
        parser.add_argument(*xmaxflags, dest="xmax", type=xlimtype, default=xmaxdefault, help=xmaxhelp)
    if include_y:
        parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis minimum")
        parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis maximum")


def color_arg(value: str) -> str:
    """Return a colour the user asked for, rejecting one matplotlib cannot parse.

    The colours are compared and resolved long before anything is drawn, so a typo caught here names the
    argument that caused it instead of surfacing from inside a plotting helper.
    """
    import matplotlib.colors as mplcolors

    if not mplcolors.is_color_like(value):
        msg = f"not a matplotlib color: {value}"
        raise argparse.ArgumentTypeError(msg)

    return value


def addarg_seriesstyle(
    parser: argparse.ArgumentParser,
    *,
    colordefault: Sequence[str] | None = None,
    include_linestyles: bool = True,
    include_linealpha: bool = False,
    include_dashes: bool = True,
) -> None:
    """Add the per-series style list arguments shared by the multi-series plotting commands."""
    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")
    parser.add_argument(
        "-color",
        "-colors",
        dest="color",
        type=color_arg,
        default=list(colordefault) if colordefault else [],
        nargs="*",
        help="List of line colors",
    )
    if include_linestyles:
        parser.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")
        parser.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")
    if include_linealpha:
        parser.add_argument("-linealpha", default=[], nargs="*", help="List of line alphas (opacities)")
    if include_dashes:
        parser.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")


def addarg_figscale(
    parser: argparse.ArgumentParser, *, figscaledefault: float = 1.0, include_figwidthscale: bool = False
) -> None:
    """Add the figure size scale factor arguments."""
    parser.add_argument(
        "-figscale", type=float, default=figscaledefault, help="Scale factor for plot area. 1.0 is for single-column"
    )
    if include_figwidthscale:
        parser.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")


def addarg_filter(parser: argparse.ArgumentParser) -> None:
    """Add the spectrum smoothing filter arguments (get_filterfunc reads exactly these dests)."""
    parser.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")
    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order, e.g. -filtersavgol 5 3",
    )


def addarg_action(parser: argparse.ArgumentParser, choices: Sequence[str], helptext: str) -> None:
    """Add the positional action argument that selects what the subcommand does."""
    parser.add_argument(
        "action",
        # optional so that main(argsraw=[], action=...) works, since parse_cli_args ignores
        # argsraw as soon as any keyword argument is given
        nargs="?",
        default=None,
        choices=choices,
        help=helptext,
    )


def suggest_names(name: str, candidates: "Collection[str]", *, count: int = 3) -> str:
    """Return a sentence that names the closest candidates, or an empty string when none is close.

    A name that differs only in case comes first, because that mistake is common and difflib scores a
    short name such as "te" against "Te" below its own threshold.
    """
    import difflib

    names = list(candidates)
    if samecase := [other for other in names if other.lower() == name.lower() and other != name]:
        return f" Did you mean {samecase[0]}?"

    matches = difflib.get_close_matches(name, names, n=count, cutoff=0.6)

    return f" Did you mean {', '.join(matches)}?" if matches else ""


def exit_with_error(message: str) -> t.NoReturn:
    """Print an error message and stop with a failing exit status.

    A mistake in the arguments earns a message rather than a traceback, and a script that runs the
    command sees that it failed.
    """
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_action(args: argparse.Namespace) -> None:
    """Stop with an error message when the caller gave no action."""
    if args.action is None:
        exit_with_error("no action given. Run with --help to see the available actions.")


def addarg_show(parser: argparse.ArgumentParser) -> None:
    """Add the --show argument that opens the figure in a window before it is saved."""
    parser.add_argument("--show", action="store_true", help="Show the plot in a window before saving it")


def addarg_quiet(parser: argparse.ArgumentParser) -> None:
    """Add the --quiet argument that hides the progress messages (__main__ reads this dest)."""
    parser.add_argument("--quiet", action="store_true", help="Hide the progress messages. An error still appears")


def addarg_dpi(parser: argparse.ArgumentParser, *, default: int = 250) -> None:
    """Add the -dpi argument setting the resolution of a raster output file."""
    parser.add_argument("-dpi", type=int, default=default, help="Dots per inch for the output file")


def addarg_yscale(parser: argparse.ArgumentParser) -> None:
    """Add the -yscale argument that selects the scale of the vertical axis.

    "auto" leaves the choice to the command, which keeps --logscaley working. "lin" means "linear".
    """
    parser.add_argument(
        "-yscale",
        choices=["log", "linear", "lin", "auto"],
        default="auto",
        help="Scale of the vertical axis. auto lets the command choose",
    )


def resolve_yscale(args: argparse.Namespace) -> None:
    """Set args.logscaley from -yscale, which every plot helper reads.

    --logscaley is the older spelling of "-yscale log". Two arguments that ask for a different scale
    get a message rather than a silent precedence.
    """
    yscale = getattr(args, "yscale", "auto")
    if yscale == "auto":
        return

    wantlog = yscale == "log"
    if getattr(args, "logscaley", False) and not wantlog:
        exit_with_error(f"specify only one of --logscaley and -yscale {yscale}")

    args.logscaley = wantlog


def addarg_notitle(parser: argparse.ArgumentParser) -> None:
    """Add the --notitle argument that suppresses the plot title (set_plot_title reads this dest)."""
    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")


def addarg_nolegend(parser: argparse.ArgumentParser) -> None:
    """Add the --nolegend argument that suppresses the plot legend."""
    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")


def addarg_maxpacketfiles(parser: argparse.ArgumentParser) -> None:
    """Add the -maxpacketfiles argument limiting how many packet files are read."""
    parser.add_argument(
        "-maxpacketfiles", "-maxpacketsfiles", type=int, default=None, help="Limit the number of packet files read"
    )


def parse_cli_args(
    addargsfunc: Callable[[argparse.ArgumentParser], None],
    description: str | None,
    args: argparse.Namespace | None,
    argsraw: Sequence[str] | None = None,
    kwargs: dict[str, t.Any] | None = None,
) -> argparse.Namespace:
    """Return args unchanged if already parsed, otherwise parse the command line using the options defined by addargsfunc.

    Any keyword arguments override the parser defaults, and when at least one is given, the command line/argsraw is ignored.
    """
    if args is not None:
        return args

    import argcomplete

    parser = argparse.ArgumentParser(formatter_class=CustomArgHelpFormatter, description=description)
    addargsfunc(parser)
    kwargs = kwargs or {}
    set_args_from_dict(parser, kwargs)
    argcomplete.autocomplete(parser)
    return parser.parse_args([] if kwargs else argsraw)


def resolve_outputfile(outputfile: Path | str | None, defaultoutputfile: Path | str) -> Path:
    """Return the output file path, appending the default filename if outputfile is unset or refers to a folder.

    A path with no file extension is treated as a folder and will be created if it does not exist.
    """
    if not outputfile:
        return Path(defaultoutputfile)

    outputfile = Path(outputfile)
    if outputfile.is_dir() or not outputfile.suffixes:
        outputfile.mkdir(parents=True, exist_ok=True)
        return outputfile / defaultoutputfile

    return outputfile


def set_args_from_dict(parser: argparse.ArgumentParser, kwargs: dict[str, t.Any]) -> None:
    """Set argparse defaults from a dictionary."""
    kwargs = kwargs.copy()  # keys are renamed to argument dests below, so don't mutate the caller's dict
    # set_defaults expects the dest of an argument. Here we allow the option strings to be used as keys
    for arg in parser._actions:  # ruff:ignore[private-member-access]
        for optstring in arg.option_strings:
            if optstring.lstrip("-") in kwargs and arg.dest not in kwargs:
                kwargs[arg.dest] = kwargs.pop(optstring.lstrip("-"))

    parser.set_defaults(**kwargs)
    # set required=False on all arguments to avoid errors about missing required arguments when we set defaults from kwargs
    for arg in parser._actions:  # ruff:ignore[private-member-access]
        if arg.default is not None:
            arg.required = False

    if unknown := {k: v for k, v in kwargs.items() if k not in (arg.dest for arg in parser._actions)}:  # ruff:ignore[private-member-access]
        msg = f"Unknown argument names: {unknown}"
        raise ValueError(msg)


def parse_range(rng: str, dictvars: dict[str, int]) -> Iterable[int]:
    """Parse a string with an integer range and return a list of numbers, replacing special variables in dictvars."""
    strparts = rng.split("-")

    if len(strparts) not in {1, 2}:
        msg = f"Bad range: '{rng}'"
        raise ValueError(msg)

    parts = [int(i) if i not in dictvars else dictvars[i] for i in strparts]
    start: int = parts[0]
    end: int = start if len(parts) == 1 else parts[1]

    if start > end:
        end, start = start, end

    return range(start, end + 1)


def parse_range_list(rngs: str | list[str] | list[int] | int, dictvars: dict[str, int] | None = None) -> list[int]:
    """Parse a string with comma-separated ranges or a list of range strings.

    Return a sorted list of integers in any of the ranges.
    """
    if isinstance(rngs, list):
        rngs = ",".join(str(x) for x in rngs)
    elif not isinstance(rngs, str):
        return [rngs]

    return sorted(set(itertools.chain.from_iterable([parse_range(rng, dictvars or {}) for rng in rngs.split(",")])))


def makelist(x: Sequence[t.Any] | str | Path | None) -> list[t.Any]:
    """If x is not a list (or is a string), make a list containing x."""
    if x is None:
        return []
    return [x] if isinstance(x, str | Path) else list(x)


def trim_or_pad(requiredlength: int, *listoflistin: t.Any) -> Sequence[Sequence[t.Any]]:
    """Make lists equal in length to requiredlength either by padding with None or truncating."""
    list_sequence = []
    for listin in listoflistin:
        listin_makelist = makelist(listin)

        listout = [listin_makelist[i] if i < len(listin_makelist) else None for i in range(requiredlength)]

        assert len(listout) == requiredlength
        list_sequence.append(listout)
    return list_sequence


def get_series_label(labels: Sequence[str | None], index: int, fallback: str) -> str:
    """Return the -label value for one series, or fallback when the user gave none for it.

    trim_or_pad pads the list with None, so an entry can be missing either as a None or, when the series
    count is not the model path count, by running off the end. An empty label is not a missing one:
    matplotlib gives no legend entry to a series labelled "", which is how one is left out of the legend.
    """
    label = labels[index] if 0 <= index < len(labels) else None

    return fallback if label is None else label


def flatten_list(listin: list[t.Any]) -> list[t.Any]:
    """Flatten a list of lists."""
    listout = []
    for elem in listin:
        if isinstance(elem, list):
            listout.extend(elem)
        else:
            listout.append(elem)
    return listout


def normalize_path_list(paths: PathArg, default: Path | str = ".") -> list[Path]:
    """Return a flat list of Paths from a scalar or (possibly nested) sequence of paths, using the default if none given."""
    if not paths:
        return [Path(default)]
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(p) for p in flatten_list(list(paths))]


def get_filterfunc(args: argparse.Namespace) -> "Callable[[npt.ArrayLike], npt.NDArray[np.float64]] | None":
    """Use command line arguments to determine the appropriate filter function."""
    filterfunc = None
    dictargs = vars(args)

    if dictargs.get("filtermovingavg", False):

        def movavgfilterfunc(ylist: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
            import numpy as np

            n = args.filtermovingavg
            arr_padded = np.pad(ylist, (n // 2, n - 1 - n // 2), mode="edge")
            return np.convolve(arr_padded, np.ones((n,)) / n, mode="valid")

        assert filterfunc is None
        filterfunc = movavgfilterfunc

    if dictargs.get("filtersavgol", False):
        from artistools.misc.general import savgol_filter

        window_length, polyorder = (int(x) for x in args.filtersavgol)

        def savgolfilterfunc(ylist: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
            return savgol_filter(ylist, window_length=window_length, polyorder=polyorder)

        assert filterfunc is None
        filterfunc = savgolfilterfunc

        print("Applying Savitzky-Golay filter")

    return filterfunc
