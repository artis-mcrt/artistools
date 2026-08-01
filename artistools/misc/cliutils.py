"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import functools
import itertools
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

from artistools.commands import CustomArgHelpFormatter


def add_viewingangle_args(parser: argparse.ArgumentParser, allow_select_all: bool = False) -> None:
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


def add_modelpath_arg(
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


def add_outputfile_arg(
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


def add_outputpath_arg(
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


def add_timestep_arg(
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


def add_timedays_arg(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "str", "float"] = "rangestr",
    helptext: str | None = None,
) -> None:
    """Add the -timedays/-time/-t argument, either as a range string like 50-100 or a single value."""
    flags = ("-timedays", "-time", "-t")
    if kind == "rangestr":
        parser.add_argument(
            *flags, dest="timedays", nargs="?", help=helptext or "Range of times in days to plot (e.g. 50-100)"
        )
    elif kind == "float":
        parser.add_argument(*flags, type=float, help=helptext or "Time in days to plot")
    else:
        parser.add_argument(*flags, help=helptext or "Time in days to plot")


def add_timeminmax_args(
    parser: argparse.ArgumentParser,
    *,
    helptext_min: str = "Lower time in days",
    helptext_max: str = "Upper time in days",
) -> None:
    """Add the -timemin and -timemax arguments bounding a time range in days."""
    parser.add_argument("-timemin", type=float, help=helptext_min)
    parser.add_argument("-timemax", type=float, help=helptext_max)


def add_axis_limit_args(
    parser: argparse.ArgumentParser,
    *,
    xlimtype: type[int] | type[float] = float,
    xmindefault: float | None = None,
    xmaxdefault: float | None = None,
    xminhelp: str = "Plot range: minimum x value",
    xmaxhelp: str = "Plot range: maximum x value",
    include_x: bool = True,
    include_y: bool = True,
) -> None:
    """Add the -xmin/-xmax and -ymin/-ymax plot range arguments."""
    if include_x:
        parser.add_argument("-xmin", type=xlimtype, default=xmindefault, help=xminhelp)
        parser.add_argument("-xmax", type=xlimtype, default=xmaxdefault, help=xmaxhelp)
    if include_y:
        parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis minimum")
        parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis maximum")


def add_series_style_args(
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


def add_figscale_args(
    parser: argparse.ArgumentParser, *, figscaledefault: float = 1.0, include_figwidthscale: bool = False
) -> None:
    """Add the figure size scale factor arguments."""
    parser.add_argument(
        "-figscale", type=float, default=figscaledefault, help="Scale factor for plot area. 1.0 is for single-column"
    )
    if include_figwidthscale:
        parser.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the spectrum smoothing filter arguments (get_filterfunc reads exactly these dests)."""
    parser.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")
    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order, e.g. -filtersavgol 5 3",
    )


def add_maxpacketfiles_arg(parser: argparse.ArgumentParser) -> None:
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
    kwargs = dict(kwargs)  # keys are renamed to argument dests below, so don't mutate the caller's dict
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


def parse_range(rng: str, dictvars: dict[str, int]) -> Iterable[t.Any]:
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


def parse_range_list(rngs: str | list[str] | list[int] | int, dictvars: dict[str, int] | None = None) -> list[t.Any]:
    """Parse a string with comma-separated ranges or a list of range strings.

    Return a sorted list of integers in any of the ranges.
    """
    if isinstance(rngs, list):
        rngs = ",".join(str(x) for x in rngs)
    elif not hasattr(rngs, "split"):
        return [rngs]

    assert isinstance(rngs, str)
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


def flatten_list(listin: list[t.Any]) -> list[t.Any]:
    """Flatten a list of lists."""
    listout = []
    for elem in listin:
        if isinstance(elem, list):
            listout.extend(elem)
        else:
            listout.append(elem)
    return listout


def normalize_path_list(paths: t.Any, default: Path | str = ".") -> list[Path]:
    """Return a flat list of Paths from a scalar or (possibly nested) sequence of paths, using the default if none given."""
    if not paths:
        return [Path(default)]
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(p) for p in flatten_list(list(paths))]


def get_filterfunc(args: argparse.Namespace) -> Callable[[t.Any], t.Any] | None:
    """Use command line arguments to determine the appropriate filter function."""
    filterfunc = None
    dictargs = vars(args)

    if dictargs.get("filtermovingavg", False):

        def movavgfilterfunc(ylist: t.Any) -> t.Any:
            import numpy as np

            n = args.filtermovingavg
            arr_padded = np.pad(ylist, (n // 2, n - 1 - n // 2), mode="edge")
            return np.convolve(arr_padded, np.ones((n,)) / n, mode="valid")

        assert filterfunc is None
        filterfunc = movavgfilterfunc

    if dictargs.get("filtersavgol", False):
        import scipy.signal

        window_length, polyorder = (int(x) for x in args.filtersavgol)

        assert filterfunc is None
        filterfunc = functools.partial(  # pyright: ignore[reportCallIssue]
            scipy.signal.savgol_filter, window_length=window_length, polyorder=polyorder, mode="interp"
        )

        print("Applying Savitzky-Golay filter")

    return filterfunc
