"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import functools
import itertools
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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

    parser.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # deprecated alias for --average_over_phi_angle kept for backwards compatibility
    parser.add_argument("--average_every_tenth_viewing_angle", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
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


def normalize_path_list(paths: t.Any, default: Path | str = Path()) -> list[Path]:
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
