"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import dataclasses as dc
import itertools
import operator
import re
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from artistools.commands import CustomArgHelpFormatter
from artistools.commands import SuggestingArgumentParser
from artistools.misc.remote import model_path_from_text

if t.TYPE_CHECKING:
    from collections.abc import Collection

    import numpy as np
    import numpy.typing as npt

# a path argument arrives as a scalar, as a sequence, or as a nested sequence from repeated -modelpath
type PathArg = Path | str | Sequence[PathArg] | None

# an error message names the ARTIS folders that are near. A folder of runs can hold many,
# thus this constant gives the maximum number of names
MAXNEARBYFOLDERS = 6


class CommaJoinAction(argparse.Action):
    """Join a repeated flag: "-ts 3 -ts 5" gives "3,5", which parse_range_list expands.

    Without this the last occurrence replaces the first, and the command drops a cell silently.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # ruff:ignore[unused-method-argument]
        namespace: argparse.Namespace,
        values: str | Sequence[t.Any] | None,
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Put the joined text of every occurrence of this flag in the namespace."""
        previous = getattr(namespace, self.dest, None)
        text = ",".join(str(value) for value in values) if isinstance(values, list) else str(values)
        isfirstoccurrence = previous is self.default or previous is None
        setattr(namespace, self.dest, text if isfirstoccurrence else f"{previous},{text}")


class CellListAction(argparse.Action):
    """Store the cells of -modelgridindex as a sorted list, and add the cells of each repeated flag.

    "-cell 3 -cell 5-7" gives [3, 5, 6, 7]. Each command then reads a list, whatever text the user gave.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # ruff:ignore[unused-method-argument]
        namespace: argparse.Namespace,
        values: str | Sequence[t.Any] | None,
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Put the cells of every occurrence of this flag in the namespace."""
        previous = getattr(namespace, self.dest, None)
        try:
            cells = get_cell_list(str(values))
        except ValueError as exc:
            # argparse names the flag in the message of an ArgumentError, and it stops with no traceback
            msg = f"'{values}' names no cells. Give a cell, a range such as 3-7, or a list such as 1,4,9"
            raise argparse.ArgumentError(self, msg) from exc
        isfirstoccurrence = previous is self.default or previous is None
        setattr(namespace, self.dest, cells if isfirstoccurrence else sorted({*previous, *cells}))


def get_cell_list(cells: "str | t.SupportsIndex | Iterable[str | t.SupportsIndex]") -> list[int]:
    """Return the sorted integers of a text of ranges, an integer, or a sequence, e.g. "3-7,9", 12, or [4, "6-7"].

    The values of -modelgridindex and -ion_stages use this function. A keyword argument of main() does not pass
    through the parser, thus the function also takes an integer or a list. A float gives a TypeError.
    """
    if isinstance(cells, str):
        return parse_range_list(cells)
    if isinstance(cells, Iterable):
        return sorted({cell for item in cells for cell in get_cell_list(item)})
    return [operator.index(cells)]


def contiguous_runs(numbers: Sequence[int]) -> list[list[int]]:
    """Split a sorted sequence of integers into the runs that have no gap."""
    runs: list[list[int]] = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])

    return runs


def format_range_list(numbers: Iterable[int]) -> str:
    """Return the text that parse_range_list reads for these numbers, e.g. "3-7,9" for [3, 4, 5, 6, 7, 9]."""
    return ",".join(
        str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in contiguous_runs(sorted(set(numbers)))
    )


def arggroup(parser: argparse.ArgumentParser, title: str) -> "argparse._ArgumentGroup":  # pyright: ignore[reportPrivateUsage]
    """Return the argument group of the parser with this title, and make it if the parser has none.

    The flagship commands hold more than 70 options, thus a flat listing is hard to read. Each shared
    helper puts its arguments into a titled group, and the help of every command gains the same shape.
    """
    for group in parser._action_groups:  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
        if group.title == title:
            return group

    return parser.add_argument_group(title)


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
        help="Show the angles of the viewing directions in degrees, and not as cos θ and radians",
    )

    # averaging over one angle leaves one bin per index of the other, so the two cannot be combined. argparse
    # enforces this once here for every command that takes these flags
    averagegroup = parser.add_mutually_exclusive_group()

    averagegroup.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # the hidden alias keeps the old spelling. argparse writes a warning when a command gives it
    averagegroup.add_argument(
        "--average_every_tenth_viewing_angle",
        dest="average_over_phi_angle",
        action="store_true",
        deprecated=True,
        help=argparse.SUPPRESS,
    )

    averagegroup.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
    )


class KeepGivenPaths(argparse.Action):
    """Store the paths of a positional argument, but keep the paths that the option form already gave.

    argparse applies a positional after an option that shares its dest, thus a positional that the user
    left out would otherwise hide the value of that option. argparse gives the default object itself as the
    value in that case, thus only that object counts as no value. A path that the user writes is a new object,
    thus it counts also when it is equal to the default.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Set the paths of the positional argument, unless the option form already gave some."""
        userwrote = bool(values) and values is not self.default
        given = getattr(namespace, self.dest, None)
        optiongave = bool(given) and given is not self.default
        if userwrote and optiongave:
            # the option form gives the paths in either order of the two forms
            if given != values:
                warn_ignored_paths(ignored=values, kept=given)
        elif userwrote or given is None:
            setattr(namespace, self.dest, values)

        if not userwrote:
            take_back_swallowed_folder(parser, namespace, self)


class ReplaceGivenPaths(argparse.Action):
    """Store the paths of the option form, and give a warning when they replace paths that the user wrote.

    argparse applies an option after a positional that the user wrote in front of it. The option then
    replaced the paths of the positional, and the user got no message.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Set the paths of the option, and name the paths that it replaces."""
        # the namespace holds the default object itself until KeepGivenPaths stores the paths that the user wrote
        given = getattr(namespace, self.dest, None)
        if bool(given) and given is not parser.get_default(self.dest) and given != values:
            warn_ignored_paths(ignored=given, kept=values)
        setattr(namespace, self.dest, values)


def warn_ignored_paths(ignored: "str | Sequence[t.Any] | None", kept: "str | Sequence[t.Any] | None") -> None:
    """Give a warning that the command reads the paths of the option form and not the other paths."""

    def as_text(paths: "str | Sequence[t.Any] | None") -> str:
        return ", ".join(f"'{path}'" for path in (paths if isinstance(paths, list) else [paths]))

    print_warning(
        f"the command ignores the path {as_text(ignored)}, because the option form gives the path {as_text(kept)}."
        " Give the paths in one form only"
    )


def trailing_folder_count(values: list[t.Any], *, anypath: bool = False) -> int:
    """Return how many values at the end of a list name an ARTIS run folder, or with anypath, a path that exists.

    The text alone decides for a value of the form host:path. A test of a remote folder asks its host through ssh,
    and a label of a list option can have that form, e.g. "W7:Kasen". An empty value is a label, and not the
    working folder.
    """
    from artistools.misc.fileio import folder_is_artis_run
    from artistools.misc.remote import is_remote_path
    from artistools.misc.remote import names_a_remote_folder

    def names_a_folder(value: str) -> bool:
        if is_remote_path(value):
            return names_a_remote_folder(value)

        return Path(value).exists() if anypath else folder_is_artis_run(value)

    count = 0
    for value in reversed(values):
        if not isinstance(value, str) or not value or not names_a_folder(value):
            break
        count += 1

    return count


def separate_trailing_folders(argsraw: "Sequence[str] | None") -> list[str]:
    """Return the command line with a "--" in front of the ARTIS folders that end it.

    argparse gives every word that follows to an option that reads a list. The separator marks the
    end of that list, thus each folder reaches the positional path argument.
    """
    tokens = list(sys.argv[1:] if argsraw is None else argsraw)
    if "--" in tokens:
        return tokens

    count = trailing_folder_count(tokens)
    # a command line of folders alone gives the folders to the positional argument already
    if count in {0, len(tokens)}:
        return tokens

    start = len(tokens) - count
    # a flag in front of the folders takes them as its own values, e.g. "-modelpath mymodel". The
    # separator would leave that option with no value at all. A negative number such as the -2 of
    # "-plotviewingangle -2 mymodel" is a value and not a flag
    if tokens[start - 1].startswith("-") and not re.match(r"-\.?\d", tokens[start - 1]):
        return tokens

    return [*tokens[:start], "--", *tokens[start:]]


def take_back_swallowed_folder(
    parser: argparse.ArgumentParser, namespace: argparse.Namespace, pathaction: argparse.Action
) -> None:
    """Give the ARTIS folders back to the positional path argument when an option took them.

    argparse gives every word that follows to an option that reads a list. Thus
    "plotspectra -label mylabel mymodel" left the model path empty and made "mymodel" a second
    label. The command then plotted the working folder, and it gave no message. The folders come
    back to the positional argument here, and the option keeps its other values.

    Only the last values of an option can be folders, because the user writes the paths last. Two
    options that each end with the name of a folder are ambiguous. The command then reads them as
    the user wrote them.

    A different file or folder stays with the option, because an option such as -reflightcurves reads files. Such a
    value can be a reference spectrum or a model folder with no input.txt, thus the user gets a warning.
    """
    given = getattr(namespace, pathaction.dest, None)
    # only the default object counts as no value, as in KeepGivenPaths
    if given and given is not pathaction.default:
        return

    def taken_folder_count(action: argparse.Action, *, anypath: bool = False) -> int:
        values = getattr(namespace, action.dest, None)
        takesalist = (
            bool(action.option_strings)
            and action.nargs in {"*", "+"}
            and isinstance(values, list)
            # the default list belongs to the parser, thus the user gave no value in that list
            and values is not action.default
            and bool(values)
        )
        if not takesalist:
            return 0

        assert isinstance(values, list)
        count = trailing_folder_count(values, anypath=anypath)

        # every value of the option names a folder, thus which one is the model path is unknown
        return 0 if count == len(values) else count

    candidates = [
        (action, count)
        for action in parser._actions  # ruff:ignore[private-member-access]
        if (count := taken_folder_count(action))
    ]
    if not candidates:
        for action in parser._actions:  # ruff:ignore[private-member-access]
            if count := taken_folder_count(action, anypath=True):
                flag = action.option_strings[0]
                paths = ", ".join(f"'{value}'" for value in getattr(namespace, action.dest)[-count:])
                print_warning(
                    f"{flag} read {paths} as values, and each one names a file or a folder. To read them as paths,"
                    f" write them in front of {flag}"
                )
    if len(candidates) != 1:
        return

    action, count = candidates[0]
    taken = getattr(namespace, action.dest)
    takesalist = pathaction.nargs in {"*", "+"}
    # a positional argument that holds one path takes the last folder alone
    folders = taken[-count:] if takesalist else taken[-1:]
    setattr(namespace, action.dest, taken[: len(taken) - len(folders)])
    converter = pathaction.type
    paths = [converter(folder) for folder in folders] if callable(converter) else list(folders)
    setattr(namespace, pathaction.dest, paths if takesalist else paths[0])

    flag = action.option_strings[0]
    folderword = "folder" if len(folders) == 1 else "folders"
    print_warning(
        f"{flag} read the ARTIS {folderword} {', '.join(folders)} as a value. The model path gets "
        f"the {folderword} back. Write the model path in front of {flag}"
    )


def addarg_pathoption(parser: argparse.ArgumentParser, flag: str, dest: str, *, multiplepaths: bool) -> None:
    """Accept an option that names the same paths as a positional argument.

    Some commands take the paths as a positional argument and others take -modelpath. A user who learns
    one form must not meet "unrecognized arguments" with the other, thus the option stands beside the
    positional. It stays out of the help text, because the positional already gives the paths a name.
    """
    optionkwargs: dict[str, t.Any] = {
        "dest": dest,
        "action": ReplaceGivenPaths,
        "type": model_path_from_text,
        "default": argparse.SUPPRESS,
        "help": argparse.SUPPRESS,
    }
    if multiplepaths:
        optionkwargs["nargs"] = "*"

    parser.add_argument(flag, **optionkwargs)


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
    kwargs: dict[str, t.Any] = {"type": model_path_from_text, "default": default, "help": helptext}
    if multiplepaths:
        kwargs["nargs"] = "*"
    if positional:
        # a positional argument with no nargs is required, and the option form could then never replace it
        parser.add_argument("modelpath", action=KeepGivenPaths, nargs=kwargs.pop("nargs", "?"), **kwargs)
        addarg_pathoption(parser, "-modelpath", "modelpath", multiplepaths=multiplepaths)
    else:
        if required:
            kwargs["required"] = True
        parser.add_argument("-modelpath", **kwargs)


def item_names_a_path(item: str) -> bool:
    """Return whether a positional argument names a path in any position.

    A path that a user writes has a parent folder that exists, e.g. "runs/mymodel", or it names a
    virtual codecomparison data set. A separator alone does not make a path, because an estimator
    variable can have one, e.g. heating_dep/total_dep.
    """
    from artistools.misc.fileio import path_is_codecomparison

    if not item:
        return False

    parent = Path(item).parent

    return (parent != Path() and parent.is_dir()) or path_is_codecomparison(item)


def item_names_a_folder(item: str) -> bool:
    """Return whether the last positional argument names the ARTIS folder and not a name of the command.

    A folder that exists names the model, even when the command has an item of the same name. A name
    that is not last stays an item. A variable thus keeps its meaning when a folder has the same name.
    """
    from artistools.misc.remote import is_remote_path

    # a name of an item has no colon, thus a last item of the form host:path names a remote folder
    return bool(item) and (is_remote_path(item) or Path(item).is_dir() or item_names_a_path(item))


def addarg_positional_items(
    parser: argparse.ArgumentParser, *, dest: str, metavar: str, helptext: str
) -> argparse.Action:
    """Add the positional arguments of a command that reads the ARTIS folder as the last one.

    A command that names its items on the command line, e.g. the estimator variables of
    plotestimators, adds them with this function. resolve_positional_modelpath then removes the ARTIS
    folder from the end. The caller can put an argcomplete completer on the action that this function
    returns.
    """
    # a user can write a flag between two items, thus this parser reads the positional arguments intermixed
    if isinstance(parser, SuggestingArgumentParser):
        parser.wantsintermixed = True

    return parser.add_argument(dest, nargs="*", default=[], metavar=metavar, help=helptext)


def resolve_positional_modelpath(args: argparse.Namespace, dest: str) -> list[str]:
    """Remove the ARTIS folder from the end of the positional arguments, and return the other items.

    The ARTIS folder comes last, thus only the last argument can name a folder. A folder in an earlier
    place gives an error, because a name in that place is a name of the command. This one rule serves
    every command that takes positional items. Thus every such command has the same order.

    The command adds -modelpath with the default of addarg_modelpath, which is None. A value that is
    not None then shows that the user gave -modelpath. The working folder applies at the end.
    """
    from artistools.misc.fileio import folder_is_artis_run

    items: list[str] = list(getattr(args, dest))

    if items and item_names_a_folder(items[-1]):
        from artistools.misc.remote import get_canonical_path

        givenpath = get_canonical_path(model_path_from_text(items.pop()))
        # the default of such a command is None, thus a different value is one that the user wrote.
        # A command can also take many paths, and then only the positional argument names the model
        given_modelpath = getattr(args, "modelpath", None)
        if isinstance(given_modelpath, str | Path) and Path(given_modelpath) != givenpath:
            exit_with_error(
                f"the folder '{givenpath}' and -modelpath '{given_modelpath}' name two different models",
                "Give the folder one time",
            )
        args.modelpath = givenpath

    # a bare name that holds an ARTIS run is a folder that the user wrote too early. A name that only
    # matches a folder stays an item, thus a variable keeps its meaning when a folder has that name
    def item_is_misplaced(item: str) -> bool:
        return item_names_a_path(item) or folder_is_artis_run(item)

    if misplaced := [item for item in items if item_is_misplaced(item)]:
        kept = [item for item in items if not item_is_misplaced(item)]
        example = " ".join([*kept, *misplaced, str(getattr(args, "modelpath", "") or "")]).strip()
        exit_with_error(
            f"'{misplaced[0]}' names a folder, and the ARTIS folder comes after the other arguments",
            f"Write the folder last, e.g. {example}",
        )

    if getattr(args, "modelpath", "") is None:
        args.modelpath = Path()

    setattr(args, dest, items)

    return items


def artis_subfolders(folder: Path) -> list[str]:
    """Return the names of the subfolders of this folder that hold an ARTIS run.

    A user often runs a command one level above the model, e.g. in the folder that holds every run.
    An error message can then name the folders that are near. The user does not have to list the folder.
    """
    from artistools.misc.fileio import folder_is_artis_run

    try:
        children = sorted(child.name for child in folder.iterdir() if folder_is_artis_run(child))
    except OSError:
        # the folder can give a permission error. A message about the model helps more than that error
        return []

    return children[:MAXNEARBYFOLDERS]


def make_output_folder(folder: Path | str, verb: str) -> Path:
    """Make the folder that holds the output of a command, and refuse a name that a file holds."""
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        msg = f"'{folder}' names a file that exists, and this command {verb} a folder of that name"
        raise ValueError(msg)

    folder.mkdir(parents=True, exist_ok=True)

    return folder


def addarg_output(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["file", "folder"],
    defaultname: str | None = None,
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the -outputfile/-o argument, and record what the command writes.

    A command writes one file or a folder of files, and kind says which. resolve_output_argument reads
    that word after the parse: it gives a file the defaultname of the command when -o names a folder,
    and it makes the folder that -o names either way. Thus every command keeps the promise of the help
    text, and no command writes that rule again.

    A command that names its own frames takes no defaultname, because resolve_frameset_paths gives each
    frame a name of its own.
    """
    rule = (
        "A path with no file extension names a folder, which the command creates"
        if kind == "file"
        else "The command creates this folder"
    )
    # -o is a Path on every command, thus no command reads a text where another reads a Path
    kwargs: dict[str, t.Any] = {
        "dest": "outputfile",
        "default": default,
        "type": Path,
        "help": f"{helptext or ('Path/filename for the output file' if kind == 'file' else 'Path for the output files')}. {rule}",
    }

    arggroup(parser, "output").add_argument("-outputfile", "-outputpath", "-o", **kwargs)
    parser.set_defaults(outputkind=kind, outputdefaultname=defaultname)


def resolve_output_argument(args: argparse.Namespace) -> None:
    """Apply the rule of -o that addarg_output recorded on the parser of the command.

    A command that writes one file takes the name of that file when -o names a folder. A command that
    writes a folder of files gets that folder. The folder exists after this either way.
    """
    kind = getattr(args, "outputkind", None)
    if kind is None:
        return

    outputfile = getattr(args, "outputfile", None)
    if kind == "folder":
        # a command that takes no -o names its own folder, thus there is nothing to make
        if outputfile:
            make_output_folder(outputfile, "writes")
    elif (defaultname := getattr(args, "outputdefaultname", None)) is not None:
        # resolve_outputfile gives the name of the command when -o names a folder or nothing
        args.outputfile = resolve_outputfile(outputfile, defaultname)


def addarg_modelgridindex(
    parser: argparse.ArgumentParser, *, default: list[int] | None = None, helptext: str | None = None
) -> None:
    """Add the -modelgridindex/-cell/-mgi argument that selects the model grid cell or cells.

    Every command reads the same text: a number, or a range such as 3-7, or a list such as 1,4,9.
    CellListAction stores the sorted list of the cells, and get_single_modelgridindex gives the one
    cell that a command which reads one cell needs.
    """
    arggroup(parser, "cell selection").add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        action=CellListAction,
        default=default,
        help=helptext or "Model grid cell to plot, e.g. 12 or a range 3-7",
    )


def get_single_modelgridindex(cells: Sequence[int] | None) -> int | None:
    """Return the one cell of the cells that -modelgridindex names, or None when it names none.

    Every command reads the same text, thus a range reaches a command that plots one cell. Such a
    range earns a message that says so, in place of a cell that the text does not name.
    """
    if not cells:
        return None

    if len(cells) > 1:
        msg = (
            f"-modelgridindex '{format_range_list(cells)}' names {len(cells)} cells, and this command reads one. "
            "Give one cell, e.g. -cell 12"
        )
        raise ValueError(msg)

    return cells[0]


class UnsupportedArgument(argparse.Action):
    """Stop the command, and name the argument to give in place of the one that the user gave."""

    def __init__(self, option_strings: "Sequence[str]", dest: str, instead: str, **kwargs: t.Any) -> None:
        """Take the name of the argument that this command does take."""
        super().__init__(option_strings, dest, nargs="?", help=argparse.SUPPRESS, **kwargs)
        self.instead = instead

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
    ) -> None:
        """Report that this command does not take the argument."""
        assert isinstance(parser, SuggestingArgumentParser), "every parser of a command is this class"
        parser.exit_with_help(f"{option_string} is not an argument of this command", f"Give {self.instead} instead")


def addarg_unsupported(parser: argparse.ArgumentParser, *flags: str, instead: str) -> None:
    """Declare an argument that this command does not take, so that a user gets a clear message.

    argparse joins a value to a single-dash flag, thus "-timestep 30" on a parser that declares -t but
    no -timestep reads as "-t imestep". A declared name gives a message that names the right argument.
    """
    parser.add_argument(*flags, action=UnsupportedArgument, instead=instead, default=argparse.SUPPRESS)


def addarg_timestep(parser: argparse.ArgumentParser, *, default: t.Any = None, helptext: str | None = None) -> None:
    """Add the -timestep/-ts argument that selects the timestep or the timesteps.

    Every command reads the same text: a number, a range such as 45-65, or "last". get_time_range
    refuses a list such as 4,9, because a plot reads one range of timesteps. get_single_timestep
    gives the one timestep that a command which plots one timestep needs.
    """
    arggroup(parser, "time selection").add_argument(
        "-timestep",
        "-ts",
        action=CommaJoinAction,
        default=default,
        help=helptext or "Timestep to plot, e.g. 40, a range 45-65, or last",
    )


def addarg_timedays(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "str", "float"] = "rangestr",
    helptext: str | None = None,
) -> None:
    """Add the -timedays/-time/-t argument, either as a range string like 50-100 or a single value.

    -t means -timedays on every command, thus a user needs no knowledge of which other arguments a
    command takes. A command that takes no -timestep calls addarg_unsupported for that name, because
    argparse joins a value to a single-dash flag and would read "-timestep 30" as "-t imestep".
    """
    group = arggroup(parser, "time selection")
    flags = ("-timedays", "-time", "-t")
    if kind == "rangestr":
        group.add_argument(
            *flags, dest="timedays", nargs="?", help=helptext or "Range of times in days to plot (e.g. 50-100)"
        )
    elif kind == "float":
        group.add_argument(*flags, type=float, help=helptext or "Time in days to plot")
    else:
        group.add_argument(*flags, help=helptext or "Time in days to plot")


def addarg_timeminmax(
    parser: argparse.ArgumentParser,
    *,
    helptext_min: str = "Lower time in days",
    helptext_max: str = "Upper time in days",
) -> None:
    """Add the -timemin and -timemax arguments bounding a time range in days."""
    group = arggroup(parser, "time selection")
    group.add_argument("-timemin", type=float, help=helptext_min)
    group.add_argument("-timemax", type=float, help=helptext_max)


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
    group = arggroup(parser, "appearance")
    if include_x:
        xminflags = ("-xmin", "-lambdamin") if wavelength_aliases else ("-xmin",)
        xmaxflags = ("-xmax", "-lambdamax") if wavelength_aliases else ("-xmax",)
        group.add_argument(*xminflags, dest="xmin", type=xlimtype, default=xmindefault, help=xminhelp)
        group.add_argument(*xmaxflags, dest="xmax", type=xlimtype, default=xmaxdefault, help=xmaxhelp)
    if include_y:
        group.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis minimum")
        group.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis maximum")


# an entry of a list of series styles that gives no value to its series, e.g. -label default Model2. A missing entry at
# the end of a list has the same effect. The token lets a list give a value to a later series only
SERIES_DEFAULT: t.Final = "default"

# the help text of each list of series styles gives the token
SERIES_DEFAULT_HELP: t.Final = f"An entry {SERIES_DEFAULT} keeps the default of its series"


def positive_int_arg(text: str) -> int:
    """Return the integer of the text, and reject a value below 1 when argparse reads the command line.

    A count of 0 or below passed argparse, and the command then failed after it read all the data.
    """
    try:
        value = int(text)
    except ValueError as exc:
        msg = f"invalid int value: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if value < 1:
        msg = f"{value} is not a positive count. Give 1 or more"
        raise argparse.ArgumentTypeError(msg)
    return value


def series_value_arg[T](convert: Callable[[str], T]) -> Callable[[str], T | None]:
    """Return an argparse type that reads SERIES_DEFAULT as None, and each other value with convert."""

    def convert_or_default(value: str) -> T | None:
        return None if value == SERIES_DEFAULT else convert(value)

    # argparse names the type in the message of a bad value, e.g. "invalid float value"
    convert_or_default.__name__ = getattr(convert, "__name__", "value")
    return convert_or_default


class KeepDefaultColors(argparse.Action):
    """Store the colours of -color, with the default colour of its series for each entry SERIES_DEFAULT.

    A command can give a default colour to each series, e.g. C0 to C9. A -color list replaces the whole default list,
    thus an entry SERIES_DEFAULT takes the default colour of its place in the list.
    """

    def with_default_colours(self, colours: "Sequence[t.Any]") -> list[t.Any]:
        """Return the colours with the default colour of its series for each entry SERIES_DEFAULT."""
        defaults: Sequence[str] = self.default or []
        return [
            defaults[index] if colour is None and index < len(defaults) else colour
            for index, colour in enumerate(colours)
        ]

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # ruff:ignore[unused-method-argument]
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Set the colours, and give each entry SERIES_DEFAULT the default colour of its series."""
        colours: list[t.Any] = [] if values is None else [values] if isinstance(values, str) else list(values)
        setattr(namespace, self.dest, self.with_default_colours(colours))


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


def dashes_arg(value: str) -> tuple[float, ...]:
    """Return the dash pattern of one line, which matplotlib reads as a sequence of lengths.

    The user writes the lengths of the dash and of the gap with a comma between them, e.g. 5,2.
    matplotlib refuses the text, thus this function converts the numbers. The error message then
    names -dashes.

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
    group = arggroup(parser, "appearance")
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
                "List of line widths. For a reference series the value gives the size of the marker."
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


def addarg_figscale(
    parser: argparse.ArgumentParser, *, include_figwidthscale: bool = False, helptext: str | None = None
) -> None:
    """Add the figure size scale factor arguments."""
    group = arggroup(parser, "appearance")
    group.add_argument(
        "-figscale",
        type=float,
        default=1.0,
        help=helptext or "Scale factor for plot area. 1.0 fills the text width of a page",
    )
    if include_figwidthscale:
        group.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")


def addarg_filter(parser: argparse.ArgumentParser) -> None:
    """Add the spectrum smoothing filter arguments (get_filterfunc reads exactly these dests)."""
    group = arggroup(parser, "appearance")
    group.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")
    group.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order, e.g. -filtersavgol 5 3",
    )


def addarg_action(parser: argparse.ArgumentParser, choices: Sequence[str], helptext: str) -> None:
    """Add the positional action argument that selects what the subcommand does."""
    parser.add_argument(
        "action",
        # optional so that main(argsraw=[], action=...) works, because the keyword gives the default
        # and the command line then holds no action
        nargs="?",
        default=None,
        choices=choices,
        help=helptext,
    )


def suggest_names(name: str, candidates: "Collection[str]") -> str:
    """Return a sentence that names the closest candidates, or an empty string when none is close.

    The sentence goes on the help line of an error, thus it carries no leading space. A name that
    differs only in case comes first, because that mistake is common and difflib scores a short name
    such as "te" against "Te" below its own threshold.
    """
    import difflib

    names = list(candidates)
    if samecase := [other for other in names if other.lower() == name.lower() and other != name]:
        return f"Did you mean {samecase[0]}?"

    matches = difflib.get_close_matches(name, names, n=3, cutoff=0.6)

    return f"Did you mean {', '.join(matches)}?" if matches else ""


def suggest_flags(given: str, visibleflags: "Collection[str]") -> str:
    """Return a sentence that names the flags that the user possibly meant, or an empty string.

    The user can give the start of a longer name, e.g. -dim for -dimensionreduce, thus a flag that starts
    with the given text comes before the closest name of difflib, which gave "-d". The shortest names
    come first, and three of them, as suggest_names gives: "-ti" on plotspectra starts 13 flags.
    """
    starts = sorted(sorted(flag for flag in visibleflags if flag.startswith(given)), key=len)
    return f"Did you mean {', '.join(starts[:3])}?" if starts else suggest_names(given, visibleflags)


def print_error(message: str, helptext: str = "") -> None:
    """Print an error to the standard error, then a help line that says what to do next.

    The error states the fault alone, thus a reader sees the remedy on its own line. rich colours the
    prefix in a terminal alone.
    """
    from rich.console import Console
    from rich.text import Text

    console = Console(stderr=True, highlight=False, soft_wrap=True)
    console.print(Text("error: ", style="bold red") + Text(message))
    if helptext:
        console.print(Text("help: ", style="bold cyan") + Text(helptext))


def print_warning(*values: object) -> None:
    """Print a warning to the standard error, thus --quiet keeps it and a script reads a clean product.

    rich colours the prefix in a terminal, and it writes plain text into a pipe or under NO_COLOR.
    """
    from rich.console import Console
    from rich.text import Text

    message = " ".join(str(value) for value in values)
    Console(stderr=True, highlight=False, soft_wrap=True).print(Text("WARNING: ", style="bold yellow") + Text(message))


def exit_with_error(message: str, helptext: str = "") -> t.NoReturn:
    """Print an error message and stop with a failing exit status.

    A mistake in the arguments earns a message rather than a traceback, and a script that runs the
    command sees that it failed. helptext names what the user can do next.
    """
    print_error(message, helptext)
    raise SystemExit(1)


def require_action(args: argparse.Namespace) -> None:
    """Stop with an error message when the caller gave no action."""
    if args.action is None:
        exit_with_error("no action was given", "Run with --help to see the available actions")


def addarg_residuals(parser: argparse.ArgumentParser) -> None:
    """Add the options for the residual panel and its baseline series."""
    if isinstance(parser, SuggestingArgumentParser):
        parser.wantsintermixed = True
    group = arggroup(parser, "appearance")
    baselinehelp = (
        "Select the baseline series for the residual panel in plot order, with the first series at index 0."
        " Without INDEX, use series 0. The plot must have one frame"
    )
    for flag, helptext in [
        ("-residualbaselineseries", baselinehelp),
        ("-residuals", argparse.SUPPRESS),
        ("-res", argparse.SUPPRESS),
    ]:
        group.add_argument(
            flag,
            dest="residualbaselineseries",
            type=int,
            nargs="?",
            const=0,
            default=None,
            metavar="INDEX",
            help=helptext,
        )
    # The old flag takes no value, because a numeric model path must stay positional.
    group.add_argument(
        "--residuals", dest="residualbaselineseries", action="store_const", const=0, help=argparse.SUPPRESS
    )
    group.add_argument(
        "-residual",
        dest="residuals",
        type=int,
        nargs="*",
        default=None,
        metavar="INDEX",
        help=(
            "Add a residual panel for the selected series."
            " By default, show series / baseline for flux and series minus baseline for magnitudes."
            " Give indices in plot order, with the first series at index 0."
            " By default, include all series except the baseline."
            " Print the root mean square (RMS) of each residual."
            " --write_data also writes this number"
        ),
    )
    group.add_argument(
        "-residualtype",
        choices=["absolute", "relative", "relativelog"],
        default="relative",
        help=(
            "Select the residual type: absolute shows series minus baseline."
            " Relative shows series / baseline on a linear y axis."
            " Relativelog shows the same ratio on a logarithmic y axis."
            " The default is relative. Magnitude panels always show series minus baseline on a linear y axis."
            " The statistics use series minus baseline for every type"
        ),
    )
    group.add_argument(
        "-residualymax",
        type=float,
        default=None,
        help=(
            "Set the maximum y value of the residual panel."
            " By default, use the data range. The magnitude axis shows this value at the bottom"
        ),
    )


def addarg_show(parser: argparse.ArgumentParser) -> None:
    """Add --show, which opens the figure in a window before the save, and --open, which opens the file after it."""
    group = arggroup(parser, "output")
    group.add_argument("--show", action="store_true", help="Show the plot in a window before saving it")
    group.add_argument("--open", action="store_true", help="Open the saved file with its default application")


def addarg_darkmode(parser: argparse.ArgumentParser) -> None:
    """Add --darkmode, which gives a plot white text, frames, and ticks for a dark background.

    Each command that writes a plot adds it, and save_figure applies it.
    """
    arggroup(parser, "output").add_argument(
        "--darkmode",
        action="store_true",
        help=(
            "Use white text, frames, and ticks for a dark background."
            " A PDF or SVG file has a transparent background, and a file in a different format has a black background"
        ),
    )


def addarg_quiet(parser: argparse.ArgumentParser) -> None:
    """Add the --quiet argument that hides the progress messages.

    A command writes its product with print_product, thus --quiet keeps that product and hides only the
    progress messages around it.
    """
    arggroup(parser, "output").add_argument(
        "--quiet", "-q", action="store_true", help="Hide the progress messages. Warnings and errors still appear"
    )


def addarg_verbose(parser: argparse.ArgumentParser) -> None:
    """Add the --verbose argument that shows the detail of each step.

    A command prints a summary of the work by default. --verbose adds the detail, e.g. the name of
    each file that the command reads.

    -v is the short form, but four commands gave -v to -velocity or to -rhoscale before this
    argument existed. A script holds such a command, thus -v keeps that meaning there, and
    --verbose is the only form. Declare the argument of the command first, then call this function.
    """
    flags = ["--verbose"] if "-v" in parser._option_string_actions else ["--verbose", "-v"]  # ruff:ignore[private-member-access]
    arggroup(parser, "output").add_argument(
        *flags, action="store_true", help="Show the detail of each step, e.g. each file that is read"
    )


def print_heading(text: str) -> None:
    """Print the name of the model or the series that the lines below it describe.

    Every command marks such a line, thus one style serves them all. rich gives the line its weight in
    a terminal alone, and a pipe takes the plain text. "====>" stood in front of it before.
    """
    from rich.console import Console
    from rich.text import Text

    Console(highlight=False, soft_wrap=True).print(Text(text, style="bold"))


def print_detail(*values: object) -> None:
    """Print one line of detail below a heading, with the indent that every command uses."""
    print(" ", *values)


def print_product(args: argparse.Namespace, *values: object) -> None:
    """Print the product of a command, which --quiet keeps.

    --quiet sends the progress messages to the null device. The product of the command, e.g. the table
    of --print_data or the listing of --listvariables, must reach the standard output even so. A script
    then reads that product with no progress message around it.
    """
    print(*values, file=getattr(args, "productstream", None) or sys.stdout)


def addarg_dpi(parser: argparse.ArgumentParser, *, default: int = 250) -> None:
    """Add the -dpi argument setting the resolution of a raster output file."""
    arggroup(parser, "output").add_argument("-dpi", type=int, default=default, help="Dots per inch for the output file")


def addarg_labelfontsize(parser: argparse.ArgumentParser) -> None:
    """Add the -labelfontsize argument that sets the font size of the tick labels and the axis labels."""
    arggroup(parser, "plot style").add_argument(
        "-labelfontsize",
        type=float,
        default=None,
        help="Font size of the tick labels and the axis labels. The default comes from the artistools matplotlibrc",
    )


def addarg_yscale(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add the -yscale argument that selects the scale of the vertical axis.

    "auto" reads the drawn values and takes a log scale when they cover more than one order of
    magnitude. It keeps --logscaley working, which asks for a log scale whatever the values are.
    "lin" means "linear".

    The argument has no default, thus resolve_yscale can find an explicit -yscale that gives a different
    scale from --logscaley. args.defaultyscale holds the default of the command.
    """
    arggroup(parser, "appearance").add_argument(
        "-yscale",
        choices=["log", "linear", "lin", "auto"],
        default=None,
        help=(
            "Scale of the vertical axis. auto takes a log scale for values that cover a wide range."
            f" The default is {default}"
        ),
    )
    parser.set_defaults(defaultyscale=default)


def resolve_yscale(args: argparse.Namespace) -> None:
    """Set args.yscale to the scale that the plot takes, and set args.logscaley from it.

    --logscaley is the older spelling of "-yscale log", thus it replaces the default of the command.
    If two arguments ask for different scales, the command stops with an error message. The result
    has the spelling "linear" for "lin", thus the code that reads args.yscale compares with one spelling.
    """
    if not hasattr(args, "yscale"):
        return

    logscaley = getattr(args, "logscaley", False)
    yscale = args.yscale
    if logscaley and yscale in {"linear", "lin"}:
        exit_with_error(f"specify only one of --logscaley and -yscale {yscale}")
    # "auto" chooses a scale, thus --logscaley gives that choice
    if yscale is None or (logscaley and yscale == "auto"):
        yscale = "log" if logscaley else getattr(args, "defaultyscale", "auto")

    args.yscale = "linear" if yscale == "lin" else yscale
    if args.yscale != "auto":
        args.logscaley = args.yscale == "log"


def addarg_notitle(parser: argparse.ArgumentParser) -> None:
    """Add the --notitle argument that suppresses the plot title (set_plot_title reads this dest)."""
    arggroup(parser, "appearance").add_argument(
        "--notitle", action="store_true", help="Suppress the top title from the plot"
    )


def addarg_legend(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of the legend: --nolegend, --legendframe, and -legendcols."""
    arggroup(parser, "appearance").add_argument(
        "--nolegend", action="store_true", help="Suppress the legend from the plot"
    )
    arggroup(parser, "appearance").add_argument(
        "-legendcols",
        type=positive_int_arg,
        default=None,
        help=(
            "Number of columns of the legend. The default gives a long legend more columns, thus the legend takes"
            " half of the frame height at most"
        ),
    )
    arggroup(parser, "appearance").add_argument(
        "--legendframe",
        action="store_true",
        help="Draw the legend on a white background. The legend text then stays readable over the data",
    )


def addarg_maxpacketfiles(parser: argparse.ArgumentParser) -> None:
    """Add the -maxpacketfiles argument limiting how many packet files are read."""
    from artistools.packets.core import RANKS_PER_BATCH

    parser.add_argument(
        "-maxpacketfiles",
        "-maxpacketsfiles",
        type=int,
        default=None,
        help=(
            f"Set the maximum number of packet files to read. The reader reads whole batches of {RANKS_PER_BATCH}"
            f" files, thus give a value of at least {RANKS_PER_BATCH}. The reader rounds the value down to a multiple"
            f" of {RANKS_PER_BATCH}"
        ),
    )


def check_time_selection(
    parser: SuggestingArgumentParser,
    args: argparse.Namespace,
    argsraw: "Sequence[str] | None" = None,
    kwargs: "Collection[str] | None" = None,
) -> None:
    """Stop when the arguments name a time range in more than one way.

    get_time_range cannot make this test. A caller assigns the times that it returns back onto its own
    arguments, thus a second call for a second model path would read its own output as a second range.

    The test reads the arguments that the user wrote, because a value can be the same as the default of
    the parser: plottransitions gives -timestep a default of "last", and a user can also type that value.

    set_args_from_dict makes a keyword argument of the API into a default of the parser, thus a value
    that differs from the default counts, and so does a name that kwargs holds. A name that carries
    None counts for nothing, because a caller can forward an argument that it does not use.
    """
    # split a joined value such as -ts70 here as well, or the scan below reads it as -t
    argstrings = parser.split_joined_flags(list(sys.argv[1:] if argsraw is None else argsraw))
    keywordnames = set(kwargs or ())
    flagsofdest = {
        action.dest: action.option_strings
        for action in parser._actions  # ruff:ignore[private-member-access]
    }
    allflags = list(itertools.chain.from_iterable(flagsofdest.values()))

    def givesflag(argstring: str, flag: str) -> bool:
        """Report whether the string of the command line gives the flag its value, as argparse reads it.

        The split above separates a value from a flag of more than one letter, thus only a flag of
        one letter still carries a joined value here, e.g. -t300. A string that names another flag,
        exactly or as a start of its name, belongs to that flag.
        """
        if argstring == flag or argstring.startswith(f"{flag}="):
            return True

        if len(flag) != 2 or flag.startswith("--") or not argstring.startswith(flag):
            return False

        base = argstring.partition("=")[0]

        return not any(other != flag and (base == other or other.startswith(base)) for other in allflags)

    def wasgiven(dest: str) -> bool:
        # a value of None selects no time, thus a caller that forwards an unused argument gives nothing
        if not hasattr(args, dest) or getattr(args, dest) is None:
            return False

        flags = flagsofdest.get(dest, [])
        # set_args_from_dict takes the dest of an argument or any of its option strings as a key
        if dest in keywordnames or any(flag.lstrip("-") in keywordnames for flag in flags):
            return True

        if any(givesflag(argstring, flag) for flag in flags for argstring in argstrings):
            return True

        return bool(getattr(args, dest) != parser.get_default(dest))

    given = [name for name in ("timestep", "timedays", "timemin", "timemax") if wasgiven(name)]
    # -timemin and -timemax bound one range, thus the pair counts as one way to give it. get_time_range
    # reads the range of -timedays alone, thus a bound beside it has no effect and must not pass
    ways = [f"-{name}" for name in ("timestep", "timedays") if name in given]
    if bounds := [f"-{name}" for name in ("timemin", "timemax") if name in given]:
        ways.append(" and ".join(bounds))

    if len(ways) > 1:
        exit_with_error(f"{', '.join(ways)} name the time range in more than one way", "Give only one of them")


def parse_cli_args(
    addargsfunc: Callable[[argparse.ArgumentParser], None],
    description: str | None,
    args: argparse.Namespace | None,
    argsraw: Sequence[str] | None = None,
    kwargs: dict[str, t.Any] | None = None,
) -> argparse.Namespace:
    """Return args if the caller parsed them, or else parse the command line with the options of addargsfunc.

    The keyword arguments replace the parser defaults, and argsraw then gives the other arguments as text. A call
    that gives keyword arguments and no argsraw reads no command line, because sys.argv holds a different command.
    """
    if args is not None:
        return args

    parser = SuggestingArgumentParser(formatter_class=CustomArgHelpFormatter, description=description)
    addargsfunc(parser)
    # the dispatcher adds --quiet to the parser that it builds, thus a direct call needs it here
    addarg_quiet(parser)
    kwargs = kwargs or {}
    set_args_from_dict(parser, kwargs)
    tokens = [] if argsraw is None and kwargs else argsraw
    args = parser.parse_args(separate_trailing_folders(tokens))
    check_time_selection(parser, args, tokens, kwargs)
    resolve_output_argument(args)
    resolve_yscale(args)

    return args


def resolve_outputfile(outputfile: Path | str | None, defaultoutputfile: Path | str) -> Path:
    """Return the output file path, appending the default filename if outputfile is unset or refers to a folder.

    A path with no file extension is treated as a folder and will be created if it does not exist.
    """
    if not outputfile:
        return Path(defaultoutputfile)

    outputfile = Path(outputfile)
    if outputfile.is_dir() or not outputfile.suffixes:
        return make_output_folder(outputfile, "needs") / defaultoutputfile

    return outputfile


def get_template_fields(template: Path | str) -> list[str]:
    """Return the names of the fields that a template of an output name holds."""
    import re

    return re.findall(r"\{(\w+)", str(template))


# the older name of each output template field, which a script can still hold. The help names one
# spelling for each field, thus the message that lists them leaves these out
OLDTEMPLATEFIELDS: t.Final[Mapping[str, str]] = MappingProxyType({"modelgridindex": "cell", "time_days": "timedays"})


def format_frame_path(frametemplate: Path | str, **fields: t.Any) -> str:
    """Return the path of one frame, and name the fields of the command for a template that holds another.

    A user writes the template, thus a field that the command does not give is a mistake of the
    arguments and not a fault of artistools.
    """
    withold = fields | {old: fields[new] for old, new in OLDTEMPLATEFIELDS.items() if new in fields}
    try:
        return str(frametemplate).format(**withold)
    except KeyError as exc:
        given = ", ".join(f"{{{name}}}" for name in fields)
        msg = f"the name of the output holds the field {exc}, and this command gives {given}"
        raise ValueError(msg) from exc
    except IndexError as exc:
        given = ", ".join(f"{{{name}}}" for name in fields)
        msg = f"the name of the output holds a field with no name, and each field needs one of: {given}"
        raise ValueError(msg) from exc


@dc.dataclass(frozen=True, slots=True)
class FrameSet:
    """The frames of a run that draws several figures, and the product that holds them.

    One value says whether the run combines its frames. The command reads it for the figure of each
    frame, and finish makes the product, thus the three parts of the run cannot disagree.
    """

    frametemplate: Path
    productpath: Path | None
    combines: bool
    gifduration: float | None = None

    def finish(self, framepaths: "Sequence[str | Path]", args: argparse.Namespace) -> Path | str | None:
        """Combine the frames into the product, for a run that combines them, and give its path."""
        if not self.combines:
            return None

        from artistools.misc.fileio import combine_frames

        return combine_frames(
            framepaths, self.productpath, openfile=getattr(args, "open", False), gifduration=self.gifduration
        )


def resolve_frameset_paths(
    outputfile: Path | str | None,
    *,
    framecount: int,
    framename: str,
    productname: str | None = None,
    combines: bool = False,
    gifduration: float | None = None,
) -> FrameSet:
    """Return the frames of the run and the product that holds them.

    A run that draws several figures combines them into one product, e.g. a gif or a merged pdf.
    combines says that such a product comes, and productname gives it a name for a -o path that names a
    folder. Without that name the combining step names it, as merge_pdf_files takes the names of the
    first frame and the last one.

    For a run that combines its frames, a -o path that has a file extension names the product itself, thus the
    frames go in the folder that holds it. A -o path with no file extension names a folder. This makes that folder
    either way. A -o name that holds a field, e.g. frame_{timestep}.pdf, names each frame and never the product.
    A run that does not combine its frames makes no product, thus its -o path names the frame.
    """
    givenpath = Path(outputfile) if outputfile else Path()

    if combines and givenpath.suffix and not givenpath.is_dir() and "{" not in givenpath.name:
        # a merge of the frames always writes pdf data, thus a different suffix would name the format incorrectly
        if gifduration is None and givenpath.suffix.lower() != ".pdf":
            msg = (
                f"'{givenpath.name}' names the merged product of {framecount} frames, and a merge writes a pdf file."
                f" Give a name that ends with .pdf, or a folder with -o"
            )
            raise ValueError(msg)

        # the folder of the product can carry a suffix of its own, e.g. results.v1, thus make it here
        # and let resolve_outputfile read it as a folder and not as the name of one frame
        givenpath.parent.mkdir(parents=True, exist_ok=True)

        return FrameSet(resolve_outputfile(givenpath.parent, framename), givenpath, combines, gifduration)

    frametemplate = resolve_outputfile(outputfile, framename)
    if framecount > 1 and "{" not in frametemplate.name:
        fields = get_template_fields(framename)
        example = f"-o 'frame_{{{fields[0]}}}{Path(framename).suffix}'" if fields else "-o myfolder"
        msg = (
            f"'{frametemplate.name}' names one file, and this command writes {framecount} frames. Give "
            f"a folder with -o, or a name that holds a field, e.g. {example}"
        )
        raise ValueError(msg)

    productpath = frametemplate.parent / productname if productname is not None else None

    return FrameSet(frametemplate, productpath, combines, gifduration)


def takes_a_list(action: argparse.Action) -> bool:
    """Return whether the command line gives this argument a list of values."""
    return action.nargs in {"*", "+"} or (isinstance(action.nargs, int) and action.nargs > 1)


def convert_keyword_items(arg: argparse.Action, items: "Sequence[t.Any]") -> list[t.Any]:
    """Return the items of a list keyword after the type of the argument, as the command line converts them.

    argparse converts a default of one text alone, thus label=["default", "B"] kept the text "default". An item that
    is not a text comes from Python code, thus it stays as it is.
    """
    converter = arg.type
    flag = arg.option_strings[0] if arg.option_strings else arg.dest
    try:
        converted = [converter(item) if callable(converter) and isinstance(item, str) else item for item in items]
    except (argparse.ArgumentTypeError, ValueError) as exc:
        msg = f"{flag}: {exc}"
        raise ValueError(msg) from exc

    return arg.with_default_colours(converted) if isinstance(arg, KeepDefaultColors) else converted


def set_args_from_dict(parser: argparse.ArgumentParser, kwargs: dict[str, t.Any]) -> None:
    """Set argparse defaults from a dictionary.

    A name that this command does not take raises. addarg_unsupported declares an old name, so that a
    user of the command line gets a message. Such a name is no argument of this command, thus a keyword
    of that name also raises.

    A keyword that names a destination takes priority over an option alias.
    """
    kwargs = kwargs.copy()  # keys are renamed to argument dests below, so don't mutate the caller's dict
    realactions = [
        action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if not isinstance(action, UnsupportedArgument)
    ]
    destinations = {arg.dest for arg in realactions}
    # set_defaults expects the dest of an argument. A keyword can also name an option string of the argument
    for arg in realactions:
        aliases = [flag.lstrip("-") for flag in arg.option_strings if flag.lstrip("-") not in destinations]
        names = list(dict.fromkeys(name for name in (arg.dest, *aliases) if name in kwargs))
        if len(names) > 1:
            msg = f"The keywords {', '.join(names)} name one argument, thus give only one of them"
            raise ValueError(msg)
        if names and names[0] != arg.dest:
            kwargs[arg.dest] = kwargs.pop(names[0])

    # an option that reads a list gets a list from the command line, thus main(plotviewingangle=0)
    # must mean the same as -plotviewingangle 0
    for arg in realactions:
        value = kwargs.get(arg.dest)
        # -dashes reads one tuple of numbers for each series, thus a single tuple of numbers is one item and not a
        # list. The type of -dashes is a wrapper of dashes_arg, thus the test reads the dest
        istupleitem = (
            arg.dest == "dashes"
            and isinstance(value, tuple)
            and all(isinstance(length, int | float) for length in value)
        )
        if value is not None and isinstance(arg, CellListAction):
            kwargs[arg.dest] = get_cell_list(value)
        elif value is not None and takes_a_list(arg):
            istextlist = isinstance(value, list | tuple) and not istupleitem  # pyrefly: ignore[implicit-any-type-argument]
            kwargs[arg.dest] = convert_keyword_items(arg, value if istextlist else [value])

    # set_defaults gives each argument of this dest the new default, thus the colours of -color are read first
    parser.set_defaults(**kwargs)
    # every argument takes required=False. A keyword argument can give the value instead, thus a
    # required argument would give an error for a value that the caller did supply
    for arg in realactions:
        if arg.default is not None:
            arg.required = False

    if unknown := {k: v for k, v in kwargs.items() if k not in (arg.dest for arg in realactions)}:
        msg = f"Unknown argument names: {unknown}"
        raise ValueError(msg)


def parse_float_range(text: str, description: str) -> tuple[float, float]:
    """Return the two numbers of a range such as 2.2-2.8, in the order that the text gives them.

    A hyphen also appears inside an exponent, e.g. 1e-2, and in front of a negative number. Thus only
    a hyphen that comes after a digit or after a decimal point separates the two ends of the range.
    The description names what the caller expects, and the message of an error gives that name.
    """
    try:
        first, second = (float(part) for part in re.split(r"(?<=[0-9.])-", text.strip()))
    except ValueError as exc:
        msg = f"Cannot read {text!r} as {description}"
        raise ValueError(msg) from exc

    return first, second


def parse_range(rng: str, dictvars: dict[str, int]) -> Iterable[int]:
    """Parse a string with an integer range and return a list of numbers, replacing special variables in dictvars.

    A hyphen also stands in front of a negative number, thus only a hyphen that follows a digit or a
    letter separates the two ends of the range. "-1" then names one number, which the caller refuses.
    """
    strparts = re.split(r"(?<=[0-9a-zA-Z])-", rng.strip())

    if len(strparts) not in {1, 2}:
        msg = f"Bad range: '{rng}'"
        raise ValueError(msg)

    parts = [int(i) if i not in dictvars else dictvars[i] for i in strparts]
    start: int = parts[0]
    end: int = start if len(parts) == 1 else parts[1]

    if start > end:
        # "last-1" reads as "the timestep before the last", thus a swap would select almost the whole run
        if any(part in dictvars for part in strparts):
            msg = f"The range '{rng}' ends before it starts. Give the two ends as numbers, e.g. 40-45"
            raise ValueError(msg)
        end, start = start, end

    return range(start, end + 1)


def parse_range_list(rngs: str, dictvars: dict[str, int] | None = None) -> list[int]:
    """Return the sorted integers in any of the comma-separated ranges of a string, e.g. "1,3-5"."""
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


def resolve_series_styles(
    args: argparse.Namespace, isreference: Sequence[bool], usercolors: Sequence[str | None], *stylenames: str
) -> list[str]:
    """Return one colour for each series.

    The function also pads each style list in args to one entry for each series. A colour in usercolors
    has priority. A reference series gets black and then greys, and an ARTIS
    model gets a colour of the cycle. A style list gets None where the user gave no entry.
    """
    from artistools.plottools import get_series_colors

    seriescount = len(isreference)
    for stylename in stylenames:
        setattr(args, stylename, trim_or_pad(seriescount, getattr(args, stylename))[0])

    return get_series_colors(isreference, makelist(usercolors))


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
    """Return a flat list of Paths from one path or a nested sequence of paths, or the default if there is no path.

    A remote path gets the canonical form of get_canonical_path, thus its parent keeps the host.
    """
    from artistools.misc.remote import get_canonical_path

    def to_path(path: str | Path) -> Path:
        return get_canonical_path(model_path_from_text(path) if isinstance(path, str) else Path(path))

    if not paths:
        return [Path(default)]
    if isinstance(paths, str | Path):
        return [to_path(paths)]
    return [to_path(p) for p in flatten_list(list(paths))]


def get_filterfunc(args: argparse.Namespace) -> "Callable[[npt.ArrayLike], npt.NDArray[np.float64]] | None":
    """Return the filter function that the command-line arguments select, or None.

    The function is a partial of a module function, thus pickle can send it to the server of a remote model.
    """
    import functools

    from artistools.misc.general import moving_average_filter
    from artistools.misc.general import savgol_filter

    filterfunc = None
    dictargs = vars(args)

    if dictargs.get("filtermovingavg", False):
        filterfunc = functools.partial(moving_average_filter, n=args.filtermovingavg)

    if dictargs.get("filtersavgol", False):
        if filterfunc is not None:
            msg = "Give only one of -filtermovingavg and -filtersavgol"
            raise ValueError(msg)

        window_length, polyorder = (int(x) for x in args.filtersavgol)
        filterfunc = functools.partial(savgol_filter, window_length=window_length, polyorder=polyorder)

        print("Applying Savitzky-Golay filter")

    return filterfunc
