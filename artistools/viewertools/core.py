"""Make, parse, and change the command tokens of a viewer, and hold the constants that the viewers share."""

import argparse
import contextlib
import dataclasses as dc
import io
import math
import re
import sys
import threading
import traceback
import typing as t
from functools import lru_cache
from pathlib import Path

from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import path_is_file
from artistools.misc import print_error
from artistools.misc import separate_trailing_folders
from artistools.misc.cliutils import SERIES_DEFAULT
from artistools.plottools import ExponentLabelFormatter
from artistools.plottools import get_series_colors

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Generator
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.axes as mplax

    from artistools.commands import SuggestingArgumentParser


# each option of the table with its values, in the order of the command
type OptionRows = tuple[tuple[str, tuple[str, ...]], ...]

# a command raises these errors for a bad argument or input file. The dispatcher of the CLI reports the same errors
USER_ERRORS: t.Final = (AssertionError, FileNotFoundError, ModuleNotFoundError, PermissionError, ValueError)

REJECTED_MESSAGE: t.Final = "The command cannot draw this plot. The terminal shows the error"

# the process that runs from the application bundle of a viewer has this environment variable
MACOS_BUNDLE_VARIABLE: t.Final = "ARTISTOOLS_VIEWER_IN_BUNDLE"

# the limits of the -figwidthscale that a viewer gives the plot to fill the plot area
MIN_FIGWIDTHSCALE: t.Final[float] = 0.3
MAX_FIGWIDTHSCALE: t.Final[float] = 4.0

# the time after the last resize of the window, before the plot takes the new shape
FIT_MILLISECONDS: t.Final[int] = 200

# a new -figwidthscale that differs by less than this part from the old one draws no new plot, and fit_canvas scales
# the figure. A new plot reads the data again, thus a small change of the window does not start one
FIT_TOLERANCE: t.Final[float] = 0.05

# each continuous slider of a window has this number of positions
SLIDER_STEPS: t.Final = 1000

# the first width of the sidebar. The user can drag the handle between the plot and the sidebar
SIDEBAR_WIDTH: t.Final = 600

# Export Animation asks before it runs the command for more frames than this
MAX_ANIMATION_FRAMES: t.Final = 200

# changes closer together than this, e.g. the steps of a slider drag, give one step of Undo
UNDO_MERGE_SECONDS: t.Final = 0.8

# Undo keeps this number of steps
UNDO_LIMIT: t.Final = 200

# the first frame rate of the Play button, in frames per second
DEFAULT_PLAY_FPS: t.Final = 2.0


def get_command_tokens(
    args: argparse.Namespace, argsraw: "Sequence[str] | None", kwargs: "Mapping[str, t.Any]", *, fromdispatcher: bool
) -> list[str]:
    """Return the arguments that the user gave to the command, without the name of the command.

    The dispatcher gives the parsed arguments alone. The tokens then come from the words of a call from Python code
    (args.dispatcherargsraw, which start with the subcommand), or else from sys.argv. A script for one command, e.g.
    plotartisspectrum, takes no word for the subcommand.
    """
    if kwargs:
        exit_with_error(
            "--interactive shows the command of the plot, and a call with keyword arguments has no command",
            "Give the arguments as a list, e.g. main(argsraw=['mymodel', '--interactive'])",
        )

    if argsraw is not None:
        return list(argsraw)

    if (dispatcherargsraw := getattr(args, "dispatcherargsraw", None)) is not None:
        return list(dispatcherargsraw[1:])

    if not fromdispatcher:
        return sys.argv[1:]

    from artistools.commands import get_subcommand_of_script

    return sys.argv[1:] if get_subcommand_of_script(Path(sys.argv[0]).stem) else sys.argv[2:]


def make_parser(addargs: "Callable[[argparse.ArgumentParser], None]") -> "SuggestingArgumentParser":
    """Return the parser of a command with the arguments of addargs, and --quiet, which the dispatcher adds."""
    from artistools.commands import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    addargs(parser)
    addarg_quiet(parser)
    return parser


class ViewerTokens(t.NamedTuple):
    """The arguments that the user gave to a viewer, in the parts that the viewer reads."""

    parser: "SuggestingArgumentParser"
    # the arguments as the command parses them, before the command resolves them
    args: argparse.Namespace
    # the paths of the command in their order, which gives the style and the -label of each series
    paths: list[str]
    # each option that no control of the viewer gives, which the table of the other options shows
    otheroptions: OptionRows
    helptexts: dict[str, str]


def get_path_option_flags(parser: "SuggestingArgumentParser") -> set[str]:
    """Return the first flag of each option that gives the paths of the positional argument, e.g. -modelpath."""
    from artistools.misc.cliutils import KeepGivenPaths

    actions = parser._actions  # ruff:ignore[private-member-access]
    pathdests = {action.dest for action in actions if isinstance(action, KeepGivenPaths)}
    return {action.option_strings[0] for action in actions if action.option_strings and action.dest in pathdests}


def parse_viewer_tokens(
    addargs: "Callable[[argparse.ArgumentParser], None]", tokens: "Sequence[str]", controlleddests: "Collection[str]"
) -> ViewerTokens:
    """Parse the arguments that the user gave to a viewer, and split them into the parts that the viewer reads.

    controlleddests gives the options that the controls of the viewer give, thus the other options leave them out.
    parse_cli_args puts "--" in front of the ARTIS folders at the end, thus an option that reads a list does not take
    a folder. The tokens that no option takes are the paths, e.g. the path of "--notitle mymodel".
    """
    from artistools.misc import parse_cli_args

    parser = make_parser(addargs)
    usertokens = remove_options(parser, tokens, {"interactive"})
    basetokens = remove_options(parser, separate_trailing_folders(usertokens), controlleddests)
    pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
    otheroptions, positionaltokens = split_option_rows(parser, basetokens[pathcount:])
    # -modelpath gives the paths of the positional argument, thus the list of the series holds them and not the options
    pathflags = get_path_option_flags(parser)
    optionpaths = [path for flag, values in otheroptions if flag in pathflags for path in values]
    otheroptions = tuple(row for row in otheroptions if row[0] not in pathflags)
    paths = [*basetokens[:pathcount], *optionpaths, *(word for word in positionaltokens if word != "--")]
    args = parse_cli_args(addargs, None, None, usertokens)
    if not paths:
        paths, otheroptions = take_back_folders(parser, args, otheroptions)
    return ViewerTokens(
        parser=parser, args=args, paths=paths, otheroptions=otheroptions, helptexts=get_helptexts(parser)
    )


def take_back_folders(
    parser: "SuggestingArgumentParser", args: argparse.Namespace, rows: OptionRows
) -> tuple[list[str], OptionRows]:
    """Return the paths that the command took back from a list option, and the rows without those paths.

    The command gives the ARTIS folders at the end of a list option back to the paths, e.g. the folder of
    "-label 'My model' mymodel -t 300". The list of the series must then hold the folders, and the row of the option
    must not hold them. A command with no such folders gives no paths and the same rows.
    """
    from artistools.misc.cliutils import KeepGivenPaths

    actions = parser._actions  # ruff:ignore[private-member-access]
    pathaction = next((action for action in actions if isinstance(action, KeepGivenPaths)), None)
    parsed = getattr(args, pathaction.dest, None) if pathaction is not None else None
    if pathaction is None or not parsed or parsed == pathaction.default:
        return [], rows
    folders = parsed if isinstance(parsed, list) else [parsed]
    convert = pathaction.type if callable(pathaction.type) else str
    listflags = {flag for flag, action in get_actions_by_flag(parser).items() if action.nargs in {"*", "+"}}
    count = len(folders)
    # the option keeps one value at least, because the command takes no folder from an option of folders alone
    for index, (flag, values) in enumerate(rows):
        if flag in listflags and len(values) > count and [convert(value) for value in values[-count:]] == folders:
            return list(values[-count:]), (*rows[:index], (flag, values[:-count]), *rows[index + 1 :])
    # a control of the window gives the option that took the folders, thus the rows do not hold them
    return [str(folder) for folder in folders], rows


def exit_for_other_actions(plotname: str, otheractions: "Mapping[str, bool]") -> None:
    """Stop when an argument selects an action that is not one plot. The window shows one plot only.

    otheractions gives each such argument and whether the command gives it.
    """
    if given := [name for name, isgiven in otheractions.items() if isgiven]:
        exit_with_error(
            f"--interactive shows one plot of {plotname}. A different action comes from: {', '.join(given)}",
            f"Remove {', '.join(given)}, or remove --interactive",
        )


def make_command_tokens(basetokens: "Sequence[str]", options: "Sequence[str]") -> list[str]:
    """Return the arguments of the command with the options of the controls after the paths at the start."""
    pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
    return [*basetokens[:pathcount], *options, *basetokens[pathcount:]]


def fix_title_position(axis: "mplax.Axes") -> None:
    """Keep the title at the top of the frame.

    matplotlib moves the title above the offset text of the y axis. For this, it measures the y axis each time that it
    draws the plot. ExponentLabelFormatter puts the offset in the label of the axis, thus the offset text is empty and
    the title stays at the top of the frame.
    """
    if axis.get_title() and isinstance(axis.yaxis.get_major_formatter(), ExponentLabelFormatter):
        axis.set_title(axis.get_title(), y=1.0)


def get_direction_kinds(runfolder: Path) -> list[str]:
    """Return the kinds of viewing direction of the run. A run with a configuration of virtual packets has observers."""
    return ["", "bin", "phi", "theta", *(["vpkt"] if path_is_file(runfolder / "vpkt.txt") else [])]


@lru_cache(maxsize=64)
def run_has_direction_data(
    runfolder: Path, kind: str, resfilenames: tuple[str, ...], vpktpattern: str, *, frompackets: bool
) -> bool:
    """Return True if a run gives the plot of a direction bin of a kind, "bin" or "vpkt", with its data source.

    A direction bin comes from a file of resfilenames, e.g. spec_res.out, or from the packets files with --frompackets.
    The commands read the packets for a direction bin only with --frompackets. An observer of the virtual packets needs
    vpkt.txt and a file of vpktpattern, e.g. vspecpol_total-0.out, or the virtual packets with --frompackets. An empty
    vpktpattern gives the observers only from the virtual packets. Reload Data clears the cache.
    """
    from artistools.misc import firstexisting_or_none
    from artistools.misc.fileio import folder_holds_match
    from artistools.packets.core import get_packets_textfilename
    from artistools.packets.core import has_packets_files

    if kind == "vpkt":
        if not path_is_file(runfolder / "vpkt.txt"):
            return False
        if vpktpattern and folder_holds_match(runfolder, vpktpattern):
            return True
        virtualfile = firstexisting_or_none(get_packets_textfilename(0, virtual=True), folder=runfolder)
        return frompackets and (virtualfile is not None or folder_holds_match(runfolder, "vpackets*"))
    if firstexisting_or_none(list(resfilenames), folder=runfolder, tryzipped=True) is not None:
        return True
    return frompackets and has_packets_files(runfolder)


def get_direction_kind(args: argparse.Namespace) -> str:
    """Return the kind of viewing direction of the arguments, in the form of ControlValues.directionkind."""
    if args.plotvspecpol:
        return "vpkt"
    if not args.plotviewingangle:
        return ""
    if args.average_over_phi_angle:
        return "phi"
    return "theta" if args.average_over_theta_angle else "bin"


def get_direction_choices(runfolder: Path, directionkind: str, *, usedegrees: bool) -> list[tuple[int, str]]:
    """Return each bin of a kind of viewing direction with its label.

    An average over the phi angle or the theta angle takes the first bin of each group, as get_dirbins gives it.
    """
    averagephi, averagetheta = directionkind == "phi", directionkind == "theta"
    labels = get_dirbin_definitions(
        runfolder,
        get_dirbins(average_over_phi=averagephi, average_over_theta=averagetheta),
        vpkt_observers=directionkind == "vpkt",
        average_over_phi=averagephi,
        average_over_theta=averagetheta,
        usedegrees=usedegrees,
    )
    return list(labels.items())


class ThreadOutput(io.TextIOBase):
    """A standard stream that sends the text of each thread to the target of that thread.

    A worker thread draws a plot and hides its output, while the thread of the window still prints to the terminal.
    contextlib.redirect_stdout changes the stream of each thread, thus it cannot do this.
    """

    def __init__(self, stream: t.TextIO) -> None:
        """Send the text of each thread to stream, until send_output gives the thread a different target."""
        super().__init__()
        self.stream = stream
        self.local = threading.local()

    def get_target(self) -> t.TextIO:
        """Return the stream of the thread that calls this method."""
        target: t.TextIO = getattr(self.local, "target", self.stream)
        return target

    @t.override
    def write(self, text: str) -> int:
        return self.get_target().write(text)

    @t.override
    def flush(self) -> None:
        self.get_target().flush()

    @t.override
    def isatty(self) -> bool:
        return self.get_target().isatty()

    @t.override
    def fileno(self) -> int:
        return self.get_target().fileno()


@contextlib.contextmanager
def send_output(stdout: t.TextIO | None, stderr: t.TextIO) -> "Generator[None]":
    """Send the output of this thread to the streams until the block ends. A stdout of None keeps the terminal.

    A window installs ThreadOutput, and then the other threads keep their output. Without it, the streams of all the
    threads change.
    """
    if not (isinstance(sys.stdout, ThreadOutput) and isinstance(sys.stderr, ThreadOutput)):
        with contextlib.ExitStack() as stack:
            if stdout is not None:
                stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            yield
        return

    routes = [(stream, target) for stream, target in ((sys.stdout, stdout), (sys.stderr, stderr)) if target is not None]
    oldtargets = [getattr(stream.local, "target", None) for stream, _ in routes]
    for stream, target in routes:
        stream.local.target = target
    try:
        yield
    finally:
        for (stream, _), oldtarget in zip(routes, oldtargets, strict=True):
            if oldtarget is None:
                del stream.local.target
            else:
                stream.local.target = oldtarget


def run_command_step_with_warning(
    step: "Callable[[], str | None]", *, quiet: bool = True, echo: bool = True
) -> tuple[str | None, str]:
    """Run a step of a command, and return its message or the first line of its error, and its last warning.

    Each plot prints the same lines again, thus a quiet step discards the standard output. With echo, the terminal
    shows the whole error. A window stays open after a failed step, thus each type of error gives a message.
    """
    errors = io.StringIO()
    message: str | None
    try:
        with send_output(io.StringIO() if quiet else None, errors):
            message = step()
    except SystemExit:
        # exit_with_error and argparse print a line that starts with "error: " before they raise SystemExit
        message = get_first_line(errors.getvalue())
    except USER_ERRORS as exc:
        if echo:
            print_error(str(exc) or type(exc).__name__)
        message = get_first_line(str(exc))
    except Exception as exc:  # ruff:ignore[blind-except]
        errors.write(traceback.format_exc())
        message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
    finally:
        if echo:
            sys.stderr.write(errors.getvalue())
    return message, get_last_warning(errors.getvalue())


def run_command_step(step: "Callable[[], str | None]", *, quiet: bool = True, echo: bool = True) -> str | None:
    """Run a step of a command, and return its message or the first line of its error for the status line."""
    message, _ = run_command_step_with_warning(step, quiet=quiet, echo=echo)
    return message


def remove_colour_codes(text: str) -> str:
    """Return the text without the colour codes of a terminal.

    rich colours the text under FORCE_COLOR or TTY_COMPATIBLE also when it writes into a capture.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def get_last_warning(errors: str) -> str:
    """Return the last warning of the standard error of a step, without its prefix, or an empty text."""
    plainerrors = remove_colour_codes(errors)
    warnings = [line.strip() for line in plainerrors.splitlines() if line.strip().startswith("WARNING: ")]
    return warnings[-1].removeprefix("WARNING: ") if warnings else ""


def find_option_action(parser: argparse.ArgumentParser, argstring: str) -> tuple[argparse.Action | None, bool]:
    """Return the option that the argument names, and whether the argument also holds its value.

    argparse accepts these forms of a flag:

    - the full flag;
    - a flag with "=value";
    - a unique start of a flag;
    - a flag of one letter with its value joined, e.g. -t300.
    """
    if not argstring.startswith("-") or re.match(r"-\.?\d", argstring):
        return None, False

    optionactions = parser._option_string_actions  # ruff:ignore[private-member-access]
    name, equals, _ = argstring.partition("=")
    if name in optionactions:
        return optionactions[name], bool(equals)

    matches = {id(action): action for flag, action in optionactions.items() if flag.startswith(name)}
    if len(matches) == 1:
        return next(iter(matches.values())), bool(equals)

    if not argstring.startswith("--") and argstring[:2] in optionactions:
        return optionactions[argstring[:2]], True

    return None, False


def is_flag(argstring: str) -> bool:
    """Return True if the argument is a flag, and not a value such as a negative number."""
    return argstring.startswith("-") and not re.match(r"-\.?\d", argstring)


def split_argstrings(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> list[str]:
    """Return the tokens with each joined flag and value apart, and each group of one-letter switches apart.

    argparse reads -qv as -q and -v. Each option has one row in the option table, thus each switch needs its own
    token. A group can end with a flag that takes a value, e.g. -qt300, and that value stays with its flag.
    """
    optionactions = parser._option_string_actions  # ruff:ignore[private-member-access]
    argstrings: list[str] = []
    for argstring in parser.split_joined_flags(tokens):
        firstaction = optionactions.get(argstring[:2])
        # each token after "--" is a positional argument, and a declared flag or a start of one is not a group
        isgroup = (
            "--" not in argstrings
            and not argstring.startswith("--")
            and len(argstring) > 2
            and firstaction is not None
            and firstaction.nargs == 0
            and not any(flag.startswith(argstring.partition("=")[0]) for flag in optionactions)
            and parser.is_switch_group(argstring)
        )
        if not isgroup:
            argstrings.append(argstring)
            continue
        for index, letter in enumerate(argstring[1:], start=2):
            argstrings.append(f"-{letter}")
            if optionactions[f"-{letter}"].nargs != 0:
                if rest := argstring[index:]:
                    argstrings.append(rest)
                break
    return argstrings


def count_option_values(action: argparse.Action, argstrings: "Sequence[str]", index: int) -> int:
    """Return the number of tokens from index that are the values of the option of this action.

    An option of nargs None or of a number takes that many tokens. An option of nargs "?" takes one value, and an
    option of nargs "*" or "+" takes each value up to the next flag.
    """
    if action.nargs == 0:
        return 0
    if action.nargs is None or isinstance(action.nargs, int):
        count = 1 if action.nargs is None else action.nargs
        return min(count, len(argstrings) - index)
    maxvalues = 1 if action.nargs == "?" else len(argstrings)
    count = 0
    while count < maxvalues and index + count < len(argstrings) and not is_flag(argstrings[index + count]):
        count += 1
    return count


def remove_options(parser: "SuggestingArgumentParser", tokens: "Sequence[str]", dests: "Collection[str]") -> list[str]:
    """Return the tokens without the options of these dests and without the values of those options."""
    argstrings = split_argstrings(parser, tokens)
    kept: list[str] = []
    index = 0
    while index < len(argstrings):
        argstring = argstrings[index]
        index += 1
        if argstring == "--":
            kept.extend(argstrings[index - 1 :])
            break

        action, holdsvalue = find_option_action(parser, argstring)
        if action is None or action.dest not in dests:
            kept.append(argstring)
            continue

        if not holdsvalue:
            index += count_option_values(action, argstrings, index)

    return kept


def get_table_actions(parser: argparse.ArgumentParser, hiddendests: "Collection[str]") -> list[argparse.Action]:
    """Return the options that the table of the window offers, which are the options that are not in hiddendests.

    hiddendests holds the options that a different control of the window sets, and the options that give a
    different action from one plot.
    """
    return [
        action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings and action.help != argparse.SUPPRESS and action.dest not in hiddendests
    ]


def get_option_kind(action: argparse.Action) -> str:
    """Return the type of control that sets the value of an option in the table of the window.

    The types are these:

    - "flag": the option takes no value;
    - "choice": one value from a list;
    - "int": one integer that has a default, thus a spin box can show it;
    - "values": a fixed number of values, each in a separate field;
    - "list": a list of values that spaces separate;
    - "text": one value, or no value for an option with nargs "?".
    """
    if action.nargs == 0:
        return "flag"
    if isinstance(action.nargs, int):
        return "values"
    if action.nargs in {"*", "+"}:
        return "list"
    if action.choices:
        return "choice"
    if action.type is int and isinstance(action.default, int) and not isinstance(action.default, bool):
        return "int"
    return "text"


def get_default_tokens(action: argparse.Action) -> tuple[str, ...] | None:
    """Return the values of a new row of the table, or None if the option needs a value that has no default."""
    kind = get_option_kind(action)
    if kind == "flag" or action.nargs == "?":
        return ()
    if kind == "choice":
        choices = [str(choice) for choice in action.choices or ()]
        return (str(action.default) if str(action.default) in choices else choices[0],)
    if kind == "int":
        return (str(action.default),)
    return None


def split_option_rows(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> tuple[OptionRows, list[str]]:
    """Return each option of the tokens with its values, and the tokens that are not part of an option.

    Each row gives the first flag of the option, thus an alias, e.g. -dx, becomes the full flag, e.g. -deltax.
    """
    rows: list[tuple[str, tuple[str, ...]]] = []
    othertokens: list[str] = []
    argstrings = split_argstrings(parser, tokens)
    index = 0
    while index < len(argstrings):
        argstring = argstrings[index]
        index += 1
        if argstring == "--":
            othertokens.extend(argstrings[index - 1 :])
            break

        action, holdsvalue = find_option_action(parser, argstring)
        if action is None:
            othertokens.append(argstring)
            continue

        values: list[str] = []
        if holdsvalue:
            _, equals, value = argstring.partition("=")
            values.append(value if equals else argstring[2:])
        else:
            values.extend(argstrings[index : index + count_option_values(action, argstrings, index)])
            index += len(values)
        rows.append((action.option_strings[0], tuple(values)))

    return tuple(rows), othertokens


def get_option_tokens(flag: str, value: str) -> list[str]:
    """Return the tokens of an option with one value.

    Python 3.13 reads a value such as -1e-13 as an option, thus a value that starts with "-" joins its flag.
    """
    return [f"{flag}={value}"] if value.startswith("-") else [flag, value]


def get_row_values(rows: OptionRows, flag: str) -> tuple[str, ...] | None:
    """Return the values of the row of an option, or None if the rows have no such option."""
    return next((values for rowflag, values in rows if rowflag == flag), None)


def set_row_values(rows: OptionRows, changes: "Mapping[str, tuple[str, ...] | None]") -> OptionRows:
    """Return the rows with new values of some options, in their old places. None removes an option.

    A new option goes after the other rows.
    """
    changed = [
        (flag, changes.get(flag, values)) for flag, values in rows if not (flag in changes and changes[flag] is None)
    ]
    present = {flag for flag, _ in rows}
    added = [(flag, values) for flag, values in changes.items() if values is not None and flag not in present]
    return tuple((flag, values) for flag, values in (*changed, *added) if values is not None)


def set_figscale_row(rows: OptionRows, figscale: float, defaultfigscale: float) -> OptionRows:
    """Return the option rows with the -figscale of the box of the Figure section. The default gives no row."""
    change = None if math.isclose(figscale, defaultfigscale) else (format(figscale, "g"),)
    return set_row_values(rows, {"-figscale": change})


def get_option_row_tokens(rows: OptionRows) -> list[str]:
    """Return the command tokens of the rows of the option table."""
    return [
        token
        for flag, optionvalues in rows
        for token in (get_option_tokens(flag, *optionvalues) if len(optionvalues) == 1 else (flag, *optionvalues))
    ]


# the options that give one value for each series, e.g. each spectrum, in the order of the paths of the command
SERIES_STYLE_FLAGS: t.Final = ("-label", "-color", "-linestyle", "-linewidth", "-linealpha", "-dashes")


def get_series_value(values: "Sequence[str]", index: int) -> str | None:
    """Return the value of a series style option for the series at index, or None if the option gives none."""
    value = values[index] if index < len(values) else None
    return None if value == SERIES_DEFAULT else value


def get_series_tokens(values: "Sequence[str | None]") -> tuple[str, ...] | None:
    """Return the values of a series style option in the order of the series, or None for no option.

    A series with no value in front of a series with a value takes the token SERIES_DEFAULT.
    """
    values = list(values)
    while values and values[-1] is None:
        values.pop()
    return tuple(SERIES_DEFAULT if value is None else value for value in values) or None


def move_series_styles(rows: OptionRows, oldpaths: "Sequence[str]", newpaths: "Sequence[str]") -> OptionRows:
    """Return the option rows with the value of each series style option on the same path in the new list.

    Thus a -label stays on its series when the order changes. The values of a removed series go out of the rows. The
    values after the paths stay after the new paths, because a command can add series after them, e.g. -obsspec.
    """
    changes: dict[str, tuple[str, ...] | None] = {}
    for flag in SERIES_STYLE_FLAGS:
        if not (values := get_row_values(rows, flag)):
            continue
        bypath = {path: get_series_value(values, index) for index, path in enumerate(oldpaths)}
        newvalues = get_series_tokens([
            *(bypath.get(path) for path in newpaths),
            *get_values_after_paths(values, len(oldpaths)),
        ])
        if newvalues != values:
            changes[flag] = newvalues
    return set_row_values(rows, changes)


def set_series_rows(
    rows: OptionRows, paths: "Sequence[str]", path: str, changes: "Mapping[str, str | None]"
) -> OptionRows:
    """Return the option rows with the value of each series style option in changes for the series of one path.

    changes gives each option by its flag, e.g. -label. A value of None gives the series the default of the command,
    and the other series keep their values, also the series that the command adds after the paths, e.g. -obsspec.
    """
    rowchanges: dict[str, tuple[str, ...] | None] = {}
    for flag, value in changes.items():
        oldvalues = get_row_values(rows, flag) or ()
        newvalues = [
            *(value if other == path else get_series_value(oldvalues, index) for index, other in enumerate(paths)),
            *get_values_after_paths(oldvalues, len(paths)),
        ]
        rowchanges[flag] = get_series_tokens(newvalues)
    return set_row_values(rows, rowchanges)


def get_values_after_paths(values: "Sequence[str]", pathcount: int) -> list[str | None]:
    """Return the values of a series style option after the series of the paths, e.g. of -obsspec or -reflightcurves."""
    return [get_series_value(values, index) for index in range(pathcount, len(values))]


def get_series_style(rows: OptionRows, paths: "Sequence[str]", path: str) -> dict[str, str | None]:
    """Return the value of each series style option of the series of one path, or None where it takes the default."""
    index = list(paths).index(path)
    return {flag: get_series_value(get_row_values(rows, flag) or (), index) for flag in SERIES_STYLE_FLAGS}


def get_path_colours(paths: "Sequence[str]", isreference: "Sequence[bool]", rows: OptionRows) -> dict[str, str]:
    """Return the colour of the series of each path, as the command gives it with the -color of the rows.

    A reference series takes black and greys, and a model takes the colours of the cycle.
    """
    usercolours = get_row_values(rows, "-color") or ()
    colours = get_series_colors(isreference, [get_series_value(usercolours, index) for index in range(len(paths))])
    return dict(zip(paths, colours, strict=True))


def get_helptexts(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Return the help text of each option by its dest. A tooltip gives it, thus the window and the CLI agree."""
    return {
        action.dest: str(action.help).replace("%(default)s", str(action.default)).replace("%%", "%")
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.help and action.help != argparse.SUPPRESS
    }


def get_actions_by_flag(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Return each option of the parser by its first flag, which is the flag of a row of the option table."""
    return {
        action.option_strings[0]: action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings
    }


def get_first_line(errortext: str) -> str:
    """Return the line of an error for the status line of the window, without the "error: " of print_error.

    argparse prints its usage line before the error, and a warning can come before an error. Thus the function
    returns the line that starts with "error: ". If no line has that start, it returns the first line. The function
    removes the colour codes of the terminal first, because a code in front of "error: " hides that start.
    """
    lines = [line.strip() for line in remove_colour_codes(errortext).splitlines() if line.strip()]
    errorline = next((line for line in lines if line.startswith("error: ")), lines[0] if lines else None)
    return errorline.removeprefix("error: ") if errorline is not None else REJECTED_MESSAGE


def get_nearest_range_start(tmids: "Sequence[float]", centre: float, count: int) -> int:
    """Return the first index of the range of count timesteps that has its centre nearest to the given time.

    The time field shows the centre of the range, thus a Return with no edit gives the same range. The centre of a
    range of an even count is near the boundary of two timesteps. Thus either timestep can hold the time.
    """

    def get_centre_offset(start: int) -> float:
        return abs((tmids[start] + tmids[start + count - 1]) / 2.0 - centre)

    return min(range(len(tmids) - count + 1), key=get_centre_offset)


class PlotValues(t.Protocol):
    """The values of the controls of a viewer: a dataclass with the width and the resolution of the figure."""

    __dataclass_fields__: t.ClassVar[dict[str, "dc.Field[t.Any]"]]

    @property
    def figwidthscale(self) -> float:
        """The -figwidthscale of the values."""

    @property
    def dpi(self) -> int | None:
        """The -dpi of the values, or None for the default of the command."""


def keep_figwidthscale[ValuesT: PlotValues](restored: ValuesT, current: ValuesT) -> ValuesT:
    """Return the values that Undo restores, with the current -figwidthscale, which the window sets."""
    return dc.replace(restored, figwidthscale=current.figwidthscale)


def get_yscale_choices(parser: argparse.ArgumentParser) -> list[str]:
    """Return the choices of -yscale for a box of a viewer.

    The alias "lin" gives the same scale as "linear", thus it has no item.
    """
    return [str(choice) for choice in get_actions_by_flag(parser)["-yscale"].choices or () if choice != "lin"]


def get_fitted_figwidthscale(
    figsize: tuple[float, float], figwidthscale: float, marginwidth: float, areawidth: float, areaheight: float
) -> float:
    """Return the -figwidthscale that gives the figure the shape of the plot area.

    The frame width is proportional to -figwidthscale, and the margins and the height stay the same.
    """
    figwidth, figheight = figsize
    widthperscale = (figwidth - marginwidth) / figwidthscale
    fitted = (figheight * areawidth / areaheight - marginwidth) / widthperscale
    # 2 decimals give a short command, and a small change of the window then keeps the frames
    return round(min(max(fitted, MIN_FIGWIDTHSCALE), MAX_FIGWIDTHSCALE), 2)


def get_short_number(value: float, digits: int = 3) -> str:
    """Return a number with the significant digits for a short command, e.g. 12300 or 1.23e-05 for 3 digits."""
    return format(float(f"{value:.{digits}g}"), f".{max(digits, 10)}g")


# a limit of a selected range moves by at most this part of the width of the range when get_short_limits rounds it
SHORT_LIMITS_TOLERANCE: t.Final = 0.01


def get_short_limits(low: float, high: float) -> tuple[str, str] | None:
    """Return the two limits of a range with few significant digits, or None for a range with no width.

    Each limit takes 3 significant digits for a short command, or more digits for a narrow range. 3 digits gave the
    same limit at each end of a narrow range, and a wider range moved by up to a third of its width. With more
    digits, each limit moves by at most SHORT_LIMITS_TOLERANCE of the width.
    """
    if not low < high:
        return None
    tolerance = SHORT_LIMITS_TOLERANCE * (high - low)
    for digits in range(3, 18):
        lowtext, hightext = get_short_number(low, digits), get_short_number(high, digits)
        if abs(float(lowtext) - low) <= tolerance and abs(float(hightext) - high) <= tolerance:
            return (lowtext, hightext) if float(lowtext) < float(hightext) else None
    return None
