"""The argument parser of the artistools commands, its help formatter, and its actions for model paths."""

import argparse
import copy
import re
import sys
import typing as t
from collections.abc import Iterable
from pathlib import Path

from artistools.misc.console import print_error
from artistools.misc.console import print_warning
from artistools.misc.console import suggest_flags
from artistools.misc.console import suggest_names
from artistools.misc.fileio import folder_is_artis_model
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import names_a_remote_folder

if t.TYPE_CHECKING:
    from collections.abc import Sequence


def get_standard_flag(flag: str) -> str:
    """Return the spelling of a flag that the help gives: two dashes in front of a name, and hyphens in the name.

    ruff, uv, and the GNU tools give one dash to a flag of one letter alone, e.g. -t. One dash in front of a
    name, e.g. -timedays, and an underscore in a name, e.g. --print_data, are the older spellings.
    """
    if flag.startswith("--"):
        return "--" + flag[2:].replace("_", "-")
    if len(flag) > 2 and flag.startswith("-"):
        return "--" + flag[1:].replace("_", "-")
    return flag


def get_shown_flags(action: argparse.Action) -> list[str]:
    """Return the spellings of the flags of an action that the help gives, in the order of the declaration.

    The parser gives a flag its standard spelling only if no other flag has it, thus the older spelling then shows.
    """
    shown: list[str] = []
    for flag in action.option_strings:
        standard = get_standard_flag(flag)
        spelling = standard if standard in action.option_strings else flag
        if spelling not in shown:
            shown.append(spelling)
    return shown


def with_shown_flags(action: argparse.Action) -> argparse.Action:
    """Return the action if the help shows all of its flags.

    Otherwise, return a copy that holds only the shown spellings.
    """
    shown = get_shown_flags(action)
    if shown == action.option_strings:
        return action
    copied = copy.copy(action)
    copied.option_strings = shown
    return copied


def with_first_sentence(action: argparse.Action) -> argparse.Action:
    """Return a copy of the action whose help is the first sentence of its help, for the short help of -h."""
    if not action.help or action.help is argparse.SUPPRESS:
        return action
    copied = copy.copy(action)
    copied.help = get_first_sentence(action.help)
    return copied


class HelpAction(argparse.Action):
    """Print the short help for -h and the full help for --help, then stop, as uv does."""

    def __init__(self, option_strings: "Sequence[str]", dest: str = argparse.SUPPRESS, **kwargs: t.Any) -> None:
        """Make an action that takes no value."""
        super().__init__(option_strings, dest=dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
    ) -> None:
        if option_string == "-h" and isinstance(parser, SuggestingArgumentParser):
            parser.print_short_help()
        else:
            parser.print_help()
        parser.exit()


# a usage line that names more flags than this takes more room than a reader gives it
MAXUSAGEFLAGS = 8


# the help of each command gives its argument groups in this order. A group that only one command has comes
# after the options of argparse and before the shared groups
ARGUMENTGROUPORDER = (
    "positional arguments",
    "options",
    "Cell selection",
    "Time selection",
    "Viewing direction",
    "Appearance",
    "Output",
)


# the column of the help text in the short help of -h. 24 is the default of argparse
SHORT_HELP_POSITION = 32


def get_first_sentence(text: str) -> str:
    """Return the first sentence of a help text. An abbreviation such as e.g. does not end a sentence."""
    import re

    return re.split(r"(?<!\be\.g)(?<!\bi\.e)\.\s+(?=\S)", text, maxsplit=1)[0].removesuffix(".")


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse formatter to show default values in help text, sorted with dashes last."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Widen the help column so long option names stay on one line."""
        kwargs["max_help_position"] = 50
        super().__init__(*args, **kwargs)
        self.short = False

    def make_short(self) -> None:
        """Give the short help of -h: the first sentence of the help of each option, as uv does.

        The column of the help starts nearer to the left, thus a sentence takes fewer lines. A long flag then goes on
        its own line. argparse limits the column by the width of the terminal, as __init__ does.
        """
        self.short = True
        self._max_help_position = min(SHORT_HELP_POSITION, max(self._width - 20, 2 * self._indent_increment))

    @t.override
    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Give the default of an argument, unless that default says nothing.

        A default of None or False means that the argument is off, which the help text already says.
        Naming it filled 45 of the 80 defaults of plotspectra with "(default: None)".
        """
        if action.default is None or action.default is False or action.default == []:
            return action.help

        return super()._get_help_string(action)

    @t.override
    def _format_usage(
        self,
        usage: str | None,
        actions: "Iterable[argparse.Action]",
        groups: "Iterable[argparse._MutuallyExclusiveGroup]",
        prefix: str | None,
    ) -> str:
        """Name the options of a long command as "[options]" in place of every flag.

        The usage of plotspectra listed 77 flags over 61 lines, which no reader takes in. The help text
        below it names each one.
        """
        if prefix is None:
            prefix = "Usage: "
        # the usage gives the standard spelling of each flag, thus each mutually exclusive group must name the copies
        shownofaction = {id(action): with_shown_flags(action) for action in actions}
        actions = list(shownofaction.values())
        shown_groups: list[argparse._MutuallyExclusiveGroup] = []
        for group in groups:
            shown_group = copy.copy(group)
            shown_group._group_actions = [  # ruff:ignore[private-member-access]
                shownofaction.get(id(action), action)
                for action in group._group_actions  # ruff:ignore[private-member-access]
            ]
            shown_groups.append(shown_group)
        groups = shown_groups
        # the usage names the flags that the help shows, thus the hidden ones do not count
        optionals = [action for action in actions if action.option_strings and action.help != argparse.SUPPRESS]
        if usage is None and len(optionals) > MAXUSAGEFLAGS:
            positionals = [action for action in actions if not action.option_strings]
            # a metavar can hold one name for each value that the argument takes, thus join them
            names = [
                self._format_args(
                    action,
                    " ".join(action.metavar) if isinstance(action.metavar, tuple) else action.metavar or action.dest,
                )
                for action in positionals
            ]
            # argparse puts the name of the command in place of %(prog)s
            usage = " ".join(["%(prog)s", "[options]", *names])

        return super()._format_usage(usage, actions, groups, prefix)

    @t.override
    def start_section(self, heading: str | None) -> None:
        """Start a section of the help. The heading starts with a capital letter, e.g. "Options".

        argparse gives its own headings in lower case, e.g. "options" and "positional arguments". argparse compares
        a heading with SUPPRESS by identity, thus this method does not change the SUPPRESS heading.
        """
        if heading and heading is not argparse.SUPPRESS:
            heading = heading[:1].upper() + heading[1:]
        super().start_section(heading)

    @t.override
    def _fill_text(self, text: str, width: int, indent: str) -> str:
        """Wrap a description of one line, and keep the lines of an epilog.

        The epilog holds the examples, which a user copies, thus its own line breaks must stay. A
        description is prose of one line, thus it wraps to the width of the terminal.
        """
        if "\n" in text.strip():
            return "".join(f"{indent}{line}\n" for line in text.splitlines())

        return super()._fill_text(text, width, indent)

    @t.override
    def add_arguments(self, actions: Iterable[argparse.Action]) -> None:
        getinvocation = super()._format_action_invocation

        def my_sort(action: argparse.Action) -> str:
            return getinvocation(action).upper().replace("-", "z")  # push dash chars below alphabet

        # the help gives the standard spelling of each flag, and the older spellings still work
        actions = sorted((with_shown_flags(action) for action in actions), key=my_sort)
        if self.short:
            actions = [with_first_sentence(action) for action in actions]
        super().add_arguments(actions)


def is_joined_value(value: str, action: argparse.Action) -> bool:
    """Report whether the text that joins a flag is a value of that flag, e.g. 300, .5, ./plot.pdf, -5, or a choice.

    The name of a flag of another command starts with a letter, "_", or "-", thus such text is no value. A "~"
    also stops, because the shell does not expand it inside a word, and the command would make a folder "~".
    """
    isnumberstart = value[:1].isdigit() or value[:1] == "."
    return (
        isnumberstart
        or value[:1] == "/"
        or (value[:1] == "-" and value[1:2] in set("0123456789."))
        or (value in (action.choices or ()))
    )


class SuggestingArgumentParser(argparse.ArgumentParser):
    """Name the closest subcommand, and the closest argument, when the given one does not match.

    Python 3.14 suggests a subcommand with suggest_on_error, which Python 3.13 does not take, and
    neither version suggests an argument. CI runs both, thus this gives the same message on each.
    """

    # Commands that accept positional arguments between options set this flag.
    wantsintermixed: bool = False

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Make the parser. The help of -h starts with a capital letter, as the help of every other argument does.

        Each parser formats its help and its usage in the same way, thus a parser of a test also gives "Usage:".
        """
        kwargs.setdefault("formatter_class", CustomArgHelpFormatter)
        addhelp = kwargs.pop("add_help", True)
        super().__init__(*args, add_help=False, **kwargs)
        if addhelp:
            self.add_argument(
                "-h",
                "--help",
                action=HelpAction,
                help="Show a summary of the options with -h, or the full help with --help, and exit",
            )

    @t.override
    def format_help(self) -> str:
        """Return the full help, with each sentence of the help of each option, and the examples."""
        return self.make_help(short=False)

    def make_formatter(self, *, short: bool) -> argparse.HelpFormatter:
        """Return the formatter of the help, which gives the short help of -h with short."""
        formatter = self._get_formatter()
        if short and isinstance(formatter, CustomArgHelpFormatter):
            formatter.make_short()
        return formatter

    def print_short_help(self, file: t.TextIO | None = None) -> None:
        """Print the short help of -h, with the first sentence of the help of each option.

        The help goes to the standard output, as print_help sends it. _print_message alone takes the standard error.
        """
        import sys

        self._print_message(self.make_help(short=True), file or sys.stdout)

    def make_help(self, *, short: bool) -> str:
        """Return the help with the argument groups in the order of ARGUMENTGROUPORDER.

        A command makes a shared group when it adds the first argument of that group. Without this order, the order
        of the groups is different in each command. The short help gives a pointer to --help in place of the examples.
        """

        def get_rank(group: "argparse._ArgumentGroup") -> float:  # pyright: ignore[reportPrivateUsage]
            # the group of the subcommands has no heading, and it comes first as the positional arguments do
            title = "positional arguments" if group.title is argparse.SUPPRESS else group.title or ""
            # 1.5 puts a group that only one command has between "options" and the shared groups
            return ARGUMENTGROUPORDER.index(title) if title in ARGUMENTGROUPORDER else 1.5

        formatter = self.make_formatter(short=short)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for group in sorted(self._action_groups, key=get_rank):
            formatter.start_section(group.title)
            formatter.add_text(group.description)
            formatter.add_arguments(group._group_actions)  # ruff:ignore[private-member-access]
            formatter.end_section()
        formatter.add_text(f"Run `{self.prog} --help` for the full help and the examples." if short else self.epilog)
        return formatter.format_help()

    @t.override
    def _check_value(self, action: argparse.Action, value: t.Any) -> None:
        """Refuse a value outside the choices with a message that names the closest choice.

        argparse composes this message here, thus the override sees the structured choices and needs
        no parse of the rendered text, which differs between Python 3.13 and 3.14.
        """
        if action.choices is None or value in action.choices:
            return

        choices = [str(choice) for choice in action.choices]
        listed = choices
        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            # the help lists no hidden command, e.g. server, thus no message names one. The list names each
            # command once, and a suggestion can also name an alias
            listed = [subaction.dest for subaction in action._get_subactions()]  # ruff:ignore[private-member-access]
            parsermap = action._name_parser_map  # ruff:ignore[private-member-access]
            listedparsers = {id(parsermap[command]) for command in listed}
            choices = [command for command, subparser in parsermap.items() if id(subparser) in listedparsers]
        # a close match answers the question. The list of the 44 commands filled the help line, and --help gives
        # each command with its description, thus the help line names --help
        suggestion = suggest_names(str(value), choices)
        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            helptext = suggestion or f"Run `{self.prog} --help` to see the commands"
            self.exit_with_help(f"'{value}' is not a command of {self.prog}", helptext)
        helptext = suggestion or f"The choices are {', '.join(listed)}"
        self.exit_with_help(f"Invalid choice '{value}' for {action.metavar or action.dest}", helptext)

    def get_visible_flags(self) -> list[str]:
        """Return the flags that the help shows. A suggestion then names no hidden alias or older spelling."""
        visibleactions = [action for action in self._actions if action.help != argparse.SUPPRESS]
        return [flag for action in visibleactions for flag in get_shown_flags(action)]

    def add_standard_flags(self) -> None:
        """Give each flag the standard spelling of get_standard_flag, if no other flag has that spelling.

        The older spelling stays, thus a script that holds -timedays or --print_data still runs. The help shows the
        standard spelling alone.
        """
        for action in self._actions:
            for flag in list(action.option_strings):
                standard = get_standard_flag(flag)
                if standard not in self._option_string_actions:
                    action.option_strings = [*action.option_strings, standard]
                    self._option_string_actions[standard] = action
            # the first flag names the option in each command that the code makes, e.g. the command of a viewer
            shown = get_shown_flags(action)
            action.option_strings = [*shown, *(flag for flag in action.option_strings if flag not in shown)]

    def show_standard_flags(self, message: str) -> str:
        """Return a message of argparse with the shown spelling of each flag.

        For example, the message gives --timedays/--time/-t for -timedays/-time/-t.
        """
        import re

        def replace(match: re.Match[str]) -> str:
            flags = match.group(0).split("/")
            action = self._option_string_actions.get(flags[0])
            if action is None:
                return match.group(0)
            if set(flags) <= set(action.option_strings) and len(flags) > 1:
                return "/".join(get_shown_flags(action))
            standard = get_standard_flag(flags[0])
            return standard if standard in action.option_strings else flags[0]

        return re.sub(r"(?<![\w/-])-{1,2}[A-Za-z][\w-]*(?:/-{1,2}[A-Za-z][\w-]*)*", replace, message)

    def exit_with_help(self, message: str, helptext: str = "") -> t.NoReturn:
        """Report a bad argument as an error line and a help line, then stop.

        argparse writes one line that holds both, thus this prints the usage itself and takes the
        place of the error method. The exit status of 2 is the one that argparse gives. With no help
        text, the help line names the --help of the command.
        """
        import sys

        self.print_usage(sys.stderr)
        print_error(message, helptext or f"Run `{self.prog} --help` to see every argument")
        raise SystemExit(2)

    def split_joined_flags(self, args: "Sequence[str]") -> list[str]:
        """Give the flag and the value of each token that joins them, e.g. -ts70 becomes -ts 70.

        argparse reads the first two characters of a single-dash token as the flag, thus "-ts70"
        gives -t the value "s70", and -ts keeps no value. The user names the longest flag that the
        token starts with, thus this splits the token there. A token that names the start of a
        longer flag stays whole, because that token is an abbreviation that argparse resolves.
        """
        # the top level declares only -h and -V, thus it must not test the flags of a subcommand, e.g. -hesmafile
        if self._subparsers is not None:
            return list(args)

        out: list[str] = []
        for index, argstring in enumerate(args):
            if argstring == "--":  # every argument after this one is a positional argument
                out.extend(args[index:])
                break

            out.extend(self.split_one_joined_flag(argstring))

        return self.keep_integer_option_paths(out)

    def keep_integer_option_paths(self, args: "Sequence[str]") -> list[str]:
        """Keep positional paths after options that take optional integers.

        argparse takes the path as the integer and then rejects it. An explicit constant keeps the path positional.
        For a list of integers, move the paths into the first positional group.
        """
        out: list[str] = []
        paths: list[str] = []
        index = 0
        while index < len(args):
            argstring = args[index]
            if argstring == "--":
                out.extend(args[index:])
                break
            flagname, equals, _ = argstring.partition("=")
            action = self._option_string_actions.get(flagname)
            if action is None and flagname.startswith("-") and self.allow_abbrev:
                matches = [value for flag, value in self._option_string_actions.items() if flag.startswith(flagname)]
                if len(matches) == 1:
                    action = matches[0]
            if action is not None and action.nargs == "*" and action.type is int:
                valueend = index + 1
                while not equals and valueend < len(args):
                    try:
                        int(args[valueend])
                    except ValueError:
                        break
                    valueend += 1
                hasseparator = valueend < len(args) and args[valueend] == "--"
                pathstart = valueend + int(hasseparator)
                pathend = pathstart
                while pathend < len(args) and not args[pathend].startswith("-"):
                    pathend += 1
                paths.extend(args[pathstart:pathend])
                out.extend(args[index:valueend])
                if hasseparator:
                    if pathend < len(args):
                        out.append("--")
                    out.extend(args[pathend:])
                    break
                index = pathend
                continue
            outstring = argstring
            if (
                action is not None
                and action.nargs == "?"
                and action.type is int
                and not equals
                and index + 1 < len(args)
            ):
                nexttoken = args[index + 1]
                if not nexttoken.startswith("-"):
                    try:
                        int(nexttoken)
                    except ValueError:
                        outstring = f"{argstring}={action.const}"
            out.append(outstring)
            index += 1
        firstoption = next((index for index, token in enumerate(out) if token.startswith("-")), len(out))
        return [*out[:firstoption], *paths, *out[firstoption:]]

    def split_one_joined_flag(self, argstring: str) -> list[str]:
        """Give the flag and the value of one argument that joins them, or stop at a flag of no command.

        argparse reads the text after a single-dash flag of one letter as its value. Thus "-obsspec 100" on a
        command that takes -o but no -obsspec wrote the plot to a file named bsspec. A joined value must look
        like a value, and a group of switches such as -qv stays whole.
        """
        if not argstring.startswith("-") or argstring.startswith("--") or len(argstring) <= 2:
            return [argstring]

        name, equals, _ = argstring.partition("=")
        declared = self._option_string_actions
        # a declared flag starts with itself, thus this one test also keeps an abbreviation of a flag,
        # which argparse resolves or reports as ambiguous
        if any(flag.startswith(name) for flag in declared):
            return [argstring]

        # the user names the longest flag that the token starts with, e.g. -ts70 is -ts 70 and not -t s70
        for length in range(len(argstring) - 1, 1, -1):
            flag = argstring[:length]
            action = declared.get(flag)
            if action is None or (action.nargs == 0 and length > 2):
                continue

            if action.nargs == 0 and self.is_switch_group(argstring):
                return [argstring]

            value = argstring[length:]
            if action.nargs != 0 and is_joined_value(value, action):
                # argparse splits a flag of one letter from its value itself. A value that starts with "-", e.g.
                # the -z of -axis-z, stays joined by "=", because argparse reads a separate -z as a flag. The "="
                # gives the flag one value alone, thus a list such as -plotviewingangle-1 0 stays separate
                if length == 2 or equals:
                    return [argstring]
                joinsvalue = value.startswith("-") and action.nargs in {None, "?"}
                return [f"{flag}={value}"] if joinsvalue else [flag, value]

            helptext = f"Did you mean {flag}?" if length > 2 else suggest_flags(name, self.get_visible_flags())
            if not helptext and action.nargs != 0:
                helptext = f"Put a space between {flag} and its value"
            self.exit_with_help(f"{name} is not an argument of this command", helptext)

        return [argstring]

    def is_switch_group(self, argstring: str) -> bool:
        """Report whether argparse reads the argument as a group of flags of one letter, e.g. -qv or -qt300.

        argparse lets the last flag of the group take a value, which is the rest of the argument or the next one.
        """
        for index, letter in enumerate(argstring[1:], start=2):
            action = self._option_string_actions.get(f"-{letter}")
            if action is None:
                return False

            if action.nargs != 0:
                rest = argstring[index:]
                return not rest or is_joined_value(rest, action)

        return True

    @t.override
    def parse_known_args(  # ty:ignore[invalid-method-override]  # pyrefly: ignore[bad-override]
        self, args: "Sequence[str] | None" = None, namespace: argparse.Namespace | None = None
    ) -> tuple[argparse.Namespace | None, list[str]]:
        """Split a joined flag and value, then parse. A subparser reads its own arguments here.

        argparse fills a positional argument from one unbroken group of arguments. A flag between two
        positional arguments hides the second group, thus "plotestimators Te -t 300 mymodel" failed.
        parse_known_intermixed_args reads both groups. Commands that accept this order set wantsintermixed.
        """
        import sys

        argstrings = self.split_joined_flags(sys.argv[1:] if args is None else args)
        if self.wantsintermixed:
            return self.parse_known_intermixed_args(argstrings, namespace)

        return super().parse_known_args(argstrings, namespace)

    @t.override
    def parse_args(  # ty:ignore[invalid-method-override]  # pyrefly: ignore[bad-override]
        self, args: "Sequence[str] | None" = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        """Parse the arguments, and name the closest flag when one is not recognised.

        The dispatcher reports a leftover flag of a subcommand, thus the suggestion must come from the
        arguments of that subcommand, which the namespace names as argparser.
        """
        parsednamespace, leftover = self.parse_known_args(args, namespace)
        assert parsednamespace is not None
        if leftover:
            # separate_trailing_folders puts "--" in front of the folders at the end, thus the user did not write it
            words = [word for word in leftover if word != "--"]
            flag = next((word.partition("=")[0] for word in words if word.startswith("-")), None)
            subparser = getattr(parsednamespace, "argparser", None) or self
            helptext = ""
            if isinstance(subparser, SuggestingArgumentParser):
                helptext = (
                    suggest_flags(flag, subparser.get_visible_flags())
                    if flag is not None
                    else subparser.get_folder_help(words)
                )
            # the usage of the command that the user ran, thus it holds the arguments of that command
            subparser.exit_with_help(f"Unrecognised arguments: {' '.join(words)}", helptext)

        return parsednamespace

    def get_folder_help(self, words: "Sequence[str]") -> str:
        """Return the help line for words that this command refuses and that name a folder, or an empty string.

        A command reads its positional paths as one group.
        """
        folder = next((word for word in words if is_remote_path(word) or Path(word).is_dir()), None)
        if folder is None:
            return ""
        if any(isinstance(action, KeepGivenPaths) for action in self._actions):
            return "Write the paths in one group, in front of the options or after them"
        return ""

    @t.override
    def error(self, message: str) -> t.NoReturn:
        """Report an error of argparse in the same two-part shape, with a suggestion where one fits.

        argparse starts its messages with a lower-case letter, e.g. "argument -t: ...", thus the first letter changes.
        """
        import re

        # argparse reads -timeday as -t with a joined value, thus its ambiguity list names -t as a
        # match. A suggestion from the real flags says what the user meant
        ambiguous = re.match(r"ambiguous option: (\S+) could match", message)
        helptext = ""
        if ambiguous is not None:
            # argparse names the whole token, thus "-ti=300" carries its value. The flag alone
            # matches a name and gives a suggestion
            given = ambiguous.group(1).partition("=")[0]
            # get_visible_flags leaves out a hidden alias, thus the suggestion names a flag that the help shows
            helptext = suggest_flags(given, self.get_visible_flags())

        message = self.show_standard_flags(message)
        self.exit_with_help(message[:1].upper() + message[1:], helptext)


class KeepGivenPaths(argparse.Action):
    """Store the paths of a positional argument, but keep the paths that the option form already gave.

    argparse applies a positional after an option that shares its dest, thus a positional that the user
    left out would otherwise hide the value of that option. argparse gives the default object itself as the
    value in that case, thus only that object counts as no value. A path that the user writes is a new object,
    thus it counts also when it is equal to the default.
    """

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
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

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
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
        f"The command ignores the path {as_text(ignored)}, because the option form gives the path {as_text(kept)}."
        " Give the paths in one form only"
    )


def trailing_folder_count(values: list[t.Any], *, anypath: bool = False) -> int:
    """Return how many values at the end of a list name an ARTIS model folder, or with anypath, a path that exists.

    The text alone decides for a value of the form host:path. A test of a remote folder asks its host through ssh,
    and a label of a list option can have that form, e.g. "W7:Kasen". An empty value is a label, and not the
    working folder.
    """

    def names_a_folder(value: str) -> bool:
        if is_remote_path(value):
            return names_a_remote_folder(value)

        return Path(value).exists() if anypath else folder_is_artis_model(value)

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
    "plotspectra --label mylabel mymodel" left the model path empty and made "mymodel" a second
    label. The command then plotted the working folder, and it gave no message. The folders come
    back to the positional argument here, and the option keeps its other values.

    Only the last values of an option can be folders, because the user writes the paths last. Two
    options that each end with the name of a folder are ambiguous. The command then reads them as
    the user wrote them.

    A different file or folder stays with the option, because an option such as --reflightcurves reads files. Such a
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
                flag = get_shown_flags(action)[0]
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

    flag = get_shown_flags(action)[0]
    folderword = "folder" if len(folders) == 1 else "folders"
    print_warning(
        f"{flag} read the ARTIS {folderword} {', '.join(folders)} as a value. The model path gets "
        f"the {folderword} back. Write the model path in front of {flag}"
    )
