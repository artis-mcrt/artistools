"""Define the artistools subcommand tree and dispatch to each subcommand's module."""

import argparse
import dataclasses as dc
import importlib
import typing as t
from collections.abc import Iterable
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

if t.TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence


def get_examples() -> tuple[tuple[str, str], ...]:
    """Return every example of the tree as (command line, description), for the help and for the test."""

    def walk(tree: CommandTree, path: str) -> "Generator[tuple[str, str]]":
        for name, spec in tree.items():
            if isinstance(spec, dict):
                yield from walk(spec, f"{path}{name} ")
            else:
                yield from ((f"{path}{name} {arguments}", description) for arguments, description in spec.examples)

    return tuple(walk(subcommandtree, ""))


def get_command_epilog(subcommand: str, spec: "CommandSpec") -> str | None:
    """Return the examples of one command for its own help, or None when it has none."""
    if not spec.examples:
        return None

    lines = [f"  artistools {subcommand} {arguments}  # {description}" for arguments, description in spec.examples]

    return "examples (a path of . reads the model in the working folder):\n" + "\n".join(lines)


def get_epilog() -> str:
    """Return the examples and the pointer to the help of one command."""
    examples = get_examples()
    width = max(len(command) for command, _ in examples)
    lines = [f"  artistools {command:{width}}  # {description}" for command, description in examples]

    return (
        "examples (a path of . reads the model in the working folder, and the path is the last argument):\n"
        + "\n".join(lines)
        + '\n\nRun "artistools <command> --help" for the arguments of one command.'
        + "\nSet ARTISTOOLS_TRACEBACK=1 to get the full traceback of an error."
    )


# the console scripts that take a subcommand as their first argument. Every other console script names
# its subcommand in the CommandSpec of that subcommand, thus get_script_subcommands finds it
DISPATCHERSCRIPTS = ("at", "artistools")


@dc.dataclass(frozen=True, slots=True)
class CommandSpec:
    """A subcommand definition: the implementing module and the static help text shown in command listings.

    A hidden command still works but is left out of the --help listing (used for deprecated duplicate names).
    """

    module: str
    funcname: str = "main"
    helptext: str = ""
    aliases: tuple[str, ...] = ()
    hidden: bool = False

    script: str = ""
    """The per-command console script that runs this subcommand, e.g. plotartisestimators."""

    note: str = ""
    """Extra text for the help of the command alone. The listing of one line has no room for it."""

    examples: tuple[tuple[str, str], ...] = ()
    """Example invocations as (arguments, description) pairs. A test runs each one against the test
    model, thus an example cannot name an argument that a later commit takes away. A path of "."
    reads the model in the working folder."""


type CommandTree = dict[str, CommandSpec | CommandTree]

# The --help listing shows the top-level commands under these headings, in this order. A command that no
# tuple names appears under the last heading, thus a new command is still listed.
COMMANDGROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "plot commands": (
        "comparetogsinetwork",
        "plotdensity",
        "plotestimators",
        "plotinitialcomposition",
        "plotlastpacketinteraction",
        "plotlightcurves",
        "plotlinefluxes",
        "plotlogfiles",
        "plotnltepops",
        "plotradfield",
        "plotspectra",
        "plotspherical",
        "plottransitions",
        "plotviewingangles",
    ),
    "model commands": ("inputmodel", "makevpktinput"),
    "data commands": (
        "deposition",
        "ejectaopacity",
        "exportmassfractions",
        "gsinetworkdecayproducts",
        "hesma",
        "spencerfano",
        "writebollightcurvedata",
        "writecodecomparisondata",
        "writespectra",
    ),
    "other commands": ("completions", "getpath", "timesteps", "version"),
})

# "artistools describeinputmodel" was a top-level command, thus a script of a user holds that name.
# One spec gives every name of this command, thus no copy of its help text can drift
DESCRIBEINPUTMODEL = CommandSpec(
    "inputmodel.describeinputmodel",
    helptext="Describe an ARTIS input model, such as the mass, velocity structure, and abundances.",
    aliases=("describeinputmodel",),
)

subcommandtree: CommandTree = {
    "comparetogsinetwork": CommandSpec(
        "gsinetwork.comparetogsinetwork",
        helptext="Compare ARTIS to a GSI Network calculation.",
        note="The comparison covers the energy release and the abundances.",
    ),
    "completions": CommandSpec(
        "completions",
        helptext="Give the instructions for tab completion.",
        note="With a shell name, e.g. zsh, the command prints the code that the instructions write to a file.",
    ),
    "deposition": CommandSpec(
        "estimators.deposition",
        helptext="Give the deposition rate per unit volume, per ion, and per unit mass.",
        note=(
            "The deposition_ estimators of each cell give the rate, thus the command needs a run of a"
            " recent ARTIS version. Every column covers the cells that hold matter."
        ),
    ),
    # the help lists this name under "inputmodel describe", thus the top level hides it
    "describeinputmodel": dc.replace(DESCRIBEINPUTMODEL, aliases=(), hidden=True),
    "ejectaopacity": CommandSpec(
        "ejectaopacity",
        helptext="Compute the opacities of the ejecta.",
        note="This gives the binned expansion opacities and the Planck-mean opacities.",
    ),
    "exportmassfractions": CommandSpec(
        "estimators.exportmassfractions",
        helptext="Write the mass fractions of the elements.",
        note="The values come from the estimators, and the command writes a text file.",
    ),
    "getpath": CommandSpec("commands", funcname="get_artistools_path", helptext="Print the folder of the package."),
    "hesma": CommandSpec(
        "hesma_scripts",
        helptext="Convert ARTIS output to the HESMA formats.",
        note="The HESMA model archive takes these file formats.",
    ),
    "gsinetworkdecayproducts": CommandSpec(
        "gsinetwork.decayproducts",
        helptext="Read the beta-decay energy of a trajectory.",
        note="The data comes from the trajectories of a nucleosynthesis calculation.",
    ),
    "inputmodel": {
        "describe": DESCRIBEINPUTMODEL,
        "energyfiles": CommandSpec(
            "inputmodel.energyinputfiles", helptext="Plot and inspect the ARTIS energy input files."
        ),
        "from_e2e": CommandSpec(
            "inputmodel.from_e2e_model",
            helptext="Prepare data for an ARTIS kilonova calculation from end-to-end hydro models.",
        ),
        "fromcmfgen": CommandSpec(
            "inputmodel.fromcmfgen.convert_to_artis",
            helptext="Convert a CMFGEN SN_HYDRO_DATA snapshot to an ARTIS model at the snapshot's own time.",
        ),
        "make1dslicefrom3dmodel": CommandSpec(
            "inputmodel.make1dslicefrom3d",
            helptext="Convert abundances.txt and model.txt from a 3D model to a one-dimensional slice.",
        ),
        "makeartismodel": CommandSpec("inputmodel.makeartismodel", helptext="Tools to create an ARTIS input model."),
        "makeartismodel1dslicefromcone": CommandSpec(
            "inputmodel.slice1dfromconein3dmodel", helptext="Make a 1D model from a cone in a 3D model."
        ),
        "makeartismodelfromparticlegridmap": CommandSpec(
            "inputmodel.modelfromhydro", helptext="Create an ARTIS format model from grid.dat."
        ),
        "makeartismodelfromshen2018": CommandSpec(
            "inputmodel.shen2018", helptext="Convert Shen et al. 2018 models to ARTIS format."
        ),
        "makeartismodelfromsingletrajectory": CommandSpec(
            "inputmodel.rprocess_from_trajectory", helptext="Create an ARTIS model from single trajectory abundances."
        ),
        "maptogrid": CommandSpec(
            "inputmodel.maptogrid", helptext="Map tracer particle trajectories to a Cartesian grid."
        ),
        "opacityfile": CommandSpec(
            "inputmodel.opacityinputfile", helptext="Write or inspect an ARTIS grey opacity.txt."
        ),
        "plotinitialabundances": CommandSpec(
            "inputmodel.plotinitialabundances",
            helptext="Plot initial abundances or mass fractions from one or more ARTIS models.",
        ),
        "to_tardis": CommandSpec("inputmodel.to_tardis", helptext="Convert an ARTIS format model to TARDIS format."),
    },
    "makevpktinput": CommandSpec(
        "make_vpkt_input",
        helptext="Write a vpkt.txt for a run.",
        note="The file holds the configuration of the virtual packets.",
    ),
    "plotdensity": CommandSpec("inputmodel.plotdensity", helptext="Plot the density against the radius."),
    "plotestimators": CommandSpec(
        "estimators.plotestimators",
        script="plotartisestimators",
        helptext="Plot ARTIS estimators.",
        examples=(
            ("Te TR . -t 300", "two estimator variables against velocity"),
            ("Te TR . -t 300 -dim 2", "each variable as a colour image at each cylindrical radius and z"),
            (". --listvariables", "every variable that a model holds"),
        ),
        aliases=("estimators",),
    ),
    "plotinitialcomposition": CommandSpec(
        "inputmodel.plotinitialcomposition", helptext="Plot ARTIS input model composition."
    ),
    "plotlastpacketinteraction": CommandSpec(
        "packets.plotlastpacketinteraction",
        helptext="Plot the last interaction of a packet.",
        note="The plot gives the properties of that interaction against the velocity of the ejecta.",
    ),
    "plotlightcurves": CommandSpec(
        "lightcurve.plotlightcurve",
        script="plotartislightcurve",
        helptext="Plot ARTIS light curves.",
        examples=((".", "the light curve of a model"),),
        aliases=("lc", "plotlightcurve"),
    ),
    "plotlinefluxes": CommandSpec(
        "plotlinefluxes",
        helptext="Plot the ratios of the line fluxes.",
        note="The ratios serve a comparison to Floers.",
    ),
    "plotlogfiles": CommandSpec(
        "plotlogfiles",
        helptext="Plot the time that each rank took.",
        note="The times come from the log files of a run.",
    ),
    "plotnltepops": CommandSpec(
        "nltepops.plotnltepops",
        helptext="Plot ARTIS non-LTE populations.",
        examples=(("-modelpath . -t 300 -modelgridindex 0", "the level populations of one cell"),),
        note=(
            "Give a time with -timedays or -timestep. A model of more than one cell also needs a cell,"
            " which -modelgridindex or -velocity gives."
        ),
    ),
    "plotradfield": CommandSpec("plotradfield", helptext="Plot the radiation field estimators."),
    "plotspectra": CommandSpec(
        "spectra.plotspectra",
        script="plotartisspectrum",
        helptext="Plot spectra from ARTIS and reference data.",
        examples=((". -t 300", "the spectrum at 300 days"),),
        aliases=("spec",),
    ),
    "plotspherical": CommandSpec("plotspherical", helptext="Plot direction maps based on escaped packets."),
    "plottransitions": CommandSpec(
        "plottransitions",
        helptext="Plot the spectrum of the transitions.",
        note="The spectrum comes from the bound-bound transitions.",
    ),
    "plotviewingangles": CommandSpec(
        "plotviewingangles",
        helptext="Plot a 3D view of a model.",
        note="The view holds an isosurface of the density and the direction bins.",
    ),
    "spencerfano": CommandSpec(
        "nonthermal.spencerfano",
        helptext="Solve the Spencer-Fano equation for a cell.",
        note="The data comes from one cell of an ARTIS run at one timestep.",
    ),
    "timesteps": CommandSpec(
        "timesteps",
        helptext="List the timesteps and their days.",
        note="The table gives the days that each timestep covers.",
    ),
    "version": CommandSpec("commands", funcname="show_version", helptext="Print the artistools version."),
    "writebollightcurvedata": CommandSpec(
        "lightcurve.writebollightcurvedata",
        helptext="Write the bolometric light curve of a model.",
        note="The command writes a plain text file.",
    ),
    "writecodecomparisondata": CommandSpec(
        "writecomparisondata",
        helptext="Write the model in the comparison format.",
        note="The code comparison workshop takes this format.",
    ),
    "writespectra": CommandSpec(
        "spectra.writespectra",
        helptext="Write the spectrum of each timestep.",
        note="The command writes one text file for each timestep.",
    ),
}


class CommandGroupHeading(argparse.Action):
    """A pseudo action that carries a heading into the help listing of the top-level commands."""


def group_subactions(subactions: "list[argparse.Action]") -> "dict[str, list[argparse.Action]] | None":
    """Return the top-level subcommands keyed by the heading of COMMANDGROUPS.

    A command that no tuple of COMMANDGROUPS names goes under the last heading, thus a new command still
    appears and the headings stay. Return None when no command at all has a heading, which is how a
    command group such as "artistools inputmodel" keeps one flat listing.
    """
    groupofcommand = {name: heading for heading, names in COMMANDGROUPS.items() for name in names}
    if all(sub.dest not in groupofcommand for sub in subactions):
        return None

    lastheading = list(COMMANDGROUPS)[-1]
    grouped: dict[str, list[argparse.Action]] = {heading: [] for heading in COMMANDGROUPS}
    for sub in subactions:
        grouped[groupofcommand.get(sub.dest, lastheading)].append(sub)

    return {heading: members for heading, members in grouped.items() if members}


# what a group of subcommands does, for the listing of the commands above it
GROUPHELPTEXT = {"inputmodel": "Make and inspect an ARTIS input model."}

# a usage line that names more flags than this takes more room than a reader gives it
MAXUSAGEFLAGS = 8


def shorten_aliases(metavar: str) -> str:
    """Return the name of a command with one alias alone, as the listing shows it.

    argparse names every alias, thus "plotlightcurves (lc, plotlightcurve)" took 36 columns of the
    listing and left 38 for the text, which then wrapped over three lines. Every alias still works, and
    the help of the command names them all.
    """
    name, _, aliases = metavar.partition(" (")
    if not aliases:
        return metavar

    return f"{name} ({aliases.rstrip(')').split(', ')[0]})"


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse formatter to show default values in help text, sorted with dashes last."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Widen the help column so long option names stay on one line."""
        kwargs["max_help_position"] = 50
        super().__init__(*args, **kwargs)

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
        actions = list(actions)
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

        actions = sorted(actions, key=my_sort)
        super().add_arguments(actions)

    @t.override
    def _format_action(self, action: argparse.Action) -> str:
        """Render a group heading as its own line, and every other action as usual."""
        if isinstance(action, CommandGroupHeading):
            return f"\n{action.dest}\n"

        return super()._format_action(action)

    @t.override
    def _iter_indented_subactions(self, action: argparse.Action) -> "Generator[argparse.Action]":
        """Yield the subcommands under a heading for each group.

        This hook also feeds the column width, thus the headings do not disturb the alignment.
        """
        if not isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            yield from super()._iter_indented_subactions(action)
            return

        grouped = group_subactions(list(action._get_subactions()))  # ruff:ignore[private-member-access]
        if grouped is None:
            yield from super()._iter_indented_subactions(action)
            return

        self._indent()
        for heading, members in grouped.items():
            yield CommandGroupHeading(option_strings=[], dest=f"{heading}:", help=None)
            for member in members:
                member.metavar = shorten_aliases(str(member.metavar or member.dest))
                yield member
        self._dedent()


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no command-line arguments (for subcommands that take none)."""


@lru_cache(maxsize=1)
def get_script_subcommands() -> Mapping[str, tuple[str, ...]]:
    """Return the subcommand path of each per-command console script, e.g. plotartisestimators.

    The CommandSpec of a subcommand names its script, thus the tree holds both names. No second table
    can disagree with the tree.
    """
    scripts: dict[str, tuple[str, ...]] = {}

    def walk(tree: CommandTree, path: tuple[str, ...]) -> None:
        for name, spec in tree.items():
            if isinstance(spec, dict):
                walk(spec, (*path, name))
            elif spec.script:
                if spec.script in scripts:
                    msg = f"Console script {spec.script} names more than one subcommand"
                    raise ValueError(msg)
                scripts[spec.script] = (*path, name)

    walk(subcommandtree, ())

    return MappingProxyType(scripts)


def get_subcommand_of_script(scriptname: str) -> tuple[str, ...]:
    """Return the subcommand that a per-command console script stands for, or an empty tuple."""
    return get_script_subcommands().get(scriptname, ())


def addcommandargs(parser: argparse.ArgumentParser, spec: CommandSpec) -> None:
    """Add the arguments of one subcommand to a parser, and record how to run it."""
    from artistools.misc import addarg_quiet

    submodule = importlib.import_module(f"artistools.{spec.module}")
    submodule.addargs(parser)
    # run_command alone implements --quiet, thus every command takes it and no module declares it
    addarg_quiet(parser)
    # __main__ tests the arguments against the defaults of this parser, thus it needs the parser itself.
    # parse_cli_args cannot make that test, because it returns at once for a parsed namespace, which is
    # what the dispatcher gives it
    parser.set_defaults(func=getattr(submodule, spec.funcname), argparser=parser)


def build_script_parser(scriptname: str) -> argparse.ArgumentParser | None:
    """Return a parser for the subcommand of a per-command console script, or None for another name.

    A script such as plotartisestimators runs one subcommand, thus its parser holds that command alone.
    The usage text then gives the name of the script and not the name of the subcommand, and argparse
    builds one parser in place of the whole tree.
    """
    words = get_subcommand_of_script(scriptname)
    if not words:
        return None

    node: CommandSpec | CommandTree = subcommandtree
    for word in words:
        assert isinstance(node, dict)
        node = node[word]

    assert isinstance(node, CommandSpec)
    parser = SuggestingArgumentParser(
        prog=scriptname, description=node.helptext, formatter_class=CustomArgHelpFormatter
    )
    addcommandargs(parser, node)

    return parser


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

    # addarg_positional_items sets this flag on the one parser that reads a positional argument after a
    # flag. A parser that does not set it keeps the argparse order, in which an option has priority
    # over a positional argument.
    wantsintermixed: bool = False

    @t.override
    def _check_value(self, action: argparse.Action, value: t.Any) -> None:
        """Refuse a value outside the choices with a message that names the closest choice.

        argparse composes this message here, thus the override sees the structured choices and needs
        no parse of the rendered text, which differs between Python 3.13 and 3.14.
        """
        if action.choices is None or value in action.choices:
            return

        from artistools.misc import suggest_names

        choices = [str(choice) for choice in action.choices]
        name = action.metavar or action.dest
        # a close match answers the question, thus the long list of every choice serves the other case
        helptext = suggest_names(str(value), choices) or f"The choices are {', '.join(choices)}"
        self.exit_with_help(f"invalid choice '{value}' for {name}", helptext)

    def get_visible_flags(self) -> list[str]:
        """Return the option strings that the help shows, thus a suggestion names no hidden alias."""
        return [flag for action in self._actions if action.help != argparse.SUPPRESS for flag in action.option_strings]

    def exit_with_help(self, message: str, helptext: str = "") -> t.NoReturn:
        """Report a bad argument as an error line and a help line, then stop.

        argparse writes one line that holds both, thus this prints the usage itself and takes the
        place of the error method. The exit status of 2 is the one that argparse gives. With no help
        text, the help line names the --help of the command.
        """
        import sys

        from artistools.misc import print_error

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

        return out

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
                # the -z of -axis-z, stays joined by "=", because argparse reads a separate -z as a flag
                if length == 2 or equals:
                    return [argstring]
                return [f"{flag}={value}"] if value.startswith("-") else [flag, value]

            from artistools.misc import suggest_flags

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
        parse_known_intermixed_args reads both groups. It also applies every positional argument after
        every option. This order is the opposite of the order that KeepGivenPaths needs, thus each
        parser must set wantsintermixed.
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
            from artistools.misc import suggest_names

            flag = next((word.partition("=")[0] for word in leftover if word.startswith("-")), None)
            subparser = getattr(parsednamespace, "argparser", None) or self
            helptext = ""
            if flag is not None and isinstance(subparser, SuggestingArgumentParser):
                helptext = suggest_names(flag, subparser.get_visible_flags())
            # the usage of the command that the user ran, thus it holds the arguments of that command
            subparser.exit_with_help(f"unrecognized arguments: {' '.join(leftover)}", helptext)

        return parsednamespace

    @t.override
    def error(self, message: str) -> t.NoReturn:
        """Report an error of argparse in the same two-part shape, with a suggestion where one fits."""
        import re

        from artistools.misc import suggest_flags

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

        self.exit_with_help(message, helptext)


def addsubparsers(parser: argparse.ArgumentParser, subcommandtree: CommandTree) -> None:
    """Register the subcommands in the tree on the parser."""

    def func(args: argparse.Namespace) -> None:  # ruff:ignore[unused-function-argument]
        parser.print_help()

    parser.set_defaults(func=func)
    subparsers = parser.add_subparsers(
        dest="subcommand", required=False, metavar="command", parser_class=SuggestingArgumentParser
    )

    for subcommand, spec in subcommandtree.items():
        if isinstance(spec, dict):
            subparser = subparsers.add_parser(
                subcommand,
                help=GROUPHELPTEXT.get(subcommand, "command group"),
                description=f"The {subcommand} commands of artistools.",
                epilog=f'Run "artistools {subcommand} <command> --help" for the arguments of one command.',
                formatter_class=CustomArgHelpFormatter,
            )
            addsubparsers(parser=subparser, subcommandtree=spec)
        else:
            # omitting help= entirely keeps a hidden entry out of the parent help listing. Do not use
            # help=argparse.SUPPRESS here: argparse only honours it for arguments, not subparsers, and
            # it would show the command with a literal ==SUPPRESS== description
            addparserkwargs: dict[str, t.Any] = {} if spec.hidden else {"help": spec.helptext}
            subparser = subparsers.add_parser(
                subcommand,
                description=f"{spec.helptext} {spec.note}".strip(),
                epilog=get_command_epilog(subcommand, spec),
                aliases=spec.aliases,
                formatter_class=CustomArgHelpFormatter,
                **addparserkwargs,
            )
            addcommandargs(subparser, spec)


def show_version(*args: t.Any, **kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Print the artistools version."""
    from importlib.metadata import version

    print(f"artistools {version('artistools')}")


def get_path(key: str) -> Path:
    """Return a well-known path by name, such as the package folder or the code comparison data folder."""
    match key:
        case "codecomparisondata1path":
            return Path(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/GitHub/sn-rad-trans/data1")
        case "codecomparisonmodelartismodelpath":
            return Path(Path.home() / "Google Drive/My Drive/artis_runs/weizmann/")
        case "artistools_repository":
            return Path(__file__).absolute().parent.parent
        case "artistools_dir":
            return Path(__file__).absolute().parent  # the package path
        case "datadir":
            return Path(__file__).absolute().parent / "data"
        case "testartismodel":
            return Path(get_path("artistools_repository"), "tests", "data", "testmodel")
        case "testdata":
            return Path(get_path("artistools_repository"), "tests", "data")
        case "testoutput":
            return Path(get_path("artistools_repository"), "tests", "output")
        case _:
            msg = f"Unknown path key: {key}"
            raise KeyError(msg)


def get_artistools_path(**kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Print the installed artistools package directory."""
    print(get_path("artistools_dir"))
