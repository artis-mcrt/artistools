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


type CommandTree = dict[str, CommandSpec | CommandTree]

# The --help listing shows the top-level commands under these headings, in this order. A command that no
# tuple names appears under the last heading, thus a new command is still listed.
COMMANDGROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "plot commands": (
        "comparetogsinetwork",
        "leptontransport",
        "plotdensity",
        "plotestimators",
        "plotinitialcomposition",
        "plotlastpacketinteraction",
        "plotlightcurves",
        "plotlinefluxes",
        "plotlogfiles",
        "plotmacroatom",
        "plotnltepops",
        "plotradfield",
        "plotspectra",
        "plotspherical",
        "plottransitions",
        "plotviewingangles",
    ),
    "model commands": ("inputmodel", "makeartismodelfromparticlegridmap", "makevpktinput", "maptogrid"),
    "data commands": (
        "ejectaopacity",
        "exportmassfractions",
        "gsinetworkdecayproducts",
        "hesma",
        "spencerfano",
        "writebollightcurvedata",
        "writecodecomparisondata",
        "writespectra",
    ),
    "other commands": ("completions", "describeinputmodel", "getpath", "version"),
})

subcommandtree: CommandTree = {
    "comparetogsinetwork": CommandSpec(
        "gsinetwork.plotqdotabund",
        helptext="Compare the energy release and abundances from ARTIS to a GSI Network calculation.",
    ),
    "completions": CommandSpec(
        "commands",
        funcname="setup_completions",
        helptext="Generate a shell tab-completion script for artistools commands.",
    ),
    "describeinputmodel": CommandSpec(
        "inputmodel.describeinputmodel",
        helptext="Describe an ARTIS input model, such as the mass, velocity structure, and abundances.",
        hidden=True,  # duplicate of "inputmodel describe"
    ),
    "ejectaopacity": CommandSpec(
        "ejectaopacity", helptext="Compute binned expansion opacities and Planck-mean opacities in postprocessing."
    ),
    "exportmassfractions": CommandSpec(
        "estimators.exportmassfractions", helptext="Export elemental mass fractions from the estimators to a text file."
    ),
    "getpath": CommandSpec(
        "commands", funcname="get_artistools_path", helptext="Print the installed artistools package directory."
    ),
    "hesma": CommandSpec(
        "hesma_scripts", helptext="Convert ARTIS output to the file formats used by the HESMA model archive."
    ),
    "gsinetworkdecayproducts": CommandSpec(
        "gsinetwork.decayproducts", helptext="Load beta-decay energy release data from nucleosynthesis trajectories."
    ),
    "inputmodel": {
        "describe": CommandSpec(
            "inputmodel.describeinputmodel",
            helptext="Describe an ARTIS input model, such as the mass, velocity structure, and abundances.",
        ),
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
        "makeartismodel": CommandSpec(
            "inputmodel.makeartismodel", script="makeartismodel", helptext="Tools to create an ARTIS input model."
        ),
        "makeartismodel1dslicefromcone": CommandSpec(
            "inputmodel.slice1dfromconein3dmodel",
            script="makeartismodel1dslicefromcone",
            helptext="Make a 1D model from a cone in a 3D model.",
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
    "leptontransport": CommandSpec(
        "nonthermal.leptontransport",
        helptext="Plot the energy loss of a fast lepton to plasma, ionisation, and excitation with distance.",
    ),
    "makeartismodelfromparticlegridmap": CommandSpec(
        "inputmodel.modelfromhydro",
        helptext="Create an ARTIS format model from grid.dat.",
        hidden=True,  # duplicate of "inputmodel makeartismodelfromparticlegridmap"
    ),
    "makevpktinput": CommandSpec(
        "make_vpkt_input", helptext="Create a vpkt.txt virtual packet configuration file for an ARTIS simulation."
    ),
    "maptogrid": CommandSpec(
        "inputmodel.maptogrid",
        helptext="Map tracer particle trajectories to a Cartesian grid.",
        hidden=True,  # duplicate of "inputmodel maptogrid"
    ),
    "plotdensity": CommandSpec(
        "inputmodel.plotdensity",
        script="plotartisdensity",
        helptext="Plot the radial density profile of an ARTIS model.",
    ),
    "plotestimators": CommandSpec(
        "estimators.plotestimators",
        script="plotartisestimators",
        helptext="Plot ARTIS estimators.",
        aliases=("estimators",),
    ),
    "plotinitialcomposition": CommandSpec(
        "inputmodel.plotinitialcomposition",
        script="plotartisinitialcomposition",
        helptext="Plot ARTIS input model composition.",
    ),
    "plotlastpacketinteraction": CommandSpec(
        "packets.packetsplots",
        helptext="Plot last packet interaction properties versus ejecta velocity for selected packets.",
    ),
    "plotlightcurves": CommandSpec(
        "lightcurve.plotlightcurve",
        script="plotartislightcurve",
        helptext="Plot ARTIS light curves.",
        aliases=("lc", "plotlightcurve"),
    ),
    "plotlinefluxes": CommandSpec(
        "linefluxes", script="plotartislinefluxes", helptext="Plot line flux ratios for comparisons to Floers."
    ),
    "plotlogfiles": CommandSpec("logfiles", helptext="Plot per-rank stage durations from ARTIS log files."),
    "plotmacroatom": CommandSpec("macroatom", script="plotartismacroatom", helptext="Plot the macroatom transitions."),
    "plotnltepops": CommandSpec(
        "nltepops.plotnltepops", script="plotartisnltepops", helptext="Plot ARTIS non-LTE populations."
    ),
    "plotradfield": CommandSpec(
        "radfield", script="plotartisradfield", helptext="Plot the radiation field estimators."
    ),
    "plotspectra": CommandSpec(
        "spectra.plotspectra",
        script="plotartisspectrum",
        helptext="Plot spectra from ARTIS and reference data.",
        aliases=("spec",),
    ),
    "plotspherical": CommandSpec("plotspherical", helptext="Plot direction maps based on escaped packets."),
    "plottransitions": CommandSpec(
        "transitions", script="plotartistransitions", helptext="Plot estimated spectra from bound-bound transitions."
    ),
    "plotviewingangles": CommandSpec(
        "viewing_angles_visualization",
        script="plotartisviewingangles",
        helptext="Generate a 3D visualization of an ARTIS model.",
    ),
    "spencerfano": CommandSpec(
        "nonthermal.solvespencerfanocmd",
        script="plotartisnonthermal",
        helptext="Solve the Spencer-Fano equation using data from an ARTIS cell at some timestep.",
    ),
    "version": CommandSpec("commands", funcname="show_version", helptext="Print the artistools version."),
    "writebollightcurvedata": CommandSpec(
        "lightcurve.writebollightcurvedata",
        helptext="Write the bolometric light curve of each model out as a plain text file.",
    ),
    "writecodecomparisondata": CommandSpec(
        "writecomparisondata", helptext="Write ARTIS model data out in code comparison workshop format."
    ),
    "writespectra": CommandSpec(
        "spectra.writespectra", helptext="Write ARTIS spectra for each timestep to individual text files."
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


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse formatter to show default values in help text, sorted with dashes last."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Widen the help column so long option names stay on one line."""
        kwargs["max_help_position"] = 50
        super().__init__(*args, **kwargs)

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
            yield from members
        self._dedent()


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no command-line arguments (for subcommands that take none)."""


@lru_cache(maxsize=1)
def get_script_subcommands() -> Mapping[str, tuple[str, ...]]:
    """Return the subcommand path of each per-command console script, e.g. plotartisestimators.

    The CommandSpec of a subcommand names its script, thus the tree holds both and no second table
    can fall out of step with it.
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
    submodule = importlib.import_module(f"artistools.{spec.module}")
    submodule.addargs(parser)
    # __main__ tests the arguments against the defaults of this parser, thus it needs the parser itself.
    # parse_cli_args cannot make that test, because it returns at once for a parsed namespace, which is
    # what the dispatcher gives it
    parser.set_defaults(func=getattr(submodule, spec.funcname), argparser=parser)


def build_script_parser(scriptname: str) -> argparse.ArgumentParser | None:
    """Return a parser for the subcommand of a per-command console script, or None for another name.

    A script such as plotartisestimators runs one subcommand. Thus its parser holds that command alone,
    and it imports one module in place of every module of the tree. The usage text also gives the name
    of the script and not the name of the subcommand.
    """
    words = get_subcommand_of_script(scriptname)
    if not words:
        return None

    node: CommandSpec | CommandTree = subcommandtree
    for word in words:
        assert isinstance(node, dict)
        node = node[word]

    assert isinstance(node, CommandSpec)
    parser = argparse.ArgumentParser(prog=scriptname, description=node.helptext, formatter_class=CustomArgHelpFormatter)
    addcommandargs(parser, node)

    return parser


def addsubparsers(parser: argparse.ArgumentParser, parentcommand: str, subcommandtree: CommandTree) -> None:
    """Register the subcommands in the tree on the parser."""

    def func(args: argparse.Namespace) -> None:  # ruff:ignore[unused-function-argument]
        parser.print_help()

    parser.set_defaults(func=func)
    subparsers = parser.add_subparsers(dest="subcommand", required=False, metavar="command")

    for subcommand, spec in subcommandtree.items():
        if isinstance(spec, dict):
            subparser = subparsers.add_parser(
                subcommand,
                help="command group",
                description=f"{parentcommand} {subcommand} command group.",
                formatter_class=CustomArgHelpFormatter,
            )
            addsubparsers(parser=subparser, parentcommand=subcommand, subcommandtree=spec)
        else:
            # omitting help= entirely keeps a hidden entry out of the parent help listing. Do not use
            # help=argparse.SUPPRESS here: argparse only honours it for arguments, not subparsers, and
            # it would show the command with a literal ==SUPPRESS== description
            addparserkwargs: dict[str, t.Any] = {} if spec.hidden else {"help": spec.helptext}
            subparser = subparsers.add_parser(
                subcommand,
                description=spec.helptext,
                aliases=spec.aliases,
                formatter_class=CustomArgHelpFormatter,
                **addparserkwargs,
            )
            addcommandargs(subparser, spec)


def setup_completions(*args: t.Any, **kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Generate a shell tab-completion script and print instructions for enabling it."""
    import subprocess

    path_package_source = Path(__file__).absolute().parent
    completionscriptpath = path_package_source / "artistoolscompletions.sh"
    with (completionscriptpath).open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env zsh\n")
        f.write("# automatically generated by artistools completions\n")

        proc = subprocess.run(
            ["register-python-argcomplete", "__MY_COMMAND__"], capture_output=True, text=True, check=True
        )

        if proc.stderr:
            print(proc.stderr)

        strfunctiondefs, strsplit, strcommandregister = proc.stdout.rpartition("}\n")

        f.write(strfunctiondefs)
        f.write(strsplit)
        f.write("\n")

        for command in (*DISPATCHERSCRIPTS, *sorted(get_script_subcommands())):
            completecommand = strcommandregister.replace("__MY_COMMAND__", command)
            f.write(f"\n{completecommand}")

    print("To enable completions, add these lines to your .zshrc or .bashrc file:")
    print("\n.zshrc:")
    print(f'source "{completionscriptpath}"')
    print("autoload -Uz compinit && compinit")

    print("\n.bashrc:")
    print(f"source {completionscriptpath}")


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
