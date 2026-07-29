import argparse
import dataclasses as dc
import importlib
import typing as t
from collections.abc import Iterable
from pathlib import Path

# top-level commands (one file installed per command)
# we generally should phase this out except for a couple of main ones like at and artistools
COMMANDS = [
    "at",
    "artistools",
    "makeartismodel1dslicefromcone",
    "makeartismodel",
    "plotartisdensity",
    "plotartisestimators",
    "plotartislightcurve",
    "plotartislinefluxes",
    "plotartismacroatom",
    "plotartisnltepops",
    "plotartisnonthermal",
    "plotartisradfield",
    "plotartisspectrum",
    "plotartistransitions",
    "plotartisinitialcomposition",
    "plotartisviewingangles",
]


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


type CommandTree = dict[str, CommandSpec | CommandTree]

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
    "gsinetworkdecayproducts": CommandSpec(
        "gsinetwork.decayproducts", helptext="Load beta-decay energy release data from nucleosynthesis trajectories."
    ),
    "inputmodel": {
        "describe": CommandSpec(
            "inputmodel.describeinputmodel",
            helptext="Describe an ARTIS input model, such as the mass, velocity structure, and abundances.",
        ),
        "from_e2e": CommandSpec(
            "inputmodel.from_e2e_model",
            helptext="Prepare data for an ARTIS kilonova calculation from end-to-end hydro models.",
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
        "plotinitialabundances": CommandSpec(
            "inputmodel.plotinitialabundances",
            helptext="Plot initial abundances or mass fractions from one or more ARTIS models.",
        ),
        "to_tardis": CommandSpec("inputmodel.to_tardis", helptext="Convert an ARTIS format model to TARDIS format."),
    },
    "makeartismodelfromparticlegridmap": CommandSpec(
        "inputmodel.modelfromhydro",
        helptext="Create an ARTIS format model from grid.dat.",
        hidden=True,  # duplicate of "inputmodel makeartismodelfromparticlegridmap"
    ),
    "maptogrid": CommandSpec(
        "inputmodel.maptogrid",
        helptext="Map tracer particle trajectories to a Cartesian grid.",
        hidden=True,  # duplicate of "inputmodel maptogrid"
    ),
    "plotdensity": CommandSpec("inputmodel.plotdensity", helptext="Plot the radial density profile of an ARTIS model."),
    "plotestimators": CommandSpec(
        "estimators.plotestimators", helptext="Plot ARTIS estimators.", aliases=("estimators",)
    ),
    "plotinitialcomposition": CommandSpec(
        "inputmodel.plotinitialcomposition", helptext="Plot ARTIS input model composition."
    ),
    "plotlastpacketinteraction": CommandSpec(
        "packets.packetsplots",
        helptext="Plot last packet interaction properties versus ejecta velocity for selected packets.",
    ),
    "plotlightcurves": CommandSpec(
        "lightcurve.plotlightcurve", helptext="Plot ARTIS light curves.", aliases=("lc", "plotlightcurve")
    ),
    "plotlinefluxes": CommandSpec("linefluxes", helptext="Plot line flux ratios for comparisons to Floers."),
    "plotmacroatom": CommandSpec("macroatom", helptext="Plot the macroatom transitions."),
    "plotnltepops": CommandSpec("nltepops.plotnltepops", helptext="Plot ARTIS non-LTE populations."),
    "plotradfield": CommandSpec("radfield", helptext="Plot the radiation field estimators."),
    "plotspectra": CommandSpec(
        "spectra.plotspectra", helptext="Plot spectra from ARTIS and reference data.", aliases=("spec",)
    ),
    "plotspherical": CommandSpec("plotspherical", helptext="Plot direction maps based on escaped packets."),
    "plottransitions": CommandSpec("transitions", helptext="Plot estimated spectra from bound-bound transitions."),
    "plotviewingangles": CommandSpec(
        "viewing_angles_visualization", helptext="Generate a 3D visualization of an ARTIS model."
    ),
    "spencerfano": CommandSpec(
        "nonthermal.solvespencerfanocmd",
        helptext="Solve the Spencer-Fano equation using data from an ARTIS cell at some timestep.",
    ),
    "version": CommandSpec("commands", funcname="show_version", helptext="Print the artistools version."),
    "writecodecomparisondata": CommandSpec(
        "writecomparisondata", helptext="Write ARTIS model data out in code comparison workshop format."
    ),
    "writespectra": CommandSpec(
        "spectra.writespectra", helptext="Write ARTIS spectra for each timestep to individual text files."
    ),
}


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse formatter to show default values in help text, sorted with dashes last."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        kwargs["max_help_position"] = 50
        super().__init__(*args, **kwargs)

    @t.override
    def add_arguments(self, actions: Iterable[argparse.Action]) -> None:
        getinvocation = super()._format_action_invocation

        def my_sort(action: argparse.Action) -> str:
            return getinvocation(action).upper().replace("-", "z")  # push dash chars below alphabet

        actions = sorted(actions, key=my_sort)
        super().add_arguments(actions)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no command-line arguments (for subcommands that take none)."""


def addsubparsers(parser: argparse.ArgumentParser, parentcommand: str, subcommandtree: CommandTree) -> None:
    """Register the subcommands in the tree on the parser."""

    def func(args: t.Any) -> None:  # ruff:ignore[unused-function-argument]
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
            submodule = importlib.import_module(f"artistools.{spec.module}")
            submodule.addargs(subparser)
            subparser.set_defaults(func=getattr(submodule, spec.funcname))


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

        for command in COMMANDS:
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
    from artistools.version import version

    print(f"artistools {version}")


def get_path(key: str) -> Path:
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
