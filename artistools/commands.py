import argparse
import importlib
import subprocess
import typing as t
from pathlib import Path

from artistools.misc import CustomArgHelpFormatter

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
# fully recursive python >= 3.12
# type CommandType = dict[str, tuple[str, str] | "CommandType"]
# fully recursive python >= 3.11
CommandType: t.TypeAlias = dict[str, t.Union[tuple[str, str], "CommandType"]]  # pyright: ignore[reportDeprecated]

# new subparser based list
subcommandtree: CommandType = {
    "comparetogsinetwork": ("gsinetwork.plotqdotabund", "main"),
    "gsinetworkdecayproducts": ("gsinetwork.decayproducts", "main"),
    "describeinputmodel": ("inputmodel.describeinputmodel", "main"),
    "estimators": ("estimators.plotestimators", "main"),
    "exportmassfractions": ("estimators.exportmassfractions", "main"),
    "getpath": ("", "get_path"),
    "lc": ("lightcurve.plotlightcurve", "main"),
    "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
    "maptogrid": ("inputmodel.maptogrid", "main"),
    "plotestimators": ("estimators.plotestimators", "main"),
    "plotinitialcomposition": ("inputmodel.plotinitialcomposition", "main"),
    "plotlightcurves": ("lightcurve.plotlightcurve", "main"),
    "plotlinefluxes": ("linefluxes", "main"),
    "plotdensity": ("inputmodel.plotdensity", "main"),
    "plotmacroatom": ("macroatom", "main"),
    "plotnltepops": ("nltepops.plotnltepops", "main"),
    "plotradfield": ("radfield", "main"),
    "plotspectra": ("spectra.plotspectra", "main"),
    "plotspherical": ("plotspherical", "main"),
    "plottransitions": ("transitions", "main"),
    "plotviewingangles": ("viewing_angles_visualization", "main"),
    "completions": ("commands", "setup_completions"),
    "version": ("commands", "show_version"),
    "spec": ("spectra.plotspectra", "main"),
    "spencerfano": ("nonthermal.solvespencerfanocmd", "main"),
    "writecodecomparisondata": ("writecomparisondata", "main"),
    "writespectra": ("spectra.writespectra", "main"),
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
        "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
        "makeartismodel": ("inputmodel.makeartismodel", "main"),
        "make1dslicefrom3dmodel": ("inputmodel.make1dslicefrom3d", "main"),
        "makeartismodel1dslicefromcone": ("inputmodel.slice1dfromconein3dmodel", "main"),
        "makeartismodelfromshen2018": ("inputmodel.shen2018", "main"),
        "makeartismodelfromsingletrajectory": ("inputmodel.rprocess_from_trajectory", "main"),
        "from_e2e": ("inputmodel.from_e2e_model", "main"),
        "to_tardis": ("inputmodel.to_tardis", "main"),
    },
}


def addargs(parser: argparse.ArgumentParser) -> None:
    pass


def addsubparsers(
    parser: argparse.ArgumentParser, parentcommand: str, subcommandtree: CommandType, depth: int = 1
) -> None:
    def func(args: t.Any) -> None:  # noqa: ARG001
        parser.print_help()

    parser.set_defaults(func=func)
    subparsers = parser.add_subparsers(dest=f"{parentcommand} command", required=False)

    for subcommand, subcommands in subcommandtree.items():
        strhelp: str | None
        if isinstance(subcommands, dict):
            strhelp = "command group"
            subparser = subparsers.add_parser(subcommand, help=strhelp, formatter_class=CustomArgHelpFormatter)
            addsubparsers(parser=subparser, parentcommand=subcommand, subcommandtree=subcommands, depth=depth + 1)
        else:
            submodulename, funcname = subcommands
            namestr = f"artistools.{submodulename.removeprefix('artistools.')}" if submodulename else "artistools"
            submodule = importlib.import_module(namestr, package="artistools")
            func = getattr(submodule, funcname)
            strhelp = func.__doc__
            subparser = subparsers.add_parser(subcommand, help=strhelp, formatter_class=CustomArgHelpFormatter)

            assert hasattr(submodule, "addargs")
            assert callable(submodule.addargs)
            submodule.addargs(subparser)
            subparser.set_defaults(func=func)


def setup_completions(*args: t.Any, **kwargs: t.Any) -> None:  # noqa: ARG001
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


def show_version(*args: t.Any, **kwargs: t.Any) -> None:  # noqa: ARG001
    from artistools.version import version

    print(f"artistools {version}")
