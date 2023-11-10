import argparse
import subprocess
import typing as t
from pathlib import Path

CommandType: t.TypeAlias = dict[str, t.Union[tuple[str, str], "CommandType"]]

# new subparser based list
dictcommands: CommandType = {
    "comparetogsinetwork": ("gsinetwork", "main"),
    "deposition": ("deposition", "main_analytical"),
    "describeinputmodel": ("inputmodel.describeinputmodel", "main"),
    "exportmassfractions": ("estimators.exportmassfractions", "main"),
    "getpath": ("", "get_path"),
    "listtimesteps": ("", "showtimesteptimes"),
    "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
    "maptogrid": ("inputmodel.maptogrid", "main"),
    "plotestimators": ("estimators.plotestimators", "main"),
    "plotinitialcomposition": ("inputmodel.plotinitialcomposition", "main"),
    "plotlightcurves": ("lightcurve.plotlightcurve", "main"),
    "plotlinefluxes": ("linefluxes", "main"),
    "plotdensity": ("inputmodel.plotdensity", "main"),
    "plotdeposition": ("deposition", "main"),
    "plotmacroatom": ("macroatom", "main"),
    "plotnltepops": ("nltepops.plotnltepops", "main"),
    "plotnonthermal": ("nonthermal.plotnonthermal", "main"),
    "plotradfield": ("radfield", "main"),
    "plotspectra": ("spectra.plotspectra", "main"),
    "plotspherical": ("plotspherical", "main"),
    "plottransitions": ("transitions", "main"),
    "plotviewingangles": ("viewing_angles_visualization", "main"),
    "setupcompletions": ("commands", "setup_completions"),
    "spencerfano": ("nonthermal.solvespencerfanocmd", "main"),
    "writecodecomparisondata": ("writecomparisondata", "main"),
    "writespectra": ("spectra.writespectra", "main"),
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
        "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
        "makeartismodel": ("inputmodel.makeartismodel", "main"),
        "make1dslicefrom3dmodel": ("inputmodel.1dslicefrom3d", "main"),
        "makeartismodel1dslicefromcone": ("inputmodel.slice1dfromconein3dmodel", "main"),
        "makeartismodelbotyanski2017": ("inputmodel.botyanski2017", "main"),
        "makeartismodelfromshen2018": ("inputmodel.shen2018", "main"),
        "makeartismodelscalevelocity": ("inputmodel.scalevelocity", "main"),
        "makeartismodelfullymixed": ("inputmodel.fullymixed", "main"),
        "makeartismodelsolar_rprocess": ("inputmodel.rprocess_solar", "main"),
        "makeartismodelfromsingletrajectory": ("inputmodel.rprocess_from_trajectory", "main"),
        "from_alcar": ("inputmodel.from_alcar", "main"),
        "to_tardis": ("inputmodel.to_tardis", "main"),
    },
}


def get_commandlist() -> dict[str, tuple[str, str]]:
    # direct commands (one file installed per command)
    # we generally should phase this out except for a couple of main ones like at and artistools
    return {
        "at": ("artistools", "main"),
        "artistools": ("artistools", "main"),
        "makeartismodel1dslicefromcone": ("artistools.inputmodel.slice1dfromconein3dmodel", "main"),
        "makeartismodel": ("artistools.inputmodel.makeartismodel", "main"),
        "plotartisdensity": ("artistools.inputmodel.plotdensity", "main"),
        "plotartisdeposition": ("artistools.deposition", "main"),
        "plotartisestimators": ("artistools.estimators.plotestimators", "main"),
        "plotartislightcurve": ("artistools.lightcurve.plotlightcurve", "main"),
        "plotartislinefluxes": ("artistools.linefluxes", "main"),
        "plotartismacroatom": ("artistools.macroatom", "main"),
        "plotartisnltepops": ("artistools.nltepops.plotnltepops", "main"),
        "plotartisnonthermal": ("artistools.nonthermal", "main"),
        "plotartisradfield": ("artistools.radfield", "main"),
        "plotartisspectrum": ("artistools.spectra.plotspectra", "main"),
        "plotartistransitions": ("artistools.transitions", "main"),
        "plotartisinitialcomposition": ("artistools.inputmodel.plotinitialcomposition", "main"),
        "plotartisviewingangles": ("artistools.viewing_angles_visualization", "main"),
    }


def get_console_scripts() -> list[str]:
    return [
        f"{command} = {submodulename}:{funcname}" for command, (submodulename, funcname) in get_commandlist().items()
    ]


def setup_completions(*args: t.Any, **kwargs: t.Any) -> None:
    # Add the following lines to your .zshrc file to get command completion:
    # autoload -U bashcompinit
    # bashcompinit
    # source artistoolscompletions.sh
    path_repo = Path(__file__).absolute().parent.parent
    completionscriptpath = path_repo / "artistoolscompletions.sh"
    with (completionscriptpath).open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env zsh\n")

        proc = subprocess.run(
            ["register-python-argcomplete", "__MY_COMMAND__"], capture_output=True, text=True, check=True
        )

        if proc.stderr:
            print(proc.stderr)

        strfunctiondefs, strsplit, strcommandregister = proc.stdout.rpartition("}\n")

        f.write(strfunctiondefs)
        f.write(strsplit)
        f.write("\n\n")

        for command in get_commandlist():
            completecommand = strcommandregister.replace("__MY_COMMAND__", command)
            f.write(completecommand + "\n")

    print("To enable completions, add these line to your .zshrc/.bashrc")
    print(f'source "{completionscriptpath}"')
    print("autoload -Uz compinit && compinit")


def addargs(parser: argparse.ArgumentParser) -> None:
    pass
