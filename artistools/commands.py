import subprocess
from pathlib import Path

commandlist = {
    "artistools-comparetogsinetwork": ("artistools.gsinetwork", "main"),
    "artistools-modeldeposition": ("artistools.deposition", "main_analytical"),
    "getartisspencerfano": ("artistools.nonthermal.solvespencerfanocmd", "main"),
    "artistools-spencerfano": ("artistools.nonthermal.solvespencerfanocmd", "main"),
    "listartistimesteps": ("artistools", "showtimesteptimes"),
    "artistools-timesteptimes": ("artistools", "showtimesteptimes"),
    "artistools-make1dslicefrom3dmodel": ("artistools.inputmodel.1dslicefrom3d", "main"),
    "makeartismodel1dslicefromcone": ("artistools.inputmodel.slice1Dfromconein3dmodel", "main"),
    "makeartismodelbotyanski2017": ("artistools.inputmodel.botyanski2017", "main"),
    "makeartismodelfromshen2018": ("artistools.inputmodel.shen2018", "main"),
    "makeartismodelfromlapuente": ("artistools.inputmodel.lapuente", "main"),
    "makeartismodelscalevelocity": ("artistools.inputmodel.scalevelocity", "main"),
    "makeartismodelfullymixed": ("artistools.inputmodel.fullymixed", "main"),
    "makeartismodelsolar_rprocess": ("artistools.inputmodel.rprocess_solar", "main"),
    "makeartismodelfromsingletrajectory": ("artistools.inputmodel.rprocess_from_trajectory", "main"),
    "makeartismodelfromparticlegridmap": ("artistools.inputmodel.modelfromhydro", "main"),
    "makeartismodel": ("artistools.inputmodel.makeartismodel", "main"),
    "artistools-maketardismodelfromartis": ("artistools.inputmodel.maketardismodelfromartis", "main"),
    "artistools-maptogrid": ("artistools.inputmodel.maptogrid", "main"),
    "plotartismodeldensity": ("artistools.inputmodel.plotdensity", "main"),
    "artistools-plotdensity": ("artistools.inputmodel.plotdensity", "main"),
    "plotartismodeldeposition": ("artistools.deposition", "main"),
    "artistools-deposition": ("artistools.deposition", "main"),
    "artistools-describeinputmodel": ("artistools.inputmodel.describeinputmodel", "main"),
    "plotartisestimators": ("artistools.estimators.plotestimators", "main"),
    "artistools-estimators": ("artistools.estimators", "main"),
    "artistools-exportmassfractions": ("artistools.estimators.exportmassfractions", "main"),
    "plotartislightcurve": ("artistools.lightcurve.plotlightcurve", "main"),
    "artistools-lightcurve": ("artistools.lightcurve", "main"),
    "plotartislinefluxes": ("artistools.linefluxes", "main"),
    "artistools-linefluxes": ("artistools.linefluxes", "main"),
    "plotartismacroatom": ("artistools.macroatom", "main"),
    "artistools-macroatom": ("artistools.macroatom", "main"),
    "plotartisnltepops": ("artistools.nltepops.plotnltepops", "main"),
    "artistools-nltepops": ("artistools.nltepops", "main"),
    "plotartisnonthermal": ("artistools.nonthermal", "main"),
    "artistools-nonthermal": ("artistools.nonthermal", "main"),
    "plotartisradfield": ("artistools.radfield", "main"),
    "artistools-radfield": ("artistools.radfield", "main"),
    "plotartisspectrum": ("artistools.spectra.plotspectra", "main"),
    "artistools-spectrum": ("artistools.spectra", "main"),
    "plotartistransitions": ("artistools.transitions", "main"),
    "artistools-transitions": ("artistools.transitions", "main"),
    "plotartisinitialcomposition": ("artistools.initial_composition", "main"),
    "artistools-initialcomposition": ("artistools.initial_composition", "main"),
    "artistools-writecodecomparisondata": ("artistools.writecomparisondata", "main"),
    "artistools-setup_completions": ("artistools.commands", "setup_completions"),
}


def get_console_scripts():
    console_scripts = [
        f"{command} = {submodulename}:{funcname}" for command, (submodulename, funcname) in commandlist.items()
    ]
    console_scripts.append("at = artistools:main")
    console_scripts.append("artistools = artistools:main")
    return console_scripts


def setup_completions():
    # Add the following lines to your .zshrc file to get command completion:
    # autoload -U bashcompinit
    # bashcompinit
    # source artistoolscompletions.sh
    path_repo = Path(__file__).absolute().parent.parent
    completioncommands = []
    with open(path_repo / "artistoolscompletions.sh", "w", encoding="utf-8") as f:
        for command in commandlist.keys():
            proc = subprocess.run(["register-python-argcomplete", command], capture_output=True, text=True)

            if proc.stderr:
                print(proc.stderr)

            completioncommands.append(proc.stdout + "\n")

        f.write("\n".join(completioncommands))

    print("To enable completions, add this line to your .zshrc/.bashrc")
    print("source artistoolscompletions.sh")


def addargs(parser):
    pass
