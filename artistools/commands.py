import subprocess

commandlist = {
    'artistools-comparetogsinetwork': ('artistools.gsinetwork', 'main'),

    'artistools-modeldeposition': ('artistools.deposition', 'main_analytical'),

    'getartisspencerfano': ('artistools.nonthermal.solvespencerfanocmd', 'main'),
    'artistools-spencerfano': ('artistools.nonthermal.solvespencerfanocmd', 'main'),

    'listartistimesteps': ('artistools', 'showtimesteptimes'),
    'artistools-timesteptimes': ('artistools', 'showtimesteptimes'),

    'artistools-make1dslicefrom3dmodel': ('artistools.inputmodel.1dslicefrom3d', 'main'),
    'makeartismodel1dslicefromcone': ('artistools.inputmodel.slice1Dfromconein3dmodel', 'main'),
    'makeartismodelbotyanski2017': ('artistools.inputmodel.botyanski2017', 'main'),
    'makeartismodelfromshen2018': ('artistools.inputmodel.shen2018', 'main'),
    'makeartismodelfromlapuente': ('artistools.inputmodel.lapuente', 'main'),
    'makeartismodelscalevelocity': ('artistools.inputmodel.scalevelocity', 'main'),
    'makeartismodelfullymixed': ('artistools.inputmodel.fullymixed', 'main'),
    'makeartismodelsolar_rprocess': ('artistools.inputmodel.rprocess_solar', 'main'),
    'makeartismodelfromsingletrajectory': ('artistools.inputmodel.rprocess_from_trajectory', 'main'),
    'makeartismodelfromparticlegridmap': ('artistools.inputmodel.modelfromhydro', 'main'),
    'makeartismodel': ('artistools.inputmodel.makeartismodel', 'main'),

    'artistools-maketardismodelfromartis': ('artistools.inputmodel.maketardismodelfromartis', 'main'),

    'artistools-maptogrid': ('artistools.inputmodel.maptogrid', 'main'),

    'plotartisdeposition': ('artistools.deposition', 'main'),
    'artistools-deposition': ('artistools.deposition', 'main'),

    'artistools-describeinputmodel': ('artistools.inputmodel.describeinputmodel', 'main'),

    'plotartisestimators': ('artistools.estimators.plotestimators', 'main'),
    'artistools-estimators': ('artistools.estimators', 'main'),

    'artistools-exportmassfractions': ('artistools.estimators.exportmassfractions', 'main'),

    'plotartislightcurve': ('artistools.lightcurve.plotlightcurve', 'main'),
    'artistools-lightcurve': ('artistools.lightcurve', 'main'),

    'plotartislinefluxes': ('artistools.linefluxes', 'main'),
    'artistools-linefluxes': ('artistools.linefluxes', 'main'),

    'plotartismacroatom': ('artistools.macroatom', 'main'),
    'artistools-macroatom': ('artistools.macroatom', 'main'),

    'plotartisnltepops': ('artistools.nltepops.plotnltepops', 'main'),
    'artistools-nltepops': ('artistools.nltepops', 'main'),

    'plotartisnonthermal': ('artistools.nonthermal', 'main'),
    'artistools-nonthermal': ('artistools.nonthermal', 'main'),

    'plotartisradfield': ('artistools.radfield', 'main'),
    'artistools-radfield': ('artistools.radfield', 'main'),

    'plotartisspectrum': ('artistools.spectra.plotspectra', 'main'),
    'artistools-spectrum': ('artistools.spectra', 'main'),

    'plotartistransitions': ('artistools.transitions', 'main'),
    'artistools-transitions': ('artistools.transitions', 'main'),

    'plotartisinitialcomposition': ('artistools.initial_composition', 'main'),
    'artistools-initialcomposition': ('artistools.initial_composition', 'main'),

    'artistools-writecodecomparisondata': ('artistools.writecomparisondata', 'main'),
}


def get_console_scripts():
    console_scripts = [f'{command} = {submodulename}:{funcname}'
                       for command, (submodulename, funcname) in commandlist.items()]
    console_scripts.append('at = artistools:main')
    console_scripts.append('artistools = artistools:main')
    return console_scripts


def get_completioncommands():
    completioncommands = []
    for command in commandlist.keys():
        result = subprocess.run(['register-python-argcomplete', command], capture_output=True, text=True).stdout
        completioncommands.append(result + '\n')
    return completioncommands
