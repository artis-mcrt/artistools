commandlist = {
    'artistools-writecodecomparisondata': ('artistools.writecomparisondata', 'main'),

    'artistools-modeldeposition': ('artistools.deposition', 'main_analytical'),

    'getartisspencerfano': ('artistools.nonthermal.solvespencerfanocmd', 'main'),
    'artistools-spencerfano': ('artistools.nonthermal.solvespencerfanocmd', 'main'),

    'listartistimesteps': ('artistools', 'showtimesteptimes'),
    'artistools-timesteptimes': ('artistools', 'showtimesteptimes'),

    'artistools-make1dslicefrom3dmodel': ('artistools.inputmodel.1dslicefrom3d', 'main'),
    'makeartismodel1dslicefromcone': ('artistools.inputmodel.1dslicefromconein3dmodel', 'main'),
    'makeartismodelbotyanski2017': ('artistools.inputmodel.botyanski2017', 'main'),
    'makeartismodelfromshen2018': ('artistools.inputmodel.shen2018', 'main'),
    'makeartismodelfromlapuente': ('artistools.inputmodel.lapuente', 'main'),
    'makeartismodelscalevelocity': ('artistools.inputmodel.scalevelocity', 'main'),
    'makeartismodelfullymixed': ('artistools.inputmodel.fullymixed', 'main'),
    'makeartismodelsolar_rprocess': ('artistools.inputmodel.solar_rprocess', 'main'),
    'makeartismodel': ('artistools.inputmodel.makeartismodel', 'main'),

    'plotartisdeposition': ('artistools.deposition', 'main'),
    'artistools-deposition': ('artistools.deposition', 'main'),

    'plotartisestimators': ('artistools.estimators.plotestimators', 'main'),
    'artistools-estimators': ('artistools.estimators', 'main'),

    'plotartislightcurve': ('artistools.lightcurve.plotlightcurve', 'main'),
    'artistools-lightcurve': ('artistools.lightcurve', 'main'),

    'plotartislinefluxes': ('artistools.linefluxes', 'main'),
    'artistools-linefluxes': ('artistools.linefluxes', 'main'),

    'plotartisnltepops': ('artistools.nltepops.plotnltepops', 'main'),
    'artistools-nltepops': ('artistools.nltepops', 'main'),

    'plotartismacroatom': ('artistools.macroatom', 'main'),
    'artistools-macroatom': ('artistools.macroatom', 'main'),

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
}

console_scripts = [f'{command} = {submodulename}:{funcname}'
                   for command, (submodulename, funcname) in commandlist.items()]
console_scripts.append('at = artistools:main')
console_scripts.append('artistools = artistools:main')
