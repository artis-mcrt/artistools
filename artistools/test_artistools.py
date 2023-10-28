#!/usr/bin/env python3
import hashlib
import importlib
import math
import typing as t

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]


def test_commands() -> None:
    # ensure that the commands are pointing to valid submodule.function() targets
    for _command, (submodulename, funcname) in sorted(at.commands.get_commandlist().items()):
        submodule = importlib.import_module(submodulename)
        assert hasattr(submodule, funcname) or (
            funcname == "main" and hasattr(importlib.import_module(f"{submodulename}.__main__"), funcname)
        )

    def recursive_check(dictcmd: dict[str, t.Any]) -> None:
        for cmdtarget in dictcmd.values():
            if isinstance(cmdtarget, dict):
                recursive_check(cmdtarget)
            else:
                submodulename, funcname = cmdtarget
                namestr = f"artistools.{submodulename.removeprefix('artistools.')}" if submodulename else "artistools"
                print(namestr)
                submodule = importlib.import_module(namestr, package="artistools")
                assert hasattr(submodule, funcname) or (
                    funcname == "main" and hasattr(importlib.import_module(f"{namestr}.__main__"), funcname)
                )

    recursive_check(at.commands.dictcommands)


def test_timestep_times() -> None:
    timestartarray = at.get_timestep_times(modelpath, loc="start")
    timedeltarray = at.get_timestep_times(modelpath, loc="delta")
    timemidarray = at.get_timestep_times(modelpath, loc="mid")
    assert len(timestartarray) == 100
    assert math.isclose(float(timemidarray[0]), 250.421, abs_tol=1e-3)
    assert math.isclose(float(timemidarray[-1]), 349.412, abs_tol=1e-3)

    assert all(
        tstart < tmid < (tstart + tdelta) for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray)
    )


def test_deposition() -> None:
    at.deposition.main(argsraw=[], modelpath=modelpath)


def test_estimator_snapshot() -> None:
    at.estimators.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)


def test_estimator_timeevolution() -> None:
    at.estimators.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, modelgridindex=0, x="time")


def test_get_inputparams() -> None:
    inputparams = at.get_inputparams(modelpath)
    dicthash = hashlib.sha256(str(sorted(inputparams.items())).encode("utf-8")).hexdigest()
    assert dicthash == "ce7d04d6944207673a105cba8d2430055d0b53b7f3e92db3964d2dca285a3adb"


def test_get_levels() -> None:
    at.atomic.get_levels(modelpath, get_transitions=True, get_photoionisations=True)


def test_macroatom() -> None:
    at.macroatom.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=10)


def test_nltepops() -> None:
    # at.nltepops.plot(modelpath=modelpath, outputfile=outputpath, timedays=300),
    #                    **benchargs)
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40)


def test_nonthermal() -> None:
    at.nonthermal.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=70)


def test_radfield() -> None:
    at.radfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=outputpath)


def test_get_ionrecombratecalibration() -> None:
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)


def test_plotspherical() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=outputpath)


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, outputfile=outputpath)


def test_spencerfano() -> None:
    at.nonthermal.solvespencerfanocmd.main(
        argsraw=[], modelpath=modelpath, timedays=300, makeplot=True, npts=200, noexcitation=True, outputfile=outputpath
    )


def test_transitions() -> None:
    at.transitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)


def test_write_comparisondata() -> None:
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=outputpath, selected_timesteps=list(range(99))
    )
