#!/usr/bin/env python3
import contextlib
import hashlib
import importlib
import inspect
import math
from pathlib import Path

import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_3d = at.get_config()["path_testdata"] / "testmodel_3d_10^3"
outputpath = at.get_config()["path_testoutput"]
outputpath.mkdir(exist_ok=True, parents=True)

REPOPATH = at.get_config("path_artistools_repository")


def funcname() -> str:
    """Get the name of the calling function."""
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr] # pyright: ignore[reportOptionalMemberAccess]
    except AttributeError as e:
        msg = "Could not get the name of the calling function."
        raise RuntimeError(msg) from e


def test_commands() -> None:
    commands: dict[str, tuple[str, str]] = {}

    # just skip the test if tomllib is not available (python < 3.11)
    with contextlib.suppress(ImportError):
        import tomllib

        assert isinstance(REPOPATH, Path)
        with (REPOPATH / "pyproject.toml").open("rb") as f:
            pyproj = tomllib.load(f)
        commands = {k: tuple(v.split(":")) for k, v in pyproj["project"]["scripts"].items()}

        # ensure that the commands are pointing to valid submodule.function() targets
        for command, (submodulename, funcname) in commands.items():
            submodule = importlib.import_module(submodulename)
            assert hasattr(submodule, funcname) or (
                funcname == "main" and hasattr(importlib.import_module(f"{submodulename}.__main__"), funcname)
            ), f"{submodulename}.{funcname} not found for command {command}"

    def recursive_check(dictcmd: at.commands.CommandType) -> None:
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

    recursive_check(at.commands.subcommandtree)


def test_timestep_times() -> None:
    timestartarray = at.get_timestep_times(modelpath, loc="start")
    timedeltarray = at.get_timestep_times(modelpath, loc="delta")
    timemidarray = at.get_timestep_times(modelpath, loc="mid")
    assert len(timestartarray) == 100
    assert math.isclose(timemidarray[0], 250.421, abs_tol=1e-3)
    assert math.isclose(timemidarray[-1], 349.412, abs_tol=1e-3)

    assert all(
        tstart < tmid < (tstart + tdelta)
        for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray, strict=False)
    )


def test_get_inputparams() -> None:
    inputparams = at.get_inputparams(modelpath)
    dicthash = hashlib.sha256(str(sorted(inputparams.items())).encode("utf-8")).hexdigest()
    assert dicthash == "1edcddd5d36cc2eaed94ad083dacfb95c6915b8fd4f62591e2b79ceca6885d1e", dicthash


def test_macroatom() -> None:
    at.macroatom.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=10)


@pytest.mark.benchmark
def test_nltepops() -> None:
    # at.nltepops.plot(modelpath=modelpath, outputfile=outputpath, timedays=300),
    #                    **benchargs)
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40)


@pytest.mark.benchmark
def test_radfield() -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.radfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=funcoutpath)


@pytest.mark.benchmark
def test_plotspherical() -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=funcoutpath)


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemax=270, outputfile=outputpath)


@pytest.mark.benchmark
def test_transitions() -> None:
    at.transitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)


@pytest.mark.benchmark
def test_writecomparisondata() -> None:
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=outputpath, selected_timesteps=list(range(99))
    )
