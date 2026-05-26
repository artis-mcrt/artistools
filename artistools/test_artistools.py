import contextlib
import hashlib
import importlib
import inspect
import math
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"
outputpath = at.get_path("testoutput")
outputpath.mkdir(exist_ok=True, parents=True)

REPOPATH = at.get_path("artistools_repository")


def funcname() -> str:
    """Get the name of the calling function."""
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr] # pyright: ignore[reportOptionalMemberAccess]
    except AttributeError as e:
        msg = "Could not get the name of the calling function."
        raise RuntimeError(msg) from e


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


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


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_singletimestep(mockplot: t.Any) -> None:
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40)

    assert len(mockplot.call_args_list) == 15
    expected_stats = {
        2: (5.31208, 6.01117e-08, 0.1588193243000988, 0.6969032639613162),
        6: (27071.5, 5.25769e-06, 1493.302228052353, 4621.593586839317),
        10: (109325.0, 5.03688e-10, 3308.8327426461733, 15522.059841599794),
        14: (35210.9, 2.5153e-08, 431.60267129328645, 3864.3843149881213),
    }
    for callindex, (expected_first, expected_last, expected_mean, expected_std) in expected_stats.items():
        _, yarr = get_plot_xy(mockplot.call_args_list[callindex])
        assert np.isclose(yarr[0], expected_first, rtol=1e-4)
        assert np.isclose(yarr[-1], expected_last, rtol=1e-4)
        assert np.isclose(yarr.mean(), expected_mean, rtol=1e-4)
        assert np.isclose(yarr.std(), expected_std, rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_velocity(mockplot: t.Any) -> None:
    at.nltepops.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        timestep=40,
        x="velocity",
        ion_stages=[1, 2],
        levels=[0, 1],
    )

    assert len(mockplot.call_args_list) == 2
    expected_yvals = [5.31208, 3.07492]
    for callargs, expected_yval in zip(mockplot.call_args_list, expected_yvals, strict=True):
        xarr, yarr = get_plot_xy(callargs)
        assert np.allclose(xarr, [8000.0], rtol=1e-4)
        assert np.allclose(yarr, [expected_yval], rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_time(mockplot: t.Any) -> None:
    at.nltepops.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        cell=0,
        x="time",
        timedays="270-275",
        ion_stages=[1, 2],
        levels=[0, 1],
    )

    assert len(mockplot.call_args_list) == 10
    expected_series = [
        ([271.48221094182054, 273.31529638210384], [7.40594, 6.39568]),
        ([271.48221094182054, 273.31529638210384], [4.71888, 3.89199]),
    ]
    for callargs, (expected_xarr, expected_yarr) in zip(mockplot.call_args_list[:2], expected_series, strict=True):
        xarr, yarr = get_plot_xy(callargs)
        assert np.allclose(xarr, expected_xarr, rtol=1e-4)
        assert np.allclose(yarr, expected_yarr, rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(mockstep: t.Any, mockplot: t.Any) -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.radfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=funcoutpath, showbinedges=True)

    plot_calls = {call.kwargs.get("label"): call for call in mockplot.call_args_list if call.kwargs.get("label")}
    dilute_xarr, dilute_yarr = get_plot_xy(plot_calls["Dilute blackbody model "])
    assert np.isclose(dilute_xarr.min(), 1000.0, rtol=1e-4)
    assert np.isclose(dilute_xarr.max(), 20000.0, rtol=1e-4)
    assert np.isclose(dilute_yarr.mean(), 21.27744616064978, rtol=1e-4)
    assert np.isclose(dilute_yarr.std(), 26.77850448874471, rtol=1e-4)

    fitted_xarr, fitted_yarr = get_plot_xy(plot_calls["Radiation field model"])
    assert np.isclose(fitted_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(fitted_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(fitted_yarr.mean(), 48.342355990852596, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(fitted_yarr, fitted_xarr)), 489588.12007010705, rtol=1e-4)

    bandavg_xarr, bandavg_yarr = get_plot_xy(mockstep.call_args_list[0])
    assert np.isclose(bandavg_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(bandavg_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(bandavg_yarr.mean(), 43.58807185509511, rtol=1e-4)
    assert np.isclose(np.trapezoid(bandavg_yarr, bandavg_xarr), -475489.00963827176, rtol=1e-4)


@pytest.mark.benchmark
def test_plotspherical() -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=funcoutpath)


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemax=270, outputfile=outputpath)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_transitions(mockplot: t.Any) -> None:
    at.transitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)

    assert len(mockplot.call_args_list) == 7
    expected_integrals = [
        0.03762393022815368,
        266.8869480321175,
        299.25457622600254,
        8.318170397519948,
        34.5598725883166,
        0.0,
        0.0,
    ]
    expected_maxima = [
        7.054096268640787e-05,
        0.3309041740131583,
        0.9619558273061346,
        0.013829945098038332,
        0.060540167233566825,
        0.0,
        0.0,
    ]
    for callargs, expected_integral, expected_max in zip(
        mockplot.call_args_list, expected_integrals, expected_maxima, strict=True
    ):
        xarr, yarr = get_plot_xy(callargs)
        assert np.isclose(xarr[0], 3500.0, rtol=1e-4)
        assert np.isclose(xarr[-1], 7996.0, rtol=1e-4)
        assert np.isclose(np.trapezoid(yarr, xarr), expected_integral, rtol=1e-4, atol=1e-8)
        assert np.isclose(yarr.max(), expected_max, rtol=1e-4, atol=1e-8)


@pytest.mark.benchmark
def test_writecomparisondata() -> None:
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=outputpath, selected_timesteps=list(range(99))
    )
