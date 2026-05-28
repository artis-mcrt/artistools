import math
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import pandas as pd
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")
outputpath.mkdir(exist_ok=True, parents=True)


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


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


def test_texifyterm_handles_multiplicity_parity_and_jvalue() -> None:
    assert at.nltepops.texifyterm("o4Fo[2]") == r"$^{4}$F$^{\rm o}_{2}$"
    assert at.nltepops.texifyterm("3P2") == r"$^{3}$P2"


def test_texifyconfiguration_formats_configuration_and_parent_terms() -> None:
    assert at.nltepops.texifyconfiguration("3d6_5D") == r"3d$^{6}$ $^{5}$D"
    assert at.nltepops.texifyconfiguration("3d7(4F)4p_z5G[2]") == r"3d$^{7}$($^{4}$F)4p $^{5}$G$_{2}$"


def test_add_lte_pops_calculates_levels_and_superlevel() -> None:
    ionlevels = pl.DataFrame({"g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    adata = pl.DataFrame({"Z": [26], "ion_stage": [2], "levels": pl.Series([ionlevels], dtype=pl.Object)})

    dfpop = pd.DataFrame({
        "modelgridindex": [0, 0, 0],
        "timestep": [1, 1, 1],
        "Z": [26, 26, 26],
        "ion_stage": [2, 2, 2],
        "level": [0, 1, -1],
        "n_NLTE": [1.0, 0.3, 0.1],
    })

    result = at.nltepops.add_lte_pops(dfpop, adata, [("lte_10000", 10000)], noprint=True)

    k_b = 8.617333262145179e-05
    expected_level1 = 4.0 / 2.0 * math.exp(-(1.0 - 0.0) / k_b / 10000)
    expected_superlevel = 6.0 / 2.0 * math.exp(-(2.0 - 0.0) / k_b / 10000)

    assert math.isclose(result.loc[result["level"] == 0, "lte_10000"].item(), 1.0, rel_tol=1e-12)
    assert math.isclose(result.loc[result["level"] == 1, "lte_10000"].item(), expected_level1, rel_tol=1e-12)
    assert math.isclose(result.loc[result["level"] == 4, "lte_10000"].item(), expected_superlevel, rel_tol=1e-12)


def test_read_files_combines_ranks_and_applies_filters(tmp_path: Path) -> None:
    rank0 = tmp_path / "nlte_0000.out"
    rank1 = tmp_path / "nlte_0001.out"

    rank0.write_text("modelgridindex timestep ionstage level n_NLTE\n7 5 2 0 1.0\n8 5 2 1 2.0\n", encoding="utf-8")
    rank1.write_text("modelgridindex timestep ionstage level n_NLTE\n7 6 2 1 3.0\n7 5 3 0 4.0\n", encoding="utf-8")

    filtered = at.nltepops.read_files(tmp_path, timestep=5, modelgridindex=7)

    assert filtered.height == 2
    assert "ion_stage" in filtered.columns
    assert "ionstage" not in filtered.columns
    assert set(filtered["ion_stage"].to_list()) == {2, 3}
    assert filtered["modelgridindex"].to_list() == [7, 7]
    assert filtered["timestep"].to_list() == [5, 5]
