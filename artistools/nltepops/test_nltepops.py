import math
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
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
def test_nltepops_singletimestep(mockplot: mock.MagicMock) -> None:
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


def make_model_without_plotted_cell_estimators(tmp_path: Path) -> None:
    """Fabricate a model whose estimator files cover timestep 40 but omit the plotted cell 0.

    The test model's files are symlinked, except that model.txt gains a second cell and the
    estimator file's timestep 40 block is reassigned to that cell, so the run folder is matched
    for timestep 40 yet read_estimators finds no (40, 0) entry.
    """
    for filename in (
        "input.txt",
        "adata.txt.xz",
        "compositiondata.txt",
        "transitiondata.txt",
        "phixsdata_v2.txt.xz",
        "nlte_0000.out.xz",
    ):
        (tmp_path / filename).symlink_to(modelpath / filename)

    _npts_line, t_model_line, cellrow = (modelpath / "model.txt").read_text(encoding="utf-8").splitlines()
    cellrow2 = "     2   16000.  " + cellrow.split(maxsplit=2)[2]
    (tmp_path / "model.txt").write_text(f"2\n{t_model_line}\n{cellrow}\n{cellrow2}\n", encoding="utf-8")

    estlines = (modelpath / "estimators_0000.out").read_text(encoding="utf-8").splitlines(keepends=True)
    (tmp_path / "estimators_0000.out").write_text(
        "".join(
            line.replace("modelgridindex 0", "modelgridindex 1", 1) if line.startswith("timestep 40 ") else line
            for line in estlines
        ),
        encoding="utf-8",
    )


@mock.patch.object(mplax.Axes, "set_title", side_effect=mplax.Axes.set_title, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_nltepops_no_estimator_data(
    mockplot: mock.MagicMock, mocktitle: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cell with NLTE populations but no estimator data must still plot, using the LTE fallback temperature."""
    make_model_without_plotted_cell_estimators(tmp_path)

    # the model has two cells, thus the command must name the plotted cell
    at.nltepops.plot(
        argsraw=[],
        modelpath=tmp_path,
        outputfile=tmp_path,
        cell=0,
        timestep=40,
        exc_temperature=5000.0,
        plotrefdata=True,
    )

    assert "WARNING: No estimator data" in capsys.readouterr().out
    titles = [callargs[0][1] for callargs in mocktitle.call_args_list]
    assert any("Te=5000 K" in ti and "nne=nan" in ti and "T$_R$=5000 K" in ti and "W=nan" in ti for ti in titles)

    # the same series as the with-estimators case are plotted, and the NLTE populations are unaffected
    assert len(mockplot.call_args_list) == 15
    _, yarr = get_plot_xy(mockplot.call_args_list[2])
    assert np.isclose(yarr[0], 5.31208, rtol=1e-4)

    assert any(tmp_path.glob("plotnlte_Fe_cell000_ts40_*.pdf"))


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_velocity(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    at.nltepops.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=tmp_path,
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

    assert (tmp_path / "plotnltelevelpops_Fe.pdf").is_file()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_time(mockplot: mock.MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # no outputfile, so this covers the default filename that -x time selects
    monkeypatch.chdir(tmp_path)
    at.nltepops.plot(
        argsraw=[], modelpath=modelpath, cell=0, x="time", timedays="270-275", ion_stages=[1, 2], levels=[0, 1]
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

    assert (tmp_path / "plotnltelevelpops_Fe.pdf").is_file()


@mock.patch.object(mplax.Axes, "legend", side_effect=mplax.Axes.legend, autospec=True)
def test_nltepops_draws_one_shared_legend(mocklegend: mock.MagicMock, tmp_path: Path) -> None:
    """One legend covers every subplot, and it names each series one time."""
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, cell=0, timestep=40)

    assert len(mocklegend.call_args_list) == 1
    labels = mocklegend.call_args_list[0].kwargs["labels"]
    assert len(labels) == len(set(labels)), f"a label appears more than one time: {labels}"
    # the ion names the subplot, thus no legend entry repeats it
    assert not any("Fe" in label for label in labels), labels


def test_texifyterm_handles_multiplicity_parity_and_jvalue() -> None:
    assert at.nltepops.texifyterm("o4Fo[2]") == r"$^{4}$F$^{\rm o}_{2}$"
    assert at.nltepops.texifyterm("3P2") == r"$^{3}$P2"


def test_texifyconfiguration_formats_configuration_and_parent_terms() -> None:
    assert at.nltepops.texifyconfiguration("3d6_5D") == r"3d$^{6}$ $^{5}$D"
    assert at.nltepops.texifyconfiguration("3d7(4F)4p_z5G[2]") == r"3d$^{7}$($^{4}$F)4p $^{5}$G$_{2}$"


def test_add_lte_pops_calculates_levels_and_superlevel() -> None:
    ionlevels = pl.DataFrame({"g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    adata = pl.DataFrame({"Z": [26], "ion_stage": [2], "levels": pl.Series([ionlevels], dtype=pl.Object)})

    dfpop = pl.DataFrame({
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

    assert math.isclose(result.filter(pl.col("level") == 0)["lte_10000"].item(), 1.0, rel_tol=1e-12)
    assert math.isclose(result.filter(pl.col("level") == 1)["lte_10000"].item(), expected_level1, rel_tol=1e-12)
    assert math.isclose(result.filter(pl.col("level") == 4)["lte_10000"].item(), expected_superlevel, rel_tol=1e-12)


@pytest.mark.parametrize("timedays", [300, "300", 300.0])
def test_nltepops_keyword_timedays_reads_a_number(timedays: float | str, tmp_path: Path) -> None:
    """A command line gives a string, and a keyword argument of the API gives a number. Both name one time."""
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, modelgridindex=0, timedays=timedays)

    assert any(tmp_path.glob("plotnlte_Fe_cell000_ts*.pdf"))


def test_nltepops_subplot_blocks_do_not_overlap() -> None:
    """Each cell must own its own block of subplots, one for each ion stage.

    A block that started at the index of the cell drew over the block of the cell in front of it, and
    it left the last block empty.
    """
    for ncells in (1, 2, 5):
        for nionstages in (1, 3, 8):
            blocks = [at.nltepops.plotnltepops.get_subplot_block(index, nionstages) for index in range(ncells)]
            covered = [axindex for first, last in blocks for axindex in range(first, last + 1)]

            # make_singletimestep_plot builds this many subplots
            assert covered == list(range(ncells * nionstages)), (ncells, nionstages)
