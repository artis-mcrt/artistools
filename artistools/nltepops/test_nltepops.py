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
        assert np.isclose(yarr[0], expected_first, rtol=1e-4, atol=0.0)
        assert np.isclose(yarr[-1], expected_last, rtol=1e-4, atol=0.0)
        assert np.isclose(yarr.mean(), expected_mean, rtol=1e-4, atol=0.0)
        assert np.isclose(yarr.std(), expected_std, rtol=1e-4, atol=0.0)


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


@mock.patch.object(mplax.Axes, "set_xticklabels", side_effect=mplax.Axes.set_xticklabels, autospec=True)
def test_nltepops_config_labels_skip_a_last_cell_without_data(mockticklabels: mock.MagicMock, tmp_path: Path) -> None:
    """The lowest panel that the command draws shows the configuration names, also when the last cell has no data.

    Only a panel of the last cell in the list took the names, thus a last cell without NLTE data left every axis blank.
    """
    make_model_without_plotted_cell_estimators(tmp_path)

    # the NLTE file holds cell 0 alone, thus cell 1 has no data
    at.nltepops.plot(argsraw=[], modelpath=tmp_path, outputfile=tmp_path, cell="0,1", timestep=40, x="config")

    # autospec records the axes as the first argument, thus the test knows which subplot took the names
    labelledaxes = [
        callargs[0][0] for callargs in mockticklabels.call_args_list if any(label for label in callargs[0][1])
    ]
    assert len(labelledaxes) == 1

    figure = labelledaxes[0].get_figure()
    assert figure is not None
    figureaxes = figure.axes
    # each cell has a block of subplots of the same size. Cell 1 holds no data, thus the last subplot
    # of the block of cell 0 shows the names. The last subplot of the figure shows no name
    assert figureaxes.index(labelledaxes[0]) == len(figureaxes) // 2 - 1


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

    # a warning goes to the standard error, thus --quiet keeps it
    assert "WARNING: No estimator data" in capsys.readouterr().err
    titles = [callargs[0][1] for callargs in mocktitle.call_args_list]
    assert any("Te=5000 K" in ti and "nne=nan" in ti and "T$_R$=5000 K" in ti and "W=nan" in ti for ti in titles)

    # the same series as the with-estimators case are plotted, and the NLTE populations are unaffected
    assert len(mockplot.call_args_list) == 15
    _, yarr = get_plot_xy(mockplot.call_args_list[2])
    assert np.isclose(yarr[0], 5.31208, rtol=1e-4)

    assert any(tmp_path.glob("plotnltepops_Fe_cell00000_ts040_*.pdf"))


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_velocity(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Each ion stage of -ion_stages gives one series for each level, and the label names the ion.

    The command plotted the first ion stage alone, and it dropped Fe II with no message.
    """
    at.nltepops.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=tmp_path,
        timestep=40,
        x="velocity",
        ion_stages=[1, 2],
        levels=[0, 1],
    )

    assert len(mockplot.call_args_list) == 4
    expected_yvals = [5.31208, 3.07492, 27071.5, 19075.8]
    for callargs, expected_yval in zip(mockplot.call_args_list, expected_yvals, strict=True):
        xarr, yarr = get_plot_xy(callargs)
        # vel_r_mid is the mid-point velocity of the only cell, whose outer velocity is 8000 km/s
        assert np.allclose(xarr, [4000.0], rtol=1e-4)
        assert np.allclose(yarr, [expected_yval], rtol=1e-4)
    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    assert [label.split()[:2] for label in labels] == [["Fe", "I"], ["Fe", "I"], ["Fe", "II"], ["Fe", "II"]]

    assert (tmp_path / "plotnltepops_Fe.pdf").is_file()


def test_nltepops_versus_velocity_needs_a_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With no time, a plot against velocity drew one series for each level at every timestep of the run."""
    with pytest.raises(SystemExit):
        at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, x="velocity", ion_stages=2, levels=[0])

    assert "-x velocity needs a time" in capsys.readouterr().err


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_nltepops_versus_time(mockplot: mock.MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # no outputfile, so this covers the default filename that -x time selects
    monkeypatch.chdir(tmp_path)
    at.nltepops.plot(
        argsraw=[], modelpath=modelpath, cell=0, x="time", timedays="270-275", ion_stages=[1, 2], levels=[0, 1]
    )

    # one call draws the full time series of each level. The command called the plot function once
    # per timestep on the same axes, thus it drew each series five times. Each ion stage of -ion_stages
    # gives its own series, and the command dropped Fe II
    assert len(mockplot.call_args_list) == 4
    dfpops = at.nltepops.read_nltepops(modelpath, modelgridindex=0).filter(pl.col("Z") == 26)
    expected_series = [([7.40594, 6.39568], (1, 0)), ([4.71888, 3.89199], (1, 1)), (None, (2, 0)), (None, (2, 1))]
    for callargs, (expected_yarr, (ion_stage, level)) in zip(mockplot.call_args_list, expected_series, strict=True):
        xarr, yarr = get_plot_xy(callargs)
        assert np.allclose(xarr, [271.48221094182054, 273.31529638210384], rtol=1e-4)
        plottedtimesteps = [at.misc.get_timestep_of_timedays(modelpath, timedays) for timedays in xarr]
        filepops = (
            dfpops
            .filter(
                (pl.col("ion_stage") == ion_stage)
                & (pl.col("level") == level)
                & pl.col("timestep").is_in(plottedtimesteps)
            )
            .sort("timestep")["n_NLTE"]
            .to_numpy()
        )
        assert np.allclose(yarr, filepops, rtol=1e-6)
        if expected_yarr is not None:
            assert np.allclose(yarr, expected_yarr, rtol=1e-4)

    assert (tmp_path / "plotnltepops_Fe.pdf").is_file()


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
    # the superlevel takes the position two places above the highest resolved level, which is level 1 here
    assert math.isclose(result.filter(pl.col("level") == 3)["lte_10000"].item(), expected_superlevel, rel_tol=1e-12)


@pytest.mark.parametrize("maxlevel", [-1, 0, 3])
def test_add_lte_pops_matches_a_row_by_row_reference(maxlevel: int) -> None:
    """The vectorised superlevel treatment must give what the loop over each cell, timestep, and ion gave.

    Each ion holds a different number of levels, and each cell holds a different highest level. Thus the
    superlevel of one ion takes a different level number in each cell, and the levels that it sums.
    """
    rng = np.random.default_rng(seed=1)
    k_b = 8.617333262145179e-05
    temperatures = [("lte_5000", 5000), ("lte_12000", 12000.0)]
    nlevels_of_ion = {(26, 1): 5, (26, 2): 7, (28, 2): 4}
    ionlevels_of_ion = {
        ion: pl.DataFrame({
            "g": rng.integers(1, 10, size=nlevels).astype(float),
            "energy_ev": np.sort(np.concatenate([[0.0], rng.uniform(0.1, 5.0, size=nlevels - 1)])),
        })
        for ion, nlevels in nlevels_of_ion.items()
    }
    adata = pl.DataFrame({
        "Z": [Z for Z, _ in ionlevels_of_ion],
        "ion_stage": [ion_stage for _, ion_stage in ionlevels_of_ion],
        "levels": pl.Series(list(ionlevels_of_ion.values()), dtype=pl.Object),
    })

    rows: list[dict[str, t.Any]] = []
    for modelgridindex in (0, 3):
        for timestep in (10, 11):
            for (Z, ion_stage), nlevels in nlevels_of_ion.items():
                # the highest level of the file differs by cell, and one ion of one cell has no superlevel
                toplevel = nlevels - 2 - modelgridindex // 3
                levels = list(range(toplevel + 1))
                if (modelgridindex, ion_stage) != (0, 1):
                    levels.append(-1)
                rows += [
                    {
                        "modelgridindex": modelgridindex,
                        "timestep": timestep,
                        "Z": Z,
                        "ion_stage": ion_stage,
                        "level": level,
                        "n_NLTE": float(rng.uniform(0.0, 1.0)),
                    }
                    for level in levels
                ]
    dfpop = pl.DataFrame(rows)

    def ltepop(ion: tuple[int, int], level: int, T_exc: float) -> float:
        ionlevels = ionlevels_of_ion[ion]
        g0 = float(ionlevels["g"].item(0))
        e0 = float(ionlevels["energy_ev"].item(0))
        g = float(ionlevels["g"].item(level))
        energy_ev = float(ionlevels["energy_ev"].item(level))
        return g / g0 * math.exp(-(energy_ev - e0) / k_b / T_exc)

    expectedrows = []
    for row in dfpop.iter_rows(named=True):
        ion = (row["Z"], row["ion_stage"])
        expected = dict(row)
        maxlevel_ion = max(
            r["level"]
            for r in rows
            if (r["modelgridindex"], r["timestep"], r["Z"], r["ion_stage"])
            == (row["modelgridindex"], row["timestep"], row["Z"], row["ion_stage"])
        )
        levelnumber_sl = maxlevel_ion + 1
        for columnname, T_exc in temperatures:
            if row["level"] == -1:
                expected[columnname] = (
                    sum(ltepop(ion, level, T_exc) for level in range(levelnumber_sl, nlevels_of_ion[ion]))
                    if maxlevel < 0 or levelnumber_sl <= maxlevel
                    else None
                )
            else:
                expected[columnname] = ltepop(ion, row["level"], T_exc)
        if row["level"] == -1:
            # the superlevel takes the position two places above the highest resolved level
            expected["level"] = levelnumber_sl + 1
        expectedrows.append(expected)

    result = at.nltepops.add_lte_pops(dfpop, adata, temperatures, noprint=True, maxlevel=maxlevel)

    assert result.columns == [*dfpop.columns, *(columnname for columnname, _ in temperatures)]
    assert result["level"].to_list() == [row["level"] for row in expectedrows]
    for columnname, _ in temperatures:
        for value, expectedrow in zip(result[columnname].to_list(), expectedrows, strict=True):
            if expectedrow[columnname] is None:
                assert value is None
            else:
                assert math.isclose(value, expectedrow[columnname], rel_tol=1e-12)


@pytest.mark.parametrize("timedays", [300, "300", 300.0])
def test_nltepops_keyword_timedays_reads_a_number(timedays: float | str, tmp_path: Path) -> None:
    """A command line gives a string, and a keyword argument of the API gives a number. Both name one time."""
    at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, modelgridindex=0, timedays=timedays)

    # 300 days is in timestep 54, thus a parse that gives another timestep writes another file name
    assert [path.name.split("_")[3] for path in tmp_path.glob("plotnltepops_Fe_cell00000_*.pdf")] == ["ts054"]


def test_nltepops_timedayslist_plots_only_the_listed_timesteps(tmp_path: Path) -> None:
    """A non-adjacent -timedayslist selects only the listed timesteps, not the range between them."""
    with mock.patch("artistools.nltepops.plotnltepops.make_singletimestep_plot") as mockplot:
        at.nltepops.plot(
            argsraw=[], modelpath=modelpath, outputfile=tmp_path, modelgridindex=0, timedayslist=[255, 340]
        )

    plotted_timesteps = [callargs.args[4] for callargs in mockplot.call_args_list]
    assert plotted_timesteps == [5, 91]


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


def test_plotnltepops_reads_the_folder_after_the_elements(tmp_path: Path) -> None:
    """The ARTIS folder is the last positional argument, as for the other commands.

    "plotnltepops Fe mymodel" read mymodel as a second element and plotted the working folder.
    """
    import artistools.__main__

    outputfile = tmp_path / "nltepops.pdf"
    artistools.__main__.main(argsraw=["plotnltepops", "Fe", str(modelpath), "-t", "300", "-o", str(outputfile)])
    assert outputfile.is_file()


def test_read_nltepops_keeps_the_last_read() -> None:
    """A figure of several subplots and a window read the same populations many times, thus the last read stays.

    Each read parsed the text files again. The read of a large run takes minutes.
    """
    from artistools.nltepops.core import read_nltepops_cached

    read_nltepops_cached.cache_clear()
    firstread = at.nltepops.read_nltepops(modelpath, timestep=50, modelgridindex=[0])
    secondread = at.nltepops.read_nltepops(str(modelpath), timestep=50, modelgridindex=(0,))
    assert secondread is firstread
    assert read_nltepops_cached.cache_info().hits == 1
    assert not firstread.is_empty()


def test_read_nltepops_of_a_numpy_cell_keeps_the_frame_of_the_plot() -> None:
    """A numpy integer names one cell, and a read of the levels for a menu does not drop the frame of the plot.

    The key of the cache took a numpy integer as a list of cells, thus the read stopped with a TypeError.
    """
    from artistools.nltepops.core import read_nltepops_cached

    read_nltepops_cached.cache_clear()
    plotframe = at.nltepops.read_nltepops(modelpath, timestep=np.int64(50), modelgridindex=np.int64(0))
    assert not plotframe.is_empty()
    at.nltepops.read_nltepops(modelpath, modelgridindex=0)
    assert at.nltepops.read_nltepops(modelpath, timestep=50, modelgridindex=0) is plotframe


def test_departure_coefficients_of_a_reference_with_a_different_count_of_levels() -> None:
    """The departure coefficient of the reference populations divides each level by the LTE population of that level.

    The division took the two arrays by position, thus a reference file with a different count of levels stopped the
    plot with a broadcast error.
    """
    import argparse

    import matplotlib.pyplot as plt

    from artistools.nltepops.plotnltepops import plot_reference_populations

    fig, ax = plt.subplots()
    # the superlevel takes the position two places above the highest resolved level
    dfpopthision = pl.DataFrame({
        "level": [0, 1, 2, 4],
        "config": ["3d6.4s", "3d7", "3d6.4s", "superlevel"],
        "n_LTE_T_e_normed": [1.0, 2.0, 4.0, 8.0],
    })
    referencepops = np.full(6, 2.0)
    plot_reference_populations(
        ax, dfpopthision, list(range(6)), referencepops, 5000.0, 5000.0, 1.0, argparse.Namespace(departuremode=True)
    )
    (referenceline,) = (line for line in ax.get_lines() if line.get_label() == "Flörs NLTE")
    assert list(np.asarray(referenceline.get_xdata())) == [0, 1, 2]
    assert np.allclose(np.asarray(referenceline.get_ydata(), dtype=float), [2.0, 1.0, 0.5])
    plt.close(fig)


@pytest.mark.parametrize("ion_stages", ["2", 2, [2]])
def test_nltepops_ion_stages_of_a_keyword_argument(ion_stages: str | int | list[int], tmp_path: Path) -> None:
    """A keyword argument of main() does not pass through the parser, thus -ion_stages can be text, a number, or a list.

    The text reader then took only text, thus a number or a list stopped the command.
    """
    outputfile = tmp_path / "nltepops.pdf"
    with mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True) as mockplot:
        at.nltepops.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputfile, cell=0, timestep=40, ion_stages=ion_stages
        )

    assert outputfile.is_file()
    plottedaxes = {id(callargs[0][0]) for callargs in mockplot.call_args_list}
    assert len(plottedaxes) == 1


def test_add_lte_pops_superlevel_holds_only_the_kept_levels() -> None:
    """ARTIS keeps the first nlevelsmax levels of an ion, thus the superlevel holds no level above them.

    The sum took every level of the atomic data, thus the superlevel had too large an LTE population.
    """
    ionlevels = pl.DataFrame({"g": [2.0, 4.0, 6.0, 8.0, 10.0], "energy_ev": [0.0, 1.0, 2.0, 3.0, 4.0]})
    adata = pl.DataFrame({"Z": [26], "ion_stage": [2], "levels": pl.Series([ionlevels], dtype=pl.Object)})
    dfpop = pl.DataFrame({
        "modelgridindex": [0, 0, 0],
        "timestep": [1, 1, 1],
        "Z": [26, 26, 26],
        "ion_stage": [2, 2, 2],
        "level": [0, 1, -1],
        "n_NLTE": [1.0, 0.3, 0.1],
    })

    k_b = 8.617333262145179e-05
    ltepops = [g / 2.0 * math.exp(-energy_ev / k_b / 10000) for g, energy_ev in ionlevels.iter_rows()]
    for nlevelsmax, expected_superlevel in ((3, ltepops[2]), (-1, sum(ltepops[2:])), (100, sum(ltepops[2:]))):
        result = at.nltepops.add_lte_pops(
            dfpop, adata, [("lte_10000", 10000)], noprint=True, nlevelsmax_of_element={26: nlevelsmax}
        )
        superlevelpop = result.filter(pl.col("level") == 3)["lte_10000"].item()
        assert math.isclose(superlevelpop, expected_superlevel, rel_tol=1e-12), nlevelsmax


def test_add_lte_pops_superlevel_agrees_with_the_nlte_file() -> None:
    """The NLTE file gives the LTE population of each superlevel, thus the sum over the kept levels must agree.

    The run of the test model took the LTE populations at the radiation temperature TJ.
    """
    dfpop = at.nltepops.read_nltepops(modelpath, timestep=40, modelgridindex=0).filter(
        (pl.col("Z") == 26) & pl.col("ion_stage").is_in([1, 2, 3])
    )
    T_J = at.estimators.read_estimators(modelpath, timestep=40, modelgridindex=0)[40, 0]["TJ"]
    compositiondata = at.get_composition_data(modelpath)
    nlevelsmax_of_element = dict(zip(compositiondata["Z"], compositiondata["nlevelsmax_readin"], strict=True))

    result = at.nltepops.add_lte_pops(
        dfpop,
        at.atomic.get_levels(modelpath, quiet=True),
        [("lte_TJ", T_J)],
        noprint=True,
        nlevelsmax_of_element=nlevelsmax_of_element,
    )

    for ion_stage in (1, 2, 3):
        dfion = result.filter(pl.col("ion_stage") == ion_stage)
        groundpop = dfion.filter(pl.col("level") == 0)["n_LTE"].item()
        superlevel = dfion.filter(pl.col("level") == pl.col("level").max())
        assert math.isclose(superlevel["lte_TJ"].item() * groundpop, superlevel["n_LTE"].item(), rel_tol=1e-3)


def test_nltepops_superlevel_takes_the_nlevelsmax_of_compositiondata(tmp_path: Path) -> None:
    """The plot gives add_lte_pops the nlevelsmax of each element of compositiondata.txt."""
    for filepath in modelpath.iterdir():
        if filepath.name != "compositiondata.txt":
            (tmp_path / filepath.name).symlink_to(filepath)
    (tmp_path / "compositiondata.txt").write_text(
        "2\n0\n0\n26  5  1  5  100 0.9 55.8450\n27  3  2  4  -1 0.1 58.9331\n", encoding="utf-8"
    )

    with mock.patch(
        "artistools.nltepops.plotnltepops.add_lte_pops", side_effect=at.nltepops.add_lte_pops
    ) as mockaddltepops:
        at.nltepops.plot(argsraw=[], modelpath=tmp_path, outputfile=tmp_path, timestep=40, ion_stages=2)

    assert mockaddltepops.call_args.kwargs["nlevelsmax_of_element"] == {26: 100, 27: -1}


@pytest.mark.parametrize(
    ("configlist", "expected_marked"),
    [(["", ""], [False, False]), (["a5D", "a5D"], [False, False]), (["3d6_5D", "3d6_3P"], [False, True])],
)
def test_config_labels_of_names_with_no_term(configlist: list[str], expected_marked: list[bool]) -> None:
    """A blank level name or a name with no underscore has no term, thus it never takes the mark of a repeat.

    The labels split each name at the underscore, thus two such names in sequence gave an IndexError.
    """
    labels = at.nltepops.plotnltepops.get_config_labels(configlist)
    assert [label.startswith('" ') for label in labels] == expected_marked


def test_nltepops_ion_stage_of_no_plot_gives_a_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An -ion_stages of no ion of more than one level stopped the command with an IndexError."""
    for ion_stages in ("5", "9"):
        with pytest.raises(SystemExit):
            at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, timestep=40, ion_stages=ion_stages)

        assert "no ion stage of Fe to plot at timestep 40" in capsys.readouterr().err


def test_radiative_decays_take_no_population_from_the_superlevel() -> None:
    """The superlevel row takes a position above the resolved levels, and it is not the level of that position.

    A join on the level number gave the population of the whole superlevel to the level of its position.
    """
    from artistools.nltepops.plotnltepops import get_radiative_decays

    dftransitions = pl.LazyFrame(
        {"lower": [0, 0], "upper": [1, 3], "A": [1.0, 1.0], "epsilon_trans_ev": [1.0, 1.0]},
        schema={"lower": pl.Int32, "upper": pl.Int32, "A": pl.Float32, "epsilon_trans_ev": pl.Float64},
    )
    dfpopthision = pl.DataFrame({"level": [0, 1, 3], "n_NLTE": [5.0, 2.0, 100.0], "config": ["a", "b", "superlevel"]})

    dfdecays = get_radiative_decays(dftransitions, dfpopthision, maxlevel_ion=3)
    assert dfdecays["emissionstrength"].to_list() == pytest.approx([2.0, 0.0])


def test_nltepops_without_axis_labels_gives_no_warning(tmp_path: Path) -> None:
    """-x none hid the tick labels with set_xticklabels, which warns for ticks that are not fixed."""
    import warnings

    with warnings.catch_warnings(record=True) as caughtwarnings:
        warnings.simplefilter("always")
        at.nltepops.plot(argsraw=[], modelpath=modelpath, outputfile=tmp_path, timestep=40, x="none")

    assert not [warning for warning in caughtwarnings if "set_ticklabels" in str(warning.message)]


def make_model_with_floers_data(modeldir: Path) -> None:
    """Make a model folder with the single-zone and the multizone reference populations of Fe II."""
    modeldir.mkdir()
    for filepath in modelpath.iterdir():
        (modeldir / filepath.name).symlink_to(filepath)

    (modeldir / "andreas_level_populations_fe2.txt").write_text(
        "energypercm frac_ionpop\n0.0 0.5\n384.8 0.3\n667.6 0.2\n", encoding="utf-8"
    )
    # the multizone file names the outer velocity of the shell, and the cell of the test model ends at 8000 km/s
    levelcolumns = [f"level{level}" for level in range(100)]
    (modeldir / "level_pops_w7-247d.csv").write_text(
        ",".join(["vel_outer", "Te", "ne", "time", *levelcolumns])
        + "\n"
        + ",".join(["8000.0", "5000.0", "1e5", "247", *(str(100.0 - level) for level in range(100))])
        + "\n",
        encoding="utf-8",
    )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_floers_populations_take_the_population_of_the_whole_ion(
    mockplot: mock.MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference fractions of the ion population take the population of all the levels of the ion.

    The population was the sum over the levels that -maxlevel shows.
    """
    modeldir = tmp_path / "floersmodel"
    make_model_with_floers_data(modeldir)
    (modeldir / "level_pops_w7-247d.csv").unlink()
    monkeypatch.chdir(tmp_path)

    at.nltepops.plot(argsraw=[], modelpath=modeldir, outputfile=tmp_path, timestep=40, ion_stages=2, maxlevel=20)

    (floerscall,) = (callargs for callargs in mockplot.call_args_list if callargs.kwargs.get("label") == "Flörs NLTE")
    ionpopulation = (
        at.nltepops
        .read_nltepops(modelpath, timestep=40, modelgridindex=0)
        .filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))["n_NLTE"]
        .sum()
    )
    _, yarr = get_plot_xy(floerscall)
    assert np.allclose(yarr, np.array([0.5, 0.3, 0.2]) * ionpopulation, rtol=1e-6)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_floers_multizone_populations_take_the_resolved_levels(
    mockplot: mock.MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The multizone file has no superlevel, thus it takes the resolved levels that the plot shows.

    The count of levels included the superlevel row, thus one reference point took the blank position below the
    superlevel. The command also read the file from the working folder alone, and not from the model folder.
    """
    modeldir = tmp_path / "w7_floers"
    make_model_with_floers_data(modeldir)
    (modeldir / "andreas_level_populations_fe2.txt").unlink()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    at.nltepops.plot(argsraw=[], modelpath=modeldir, outputfile=tmp_path, timestep=40, ion_stages=2)

    (floerscall,) = (callargs for callargs in mockplot.call_args_list if callargs.kwargs.get("label") == "Flörs NLTE")
    xarr, yarr = get_plot_xy(floerscall)
    dfresolved = (
        at.nltepops
        .read_nltepops(modelpath, timestep=40, modelgridindex=0)
        .filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2) & (pl.col("level") >= 0))
        .sort("level")
    )
    assert list(xarr) == list(range(dfresolved.height))
    assert np.isclose(yarr.sum(), float(dfresolved["n_NLTE"].sum()), rtol=1e-6)
