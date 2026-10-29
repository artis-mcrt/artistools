import argparse
import json
import math
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_linefluxes_emfeaturesearch_parsing() -> None:
    """Emission features given on the command line must arrive as tuples of ints, not as raw strings."""
    parser = argparse.ArgumentParser()
    at.plotlinefluxes.addargs(parser)

    args = parser.parse_args(["-emfeaturesearch", "(26, 2, 7155, 7150, 7160)", "(28, 2, 7378, 7373, 7383)"])
    assert args.emfeaturesearch == [(26, 2, 7155, 7150, 7160), (28, 2, 7378, 7373, 7383)]

    # the default must already be usable for the two-feature flux ratio plot
    assert len(parser.parse_args([]).emfeaturesearch) >= 2

    # time bins are floats, not appended lists of strings
    args = parser.parse_args(["-timebins_tstart", "200", "250", "-timebins_tend", "250", "300"])
    assert args.timebins_tstart == [200.0, 250.0]
    assert args.timebins_tend == [250.0, 300.0]

    # unset time bins stay None, so that each model falls back to its own timestep grid
    args = parser.parse_args([])
    assert args.timebins_tstart is None
    assert args.timebins_tend is None

    # the wavelengths may be fractional, but the atomic number, ion stage, and level indices must be integers
    assert parser.parse_args(["-emfeaturesearch", "(26, 2, 12570.5, 12470.5, 12670.5)"]).emfeaturesearch == [
        (26, 2, 12570.5, 12470.5, 12670.5)
    ]

    for badfeature in ("not a tuple", "(26.0, 2, 7155)", "(26, True, 7155)", "(26, 2, 7155, 7150, 7160, 1.5, 2)"):
        with pytest.raises(SystemExit):
            parser.parse_args(["-emfeaturesearch", badfeature])


def test_linefluxes_default_timebins_use_each_models_timesteps() -> None:
    """With no explicit time bins, the packet binning must fall back to the model's own timestep grid."""
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_packets

    emfeatures = [get_closelines(modelpath_classic_3d, 26, 2, 7155, 7100, 7200)]

    dflcdata_default = get_line_luminosities_from_packets("trueemissiontype", emfeatures, modelpath_classic_3d)
    dflcdata_explicit = get_line_luminosities_from_packets(
        "trueemissiontype",
        emfeatures,
        modelpath_classic_3d,
        arr_tstart=at.get_timestep_times(modelpath_classic_3d, loc="start"),
        arr_tend=at.get_timestep_times(modelpath_classic_3d, loc="end"),
    )

    pltest.assert_frame_equal(dflcdata_default, dflcdata_explicit)


def test_linefluxes_timebins_keep_a_rounding_gap_and_drop_a_real_gap() -> None:
    """A packet between two bins that meet within rounding takes a bin, and a packet in a real gap takes none.

    Each bin was [tstart, tstart + twidth), and timesteps.out gives six significant figures. A packet in the
    rounding gap before the next start then had no bin, and the sums did not include it.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [0.9, 1.0, 1.999995, 2.0, 2.9999, 3.0, 3.5, 4.0, 5.0, 5.1]})
    timebins = dftimes.select(get_timebin_expr(pl.col("t"), [1.0, 2.0, 4.0], [1.99999, 3.0, 5.0]))
    assert timebins.to_series().to_list() == [None, 0, 0, 1, 1, None, None, 2, 2, None]


def test_linefluxes_from_pops_takes_the_cell_volumes_of_a_3d_model(tmp_path: Path) -> None:
    """A 3D model has no shell velocities, thus the luminosity takes the volume of each cell.

    The function selected vel_r_min_kmps, which only a 1D model has, and a 3D model stopped with ColumnNotFoundError.
    """
    import itertools

    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    # a 3D grid of 2^3 cells with the files of the 1D test model. Only cell 0 has NLTE populations
    model3dpath = tmp_path / "model3d"
    model3dpath.mkdir()
    for filepath in modelpath.iterdir():
        if filepath.is_file() and filepath.name != "model.txt" and not filepath.name.startswith("."):
            (model3dpath / filepath.name).symlink_to(filepath)
    t_model_days = 0.00115740740741
    vmax = 8.0e8
    cellwidth = vmax * t_model_days * 86400.0
    cellrows = "".join(
        f"{cellid} {x} {y} {z} {10**-0.18}\n1.0 0.9 0.0 0.0 0.0\n"
        for cellid, (z, y, x) in enumerate(itertools.product((-cellwidth, 0.0), repeat=3), start=1)
    )
    (model3dpath / "model.txt").write_text(f"8\n{t_model_days}\n{vmax}\n{cellrows}", encoding="utf-8")

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    arr_tstart = [at.get_timestep_times(modelpath, loc="start")[10]]
    arr_tend = [at.get_timestep_times(modelpath, loc="end")[10]]
    lum1d = get_line_luminosities_from_pops(emfeatures, modelpath, arr_tstart, arr_tend)["Fe II 1-0"].item()
    lum3d = get_line_luminosities_from_pops(emfeatures, model3dpath, arr_tstart, arr_tend)["Fe II 1-0"].item()

    # the cell of the 3D model is a cube of side vmax, and the shell of the 1D model is a sphere of radius vmax
    assert np.isclose(lum3d / lum1d, 3.0 / (4.0 * math.pi), rtol=1e-5)


def test_linefluxes_from_pops_names_a_level_or_a_time_with_no_populations() -> None:
    """A line with no NLTE population at a time gives a message, not a bare AssertionError.

    The test model writes NLTE populations only at even timesteps from 10.
    """
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    with pytest.raises(ValueError, match="hold no value of the upper level 1 at timestep 11"):
        get_line_luminosities_from_pops(
            emfeatures,
            modelpath,
            arr_tstart=[at.get_timestep_times(modelpath, loc="start")[11]],
            arr_tend=[at.get_timestep_times(modelpath, loc="end")[11]],
        )

    with pytest.raises(ValueError, match="holds no Fe II line for the feature 7155"):
        get_closelines(modelpath_classic_3d, 26, 2, 7155, 7155.0, 7155.01)


def test_linefluxes_feature_includes_a_line_at_a_wavelength_bound() -> None:
    """A line at lower_wavelength or upper_wavelength of a feature is part of the feature."""
    from artistools.plotlinefluxes import get_closelines

    linelambda = (
        at.atomic
        .scan_linelist(modelpath_classic_3d)
        .filter(pl.col("atomic_number") == 26, pl.col("ion_stage") == 2)
        .select(pl.col("lambda_angstroms").first())
        .collect()
        .item()
    )
    feature = get_closelines(modelpath_classic_3d, 26, 2, "bound", linelambda, linelambda)
    assert feature.lowestlambda == feature.highestlambda == linelambda


def test_linefluxes_pops_luminosity_matches_a_loop_over_the_cells() -> None:
    """The vectorised sum over the lines and the cells must give what the loop over each cell gave.

    A cell without population data gives its volume to the next cell outward that has data, and the
    outermost empty cells give their volume to no cell. The loop here is the former algorithm.
    """
    from artistools.plotlinefluxes import sum_line_luminosities

    rng = np.random.default_rng(seed=3)
    ncells = 6
    shell_volumes_at_1s = rng.uniform(1e40, 2e40, size=ncells)
    dftimes = pl.DataFrame({"timeindex": [0, 1, 2], "timestep": [4, 7, 7], "t_sec_cubed": [1e21, 8e21, 8e21]})
    dflines = pl.DataFrame({"lineindex": [0, 1], "level": [12, 9], "A": [0.3, 0.02], "delta_ergs": [2e-12, 1.5e-12]})

    # cell 0 and cell 3 hold no data in timestep 4, and the last two cells hold no data in timestep 7
    poprows = [
        (timestep, level, modelgridindex, float(rng.uniform(1e5, 1e6)))
        for timestep, emptycells in ((4, {0, 3}), (7, {4, 5}))
        for level in (12, 9)
        for modelgridindex in range(ncells)
        if modelgridindex not in emptycells
    ]
    dfnltepops = pl.DataFrame(
        poprows,
        schema={"timestep": pl.Int32, "level": pl.Int32, "modelgridindex": pl.Int32, "n_NLTE": pl.Float64},
        orient="row",
    )
    levelpop_of_key = {(timestep, level, mgi): n_nlte for timestep, level, mgi, n_nlte in poprows}

    expected = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        shell_volumes = shell_volumes_at_1s * t_sec_cubed
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            unaccounted_shellvol = 0.0
            for modelgridindex in range(ncells):
                levelpop = levelpop_of_key.get((timestep, level, modelgridindex))
                if levelpop is None:
                    unaccounted_shellvol += shell_volumes[modelgridindex]
                    continue
                expected[timeindex] += (
                    delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                )
                unaccounted_shellvol = 0.0

    lumdata = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=True)

    assert lumdata.shape == expected.shape
    assert np.allclose(lumdata, expected, rtol=1e-13)

    # the cells of a 2D or 3D model have no radial order, thus an empty cell gives its volume to no other cell
    expected_cells = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            for modelgridindex in range(ncells):
                if (levelpop := levelpop_of_key.get((timestep, level, modelgridindex))) is not None:
                    expected_cells[timeindex] += (
                        delta_ergs * A_val * levelpop * shell_volumes_at_1s[modelgridindex] * t_sec_cubed
                    )
    lumdata_cells = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=False)
    assert np.allclose(lumdata_cells, expected_cells, rtol=1e-13)
    assert not np.allclose(lumdata_cells, expected, rtol=1e-6)


def test_linefluxes_rejects_lone_timebin_argument() -> None:
    """Giving only one of the two time bin edge lists must be rejected before any data is read."""
    with pytest.raises(ValueError, match="must be given together"):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], timebins_tstart=[200.0, 250.0])


def test_linefluxes_gives_a_colour_to_each_model_after_the_default_palette() -> None:
    """Each model gets a colour, also after the ten colours of the default palette.

    The command gave None to each model after the tenth, thus --plotemittingregions stopped with an error.
    """
    with mock.patch.object(at.plotlinefluxes, "make_emitting_regions_plot") as mockplot:
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d] * 11, plotemittingregions=True)
    colors = mockplot.call_args.args[0].color
    assert len(colors) == 11
    assert all(color is not None for color in colors)


def test_linefluxes_lineflux_ratio_plot(tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]) -> None:
    """The line flux ratio plot must run with no arguments beyond the model path.

    The plot of Fe II 7155 Å and 12570 Å also draws the fit of Flörs et al. (2020), and --write_data gave no table of
    it.
    """
    # the fit is near 0.02 at the 3 to 7 days of the test model, which the fixed bottom of 0.05 hides
    at.plotlinefluxes.main(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
        ymin=0.001,
        outputfile=tmp_path / "linefluxes.pdf",
        write_data=True,
    )
    assert (tmp_path / "linefluxes.pdf").is_file()
    dfcomparisons = read_data_file(tmp_path / "linefluxes_comparisons.txt")
    assert dfcomparisons["series"].unique().to_list() == ["Flörs+2020 fit"]
    assert dfcomparisons.height == 3


def test_linefluxes_table_of_three_models_with_one_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Three models of one name gave the same column, and the join stopped the plot with a repeated column.

    The axis shows the times from -xmin to -xmax, and the table also gave the points outside that range. The test model
    has no packets of these lines, thus the ratio of each time is NaN, and the test gives each line a luminosity.
    """

    def get_line_luminosities(
        _emtypecolumn: str, emfeatures: Sequence[t.Any], _modelpath: Path, **_kwargs: t.Any
    ) -> pl.DataFrame:
        times = np.linspace(3.0, 8.0, 11)
        return pl.DataFrame({"time": times, **{feature.colname: np.full(len(times), 1e38) for feature in emfeatures}})

    with mock.patch.object(at.plotlinefluxes, "get_line_luminosities_from_packets", get_line_luminosities):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d] * 3,
            emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
            xmin=4.5,
            xmax=6.5,
            ymin=0.001,
            outputfile=tmp_path / "linefluxes.pdf",
            write_data=True,
        )
    assert "The data table gives the column of" in capsys.readouterr().err
    dftable = read_data_file(tmp_path / "linefluxes.txt")
    assert len([column for column in dftable.columns if column.startswith("fratio_")]) == 3
    assert dftable["time"].is_between(4.5, 6.5).all()
    assert dftable.height > 0
    assert read_data_file(tmp_path / "linefluxes_comparisons.txt")["time"].is_between(4.5, 6.5).all()


def test_linefluxes_refuse_two_features_that_share_a_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A packet of a line that both features hold counted in both sums, thus the ratio noise was not correct.

    Two identical features gave a ratio of exactly 1 and a noise above 0.
    """
    with pytest.raises(SystemExit):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 7155, 7100, 7200)],
            outputfile=tmp_path / "linefluxes.pdf",
        )
    assert "of the ratio share" in capsys.readouterr().err
    assert not (tmp_path / "linefluxes.pdf").exists()


def test_linefluxes_floers_ratio_file_with_no_w7_model(tmp_path: Path) -> None:
    """A file with no W7 model gave a frame with no columns, and the concatenation of the comparisons then failed."""
    ratiofile = tmp_path / "floersratios.csv"
    ratiofile.write_text("file,epoch,NIR_VIS_ratio\nfig-nne_Te_allcells-subch-300d.txt,300,1.5\n", encoding="utf-8")
    at.plotlinefluxes.main(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
        floersmodelratiofile=str(ratiofile),
        outputfile=tmp_path / "linefluxes.pdf",
    )
    assert (tmp_path / "linefluxes.pdf").is_file()


def test_linefluxes_refuse_overlapping_time_bins() -> None:
    """Overlapping bins gave each shared interval to the last bin, and each bin still divided by its full width.

    Thus the bins [100, 200) and [150, 250) gave the first bin the packets of 100 to 150 days alone.
    """
    with pytest.raises(ValueError, match="time bins overlap"):
        at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 150.0], [200.0, 250.0])

    # bins that meet at an edge are no overlap, in either order
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 200.0], [200.0, 300.0])
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [200.0, 100.0], [300.0, 200.0])


def test_linefluxes_timebins_in_reverse_order_take_the_order_in_time() -> None:
    """The bins in reverse order must give each time the same bin as the bins in the order of time.

    The upper edge went to the last bin of the list, and a rounding gap closed only between neighbours of the list.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [100.0, 150.0, 200.0, 250.0, 300.0]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [200.0, 100.0], [300.0, 200.0])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        0,
    ]

    dftimes = pl.DataFrame({"t": [1.0, 1.999995, 2.0, 3.0, 3.1]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [2.0, 1.0], [3.0, 1.99999])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        None,
    ]


def test_linefluxes_emitting_regions_give_one_file_for_each_time_bin(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Each time bin of the emitting regions has its own figure, thus two bins can overlap.

    The command wrote each figure to the one name that -o gives, thus only the last one stayed.
    """
    # the test data holds no floers_te_nne.json, thus one reference point stands in for it
    refdata = (["5"], np.array([5.0]), [{"ne": [5.0], "temp": [5000.0]}])
    with mock.patch.object(at.plotlinefluxes, "read_te_nne_refdata", return_value=refdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            use_lastemissiontype=True,
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
        )

    assert sorted(path.name for path in tmp_path.glob("*.pdf")) == ["emreg_5.0d.pdf", "emreg_6.0d.pdf"]
    # the model has no packets in the features, and the table left out the reference curve, thus it gave no file
    dfpoints = read_data_file(tmp_path / "emreg_5.0d.txt")
    assert dfpoints.select("modelpath", "label", "feature", "log10nne", "Te").rows() == [
        (None, "Flörs+2020 +5d", "reference", 5.0, 5000.0)
    ]


def test_linefluxes_emitting_regions_write_the_drawn_points(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--write_data wrote no file for the emitting regions. The 3D test model has no packets in these features.

    --print-data --output-format json printed one JSON document for each time bin, which a parser of JSON refused.
    """
    import collections

    emdata = collections.defaultdict(lambda: {"em_log10nne": np.array([5.0, 6.0]), "em_Te": np.array([4e3, 5e3])})
    with mock.patch.object(at.plotlinefluxes, "get_emitting_regions_data", return_value=emdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            norefdata=True,
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
            print_data=True,
            output_format="json",
        )

    # a direct call of main prints the progress messages to the standard output, thus the JSON starts at "{"
    stdout = capsys.readouterr().out
    document = json.loads(stdout[stdout.index("\n{") :])
    assert {row["figure_time_days"] for row in document["data"]} == {5.0, 6.0}

    # --output-format json also gives each data file the JSON format
    for tmid in (5.0, 6.0):
        dfpoints = pl.DataFrame(json.loads((tmp_path / f"emreg_{tmid:.1f}d.json").read_text(encoding="utf-8"))["rows"])
        assert dfpoints.columns == ["modelpath", "label", "feature", "time_days", "log10nne", "Te"]
        assert dfpoints["time_days"].unique().to_list() == [tmid]
        assert dfpoints["log10nne"].to_list()[:2] == [5.0, 6.0]


def test_linefluxes_emitting_regions_keep_two_models_with_one_label_apart(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The table named each model by its name alone, thus two models with one label gave rows with one key."""
    import collections

    # a second path to the same model folder gives a second model that has the line list
    secondpath = tmp_path / "secondmodel"
    secondpath.symlink_to(modelpath_classic_3d, target_is_directory=True)
    emdata = collections.defaultdict(lambda: {"em_log10nne": np.array([5.0]), "em_Te": np.array([4e3])})
    with mock.patch.object(at.plotlinefluxes, "get_emitting_regions_data", return_value=emdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d, secondpath],
            label=["same", "same"],
            plotemittingregions=True,
            norefdata=True,
            timebins_tstart=[4.0],
            timebins_tend=[6.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
        )

    dfpoints = read_data_file(tmp_path / "emreg.txt")
    assert set(dfpoints["label"].to_list()) == {"same"}
    assert set(dfpoints["modelpath"].to_list()) == {str(modelpath_classic_3d), str(secondpath)}


def test_linefluxes_emitting_regions_take_the_timestep_of_the_thermal_emission() -> None:
    """The cell and the timestep of the emitting regions must come from the same event.

    The cell came from the last thermal emission and the timestep from the last interaction, thus a
    packet that scattered in a later timestep read the estimators of the wrong time.
    """
    dfpackets = pl.LazyFrame({
        "em_timestep": [52, 52, 3],
        "emtrue_timestep": [50, -1, 3],
        "emtrue_modelgridindex": [40, 40, 2],
    })

    assert at.plotlinefluxes.get_emission_columns(dfpackets, "trueemissiontype") == (
        "emtrue_timestep",
        "emtrue_modelgridindex",
    )

    # the last interaction keeps its own timestep and cell
    assert at.plotlinefluxes.get_emission_columns(dfpackets, "emissiontype") == ("em_timestep", "em_modelgridindex")

    # an old packets file has no trueem_time, thus the cell and the timestep both come from the last interaction
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_timestep"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )

    # a 3D packets file with trueem_time and no trueem_posx gives no cell of the thermal emission. The join on
    # emtrue_modelgridindex then stopped with ColumnNotFoundError
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_modelgridindex"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )


def test_linefluxes_emitting_regions_without_the_reference_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The package holds no floers_te_nne.json, thus the message names --norefdata, which plots the model alone.

    Each figure of a time bin has its own legend, and a label that holds LaTeX braces stays as it is.
    str.format read the braces of such a label as a field and raised KeyError, and only the last figure had a legend.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], plotemittingregions=True)
    assert "--norefdata" in capsys.readouterr().err

    # the packets give no position of the thermal emission, thus the cell comes from the last interaction
    with (
        mock.patch.object(at.plotlinefluxes, "set_legend") as mocklegend,
        mock.patch.object(
            at.plotlinefluxes, "plot_nne_te_points", side_effect=at.plotlinefluxes.plot_nne_te_points
        ) as mockpoints,
    ):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            norefdata=True,
            label=["$M_{\\rm ej}$ = 1 at {timeavg:.0f}d"],
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.png",
        )

    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["emreg_5.0d.png", "emreg_6.0d.png"]
    assert mocklegend.call_count == 2, "each figure needs a legend"
    assert [callargs.args[1] for callargs in mockpoints.call_args_list] == [
        "$M_{\\rm ej}$ = 1 at 5d",
        "$M_{\\rm ej}$ = 1 at 6d",
    ]


def test_linefluxes_label_fields_keep_latex_braces() -> None:
    """Only the fields timeavg and modeltag change, thus a LaTeX brace of a label stays."""
    from artistools.plotlinefluxes import format_label_fields

    assert (
        format_label_fields("$M_{\\rm ej}$ {modeltag} {timeavg:.0f}d", timeavg=250.4, modeltag="w7")
        == "$M_{\\rm ej}$ w7 250d"
    )


def test_linefluxes_ratio_noise_adds_the_noise_of_the_two_features_in_quadrature() -> None:
    """Give each feature the noise sqrt(sum(e^2)) / sum(e) of its packets, and the ratio the root of the sum of squares.

    A time bin with no packet of a feature has no noise value, thus its ratio has none either.
    """
    from artistools.plotlinefluxes import add_ratio_noise
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_packets

    features = [
        FeatureTuple(f"feature{index}", "label", 1000.0, [index], 990.0, 1010.0, 26, 2, [1], [0]) for index in (1, 2)
    ]
    dfpackets = pl.LazyFrame({
        "trueemissiontype": [1, 1, 1, 2, 2],
        "e_rf": [1.0, 2.0, 3.0, 4.0, 4.0],
        "t_arrive_d": [1.5, 1.5, 1.5, 1.5, 2.5],
    })
    with mock.patch.object(at.plotlinefluxes, "scan_packets", return_value=(1, dfpackets)):
        dflcdata = get_line_luminosities_from_packets(
            "trueemissiontype", features, modelpath_classic_3d, arr_tstart=[1.0, 2.0], arr_tend=[2.0, 3.0]
        )
    assert np.isclose(dflcdata["feature1_relnoise"][0], math.sqrt(1 + 4 + 9) / 6, rtol=1e-12)
    assert dflcdata["feature1_relnoise"][1] is None
    assert np.allclose(dflcdata["feature2_relnoise"].to_numpy(), [1.0, 1.0], rtol=1e-12, atol=0)

    dfratio = add_ratio_noise(dflcdata.with_columns(fratio=pl.lit(1.0)), ["feature1", "feature2"])
    assert np.isclose(dfratio["fratio_relnoise"][0], math.sqrt((math.sqrt(14) / 6) ** 2 + 1), rtol=1e-12)
    assert dfratio["fratio_relnoise"][1] is None
