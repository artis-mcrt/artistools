import math
import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest

import artistools as at
from artistools.constants import day_to_s

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_decayproducts(mockplot: mock.MagicMock, tmp_path: Path, trajectory_copy: Path) -> None:
    at.gsinetwork.decayproducts.main(
        argsraw=[], trajectoryroot=trajectory_copy, timemin=0.1, timemax=0.1, nsteps=1, outputpath=tmp_path
    )

    expected_y_arrays = [
        [0.45123906],
        [0.20325522],
        [0.34550572],
        [6.40901247e39],
        [5.58169828e39],
        [5.76986032e39],
        [0.43677441],
        [0.21151689],
        [0.35170869],
        [6.03183992e39],
        [5.3560247e39],
        [5.54429711e39],
        [0.79453598],
        [0.00717668],
        [0.19828734],
        [3.77172547e38],
        [2.25673577e38],
        [2.25563208e38],
    ]
    for x, expected_y_arr in zip(mockplot.call_args_list, expected_y_arrays, strict=True):
        x_arr = x[0][1]
        assert len(x_arr) == 1
        assert math.isclose(x_arr[0], 0.1, rel_tol=1e-3)
        y_arr = x[0][2]
        assert len(y_arr) == 1
        assert math.isclose(y_arr[0], expected_y_arr[0], rel_tol=1e-3)


def test_decayproducts_parquet_output(tmp_path: Path, trajectory_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet files must be written under the requested output path, not a relative 'parquet' folder."""
    outputpath_requested = tmp_path / "requested"
    outputpath_requested.mkdir()

    # run from an empty directory so that a relative "parquet/..." write is visible and cannot hit a leftover folder
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    at.gsinetwork.decayproducts.main(
        argsraw=[],
        trajectoryroot=trajectory_copy,
        timemin=0.1,
        timemax=0.1,
        nsteps=1,
        outputpath=outputpath_requested,
        parquet=True,
        trajparquet=True,
    )

    parquetfiles = sorted(p.name for p in (outputpath_requested / "parquet").glob("*.parquet"))
    assert parquetfiles, "no parquet files written under the output path"
    assert not (cwd / "parquet").exists(), "parquet files must not be written relative to the working directory"


def test_decayproducts_process_trajectory_reads_the_row_of_the_timestep(
    tmp_path: Path, write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path]
) -> None:
    """A gap in the timesteps must not move the heating values, because the row index is not the timestep minus one.

    The old code read row nstep - 1, which an archive with a gap or a left-out timestep does not have.
    """
    # the reader asserts more than 100 nuclides in a file
    nuclidelines = "".join(f"{nneutrons:4d}{1:4d}{-20.0:13.5E}  0.00000E+00  0.00000E+00\n" for nneutrons in range(120))
    steps = {1: 1.0e4, 2: 2.0e4, 4: 4.0e4}
    members = {
        "energy_thermo.dat": "#count time/s Qdot\n"
        + "".join(f"{nstep:6d}  {time_s:.6E}  {nstep:.6E}\n" for nstep, time_s in steps.items()),
        "heating.dat": "#count time/s hbeta htot\n"
        + "".join(
            f"{nstep:6d}  {time_s:.6E}  {10.0 * nstep:.6E}  {100.0 * nstep:.6E}\n" for nstep, time_s in steps.items()
        ),
    } | {
        f"nz-plane{nstep:05d}": f"  {nstep}  {time_s:.6E}   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n" + nuclidelines
        for nstep, time_s in steps.items()
    }
    write_trajectory_archive(tmp_path, 9, members, ".tar.xz")

    decay_powers = at.gsinetwork.decayproducts.process_trajectory(
        nuc_data=at.gsinetwork.decayproducts.get_nuc_data("Hotokezaka"),
        traj_root=tmp_path,
        traj_masses_g={9: 1.0},
        # a time between the timesteps 2 and 4, and nearer to 4
        arr_t_day=np.array([3.9e4 / day_to_s]),
        nuclide_contrib=False,
        traj_parquet_dir=None,
        traj_ID=9,
    )

    assert decay_powers is not None
    assert decay_powers["hbeta"].tolist() == pytest.approx([40.0])
    assert decay_powers["htot"].tolist() == pytest.approx([400.0])
    assert decay_powers["Qdot"].tolist() == pytest.approx([4.0])

    # a heating.dat with a header alone gives no heating row, and the sums leave out the trajectory
    write_trajectory_archive(tmp_path, 10, members | {"heating.dat": "#count time/s hbeta htot\n"}, ".tar.xz")
    assert (
        at.gsinetwork.decayproducts.process_trajectory(
            nuc_data=at.gsinetwork.decayproducts.get_nuc_data("Hotokezaka"),
            traj_root=tmp_path,
            traj_masses_g={10: 1.0},
            arr_t_day=np.array([3.9e4 / day_to_s]),
            nuclide_contrib=False,
            traj_parquet_dir=None,
            traj_ID=10,
        )
        is None
    )


def test_decayproducts_process_trajectory_takes_no_plot_times(trajectory_copy: Path) -> None:
    """An empty list of plot times gives empty arrays. The index array of the heating rows was a float array."""
    nuc_data = at.gsinetwork.decayproducts.get_nuc_data("Hotokezaka")

    decay_powers = at.gsinetwork.decayproducts.process_trajectory(
        nuc_data=nuc_data,
        traj_root=trajectory_copy,
        traj_masses_g={109215: 1.0e30},
        arr_t_day=np.array([]),
        nuclide_contrib=False,
        traj_parquet_dir=None,
        traj_ID=109215,
    )

    assert decay_powers is not None
    assert all(len(values) == 0 for values in decay_powers.values())


def test_electroncapture_betaplus_energies_sum_to_the_q_value() -> None:
    """Each electron capture and beta-plus row must split Q between the gamma, the electron, and the neutrino.

    The Mn52 gamma energy was 5.857 MeV, which was above the Q value of 4.711 MeV.
    """
    import polars as pl

    from artistools.gsinetwork.decayproducts import append_electroncapture_betaplus_nuclei

    colnames = ["A", "Z", "Q[MeV]", "Egamma[MeV]", "Eelec[MeV]", "Eneutrino[MeV]", "tau[s]"]
    dfnuc = append_electroncapture_betaplus_nuclei(pl.DataFrame({name: [] for name in colnames}), "Hotokezaka")

    assert dfnuc.height == 7

    residual = dfnuc.select(
        (pl.col("Q[MeV]") - pl.col("Egamma[MeV]") - pl.col("Eelec[MeV]") - pl.col("Eneutrino[MeV]")).abs().max()
    ).item()
    assert residual < 1e-3


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_comparetogsinetwork_plots_the_global_qdot_with_no_cell(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """With no -modelgridindex, the command plots the global Qdot of ARTIS into the folder of -o.

    The empty default reached parse_range_list, which stopped the command on int(""). The command also wrote the
    plot into the model folder, and it did not read -o.
    """
    at.gsinetwork.plot(argsraw=[], modelpath=modelpath_classic_3d, nogsinet=True, outputfile=tmp_path / "gsiout")

    assert (tmp_path / "gsiout" / "gsinetwork_global-qdot.pdf").is_file()
    labels = [callargs.kwargs.get("label") for callargs in mockplot.call_args_list]
    assert labels == [r"$\dot{Q}_\beta$ ARTIS", r"$\dot{Q}_\alpha$ ARTIS"]
    depdata = at.misc.scan_deposition(modelpath_classic_3d).collect()
    assert np.allclose(np.asarray(mockplot.call_args_list[0][0][2]), depdata["Qdot_betaminus_ana_erg/s/g"].to_numpy())


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_comparetogsinetwork_write_data_gives_the_drawn_series(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """--write_data gives one row for each drawn point of the Qdot figure and of the abundance figure of a cell.

    The command had no data flags, and the rule of AGENTS.md gives them to each plot command.
    """
    cell = 211
    at.gsinetwork.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        nogsinet=True,
        modelgridindex=[cell],
        species=["Fe", "Ni56"],
        outputfile=tmp_path,
        write_data=True,
    )

    def read_table(path: Path) -> pl.DataFrame:
        return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")

    # each heating rate of the test model is zero
    assert (read_table(tmp_path / "gsinetwork_global-qdot.txt")["qdot_erg_per_s_per_g"] == 0).all()
    dfabund = read_table(tmp_path / f"gsinetwork_mgi{cell}-abundance.txt")
    drawncount = 0
    for callargs in mockplot.call_args_list:
        _, drawnx, drawny = callargs.args
        if "dot{Q}" in callargs.kwargs["label"]:
            continue
        # each series has its own rows, thus the rows of the series follow the order of the plot calls
        dfseries = dfabund.slice(drawncount, len(drawnx))
        assert np.allclose(dfseries["time_days"].to_numpy(), np.asarray(drawnx), rtol=1e-12, atol=0.0)
        assert np.allclose(dfseries["massfraction"].to_numpy(), np.asarray(drawny), rtol=1e-12, atol=0.0)
        drawncount += len(drawnx)
    assert drawncount == dfabund.height > 0
    assert set(dfabund["series"]) <= {"Fe", "Ni56"}
    assert set(dfabund["source"]) == {"ARTIS"}


def test_particledata_holds_the_last_abundance_after_the_network_ends(tmp_path: Path) -> None:
    """A time after the last network step has no step above it, and the abundance must then stay at its end value.

    get_closest_network_timesteps gives None for such a time, and the sort of the steps stopped with a TypeError.
    """
    import shutil

    from artistools.gsinetwork import comparetogsinetwork

    shutil.copy(at.get_path("testdata") / "kilonova" / "trajectories" / "114511.tar.xz", tmp_path)
    # the test tar holds the abundances of step 223 alone, thus the network ends at that step here
    dfsteps = at.inputmodel.rprocess_from_trajectory.get_traj_network_timesteps(tmp_path, 114511)
    laststep_time_s = dfsteps.filter(pl.col("nstep") == 223)["timesec"].item()

    def get_steps_to_223(
        traj_root: Path, particleid: int, timesec: list[float], cond: str = "nearest"
    ) -> list[int | None]:
        assert (traj_root, particleid) == (tmp_path, 114511)
        return [223 if cond == "lessorequal" else None for _ in timesec]

    with mock.patch.object(comparetogsinetwork, "get_closest_network_timesteps", side_effect=get_steps_to_223):
        particledata = comparetogsinetwork.get_particledata(
            [laststep_time_s * 2, laststep_time_s * 10], [("Sr", 38, None)], tmp_path, 114511
        )

    abundances = particledata["Sr"][0].to_numpy()
    assert abundances[0] > 0.0
    assert np.isclose(abundances[0], abundances[1], rtol=1e-6, atol=0.0)


def test_particledata_reads_the_exact_step_at_the_time_of_a_network_step(tmp_path: Path) -> None:
    """A time that equals the time of a network step must read the abundances of that step alone.

    The bracket held only the steps strictly before and after the time, thus it read the two neighbour steps.
    """
    import shutil

    from artistools.gsinetwork import comparetogsinetwork
    from artistools.inputmodel.rprocess_from_trajectory import get_traj_network_timesteps
    from artistools.inputmodel.rprocess_from_trajectory import get_trajectory_timestepfile_nuc_abund

    shutil.copy(at.get_path("testdata") / "kilonova" / "trajectories" / "114511.tar.xz", tmp_path)
    # the test tar holds the abundances of step 223 alone, and not the abundances of steps 222 and 224
    dfsteps = get_traj_network_timesteps(tmp_path, 114511)
    step_time_s = dfsteps.filter(pl.col("nstep") == 223)["timesec"].item()

    particledata = comparetogsinetwork.get_particledata([step_time_s], [("Sr", 38, None)], tmp_path, 114511)

    dfnucabund, _ = get_trajectory_timestepfile_nuc_abund(tmp_path, 114511, "nz-plane00223")
    expected_sr = float(dfnucabund.filter(pl.col("Z") == 38)["massfrac"].sum())
    assert expected_sr > 0.0
    assert np.isclose(particledata["Sr"][0][0], expected_sr, rtol=1e-6, atol=0.0)


def test_decayproducts_counts_each_trajectory_in_one_ye_bin(tmp_path: Path, trajectory_copy: Path) -> None:
    """A Ye on the boundary of two bins goes to the upper bin, and a trajectory with no network data is left out.

    The bins included both ends, thus such a trajectory was summed into two bins. A trajectory of summary-all.dat
    with no network data stopped the whole run.
    """
    from artistools.gsinetwork import decayproducts

    summarylines = (trajectory_copy / "summary-all.dat").read_text(encoding="utf-8").splitlines()
    # trajectory 109215 takes Ye 0.25, which is the boundary of the low and the mid bin for -yemax 0.75
    rows = [line.split() for line in summarylines[1:]]
    rows[0][4] = "0.25"
    # a trajectory with no tar file has no network data
    rows.append(["999999", *rows[1][1:]])
    (trajectory_copy / "summary-all.dat").write_text(
        "\n".join([summarylines[0], *(" ".join(row) for row in rows)]) + "\n", encoding="utf-8"
    )

    qdot_of_bin: dict[str, float] = {}

    def record_decay_powers(*callargs: t.Any, outfilepath: Path) -> None:
        # plot_decay_powers takes args, decay_powers, and labelfull by position
        decay_powers: dict[str, np.ndarray] = callargs[1]
        qdot_of_bin[outfilepath.stem.split("_Ye")[-1]] = float(decay_powers["Qdot"][0])

    with mock.patch.object(decayproducts, "plot_decay_powers", side_effect=record_decay_powers):
        decayproducts.main(
            argsraw=[],
            trajectoryroot=trajectory_copy,
            timemin=0.1,
            timemax=0.1,
            nsteps=1,
            yemax=0.75,
            outputpath=tmp_path,
        )

    # the low bin holds no trajectory below Ye 0.25, thus the trajectory on the boundary is in the mid bin alone
    assert qdot_of_bin.keys() == {"all", "mid", "high"}
    assert qdot_of_bin["all"] > 0.0
    assert math.isclose(qdot_of_bin["mid"] + qdot_of_bin["high"], qdot_of_bin["all"], rel_tol=1e-9)


def test_decayproducts_skips_a_ye_bin_with_no_network_data(
    tmp_path: Path, trajectory_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Ye bin that holds only trajectories with no network data gives no plot and a warning.

    The sum of no trajectory gave zero rates, and the ratios of the plot then divided zero by zero.
    """
    from artistools.gsinetwork import decayproducts

    summarylines = (trajectory_copy / "summary-all.dat").read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in summarylines[1:]]
    # trajectory 109215 moves to the mid bin. A trajectory with no tar file has no network data, and Ye 0.05 puts it
    # alone in the low bin
    rows[0][4] = "0.3"
    rows.append(["999999", *rows[0][1:4], "0.05", *rows[0][5:]])
    (trajectory_copy / "summary-all.dat").write_text(
        "\n".join([summarylines[0], *(" ".join(row) for row in rows)]) + "\n", encoding="utf-8"
    )

    plottedbins: list[str] = []

    def record_bin(*callargs: t.Any, outfilepath: Path) -> None:
        del callargs
        plottedbins.append(outfilepath.stem.split("_Ye")[-1])

    with mock.patch.object(decayproducts, "plot_decay_powers", side_effect=record_bin):
        decayproducts.main(
            argsraw=[],
            trajectoryroot=trajectory_copy,
            timemin=0.1,
            timemax=0.1,
            nsteps=1,
            yemax=0.75,
            outputpath=tmp_path,
        )

    assert "low" not in plottedbins
    assert "all" in plottedbins
    assert "has network data" in capsys.readouterr().err


def make_estimators_of_strontium(tmp_path: Path) -> pl.LazyFrame:
    """Return the estimators of one cell that hold an isotope with no init_X column and the other stable isotopes."""
    (tmp_path / "compositiondata.txt").write_text("1\n0\n0\n38 3 1 3 -1 0.0 87.62\n", encoding="utf-8")
    return pl.LazyFrame({
        "modelgridindex": [0],
        "timestep": [0],
        "tmid_days": [1.0],
        "mass_g": [1.0e30],
        "rho": [1.0e-10],
        "nniso_Sr88": [1.0e12],
        "nniso_Sr_otherstable": [2.0e12],
        "init_X_Sr89": [0.1],
        "nniso_Sr89": [3.0e11],
    })


def test_artis_abundance_of_an_element_takes_every_isotope(tmp_path: Path) -> None:
    """The mass fraction of an element sums each isotope and the other stable isotopes of the element.

    A decay daughter that model.txt does not hold has no init_X column, and its term stopped the plot. The sum
    also left out the other stable isotopes, which ARTIS gives as <El>_otherstable.
    """
    from artistools.constants import amu_g
    from artistools.gsinetwork import comparetogsinetwork

    dfestimators = make_estimators_of_strontium(tmp_path)
    with mock.patch.object(comparetogsinetwork, "scan_estimators", return_value=dfestimators):
        abund_of_mgi = comparetogsinetwork.get_artis_abund_sequences(
            tmp_path, pl.DataFrame({"timestep": [0]}), [0], ["Sr", "Sr88"], {"Sr89": 1.5}
        )

    expected_sr88 = 1.0e12 * 88 * amu_g / 1.0e-10
    expected_sr89 = 3.0e11 * 89 * amu_g / 1.0e-10 + 0.1 * (1.5 - 1.0)
    expected_otherstable = 2.0e12 * 87.62 * amu_g / 1.0e-10
    assert math.isclose(abund_of_mgi[0]["X_Sr88"].item(), expected_sr88, rel_tol=1e-9)
    assert math.isclose(
        abund_of_mgi[0]["X_Sr"].item(), expected_sr88 + expected_sr89 + expected_otherstable, rel_tol=1e-9
    )


def test_artis_abundance_converts_nniso_with_the_atomic_mass_unit(tmp_path: Path) -> None:
    """A cell that holds only Sr88 gives a mass fraction of 1.

    After artis-mcrt/artis#670, ARTIS converts a mass number to grams with the atomic mass unit. The mass of the
    hydrogen atom gave a mass fraction that was 0.78% too large.
    """
    from artistools.constants import amu_g
    from artistools.gsinetwork import comparetogsinetwork

    rho = 1.0e-10
    dfestimators = pl.LazyFrame({
        "modelgridindex": [0],
        "timestep": [0],
        "tmid_days": [1.0],
        "mass_g": [1.0e30],
        "rho": [rho],
        "nniso_Sr88": [rho / (88 * amu_g)],
    })
    with mock.patch.object(comparetogsinetwork, "scan_estimators", return_value=dfestimators):
        abund_of_mgi = comparetogsinetwork.get_artis_abund_sequences(
            tmp_path, pl.DataFrame({"timestep": [0]}), [0], ["Sr88"], {}
        )

    assert math.isclose(abund_of_mgi[0]["X_Sr88"].item(), 1.0, rel_tol=1e-9)


def test_artis_abundance_of_a_model_with_no_compositiondata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A model with no compositiondata.txt still gives each curve, and an element leaves out its other stable isotopes.

    The abundances read compositiondata.txt for every species, thus a missing file gave no ARTIS curve at all.
    output_0-0.txt gives no masses of the elements, thus it is no replacement for the file.
    """
    from artistools.constants import amu_g
    from artistools.gsinetwork import comparetogsinetwork

    dfestimators = make_estimators_of_strontium(tmp_path)
    (tmp_path / "compositiondata.txt").unlink()
    (tmp_path / "output_0-0.txt").write_text(
        "[info]  element 0 (Z=38 Sr)\n[info]    ionstage 1:  50 levels ( 50 ionising)\nend of the list\n",
        encoding="utf-8",
    )
    with mock.patch.object(comparetogsinetwork, "scan_estimators", return_value=dfestimators):
        abund_of_mgi = comparetogsinetwork.get_artis_abund_sequences(
            tmp_path, pl.DataFrame({"timestep": [0]}), [0], ["Sr", "Sr88"], {"Sr89": 1.5}
        )

    expected_sr88 = 1.0e12 * 88 * amu_g / 1.0e-10
    expected_sr89 = 3.0e11 * 89 * amu_g / 1.0e-10 + 0.1 * (1.5 - 1.0)
    assert math.isclose(abund_of_mgi[0]["X_Sr88"].item(), expected_sr88, rel_tol=1e-9)
    assert math.isclose(abund_of_mgi[0]["X_Sr"].item(), expected_sr88 + expected_sr89, rel_tol=1e-9)
    assert "leaves out the stable isotopes" in capsys.readouterr().err


def test_comparetogsinetwork_takes_particles_with_no_network_data(tmp_path: Path) -> None:
    """The pairs of particle and cell are empty if no particle has network data, and the plots then show ARTIS alone.

    The concat of the frames of such particles had no particleid column, thus the join stopped the command.
    """
    from artistools.gsinetwork import comparetogsinetwork

    dfcontributions = pl.DataFrame({"particleid": [1, 2], "cellindex": [1, 1], "frac_of_cellmass": [0.5, 0.5]})
    lzdfmodel = pl.LazyFrame({"modelgridindex": [0], "cellmass_on_mtot": [1.0]})

    def map_in_sequence(function: t.Any, items: t.Any, **_kwargs: t.Any) -> list[t.Any]:
        return [function(item) for item in items]

    with (
        mock.patch.object(comparetogsinetwork, "get_merger_time_geomunits", return_value=0.0),
        mock.patch.object(comparetogsinetwork, "get_gridparticlecontributions", return_value=dfcontributions),
        mock.patch.object(comparetogsinetwork, "parallel_map", side_effect=map_in_sequence),
    ):
        dfpairs, dfparticledata = comparetogsinetwork.get_dfcontribsparticledata(
            modelpath=tmp_path,
            mgiplotlist=[],
            arr_strnuc_z_n=[],
            traj_root=tmp_path,
            griddata_root=tmp_path,
            lzdfmodel=lzdfmodel,
            arr_time_gsi_days=[1.0],
        )

    assert dfpairs.is_empty()
    assert dfparticledata.is_empty()


def test_abundance_correction_factors_need_the_grid_file_of_a_1d_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 1D model with no grid.out took a direct map, thus it lost the factors of a map to a 3D grid with no warning.

    A 3D model grid is also the propagation grid, thus such a model needs no grid.out.
    """
    get_abundance_correction_factors = at.gsinetwork.comparetogsinetwork.get_abundance_correction_factors
    modelpath_1d = at.get_path("testdata") / "test-classicmode_1d"
    lzdfmodel, modelmeta = at.scan_modeldata(modelpath_1d)
    with pytest.raises(SystemExit):
        get_abundance_correction_factors(lzdfmodel, [0], ["Fe"], modelpath_1d, modelmeta)
    assert "grid.out does not exist" in capsys.readouterr().err

    for filepath in modelpath_classic_3d.iterdir():
        if filepath.name != "grid.out":
            (tmp_path / filepath.name).symlink_to(filepath)
    lzdfmodel, modelmeta = at.scan_modeldata(tmp_path)
    assert get_abundance_correction_factors(lzdfmodel, [211], ["Fe"], tmp_path, modelmeta) == {}
