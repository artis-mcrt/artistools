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
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def copy_trajectories(tmp_path: Path) -> Path:
    """Return a copy of the folder of the test trajectories.

    A read extracts the members of each tar file next to it, thus a test of the folder of the repository
    extracted them only on the first run, and it wrote outside the folder of the test output.
    """
    import shutil

    trajpath = tmp_path / "trajectories"
    # CodSpeed runs a benchmark test more than one time in one process, with the same tmp_path
    trajpath.mkdir(exist_ok=True)
    for filepath in (at.get_path("testdata") / "kilonova" / "trajectories").iterdir():
        if filepath.is_file() and filepath.name != ".gitignore":
            shutil.copy(filepath, trajpath)

    return trajpath


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_decayproducts(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    trajpath = copy_trajectories(tmp_path)
    at.gsinetwork.decayproducts.main(
        argsraw=[], trajectoryroot=trajpath, tmin=0.1, tmax=0.1, nsteps=1, outputpath=tmp_path
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


def test_decayproducts_parquet_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet files must be written under the requested output path, not a relative 'parquet' folder."""
    trajpath = copy_trajectories(tmp_path)
    outputpath_requested = tmp_path / "requested"
    outputpath_requested.mkdir()

    # run from an empty directory so that a relative "parquet/..." write is visible and cannot hit a leftover folder
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    at.gsinetwork.decayproducts.main(
        argsraw=[],
        trajectoryroot=trajpath,
        tmin=0.1,
        tmax=0.1,
        nsteps=1,
        outputpath=outputpath_requested,
        parquet=True,
        trajparquet=True,
    )

    parquetfiles = sorted(p.name for p in (outputpath_requested / "parquet").glob("*.parquet"))
    assert parquetfiles, "no parquet files written under the output path"
    assert not (cwd / "parquet").exists(), "parquet files must not be written relative to the working directory"


def test_decayproducts_process_trajectory_takes_no_plot_times(tmp_path: Path) -> None:
    """An empty list of plot times gives empty arrays. The index array of the heating rows was a float array."""
    trajpath = copy_trajectories(tmp_path)
    nuc_data = at.gsinetwork.decayproducts.get_nuc_data("Hotokezaka")

    decay_powers = at.gsinetwork.decayproducts.process_trajectory(
        nuc_data=nuc_data,
        traj_root=trajpath,
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
    depdata = at.misc.get_deposition(modelpath_classic_3d).collect()
    assert np.allclose(np.asarray(mockplot.call_args_list[0][0][2]), depdata["Qdot_betaminus_ana_erg/s/g"].to_numpy())


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
    assert np.isclose(abundances[0], abundances[1], rtol=1e-6)


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

    dfnucabund, _ = get_trajectory_timestepfile_nuc_abund(tmp_path, 114511, "./Run_rprocess/nz-plane00223")
    expected_sr = float(dfnucabund.filter(pl.col("Z") == 38)["massfrac"].sum())
    assert expected_sr > 0.0
    assert np.isclose(particledata["Sr"][0][0], expected_sr, rtol=1e-6)


def test_decayproducts_counts_each_trajectory_in_one_ye_bin(tmp_path: Path) -> None:
    """A Ye on the boundary of two bins goes to the upper bin, and a trajectory with no network data is left out.

    The bins included both ends, thus such a trajectory was summed into two bins. A trajectory of summary-all.dat
    with no network data stopped the whole run.
    """
    from artistools.gsinetwork import decayproducts

    trajpath = copy_trajectories(tmp_path)
    summarylines = (trajpath / "summary-all.dat").read_text(encoding="utf-8").splitlines()
    # trajectory 109215 takes Ye 0.25, which is the boundary of the low and the mid bin for -yemax 0.75
    rows = [line.split() for line in summarylines[1:]]
    rows[0][4] = "0.25"
    # a trajectory with no tar file has no network data
    rows.append(["999999", *rows[1][1:]])
    (trajpath / "summary-all.dat").write_text(
        "\n".join([summarylines[0], *(" ".join(row) for row in rows)]) + "\n", encoding="utf-8"
    )

    qdot_of_bin: dict[str, float] = {}

    def record_decay_powers(*callargs: t.Any, outfilepath: Path) -> None:
        # plot_decay_powers takes args, decay_powers, and labelfull by position
        decay_powers: dict[str, np.ndarray] = callargs[1]
        qdot_of_bin[outfilepath.stem.split("_Ye")[-1]] = float(decay_powers["Qdot"][0])

    with mock.patch.object(decayproducts, "plot_decay_powers", side_effect=record_decay_powers):
        decayproducts.main(
            argsraw=[], trajectoryroot=trajpath, tmin=0.1, tmax=0.1, nsteps=1, yemax=0.75, outputpath=tmp_path
        )

    # the low bin holds no trajectory below Ye 0.25, thus the trajectory on the boundary is in the mid bin alone
    assert qdot_of_bin.keys() == {"all", "mid", "high"}
    assert qdot_of_bin["all"] > 0.0
    assert math.isclose(qdot_of_bin["mid"] + qdot_of_bin["high"], qdot_of_bin["all"], rel_tol=1e-9)


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
    from artistools.constants import MH_g
    from artistools.gsinetwork import comparetogsinetwork

    dfestimators = make_estimators_of_strontium(tmp_path)
    with mock.patch.object(comparetogsinetwork, "scan_estimators", return_value=dfestimators):
        abund_of_mgi = comparetogsinetwork.get_artis_abund_sequences(
            tmp_path, pl.DataFrame({"timestep": [0]}), [0], ["Sr", "Sr88"], {"Sr89": 1.5}
        )

    expected_sr88 = 1.0e12 * 88 * MH_g / 1.0e-10
    expected_sr89 = 3.0e11 * 89 * MH_g / 1.0e-10 + 0.1 * (1.5 - 1.0)
    expected_otherstable = 2.0e12 * 87.62 * MH_g / 1.0e-10
    assert math.isclose(abund_of_mgi[0]["X_Sr88"].item(), expected_sr88, rel_tol=1e-9)
    assert math.isclose(
        abund_of_mgi[0]["X_Sr"].item(), expected_sr88 + expected_sr89 + expected_otherstable, rel_tol=1e-9
    )


def test_artis_abundance_of_a_run_with_no_compositiondata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A run with no compositiondata.txt still gives each curve, and an element leaves out its other stable isotopes.

    The abundances read compositiondata.txt for every species, thus a missing file gave no ARTIS curve at all.
    output_0-0.txt gives no masses of the elements, thus it is no replacement for the file.
    """
    from artistools.constants import MH_g
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

    expected_sr88 = 1.0e12 * 88 * MH_g / 1.0e-10
    expected_sr89 = 3.0e11 * 89 * MH_g / 1.0e-10 + 0.1 * (1.5 - 1.0)
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
