import re
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_check_counts_the_cells_at_the_temperature_limits() -> None:
    """The worst time step and the fraction over all time steps agree with a count of the estimators."""
    from artistools.check import get_check_results

    dfresults = get_check_results(modelpath_classic_3d)
    assert dfresults.columns == ["check", "status", "detail"]
    status, detail = dfresults.filter(pl.col("check") == "temperature limits").select("status", "detail").row(0)

    dfte = at.scan_estimators(modelpath_classic_3d).select("timestep", "Te").collect()
    atlimit = dfte["Te"].is_in([3500.0, 140000.0])
    assert dfte.filter(pl.col("timestep") == 0)["Te"].eq(3500.0).sum() == 562
    overallpercent = 100 * float(atlimit.to_numpy().mean())
    assert np.isclose(overallpercent, 54.6, atol=0.05)
    assert status == "warning"
    assert "at the worst time step 0 (3.05 d), of the 695 cells with Te, 562 (80.9%) are at MINTEMP = 3500 K" in detail
    assert f"over all 30 time steps, {overallpercent:.1f}% of the values of Te are at a limit" in detail


def test_check_gives_an_error_for_a_te_that_is_not_finite() -> None:
    """A NaN Te counted as a cell with Te at no limit, thus it lowered the fraction at the limits."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({"timestep": [0, 0, 0, 1], "Te": [3500.0, 5000.0, float("nan"), float("inf")]})
    with mock.patch("artistools.check.scan_estimators", return_value=dfte):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "error"
    assert "of the 2 cells with Te, 1 (50.0%) are at MINTEMP = 3500 K" in detail
    assert "2 values of Te are NaN or infinite" in detail


def test_check_temperature_limits_warns_about_one_bad_timestep() -> None:
    """The mean over 20 time steps was 5% for one time step with each cell at MINTEMP, thus the status was ok."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({
        "timestep": [timestep for timestep in range(20) for _ in range(10)],
        "Te": [3500.0 if timestep == 0 else 5000.0 for timestep in range(20) for _ in range(10)],
    })
    with (
        mock.patch("artistools.check.scan_estimators", return_value=dfte),
        mock.patch("artistools.check.get_run_versions", return_value={"job0": {"git version": "v2023.10"}}),
    ):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "warning"
    assert "over all 20 time steps, 5.0% of the values of Te are at a limit" in detail


def test_check_names_runs_of_other_builds_for_the_temperature_limits() -> None:
    """artisoptions.h gives the limits of the current build, and a run of an earlier build can have other limits."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({"timestep": [0, 0], "Te": [3500.0, 5000.0]})
    versions = {"job0": {"git version": "v2023.10"}, "job1": {"git version": "v2026.5"}}
    with (
        mock.patch("artistools.check.scan_estimators", return_value=dfte),
        mock.patch("artistools.check.get_run_versions", return_value=versions),
    ):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "warning"
    assert "the runs used 2 ARTIS versions (v2023.10, v2026.5)" in detail


def test_check_complete_timesteps_finds_a_missing_row() -> None:
    """The last row gave the last complete time step, thus a row that a bad restart left out gave the status ok."""
    from artistools.check import check_complete_timesteps

    dfdeposition = pl.LazyFrame({"timestep": [0, 2]})
    with mock.patch("artistools.check.scan_deposition", return_value=dfdeposition):
        status, detail = check_complete_timesteps(modelpath_classic_3d)
    assert status == "warning"
    assert "deposition.out has no row for the time steps 1" in detail

    # each row through one time step past the model gave the status ok, and more complete time steps than the model
    ntimesteps = len(at.get_timestep_times(modelpath_classic_3d))
    with mock.patch(
        "artistools.check.scan_deposition", return_value=pl.LazyFrame({"timestep": list(range(ntimesteps + 1))})
    ):
        status, detail = check_complete_timesteps(modelpath_classic_3d)
    assert status == "warning"
    assert f"deposition.out has a row for time step {ntimesteps}" in detail


def test_check_restarts_takes_the_number_of_ranks_from_the_rank_0_log(tmp_path: Path) -> None:
    """A run of 4 ranks that lost one log and a run of 3 ranks each held 3 logs, and the check gave the status ok."""
    from artistools.check import check_restarts

    runfolders = []
    for foldername, rankcount in [("100.slurm", 4), ("200.slurm", 3)]:
        folder = tmp_path / foldername
        folder.mkdir()
        runfolders.append(folder)
        for rank in range(3):
            mpiline = f"  rank {rank} of [0..{rankcount - 1}] in MPI_COMM_WORLD\n"
            (folder / f"output_{rank}-0.txt").write_text(mpiline, encoding="utf-8")
    with (
        mock.patch("artistools.check.get_runfolders", return_value=runfolders),
        mock.patch("artistools.check.get_run_versions", return_value={}),
    ):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "the number of ranks changes from 4 in 100.slurm to 3 in 200.slurm" in detail
    assert "100.slurm has no log of the ranks 3 of its 4 ranks" in detail

    # logs of higher ranks of an earlier run made the count of the logs equal to the count of the ranks
    (runfolders[0] / "output_2-0.txt").unlink()
    for rank in (4, 5):
        (runfolders[0] / f"output_{rank}-0.txt").write_text("", encoding="utf-8")
    with (
        mock.patch("artistools.check.get_runfolders", return_value=runfolders[:1] * 2),
        mock.patch("artistools.check.get_run_versions", return_value={}),
    ):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "100.slurm has no log of the ranks 2-3 of its 4 ranks" in detail
    assert "100.slurm holds logs of the ranks 4-5, above its 4 ranks" in detail


def test_check_restarts_counts_a_run_that_stopped_before_its_estimators(tmp_path: Path) -> None:
    """A run that stops before it writes an estimator has a log, and the check said that the model had no restart."""
    from artistools.check import check_restarts

    for foldername, version in [("100.slurm", "v1"), ("200.slurm", "v2")]:
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "output_0-0.txt").write_text(
            f"git version: {version}\n  rank 0 of [0..0] in MPI_COMM_WORLD\n", encoding="utf-8"
        )
    # only the first run wrote estimators
    with mock.patch("artistools.check.get_runfolders", return_value=[tmp_path / "100.slurm"]):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "2 run folders with estimators or logs (100.slurm, 200.slurm)" in detail
    assert "the ARTIS version changes from v1" in detail


def test_check_restarts_gives_no_rank_count_from_the_names_of_the_logs(tmp_path: Path) -> None:
    """With no line of MPI_COMM_WORLD, the number of the logs stood for the number of ranks, also with a log missing."""
    from artistools.check import check_restarts

    runfolders = []
    for foldername in ("100.slurm", "200.slurm"):
        folder = tmp_path / foldername
        folder.mkdir()
        runfolders.append(folder)
        for rank in range(3):
            (folder / f"output_{rank}-0.txt").write_text("git version: v1\n", encoding="utf-8")
    with mock.patch("artistools.check.get_runfolders", return_value=runfolders):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "gives no line of MPI_COMM_WORLD, thus the number of its ranks is not known" in detail


def test_check_restarts_warns_about_a_run_with_no_version() -> None:
    """A run with no version left the comparison, and the agreement of the other runs then gave the status ok."""
    from artistools.check import check_restarts

    versions = {"job0": {"git version": "v2023.10-106-gfcb7e2c"}}
    with mock.patch("artistools.check.get_run_versions", return_value=versions):
        status, detail = check_restarts(modelpath_classic_3d)
    assert status == "warning"
    assert "job1 gives no ARTIS version, thus the check cannot compare the versions" in detail

    # a branch alone does not name the build, and two runs of one branch gave the status ok
    branches = {"job0": {"git branch": "main"}, "job1": {"git branch": "main"}}
    with mock.patch("artistools.check.get_run_versions", return_value=branches):
        status, detail = check_restarts(modelpath_classic_3d)
    assert status == "warning"
    assert "job0, job1 give no ARTIS version" in detail


def test_check_reads_each_log_once_and_finds_the_restarts() -> None:
    """The model folder holds a copy of the log of job1, thus the log check reads the logs of job0 and job1 alone.

    Almost each run logs warnings, thus a log with no error gives the status ok.
    """
    from artistools.check import get_check_results

    dfresults = get_check_results(modelpath_classic_3d)
    statusofcheck = dict(zip(dfresults["check"], dfresults["status"], strict=True))
    detailofcheck = dict(zip(dfresults["check"], dfresults["detail"], strict=True))
    assert statusofcheck["log messages"] == "ok"
    assert (
        "the rank-0 logs of 2 run folders (job0, job1) hold 0 errors and 7419 warnings" in detailofcheck["log messages"]
    )
    assert statusofcheck["restarts"] == "ok"
    assert "2 run folders with estimators or logs (job0, job1)" in detailofcheck["restarts"]
    assert "v2023.10-106-gfcb7e2c" in detailofcheck["restarts"]
    assert statusofcheck["complete time steps"] == "ok"
    assert "30 of 30 time steps as complete" in detailofcheck["complete time steps"]
    assert statusofcheck["packet noise"] == "not checked"
    assert detailofcheck["packet noise"] == "give --frompackets to read the packets"


def test_check_katz_ratio_agrees_with_a_direct_integration(
    tmp_path: Path, copy_model_inputs: Callable[[Path, Path, Sequence[str]], None]
) -> None:
    """The energy check reads only light_curve.out and deposition.out, thus a model with no estimators also has it.

    The time range of the test model starts at 3 days with hot ejecta, thus the ratio is far from 1 with no fault.
    """
    from artistools.check import get_check_results

    copy_model_inputs(modelpath_classic_3d, tmp_path, ["light_curve.out", "spec.out", "deposition.out"])

    # light_curve.out holds a second table of 30 rows, which repeats the times
    lightcurve = np.loadtxt(modelpath_classic_3d / "light_curve.out")[:30]
    deposition = np.loadtxt(modelpath_classic_3d / "deposition.out")
    lumtime_s = lightcurve[:, 0] * at.constants.day_to_s
    deptime_s = deposition[:, 1] * at.constants.day_to_s
    lumintegral = np.trapezoid(lightcurve[:, 1] * at.constants.Lsun_to_erg_per_s * lumtime_s, x=lumtime_s)
    depintegral = np.trapezoid(deposition[:, 3] * at.constants.Lsun_to_erg_per_s * deptime_s, x=deptime_s)

    for model in (modelpath_classic_3d, tmp_path):
        dfresults = get_check_results(model)
        status, detail = dfresults.filter(pl.col("check") == "energy").select("status", "detail").row(0)
        assert status == "warning"
        assert f"∫ t L dt = {lumintegral:.3e} erg s" in detail
        assert f"∫ t Q dt = {depintegral:.3e} erg s" in detail
        assert f"L/Q = {lumintegral / depintegral:.3f}" in detail
        assert "gives a ratio far from 1 with no fault in the model" in detail

    dfresults = get_check_results(tmp_path)
    statusofcheck = dict(zip(dfresults["check"], dfresults["status"], strict=True))
    assert statusofcheck["restarts"] == "not checked"
    assert statusofcheck["temperature limits"] == "not checked"
    assert statusofcheck["log messages"] == "not checked"

    # a light curve that starts late has no NaN row for its first times, thus the check said that L covers each time
    np.savetxt(tmp_path / "light_curve.out", lightcurve[3:])
    detail = get_check_results(tmp_path).filter(pl.col("check") == "energy")["detail"].item()
    omitteddays = lightcurve[3, 0] - deposition[:, 1].min()
    assert f"leave out {omitteddays:.3g} d of L and 0 d of Q with no value or no row" in detail

    # a light curve with no row at one inner time has no NaN either, and the trapezoid rule spanned the gap
    np.savetxt(tmp_path / "light_curve.out", np.delete(lightcurve, 10, axis=0))
    status, detail = get_check_results(tmp_path).filter(pl.col("check") == "energy").select("status", "detail").row(0)
    assert status == "warning"
    assert f"light_curve.out has no row at 1 mid times of deposition.out, e.g. at {lightcurve[10, 0]:.2f} d" in detail


def test_check_that_fails_gives_an_error_row_and_the_other_checks_run() -> None:
    """An exception in one check gives a row of status error, and the other checks still run."""
    from artistools.check import get_check_results

    with mock.patch("artistools.check.check_temperature_limits", side_effect=RuntimeError("no\nestimators")):
        dfresults = get_check_results(modelpath_classic_3d)
    assert dfresults.height == 9
    assert dfresults.filter(pl.col("check") == "temperature limits").select("status", "detail").row(0) == (
        "error",
        "RuntimeError: no estimators",
    )
    assert dfresults.filter(pl.col("check") == "complete time steps")["status"].item() == "ok"


def test_check_from_packets_gives_the_largest_noise(capsys: pytest.CaptureFixture[str]) -> None:
    """--frompackets gives the largest relative noise of the light curve and its time, with a warning."""
    from artistools.check import main

    dflightcurve = (
        at.lightcurve.scan_lightcurve_from_packets(modelpath)[-1].collect().filter(pl.col("relnoise").is_not_null())
    )
    worst = dflightcurve.sort("relnoise", descending=True).row(0, named=True)
    # a check with a warning gives the exit status 1, as ruff check does for a finding
    with pytest.raises(SystemExit) as exitinfo:
        main(argsraw=[str(modelpath), "--frompackets"])
    assert exitinfo.value.code == 1
    captured = capsys.readouterr()
    assert "\ncheck status detail\n" in captured.out
    noiseline = next(line for line in captured.out.splitlines() if line.startswith('"packet noise"'))
    assert noiseline.startswith('"packet noise" warning ')
    assert f"the largest relative noise is {100 * worst['relnoise']:.1f}% at {worst['time_days']:.2f} d" in noiseline
    assert re.search(r"warning: packet noise \(warning\): the largest relative noise", captured.err)
    # a check that is not checked gives no warning line
    assert "complete time steps (" not in captured.err

    # --exit-zero gives the table and the warnings, and the status 0
    main(argsraw=[str(modelpath), "--frompackets", "--exit-zero"])
    assert "warning: packet noise (warning)" in capsys.readouterr().err


def test_check_packet_noise_names_a_timestep_with_no_escaped_packet() -> None:
    """A time step with no escaped packet has a null noise, and the check left it out and gave the status ok."""
    from artistools.check import check_packet_noise

    dflightcurve = pl.LazyFrame({
        "timestep": [0, 1, 2],
        "time_days": [1.0, 2.0, 3.0],
        "packetcount": [1000, 0, 1000],
        "luminosity_Lsun": [1e8, 0.0, 1e8],
        "relnoise": [0.01, None, 0.01],
    })
    with mock.patch("artistools.check.scan_lightcurve_from_packets", return_value={-1: dflightcurve}):
        status, detail = check_packet_noise(modelpath_classic_3d, frompackets=True)
    assert status == "warning"
    assert "the time steps 1 have no escaped r-packet" in detail


def test_check_names_the_missing_input_files_and_the_leftover_files(tmp_path: Path) -> None:
    """Give a warning for each missing input file and for the files that a stopped conversion left."""
    from artistools.check import check_caches
    from artistools.check import check_input_files

    (tmp_path / "input.txt").write_text("")
    (tmp_path / "model.txt.zst").write_text("")
    (tmp_path / "packets").mkdir()
    (tmp_path / "packets" / ".packetsbatch00_0000_0003.out.parquet.partialab12.partial").write_text("x")

    status, detail = check_input_files(tmp_path)
    assert status == "warning"
    assert "model.txt" not in detail
    assert "no abundances.txt" in detail
    assert "no adata.txt" in detail
    status, detail = check_caches(tmp_path)
    assert status == "warning"
    assert "1 leftover files of conversions that stopped" in detail
