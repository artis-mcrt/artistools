import argparse
import io
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_logfiles(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Log file timings are parsed for every stage and rank, and plotted one page per timestep."""
    logfilepaths = at.logs.get_logfiles(Path(modelpath_classic_3d), None)
    # compressed log files must be read too, not skipped
    assert sorted(path.name for path in logfilepaths) == [
        "output_0-0.txt",
        "output_0-0.txt",
        "output_1-0.txt.zst",
        "output_1-0.txt.zst",
    ]

    timetaken = at.logs.core.read_time_taken(logfilepaths)
    assert set(timetaken) == {"update_grid", "update_packets", "communicate_estimators"}
    for stage, bytimestep in timetaken.items():
        assert len(bytimestep) == 30, f"expected 30 timesteps of {stage} timings"
        for byrank in bytimestep.values():
            assert set(byrank) == {0, 1}, f"expected both mpi ranks for {stage}"
    assert timetaken["update_grid"][0] == {0: 2, 1: 2}
    assert timetaken["update_packets"][0] == {0: 1, 1: 1}
    assert timetaken["communicate_estimators"][2] == {0: 1, 1: 1}

    at.logs.plotlogfiles.main(argsraw=[], modelpath=[modelpath_classic_3d], outputfile=tmp_path / "logfiles.pdf")

    # one line per stage on each of the 30 per-timestep pages
    assert len(mockplot.call_args_list) == 3 * 30


def test_logfiles_read_the_old_and_the_current_log_format(tmp_path: Path) -> None:
    """The timings of the current ARTIS log lines and of the old log lines must both be read.

    The patterns held the Unix time and the whole seconds of the old lines. The current lines hold no Unix time,
    the grid line says "on all processes", and the seconds have decimals. Thus the command found no timing data.
    """
    (tmp_path / "output_0-0.txt").write_text(
        "2026-10-04T12:00:00Z timestep 3: time after update grid on all processes"
        " (rank 0 took 1.2s, waited 0.3s, total 1.5s)\n"
        "2026-10-04T12:00:05Z timestep 3: time after update packets for all processes"
        " (rank 0 took 3.5s, waited 0.1s, total 3.6s)\n"
        "2026-10-04T12:00:06Z timestep 3: time after estimators have been communicated (took 0.4 seconds)\n"
        "2023-11-14T15:01:59Z timestep 2: time after update grid for all processes 1699974119"
        " (rank 0 took 2s, waited 0s, total 2s)\n"
        "2023-11-14T15:02:09Z timestep 2: time after update packets for all processes 1699974129"
        " (rank 0 took 7s, waited 0s, total 7s)\n"
        "2023-11-14T15:02:10Z timestep 2: time after estimators have been communicated 1699974130 (took 1 seconds)\n",
        encoding="utf-8",
    )

    timetaken = at.logs.core.read_time_taken([tmp_path / "output_0-0.txt"])

    assert timetaken["update_grid"] == {3: {0: pytest.approx(1.2)}, 2: {0: pytest.approx(2.0)}}
    assert timetaken["update_packets"] == {3: {0: pytest.approx(3.5)}, 2: {0: pytest.approx(7.0)}}
    assert timetaken["communicate_estimators"] == {3: {0: pytest.approx(0.4)}, 2: {0: pytest.approx(1.0)}}


def test_corehours_counts_the_link_to_a_log_one_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The model folder holds a link to the rank-0 log of the last run, and the sum counts that run one time.

    The second run stopped early and has no last line with its core hours, thus its value is an estimate from the
    timestamps: 1 hour * 4 processes * 1 thread.
    """
    import json

    import artistools.__main__

    modelfolder = tmp_path / "model"
    nprocsline = "Simulation propagates 100 packets on this rank (total 400 with nprocs 4)"
    logtext_of_runfolder = {
        "job_from_ts0000": (
            f"2026-10-15T09:00:00Z sn3d launched\n2026-10-15T09:00:00Z {nprocsline}\n"
            "2026-10-15T11:00:00Z sn3d finished (job: pktprop ts 0 to ts 9 grid-preprop, 2.000 wallclock hours"
            " * 4 processes * 1 threads = 8.000 core hours)\n"
        ),
        "job_from_ts0010": (
            f"2026-10-15T12:00:00Z sn3d launched\n2026-10-15T12:00:00Z {nprocsline}\n"
            "2026-10-15T12:30:00Z timestep 10: start update_packets\n"
            "2026-10-15T13:00:00Z timestep 10: time after update grid on all processes\n"
        ),
    }
    for runfolder, logtext in logtext_of_runfolder.items():
        (modelfolder / runfolder).mkdir(parents=True)
        (modelfolder / runfolder / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    (modelfolder / "output_0-0.txt").symlink_to(Path("job_from_ts0010", "output_0-0.txt"))

    artistools.__main__.main(argsraw=["corehours", str(modelfolder), "--json"])
    corehours = json.loads(capsys.readouterr().out)
    assert [job["duplicate"] for job in corehours["jobs"]] == [False, False, True]
    assert [job["estimate"] for job in corehours["jobs"]] == [False, True, False]
    # the JSON gives the inputs of the estimate
    assert np.isclose(corehours["jobs"][1]["wallclock_hours"], 1.0)
    assert corehours["jobs"][1]["tasks"] == 4
    assert np.isclose(corehours["total_core_hours"], 12.0)
    assert np.isclose(corehours["wallclock_hours"], 3.0)
    assert corehours["tasks"] == 4
    assert corehours["timestep_ranges"] == [[0, 9], [10, 10]]

    artistools.__main__.main(argsraw=["corehours", str(modelfolder)])
    table = capsys.readouterr().out
    assert "ignored duplicate log" in table
    assert "warning: sn3d did not finish. Estimated: (job: pktprop ts 10 to ts 10 grid-preprop" in table
    assert "CPU time:             12.000  core-h" in table

    # a damaged log gives no core hours and no task count, thus the sum is not complete and the tasks are not known.
    # The run folder comes before the target of the link, thus the search for the copy also reads the damaged log
    (modelfolder / "job_from_ts0005").mkdir()
    (modelfolder / "job_from_ts0005" / "output_0-0.txt.gz").write_bytes(b"not gzip data")
    # a run folder with no rank-0 log and a last line with no number also give a run with no core hours
    (modelfolder / "job_from_ts0030").mkdir()
    (modelfolder / "job_from_ts0030" / "output_1-0.txt").write_text(
        logtext_of_runfolder["job_from_ts0000"], encoding="utf-8"
    )
    (modelfolder / "job_from_ts0040").mkdir()
    (modelfolder / "job_from_ts0040" / "output_0-0.txt").write_text("sn3d finished = core hours\n", encoding="utf-8")
    (modelfolder / "light_curve.out").write_text("", encoding="utf-8")
    artistools.__main__.main(argsraw=["corehours", str(modelfolder / "light_curve.out"), "--json"])
    captured = capsys.readouterr()
    corehours = json.loads(captured.out)
    assert "output_0-0.txt.gz is unreadable" in captured.err
    assert "job_from_ts0030 has the logs of other ranks and no rank-0 log" in captured.err
    assert "job_from_ts0040/output_0-0.txt has a line with values that the command cannot read" in captured.err
    assert [job["core_hours"] is None for job in corehours["jobs"]] == [False, True, False, True, True, True]
    assert corehours["total_core_hours"] is None
    assert corehours["tasks"] is None

    artistools.__main__.main(argsraw=["corehours", str(modelfolder)])
    table = capsys.readouterr().out
    assert "warning: The log is missing, empty, or unreadable." in table
    assert "warning: Some logs give no core hours. The CPU time omits their runs." in table
    assert "Tasks:" not in table


def test_logsummary_reads_a_run_that_stopped_before_its_estimators(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path | io.StringIO], pl.DataFrame]
) -> None:
    """A run that stops early writes a log and no estimator file, and its log holds the error that stopped it."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "12411848.slurm" / "output_0-0.txt").write_text(
        "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n",
        encoding="utf-8",
    )

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = read_data_file(io.StringIO(capsys.readouterr().out))
    assert dfsummary["level"].to_list() == ["error"]
    assert dfsummary["first_timestep"].item() == 0


def test_logsummary_all_ranks_reads_the_ranks_of_each_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path | io.StringIO], pl.DataFrame]
) -> None:
    """A restart can use fewer ranks, and -ranks all read only the ranks of input.txt, which left out an error."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    for runfolder, rank, text in [
        ("100.slurm", 7, "[error] cell 3 has no matter\n"),
        ("200.slurm", 0, "[warning] cell 4 is cold\n"),
    ]:
        (modelfolder / runfolder).mkdir()
        (modelfolder / runfolder / f"output_{rank}-0.txt").write_text(
            f"timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n{text}", encoding="utf-8"
        )

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-ranks", "all", "--quiet"])
    dfsummary = read_data_file(io.StringIO(capsys.readouterr().out))
    assert sorted(dfsummary["level"].to_list()) == ["error", "warning"]


def test_logsummary_cell_reads_the_rank_of_the_cell_in_each_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A restart can use more ranks, and the rank of a cell in the current input.txt is a different cell in a run.

    With 1000 cells, the blocks of ARTIS give cell 999 to rank 1 of 2 ranks, and to rank 3 of 4 ranks.
    """
    import artistools.__main__

    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "model.txt").write_text("1000\n", encoding="utf-8")
    header = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n"
    for runfolder, rankcount, cellrank in [("100.slurm", 2, 1), ("200.slurm", 4, 3)]:
        (modelfolder / runfolder).mkdir()
        for rank in range(rankcount):
            # ARTIS writes the MPI layout near the start of each log
            mpiline = f"  rank {rank} of [0..{rankcount - 1}] in MPI_COMM_WORLD\n"
            message = "[error] cell 999 has no matter\n" if rank == cellrank else "[warning] cell 5 is cold\n"
            (modelfolder / runfolder / f"output_{rank}-0.txt").write_text(mpiline + header + message, encoding="utf-8")

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["level"].to_list() == ["error"]
    assert dfsummary["count"].item() == 2

    # the names of the logs gave 2 ranks when the logs of ranks 2 and 3 were missing, and then rank 1 took cell 999
    for rank in (2, 3):
        (modelfolder / "200.slurm" / f"output_{rank}-0.txt").unlink()
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    captured = capsys.readouterr()
    assert pl.read_csv(io.StringIO(captured.out), separator=" ", null_values="nan")["count"].item() == 1
    # the summary left out the run of 200.slurm with no message
    assert "200.slurm has no log of the ranks 3" in captured.err

    # a folder that gives no number of ranks gives a warning, and the summary leaves out its logs
    (modelfolder / "300.slurm").mkdir()
    (modelfolder / "300.slurm" / "output_0-0.txt").write_text(header + "[error] cell 999 is odd\n", encoding="utf-8")
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    captured = capsys.readouterr()
    assert "300.slurm has no modelgridrankassignments.out" in captured.err
    assert pl.read_csv(io.StringIO(captured.out), separator=" ", null_values="nan")["count"].item() == 1


def test_logsummary_log_that_cannot_be_read_is_not_a_copy(tmp_path: Path) -> None:
    """A log with no read permission stopped the search for a copy, and the reader of that log reports the problem."""
    from artistools.logs.core import are_copies

    modellog = tmp_path / "output_0-0.txt"
    modellog.write_text("sn3d launched\n", encoding="utf-8")
    (tmp_path / "100.slurm").mkdir()
    runlog = tmp_path / "100.slurm" / "output_0-0.txt"
    runlog.write_text("sn3d launched\n", encoding="utf-8")
    runlog.chmod(0)
    try:
        if os.access(runlog, os.R_OK):
            pytest.skip("the user can read a file with no read permission, e.g. root")
        assert not are_copies(modellog, runlog, comparetext=True)
    finally:
        runlog.chmod(0o644)


def test_logsummary_reads_a_compressed_copy_of_a_log_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A gzip copy of the log of a run in the model folder had a different name, and its messages counted twice."""
    import gzip

    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "100.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    logtext = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    (modelfolder / "100.slurm" / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    with gzip.open(modelfolder / "output_0-0.txt.gz", "wt", encoding="utf-8") as fcopy:
        fcopy.write(logtext)

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [1]


def test_logsummary_combines_the_summary_of_each_log() -> None:
    """The messages of all the logs did not fit in the memory, thus each log gives its own summary first.

    The combination must give the summary of all the messages: the sums, the bounds, the cells, and the first example.
    """
    from artistools.logs.core import get_log_summary
    from artistools.logs.core import get_logfiles
    from artistools.logs.core import read_log_messages
    from artistools.logs.core import summarise_log_messages

    logfiles = get_logfiles(modelpath_classic_3d, None)
    assert len(logfiles) > 2
    dfall = summarise_log_messages(pl.concat([read_log_messages(logfile) for logfile in logfiles]))
    pltest.assert_frame_equal(get_log_summary(logfiles), dfall)


def test_logsummary_reads_two_gzip_copies_of_a_log_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two gzip files of one text can have different bytes, e.g. another time in the header of gzip."""
    import gzip

    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "100.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    logtext = b"timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    for folder, mtime in ((modelfolder / "100.slurm", 1), (modelfolder, 2)):
        with gzip.GzipFile(folder / "output_0-0.txt.gz", "wb", mtime=mtime) as fcopy:
            fcopy.write(logtext)
    assert (modelfolder / "output_0-0.txt.gz").read_bytes() != (
        modelfolder / "100.slurm" / "output_0-0.txt.gz"
    ).read_bytes()

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [1]


def test_log_reader_finds_a_message_on_the_line_of_a_timestep(tmp_path: Path) -> None:
    """The reader took the timestep from the line and went on to the next line, thus it left out the message."""
    from artistools.logs.core import read_log_messages

    logfile = tmp_path / "output_0-0.txt"
    logfile.write_text(
        "timestep 3: time before update grid 0.1 (tstart + tdelta/2) 2.0 days ...[error] cell 7 has no matter\n",
        encoding="utf-8",
    )
    dfmessages = read_log_messages(logfile)
    assert dfmessages.select("level", "timestep", "modelgridindex").rows() == [("error", 3, 7)]

    # the progress text names the cell, and the text after the tag does not
    logfile.write_text("Finding T_e in cell 4...[warning] the solver did not converge\n", encoding="utf-8")
    assert read_log_messages(logfile).select("level", "modelgridindex").rows() == [("warning", 4)]


def test_logsummary_keeps_identical_logs_of_two_run_folders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two retries can stop with the same error and write the same text, and the second retry counted as a copy."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    logtext = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    for foldername in ("100.slurm", "200.slurm"):
        (modelfolder / foldername).mkdir(parents=True)
        (modelfolder / foldername / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [2]

    # a local run writes its estimators and its log into the model folder, and its log is a run and not a copy
    (modelfolder / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    (modelfolder / "estimators_0000.out").write_text("", encoding="utf-8")
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [3]


def test_logsummary_counts_each_kind_of_warning(
    capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path | io.StringIO], pl.DataFrame]
) -> None:
    """A log holds thousands of warnings of a few kinds, thus the table names each kind, its timesteps, and its cells.

    In the classic-mode 3D test, heating is bigger than cooling in many cells, thus ARTIS sets Te to MAXTEMP. The first
    timesteps are in LTE, where ARTIS solves no thermal balance.
    """
    import artistools.__main__

    artistools.__main__.main(argsraw=["logsummary", str(modelpath_classic_3d), "--quiet"])
    dfsummary = read_data_file(io.StringIO(capsys.readouterr().out))
    dfmaxtemp = dfsummary.filter(pl.col("kind").str.contains("T_e forced to be MAXTEMP"))
    assert dfmaxtemp.height == 1
    assert dfmaxtemp["count"].item() > 1000
    assert dfmaxtemp["first_timestep"].item() == 5
    assert dfmaxtemp["cells"].item() > 100
    assert dfsummary["count"].is_sorted(descending=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_logfiles_write_data_holds_the_plotted_times(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The table of --write_data must give each line of each page: the mpi ranks and the seconds of each stage."""
    at.logs.plotlogfiles.main(
        argsraw=[], modelpath=[modelpath_classic_3d], outputfile=tmp_path / "logfiles.pdf", write_data=True
    )

    dftable = read_data_file(tmp_path / "logfiles.txt")
    assert dftable.columns == ["timestep", "mpirank", "update_grid_s", "update_packets_s", "communicate_estimators_s"]
    assert len(mockplot.call_args_list) == 3 * 30
    for call in mockplot.call_args_list:
        axis, mpirank, timetaken = call.args
        timestep = int(axis.get_title().split()[-1])
        dfpage = dftable.filter(pl.col("timestep") == timestep).sort("mpirank")
        assert dfpage["mpirank"].to_list() == list(mpirank)
        assert np.allclose(dfpage[f"{call.kwargs['label']}_s"].to_numpy(), timetaken, rtol=1e-12, atol=0)


def test_logfiles_with_no_timing_data_replace_the_table_of_an_earlier_run(tmp_path: Path) -> None:
    """Logs with no timing line returned before the data output, thus the table of an earlier run stayed."""
    tablefile = tmp_path / "logfiles.txt"
    tablefile.write_text("timestep mpirank update_grid_s\n5 0 1.0\n", encoding="utf-8")
    args = argparse.Namespace(print_data=False, write_data=True)
    at.logs.plotlogfiles.make_plot({}, tmp_path / "logfiles.pdf", args=args)
    assert "the table has no rows" in tablefile.read_text(encoding="utf-8")
