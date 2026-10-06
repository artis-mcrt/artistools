"""Unit tests for the shared helpers in artistools.misc.

Most tests write synthetic data under tmp_path. Some tests read the test models that setuptestdata.sh extracts into
tests/data, e.g. testmodel and test-classicmode_3d.
"""

import argparse
import gzip
import importlib.metadata
import io
import lzma
import math
import os
import subprocess
import sys
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.testing as pltest
import pytest
import yaml

import artistools as at
from artistools.estimators import plotestimators
from artistools.estimators.core import join_cell_modeldata
from artistools.misc import dirbins
from artistools.misc import fileio
from artistools.misc import modelinfo
from artistools.misc import parse_cli_args
from artistools.misc import remote
from artistools.misc.general import get_bin_index_expr
from artistools.viewertools.core import run_command_step_with_warning


def write_timesteps_out(modeldir: Path) -> None:
    """Write a minimal timesteps.out with 5 evenly spaced timesteps (mids 105..145)."""
    lines = ["#timestep tmid_days tstart_days twidth_days"]
    for ts in range(5):
        tstart = 100 + ts * 10
        lines.append(f"{ts} {tstart + 5} {tstart} 10")
    (modeldir / "timesteps.out").write_text("\n".join(lines) + "\n")


# --- cliutils.py -------------------------------------------------------------------------------


def get_frame_sizes(fig: t.Any, axes: t.Any) -> set[tuple[float, float]]:
    """Return the size in inches of each frame of a figure, to four decimal places."""
    fig.canvas.draw()
    figwidth, figheight = fig.get_size_inches()
    return {
        (round(axis.get_position().width * figwidth, 4), round(axis.get_position().height * figheight, 4))
        for axis in axes.flat
    }


@pytest.mark.parametrize(
    ("name", "rows", "cols", "ylabel", "sharex"),
    [
        ("one frame", 1, 1, "T$_e$ [K]", True),
        ("a y label that is long", 1, 1, "T$_e$ [K] " + "x" * 30, True),
        ("three rows that share an x axis", 3, 1, "T$_e$ [K]", True),
        ("three rows with an x label each", 3, 1, "T$_e$ [K]", False),
        ("two rows and three columns", 2, 3, "T$_e$ [K]", True),
    ],
)
def test_a_frame_figure_holds_the_size_of_every_frame(
    name: str, rows: int, cols: int, ylabel: str, sharex: bool
) -> None:
    """Every frame takes the same size in inches, whatever the labels and the number of rows.

    A paper puts these files in a grid that the author builds by hand, thus a frame that follows the
    length of a tick number or the number of rows draws a panel of the wrong size beside its
    neighbour.
    """
    import artistools.plottools as pt

    args = argparse.Namespace(figscale=1.0, figwidthscale=1.0)
    fig, axes = pt.make_frame_figure(args, rows=rows, cols=cols, sharex=sharex)
    for axis in axes.flat:
        axis.plot([0, 1e7], [0, 1e-12])
        axis.set_ylabel(ylabel)
        axis.set_xlabel("velocity [km/s]")

    assert get_frame_sizes(fig, axes) == {(pt.FRAMEWIDTH_INCHES, pt.FRAMEHEIGHT_INCHES)}, name
    plt.close(fig)


def test_a_frame_figure_keeps_its_frame_when_a_command_hides_the_labels(tmp_path: Path) -> None:
    """The frame and the width of a file hold when the x tick labels go, and the height falls.

    A crop takes the part of a margin that no label fills, thus the file loses the height of the
    labels that went. It moves no artist, thus the frame keeps the size that it holds in inches, and
    each panel of a grid draws the same frame.
    """
    import pypdf

    import artistools.plottools as pt

    args = argparse.Namespace(figscale=1.0, figwidthscale=1.0)
    sizes = {}
    for name, hide in (("shown", False), ("hidden", True)):
        fig, axes = pt.make_frame_figure(args)
        axes[0][0].plot([0, 1e7], [0, 1e-12])
        axes[0][0].set_ylabel("T$_e$ [K]")
        axes[0][0].set_xlabel("velocity [km/s]")
        if hide:
            axes[0][0].tick_params(axis="x", which="both", labelbottom=False)

        frames = get_frame_sizes(fig, axes)
        outpath = tmp_path / f"{name}.pdf"
        pt.save_figure(fig, outpath)
        page = pypdf.PdfReader(outpath).pages[0].mediabox
        sizes[name] = (round(float(page.width) / 72.0, 3), round(float(page.height) / 72.0, 3), frames)

    # the width and the frame hold, and the height falls by the labels that the file no longer draws
    assert sizes["shown"][0] == sizes["hidden"][0], sizes
    assert sizes["shown"][2] == sizes["hidden"][2], sizes
    assert sizes["hidden"][1] < sizes["shown"][1], sizes


def test_a_saved_figure_keeps_an_artist_that_reaches_past_it(tmp_path: Path) -> None:
    """A file of a plot holds every artist, thus a long label or an annotation is not cut.

    The page took the size of the figure, thus an artist outside it went. It takes the size of the
    artists now, which also leaves no border of white around the plot.
    """
    import pypdf
    from PIL import Image

    import artistools.plottools as pt

    figwidth, figheight = 3.0, 2.0
    sizes = {}
    for name, overflows in (("plain", False), ("overflowing", True)):
        fig, axis = plt.subplots(figsize=(figwidth, figheight), tight_layout={"pad": 0.2})
        axis.plot([0, 1], [0, 1])
        if overflows:
            # annotation_clip=False draws the text outside the axes, and outside the figure
            axis.annotate("A" * 30, xy=(0.5, 1.6), xycoords="axes fraction", annotation_clip=False, fontsize=14)

        outpath = tmp_path / f"{name}.png"
        pt.save_figure(fig, outpath)
        sizes[name] = Image.open(outpath).size

    # the figure of both is the same size, thus the file grows only because it holds the annotation
    assert sizes["overflowing"][1] > sizes["plain"][1], sizes

    # a pdf of a plot that fits carries no border: its page is the size of the artists and no more
    fig, axis = plt.subplots(figsize=(figwidth, figheight), tight_layout={"pad": 0.2})
    axis.plot([0, 1], [0, 1])
    pdfpath = tmp_path / "plot.pdf"
    pt.save_figure(fig, pdfpath)
    page = pypdf.PdfReader(pdfpath).pages[0].mediabox
    assert float(page.width) / 72.0 < figwidth, "the page must lose the border of the figure"
    assert float(page.height) / 72.0 < figheight, "the page must lose the border of the figure"


def test_a_gif_holds_the_largest_of_its_frames(tmp_path: Path) -> None:
    """Every frame of a gif keeps its content, whatever size the other frames take.

    A gif holds one canvas, and the first frame gave its size, thus a frame that is wider or taller
    lost what lay outside it.
    """
    from PIL import Image

    framepaths = []
    for index, figsize in enumerate(((3, 2), (4, 3))):
        fig, axis = plt.subplots(figsize=figsize)
        axis.plot([0, 1], [0, 1])
        framepath = tmp_path / f"frame{index}.png"
        fig.savefig(framepath)
        plt.close(fig)
        framepaths.append(framepath)

    widths, heights = zip(*(Image.open(framepath).size for framepath in framepaths), strict=True)
    assert len(set(widths)) > 1, "this test needs frames that differ in size"

    gifpath = tmp_path / "frames.gif"
    at.misc.write_gif(gifpath, framepaths, duration=200.0)

    assert Image.open(gifpath).size == (max(widths), max(heights))


def test_one_rule_finds_the_reference_data_of_each_kind(tmp_path: Path) -> None:
    """The light curves and the spectra hold reference data in folders of their own, under one rule.

    Each command carried its own copy of "look here, then in the folder of the package, and accept a
    compressed name", thus a change to that rule reached one command and not the other.
    """
    # the folder of the package holds the reference data of both kinds
    assert at.misc.find_reference_data_file("2003du_20031213_3219_8822_00.txt", "data/refspectra") is not None
    assert (
        at.misc.find_reference_data_file("AT2017gfo_smarttetal2017.txt", "data/lightcurves/bollightcurves") is not None
    )

    # a name that no folder holds gives None, and the kind of the data selects the folder
    assert (
        at.misc.find_reference_data_file("2003du_20031213_3219_8822_00.txt", "data/lightcurves/bollightcurves") is None
    )
    assert at.misc.find_reference_data_file("nosuchfile.txt", "data/refspectra") is None

    # a file of the working folder comes first, and a compressed name of it counts
    reffile = tmp_path / "myref.txt.xz"
    reffile.write_bytes(b"")
    assert at.misc.find_reference_data_file(tmp_path / "myref.txt", "data/refspectra") == reffile

    # a folder is no file of reference data, and neither is a name that no folder holds
    assert not at.misc.path_is_reference_data(tmp_path, "data/refspectra")
    assert not at.misc.path_is_reference_data(tmp_path / "nosuchfile.txt", "data/refspectra")


def test_reference_data_search_follows_the_working_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative name must not match after a change of the working folder.

    A cache held the full result. Thus the first answer kept a working folder that no longer
    applied, and a miss stayed None after the file appeared.
    """
    folder_with_file = tmp_path / "a"
    folder_with_file.mkdir()
    (folder_with_file / "myref.txt").write_bytes(b"")
    empty_folder = tmp_path / "b"
    empty_folder.mkdir()

    monkeypatch.chdir(empty_folder)
    assert at.misc.find_reference_data_file("myref.txt", "data/refspectra") is None

    monkeypatch.chdir(folder_with_file)
    assert at.misc.find_reference_data_file("myref.txt", "data/refspectra") == Path("myref.txt")

    monkeypatch.chdir(empty_folder)
    assert at.misc.find_reference_data_file("myref.txt", "data/refspectra") is None


def test_a_rejected_parquet_cache_gives_the_reason(tmp_path: Path) -> None:
    """A rejected cache must say which check failed, because a regeneration costs minutes.

    The message said only that the cache was not current. Thus a user could not see whether a new
    cache format version, a rewritten text file, or a damaged file caused the conversion.
    """
    from artistools.misc.fileio import MTIME_TOLERANCE_S

    cacheversion = 3
    mtime = 1000.0

    current = tmp_path / "current.parquet"
    at.misc.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}),
        current,
        metadata={"cacheversion": str(cacheversion), "textsource_mtime": str(mtime)},
    )
    pqmetadata, stalereason = at.misc.read_parquet_cache_metadata(current, cacheversion, mtime)
    assert stalereason is None
    assert pqmetadata is not None

    def get_reason(parquetfilepath: Path, textsource_mtime: float) -> str:
        pqmetadata, stalereason = at.misc.read_parquet_cache_metadata(parquetfilepath, cacheversion, textsource_mtime)
        assert pqmetadata is None
        assert stalereason is not None
        return stalereason

    # a file system can move the time of a file, even when the run wrote nothing. Thus a small
    # difference counts as the same time
    assert at.misc.read_parquet_cache_metadata(current, cacheversion, mtime + 6.0)[1] is None

    # a difference above the tolerance is a rewrite, in either direction
    assert "the text source changed" in str(
        at.misc.read_parquet_cache_metadata(current, cacheversion, mtime - MTIME_TOLERANCE_S - 10.0)[1]
    )

    # the text file changed after the run wrote the cache, thus the reason names both times
    changedmtime = mtime + MTIME_TOLERANCE_S + 10.0
    changedsource = get_reason(current, changedmtime)
    assert "the text source changed" in changedsource
    assert str(mtime) in changedsource
    assert str(changedmtime) in changedsource

    # a cache that holds no stamp is stale by default, because a rebuild of a cheap cache costs less
    # than a wrong number. Only a reader that gives accept_unstamped keeps it
    unstamped = tmp_path / "unstamped.parquet"
    at.misc.write_parquet_atomic(pl.DataFrame({"timestep": [0]}), unstamped)
    assert "no cacheversion stamp" in get_reason(unstamped, mtime)
    assert at.misc.read_parquet_cache_metadata(unstamped, cacheversion, mtime, accept_unstamped=True)[1] is None

    oldversion = tmp_path / "oldversion.parquet"
    at.misc.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}), oldversion, metadata={"cacheversion": "0", "textsource_mtime": str(mtime)}
    )
    assert f"version is 0, but this artistools version writes {cacheversion}" in get_reason(oldversion, mtime)

    unreadable = tmp_path / "unreadable.parquet"
    unreadable.write_bytes(b"not parquet")
    assert "not a readable parquet file" in get_reason(unreadable, mtime)

    assert get_reason(tmp_path / "nosuchfile.parquet", mtime) == "the file does not exist"


def test_a_stale_estimator_cache_does_not_hide_new_timesteps(tmp_path: Path) -> None:
    """A run that appends timesteps makes the batch cache stale, thus the text files answer.

    get_runfolder_timesteps read the first batch cache with no freshness check. Thus a plot during
    a run excluded the new timesteps of the folder.
    """
    from artistools.estimators import CACHEVERSION
    from artistools.misc.modelinfo import get_runfolder_timesteps

    stale_folder = tmp_path / "stale"
    stale_folder.mkdir()
    (stale_folder / "estimators_0000.out").write_text("timestep 0 header\n\ntimestep 1 header\n\n")
    at.misc.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}),
        stale_folder / "estimbatch00_0000_0000.out.parquet.tmp",
        metadata={"cacheversion": str(CACHEVERSION), "textsource_mtime": "1.0"},
    )
    assert get_runfolder_timesteps(stale_folder) == (0, 1)

    current_folder = tmp_path / "current"
    current_folder.mkdir()
    textfile = current_folder / "estimators_0000.out"
    textfile.write_text("timestep 0 header\n\ntimestep 1 header\n\n")
    at.misc.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}),
        current_folder / "estimbatch00_0000_0000.out.parquet.tmp",
        metadata={"cacheversion": str(CACHEVERSION), "textsource_mtime": str(textfile.stat().st_mtime)},
    )
    # the current cache answers, thus the extra timestep of the text file stays unread
    assert get_runfolder_timesteps(current_folder) == (0,)


def test_add_cli_arg_helpers() -> None:
    """The shared argument helpers must define the standard flags, types, and defaults."""
    parser = argparse.ArgumentParser()
    at.misc.addarg_modelpath(parser, multiplepaths=True, default=[])
    at.misc.addarg_output(parser, kind="file", default=Path("out.pdf"))
    at.misc.addarg_timestep(parser)
    at.misc.addarg_timedays(parser)
    at.misc.addarg_timeminmax(parser)
    at.misc.addarg_axislimits(parser, xlimtype=int, xmindefault=1000, xmaxdefault=2000)
    at.misc.addarg_seriesstyle(parser, colordefault=["C0", "C1"], include_linealpha=True)
    at.misc.addarg_figscale(parser, include_figwidthscale=True)
    at.misc.addarg_filter(parser)
    at.misc.addarg_maxpacketfiles(parser)

    args = parser.parse_args([])
    assert args.modelpath == []
    assert args.outputfile == Path("out.pdf")
    assert args.timestep is None
    assert args.timedays is None
    assert args.figscale == 1.0
    assert args.figwidthscale == 1.0
    assert args.xmin == 1000
    assert args.xmax == 2000
    assert args.color == ["C0", "C1"]
    assert args.linealpha == []
    assert args.filtermovingavg == 0
    assert args.maxpacketfiles is None

    args = parser.parse_args([
        "-modelpath",
        "model1",
        "model2",
        "-ts",
        "45-65",
        "-t",
        "50-100",
        "-colors",
        "red",
        "blue",
        "-o",
        "other.pdf",
        "-maxpacketsfiles",
        "5",
        "-xmin",
        "1500",
        "-filtersavgol",
        "5",
        "3",
    ])
    assert args.modelpath == [Path("model1"), Path("model2")]
    assert args.timestep == "45-65"
    assert args.timedays == "50-100"
    assert args.color == ["red", "blue"]
    assert args.outputfile == Path("other.pdf")
    assert args.maxpacketfiles == 5
    assert args.xmin == 1500
    assert args.filtersavgol == ["5", "3"]

    # the token default gives no value to its series, thus a list can give a value to a later series only
    styles = {"-label": "Two", "-colors": "blue", "-linewidth": "2", "-linealpha": "0.5", "-dashes": "5,2"}
    args = parser.parse_args([
        *(token for flag, value in {**styles, "-linestyle": ":"}.items() for token in (flag, "default", value))
    ])
    assert args.label == [None, "Two"]
    # the parser gives the colours C0 and C1 by default, thus a default entry takes the colour of its place
    assert args.color == ["C0", "blue"]
    assert args.linewidth == [None, 2.0]
    assert args.linealpha == [None, 0.5]
    assert args.dashes == [None, (5.0, 2.0)]
    assert args.linestyle == [None, ":"]
    with pytest.raises(SystemExit):
        parser.parse_args(["-linewidth", "thick"])


def test_add_cli_arg_helper_variants() -> None:
    """The non-default helper modes must reproduce the per-command argument shapes."""
    parser = argparse.ArgumentParser()
    at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])
    at.misc.addarg_timestep(parser, default=70)
    at.misc.addarg_timedays(parser, kind="float")
    at.misc.addarg_output(parser, kind="folder", default=Path())
    args = parser.parse_args(["model1", "-timestep", "12", "-timedays", "45.5"])
    assert args.modelpath == [Path("model1")]
    # -timestep holds the text that the user wrote, and get_single_timestep reads one timestep
    assert args.timestep == "12"
    assert args.timedays == 45.5
    # one helper serves both kinds, thus the folder of a command is a Path as the file of one is
    assert args.outputfile == Path()
    assert args.outputkind == "folder"

    # a repeated flag joins with a comma, thus no occurrence takes the place of an earlier one
    parserrepeat = argparse.ArgumentParser()
    at.misc.addarg_timestep(parserrepeat, default=70)
    at.misc.addarg_modelgridindex(parserrepeat)
    argsrepeat = parserrepeat.parse_args(["-ts", "5", "-ts", "6", "-mgi", "3", "-mgi", "5-7"])
    assert argsrepeat.timestep == "5,6"
    assert argsrepeat.modelgridindex == [3, 5, 6, 7]
    # one occurrence replaces the default and does not join to it
    assert parserrepeat.parse_args(["-ts", "5"]).timestep == "5"

    parserrequired = argparse.ArgumentParser()
    at.misc.addarg_modelpath(parserrequired, required=True)
    with pytest.raises(SystemExit):
        parserrequired.parse_args([])


def test_the_cells_of_modelgridindex_are_a_list_from_every_source() -> None:
    """The parser, the default, and a keyword argument each give -modelgridindex as a sorted list of cells.

    The parser stored the text, thus each command expanded it again, and the defaults had four different types.
    """
    parser = argparse.ArgumentParser()
    at.misc.addarg_modelgridindex(parser, default=[0])
    assert parser.parse_args([]).modelgridindex == [0]
    assert parser.parse_args(["-cell", "7-9", "-mgi", "4", "-cell", "8"]).modelgridindex == [4, 7, 8, 9]

    for keywordvalue, expectedcells in ((5, [5]), ("0,2-3", [0, 2, 3]), ([6, 2], [2, 6])):
        keywordparser = argparse.ArgumentParser()
        at.misc.addarg_modelgridindex(keywordparser)
        at.misc.set_args_from_dict(keywordparser, {"cell": keywordvalue})
        assert keywordparser.parse_args([]).modelgridindex == expectedcells

    assert at.misc.cliutils.format_range_list([9, 3, 4, 5, 7, 6, 12]) == "3-7,9,12"
    assert at.misc.parse_range_list(at.misc.cliutils.format_range_list([-1, 0, 2])) == [-1, 0, 2]


def test_get_cell_list_takes_a_numpy_integer_and_rejects_a_float() -> None:
    """A numpy integer is one cell, and a float is an error, because a cell index is an integer.

    The function took only an int as a number, thus set() iterated a numpy integer and raised a TypeError.
    """
    assert at.misc.cliutils.get_cell_list(np.int64(5)) == [5]
    assert at.misc.cliutils.get_cell_list(np.array([7, 3])) == [3, 7]
    assert at.misc.cliutils.get_cell_list([np.int32(4), "6-7"]) == [4, 6, 7]
    with pytest.raises(TypeError):
        at.misc.cliutils.get_cell_list(5.0)  # ty:ignore[invalid-argument-type]  # pyrefly: ignore[bad-argument-type]


def test_the_old_spelling_of_the_phi_average_sets_the_phi_average(capsys: pytest.CaptureFixture[str]) -> None:
    """--average_every_tenth_viewing_angle sets average_over_phi_angle, thus no command copies it to that dest."""
    parser = argparse.ArgumentParser()
    at.misc.addarg_viewingangle(parser)
    args = parser.parse_args(["--average_every_tenth_viewing_angle"])
    assert args.average_over_phi_angle
    assert not hasattr(args, "average_every_tenth_viewing_angle")
    assert "is deprecated" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser.parse_args(["--average_every_tenth_viewing_angle", "--average_over_theta_angle"])


def test_legendcols_rejects_a_count_below_one() -> None:
    """-legendcols 0 must fail when argparse reads it, and not in matplotlib after the data is read."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    at.misc.addarg_legend(parser)
    assert parser.parse_args(["-legendcols", "3"]).legendcols == 3
    for badvalue in ("0", "-2"):
        with pytest.raises(argparse.ArgumentError, match="positive count"):
            parser.parse_args([f"-legendcols={badvalue}"])


def test_set_args_from_dict_keeps_one_dash_pattern_as_one_series() -> None:
    """A dash pattern from the Python API must give one series, also after the type of -dashes became a wrapper."""
    parser = argparse.ArgumentParser()
    at.lightcurve.addargs(parser)
    at.misc.set_args_from_dict(parser, {"dashes": (5, 2)})
    assert parser.parse_args([]).dashes == [(5, 2)]
    at.misc.set_args_from_dict(parser, {"dashes": [(5, 2), (1, 1)]})
    assert parser.parse_args([]).dashes == [(5, 2), (1, 1)]


def test_set_args_from_dict_does_not_mutate_caller() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-outputfile", "-o", type=Path)
    kwargs = {"o": "somefile.pdf"}
    at.misc.set_args_from_dict(parser, kwargs)
    assert kwargs == {"o": "somefile.pdf"}
    assert parser.parse_args([]).outputfile == Path("somefile.pdf")

    with pytest.raises(ValueError, match="badargname"):
        at.misc.set_args_from_dict(parser, {"badargname": 1})


# --- fileio.py (print_saved) -------------------------------------------------------------------


def test_print_saved(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """print_saved must emit a runnable open command with a path relative to the working directory."""
    monkeypatch.chdir(tmp_path)

    at.misc.print_saved(tmp_path / "subdir" / "out.pdf")
    opencommand = "open" if sys.platform == "darwin" else "xdg-open"
    assert capsys.readouterr().out == f"{opencommand} subdir/out.pdf\n"

    at.misc.print_saved("out.pdf")
    assert capsys.readouterr().out == f"{opencommand} out.pdf\n"

    at.misc.print_saved(tmp_path / "subdir" / ".." / "out.pdf")
    assert capsys.readouterr().out == f"{opencommand} out.pdf\n"

    at.misc.print_saved(tmp_path / "with space.pdf")
    assert capsys.readouterr().out == f"{opencommand} 'with space.pdf'\n"

    # each platform gets its own verb, thus a run on one of them covers the lines of the others.
    # cmd.exe reads the first quoted argument of start as the title of a window, thus an empty title
    # stands in front of the path there
    for platform, verb, line in (
        ("darwin", "open", "open out.pdf"),
        ("linux", "xdg-open", "xdg-open out.pdf"),
        ("win32", "start", 'start "" out.pdf'),
    ):
        monkeypatch.setattr(sys, "platform", platform)
        assert at.misc.fileio.get_open_command() == verb
        at.misc.print_saved("out.pdf")
        assert capsys.readouterr().out == f"{line}\n"

    # a name that holds a space takes the quotation marks that the platform reads
    monkeypatch.setattr(sys, "platform", "win32")
    at.misc.print_saved("with space.pdf")
    assert capsys.readouterr().out == 'start "" "with space.pdf"\n'

    monkeypatch.setattr(sys, "platform", "linux")
    at.misc.print_saved("with space.pdf")
    assert capsys.readouterr().out == "xdg-open 'with space.pdf'\n"


def test_open_file_takes_the_call_of_the_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Windows has no xdg-open, thus it opens a file through its own call and not through a command."""
    somefile = tmp_path / "out.pdf"
    somefile.touch()

    monkeypatch.setattr(sys, "platform", "linux")
    with mock.patch("subprocess.run") as mockrun:
        at.misc.open_file(somefile)
    assert mockrun.call_args.args[0] == ["xdg-open", str(somefile)]

    monkeypatch.setattr(sys, "platform", "win32")
    with mock.patch.object(os, "startfile", create=True) as mockstart, mock.patch("subprocess.run") as mockrun:
        at.misc.open_file(somefile)
    assert mockstart.call_args.args[0] == somefile
    assert not mockrun.called, "Windows must not run a command that it does not have"


def test_the_positional_items_read_the_folder_last(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every command that takes positional items reads the ARTIS folder as the last one."""
    (tmp_path / "mymodel").mkdir()
    monkeypatch.chdir(tmp_path)

    def parse(argsraw: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        at.misc.addarg_positional_items(parser, dest="items", metavar="item", helptext="items")
        at.misc.addarg_modelpath(parser)  # the default of None shows that the user named no folder
        args = parser.parse_args(argsraw)
        at.misc.resolve_positional_modelpath(args, "items")

        return args

    args = parse(["Te", "TR", "mymodel"])
    assert args.modelpath == Path("mymodel")
    assert args.items == ["Te", "TR"]

    # no folder at the end leaves every item, and the working folder stays the default
    args = parse(["Te", "TR"])
    assert args.modelpath == Path()
    assert args.items == ["Te", "TR"]

    # a path before the last item names a folder in the wrong place
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit):
        parse(["runs/mymodel", "Te"])

    # a name that has a separator but no parent folder is an item, e.g. heating_dep/total_dep
    args = parse(["heating_dep/total_dep"])
    assert args.items == ["heating_dep/total_dep"]

    # a bare name that holds an ARTIS run is a folder that the user wrote too early
    (tmp_path / "mymodel" / "input.txt").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse(["mymodel", "Te"])

    # a folder that holds no run keeps the meaning of the item, e.g. a folder of the name of a variable
    (tmp_path / "Te").mkdir()
    args = parse(["Te", "TR"])
    assert args.items == ["Te", "TR"]


def test_artis_subfolders_names_the_runs_of_a_folder(tmp_path: Path) -> None:
    """A folder that holds input.txt is an ARTIS run, thus an error can name the runs that are near."""
    for name in ("run1", "run2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "input.txt").write_text("", encoding="utf-8")
    (tmp_path / "notarun").mkdir()

    assert at.misc.artis_subfolders(tmp_path) == ["run1", "run2"]
    assert at.misc.artis_subfolders(tmp_path / "absent") == []
    assert at.misc.folder_is_artis_run(tmp_path / "run1")
    assert not at.misc.folder_is_artis_run(tmp_path / "notarun")


# --- modelinfo.py ------------------------------------------------------------------------------


def test_missing_inputfile_error_names_the_path(tmp_path: Path) -> None:
    """A non-ARTIS directory must produce an error naming the path, not a bare 'input.txt' message."""
    with pytest.raises(FileNotFoundError, match="ARTIS folder") as excinfo:
        at.get_inputparams(tmp_path / "nonexistentmodel")
    assert "nonexistentmodel" in str(excinfo.value)

    with pytest.raises(FileNotFoundError, match="ARTIS folder"):
        at.get_nprocs(tmp_path / "nonexistentmodel")


# --- fileio.py ---------------------------------------------------------------------------------


def test_zopen_zopenpl(tmp_path: Path) -> None:
    # plaintext with no compressed sibling
    (tmp_path / "plain.txt").write_text("plain contents\n")
    # gzip and xz siblings, addressed by their bare name
    with gzip.open(tmp_path / "gz.txt.gz", "wt", encoding="utf-8") as f:
        f.write("gzip contents\n")
    with lzma.open(tmp_path / "xz.txt.xz", "wt", encoding="utf-8") as f:
        f.write("xz contents\n")

    with at.zopen(tmp_path / "plain.txt") as f:
        assert f.read() == "plain contents\n"
    # bare name resolves to the compressed sibling
    with at.zopen(tmp_path / "gz.txt") as f:
        assert f.read() == "gzip contents\n"
    with at.zopen(tmp_path / "xz.txt") as f:
        assert f.read() == "xz contents\n"

    # zopenpl returns a Path for formats polars can read directly (uncompressed and .gz)...
    assert at.misc.zopenpl(tmp_path / "plain.txt") == tmp_path / "plain.txt"
    assert at.misc.zopenpl(tmp_path / "gz.txt") == tmp_path / "gz.txt.gz"
    # ...but an opened file object for .xz
    result_xz = at.misc.zopenpl(tmp_path / "xz.txt")
    assert not isinstance(result_xz, Path)
    with result_xz as f:
        assert f.read().decode("utf-8") == "xz contents\n"


def test_zopen_does_not_let_a_stale_compressed_sibling_shadow_a_named_file(tmp_path: Path) -> None:
    """A stale compressed sibling never shadows a freshly written uncompressed file."""
    (tmp_path / "both.txt").write_text("fresh contents\n")
    with gzip.open(tmp_path / "both.txt.gz", "wt", encoding="utf-8") as f:
        f.write("stale contents\n")

    # the named file wins, thus re-running the simulation is enough to change what a plot shows
    with at.zopen(tmp_path / "both.txt") as f:
        assert f.read() == "fresh contents\n"

    # zopenpl applies the same precedence, so the two readers never disagree about which file to read
    assert at.misc.zopenpl(tmp_path / "both.txt") == tmp_path / "both.txt"

    # with the plain file gone, the compressed sibling is used after all
    (tmp_path / "both.txt").unlink()
    with at.zopen(tmp_path / "both.txt") as f:
        assert f.read() == "stale contents\n"
    assert at.misc.zopenpl(tmp_path / "both.txt") == tmp_path / "both.txt.gz"

    # a compressed file addressed by its own name is decompressed, not opened raw
    with at.zopen(tmp_path / "both.txt.gz") as f:
        assert f.read() == "stale contents\n"

    # a name with no file and no compressed sibling reports the name the caller asked for
    with pytest.raises(FileNotFoundError):
        at.zopen(tmp_path / "absent.txt")


def test_read_wsv(tmp_path: Path) -> None:
    """Columns aligned with variable whitespace parse correctly, with comments and blank lines removed."""
    filepath = tmp_path / "aligned.txt"
    filepath.write_text("# file header comment\n  colA   colB  colC\n1    2.5  x # inline comment\n\n 4\t5\ty\n")

    df = at.misc.read_wsv(filepath, comment_prefix="#")
    pltest.assert_frame_equal(df, pl.DataFrame({"colA": [1, 4], "colB": [2.5, 5.0], "colC": ["x", "y"]}))

    # skip_rows applies before comment handling, and new_columns names a headerless read
    df_noheader = at.misc.read_wsv(
        filepath, has_header=False, skip_rows=2, new_columns=["a", "b", "c"], comment_prefix="#"
    )
    assert df_noheader.columns == ["a", "b", "c"]
    assert df_noheader.height == 2

    # a compressed file is read transparently
    with gzip.open(tmp_path / "data.txt.gz", "wt", encoding="utf-8") as f:
        f.write("p   q\n1   2\n")
    pltest.assert_frame_equal(at.misc.read_wsv(tmp_path / "data.txt"), pl.DataFrame({"p": [1], "q": [2]}))

    # trailing whitespace on every line must not become a trailing null column
    (tmp_path / "trailing.txt").write_text("colA colB  \n1 2 \n3 4 \t\n", encoding="utf-8")
    pltest.assert_frame_equal(
        at.misc.read_wsv(tmp_path / "trailing.txt"), pl.DataFrame({"colA": [1, 3], "colB": [2, 4]})
    )

    # a name-based projection parses only the requested columns, in the requested order
    dfprojected = at.misc.read_wsv(
        tmp_path / "aligned.txt",
        has_header=False,
        skip_rows=2,
        new_columns=["a", "b", "c"],
        columns=["c", "a"],
        comment_prefix="#",
    )
    assert dfprojected.columns == ["c", "a"]
    assert dfprojected["a"].to_list() == [1, 4]
    assert dfprojected["c"].to_list() == ["x", "y"]


def test_read_wsv_whitespace_runs(tmp_path: Path) -> None:
    """A run of ASCII whitespace between two fields acts as a single separator."""
    filepath = tmp_path / "whitespace.txt"
    # line 3 holds whitespace only, and line 5 holds a carriage return between two fields
    filepath.write_bytes(b"a\t\tb   c\r\n 1 \t 2\t\t\t3 \r\n\t \r\n4\t5     6\r\n7 8\r9\n")

    df = at.misc.read_wsv(filepath)
    pltest.assert_frame_equal(df, pl.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}))

    # a no-break space is not ASCII whitespace, thus it must stay inside its field
    (tmp_path / "nbsp.txt").write_bytes("ion pop\nFe\u00a0II 1.0\n".encode())
    dfnbsp = at.misc.read_wsv(tmp_path / "nbsp.txt")
    assert dfnbsp.columns == ["ion", "pop"]
    assert dfnbsp["ion"].to_list() == ["Fe\u00a0II"]

    # a compressed file gives the same result, including an xz file, which polars cannot read itself
    with lzma.open(tmp_path / "whitespace_xz.txt.xz", "wb") as f:
        f.write(filepath.read_bytes())
    pltest.assert_frame_equal(at.misc.read_wsv(tmp_path / "whitespace_xz.txt"), df)


def test_read_wsv_invalid_utf8(tmp_path: Path) -> None:
    """A byte that is not valid UTF-8 stops the read, unless a comment holds it."""
    # a comment of a file from a different source can hold e.g. a degree sign in Latin-1
    (tmp_path / "latin1comment.txt").write_bytes(b"colA colB\n1 2   # 30\xb0C\n3 4\n")
    pltest.assert_frame_equal(
        at.misc.read_wsv(tmp_path / "latin1comment.txt", comment_prefix="#"),
        pl.DataFrame({"colA": [1, 3], "colB": [2, 4]}),
    )

    # the header comment holds the column names, thus it must take a bad byte as the other comments do
    (tmp_path / "latin1header.txt").write_bytes(b"# t_days mag_30\xb0C\n1.0 2.0\n")
    dfheader = at.misc.read_wsv(tmp_path / "latin1header.txt", header_from_comment=True, comment_prefix="#")
    assert dfheader.columns == ["t_days", "mag_30\ufffdC"]
    assert dfheader["t_days"].to_list() == [1.0]

    # the same byte in the data of a column gives an error, and not a value that holds bad text
    (tmp_path / "latin1data.txt").write_bytes(b"colA colB\n1 2\n3 4\xb0\n")
    with pytest.raises(pl.exceptions.ComputeError):
        at.misc.read_wsv(tmp_path / "latin1data.txt", comment_prefix="#")

    # a column can hold the replacement character as data, even when a comment holds a bad byte. The
    # check reads the bytes, thus it tells the two apart
    (tmp_path / "mixed.txt").write_bytes("colA name\n1 x\ufffdy  # 30".encode() + b"\xb0C\n2 z\n")
    dfmixed = at.misc.read_wsv(tmp_path / "mixed.txt", comment_prefix="#")
    assert dfmixed["name"].to_list() == ["x\ufffdy", "z"]


def test_read_wsv_no_data(tmp_path: Path) -> None:
    """A file that holds no data raises a polars error that names the file."""
    for filename, contents in (("empty.txt", b""), ("comments.txt", b"# only a comment\n")):
        filepath = tmp_path / filename
        filepath.write_bytes(contents)

        with pytest.raises(pl.exceptions.PolarsError) as excinfo:
            at.misc.read_wsv(filepath, comment_prefix="#")

        assert any(str(filepath) in note for note in excinfo.value.__notes__ or [])


def test_read_wsv_prefers_uncompressed_file(tmp_path: Path) -> None:
    """A freshly written uncompressed file must win over a stale compressed sibling of the same name."""
    (tmp_path / "f.txt").write_text("v\n2\n", encoding="utf-8")
    with gzip.open(tmp_path / "f.txt.gz", "wt", encoding="utf-8") as f:
        f.write("v\n1\n")

    assert at.misc.read_wsv(tmp_path / "f.txt")["v"].to_list() == [2]

    # the header comment is read through a second open, which must apply the same precedence: reading it
    # from the stale sibling would label the fresh data with the stale column names
    (tmp_path / "h.txt").write_text("# freshA freshB\n1 2\n", encoding="utf-8")
    with gzip.open(tmp_path / "h.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# staleX staleY\n9 9\n")

    dfheader = at.misc.read_wsv(tmp_path / "h.txt", header_from_comment=True, comment_prefix="#")
    pltest.assert_frame_equal(dfheader, pl.DataFrame({"freshA": [1], "freshB": [2]}))


def test_read_wsv_all_null_inference_sample(tmp_path: Path) -> None:
    """A column whose first 10000 rows are all null tokens must still infer as numeric from the later rows."""
    filepath = tmp_path / "nullsample.txt"
    nnullrows = 12000  # more rows than the schema inference sample
    filepath.write_text("a b\n" + "".join(f"{i} nan\n" for i in range(nnullrows)) + f"{nnullrows} 3.5\n")

    df = at.misc.read_wsv(filepath)
    assert df["b"].dtype == pl.Float64
    assert df["b"].item(-1) == pytest.approx(3.5)
    assert df["b"].null_count() == nnullrows


def test_read_wsv_late_type_change(tmp_path: Path) -> None:
    """A column that turns from integer to float beyond the schema inference sample is read as floats."""
    filepath = tmp_path / "latefloat.txt"
    nintrows = 20000  # more rows than the schema inference sample
    filepath.write_text("a b\n" + "".join(f"{i} 1\n" for i in range(nintrows)) + f"{nintrows} 2.5\n")

    df = at.misc.read_wsv(filepath)
    assert df["b"].dtype == pl.Float64
    assert df["b"].item(-1) == pytest.approx(2.5)
    assert df.height == nintrows + 1


def test_firstexisting_anyexist(tmp_path: Path) -> None:
    # first existing entry in the list wins
    firstdir = tmp_path / "first"
    firstdir.mkdir()
    (firstdir / "a.txt").write_text("a")
    (firstdir / "b.txt").write_text("b")
    assert at.firstexisting(["a.txt", "b.txt"], folder=firstdir) == firstdir / "a.txt"
    assert at.firstexisting(["missing.txt", "b.txt"], folder=firstdir) == firstdir / "b.txt"

    # search one level into subfolders
    subdir = tmp_path / "sub"
    (subdir / "nested").mkdir(parents=True)
    (subdir / "nested" / "deep.txt").write_text("deep")
    assert at.firstexisting(["deep.txt"], folder=subdir) == subdir / "nested" / "deep.txt"

    # tryzipped locates a compressed variant
    zipdir = tmp_path / "zipped"
    zipdir.mkdir()
    (zipdir / "data.txt.xz").write_text("compressed")
    assert at.firstexisting(["data.txt"], folder=zipdir, tryzipped=True) == zipdir / "data.txt.xz"

    # nothing found raises with a helpful message
    with pytest.raises(FileNotFoundError, match="None of these files exist"):
        at.firstexisting(["nope.txt"], folder=zipdir)

    # firstexisting_or_none returns the path if found, else None
    assert at.misc.firstexisting_or_none(["a.txt"], folder=firstdir) == firstdir / "a.txt"
    assert at.misc.firstexisting_or_none(["nope.txt"], folder=firstdir) is None


def test_firstexisting_and_the_packets_cache_take_one_folder_order(tmp_path: Path) -> None:
    """The reader and the freshness check of the packets cache must take the file from one folder.

    firstexisting sorted the paths of the files, thus run2/x came before run/x, but the check sorted the
    folders. The cache then took the time stamp of a file that the conversion did not read.
    """
    from artistools.packets.core import get_packets_textsource_mtimes

    for foldername, mtime in (("run", 1000.0), ("run2", 2000.0)):
        (tmp_path / foldername).mkdir()
        packetsfile = tmp_path / foldername / "packets00_0000.out"
        packetsfile.write_text("")
        os.utime(packetsfile, (mtime, mtime))

    assert at.firstexisting("packets00_0000.out", folder=tmp_path) == tmp_path / "run" / "packets00_0000.out"
    assert get_packets_textsource_mtimes(tmp_path, ["packets00_0000.out"]) == [1000.0]


def test_firstexisting_with_an_absolute_path(tmp_path: Path) -> None:
    """An absolute path is not below the default folder, but the message must not raise a ValueError."""
    (tmp_path / "here.txt").write_text("here")
    assert at.firstexisting(tmp_path / "here.txt") == tmp_path / "here.txt"

    missingpath = tmp_path / "notafile.txt"
    with pytest.raises(FileNotFoundError, match=str(missingpath)):
        at.firstexisting(missingpath)

    assert at.misc.firstexisting_or_none(missingpath) is None


def test_readnoncommentline() -> None:
    stream = io.StringIO("\n# a comment\n   # indented comment\nreal data line\nsecond\n")
    assert at.misc.readnoncommentline(stream) == "real data line\n"
    # the next call continues from where the last one stopped
    assert at.misc.readnoncommentline(stream) == "second\n"

    # reaching EOF without a data line raises rather than looping forever
    with pytest.raises(EOFError, match="end of file"):
        at.misc.readnoncommentline(io.StringIO(""))
    with pytest.raises(EOFError, match="end of file"):
        at.misc.readnoncommentline(io.StringIO("\n# only comments\n   \n"))


def test_get_file_metadata(tmp_path: Path) -> None:
    # r_v is derived from a_v and e_bminusv
    (tmp_path / "rv.txt").write_text("data")
    (tmp_path / "rv.txt.meta.yml").write_text("a_v: 1.0\ne_bminusv: 0.5\n")
    assert at.misc.get_file_metadata(tmp_path / "rv.txt")["r_v"] == pytest.approx(2.0)

    # a_v is derived from e_bminusv and r_v
    (tmp_path / "av.txt").write_text("data")
    (tmp_path / "av.txt.meta.yml").write_text("e_bminusv: 0.4\nr_v: 3.0\n")
    assert at.misc.get_file_metadata(tmp_path / "av.txt")["a_v"] == pytest.approx(1.2)

    # e_bminusv is derived from a_v and r_v
    (tmp_path / "ebv.txt").write_text("data")
    (tmp_path / "ebv.txt.meta.yml").write_text("a_v: 2.0\nr_v: 4.0\n")
    assert at.misc.get_file_metadata(tmp_path / "ebv.txt")["e_bminusv"] == pytest.approx(0.5)

    # metadata can also come from a combined metadata.yml keyed by the file path
    combineddir = tmp_path / "combined"
    combineddir.mkdir()
    combinedfile = combineddir / "spectrum.txt"
    combinedfile.write_text("data")
    (combineddir / "metadata.yml").write_text(yaml.safe_dump({str(combinedfile): {"a_v": 1.0, "e_bminusv": 0.25}}))
    combined_metadata = at.misc.get_file_metadata(combinedfile)
    assert combined_metadata["r_v"] == pytest.approx(4.0)

    # no metadata file present -> empty dict
    (tmp_path / "nometa.txt").write_text("data")
    assert at.misc.get_file_metadata(tmp_path / "nometa.txt") == {}


def test_write_parquet_atomic(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    parquetpath = tmp_path / "out.parquet"

    at.misc.write_parquet_atomic(df, parquetpath)

    assert parquetpath.exists()
    pltest.assert_frame_equal(pl.read_parquet(parquetpath), df)
    # the temporary partial file must not be left behind
    assert list(tmp_path.glob("*.partial*")) == []


def test_write_parquet_atomic_names_a_read_only_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read-only mount gives errno EROFS, which is no PermissionError, thus the message came as a traceback."""
    import errno
    import tempfile

    def readonly_mkstemp(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, str]:
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(tempfile, "mkstemp", readonly_mkstemp)
    with pytest.raises(PermissionError, match="is read-only"):
        at.misc.write_parquet_atomic(pl.DataFrame({"a": [1]}), tmp_path / "out.parquet")


def test_write_parquet_atomic_temp_file_is_invisible_to_globs(tmp_path: Path) -> None:
    """A reader globbing for the destination name must not pick up the in-flight temporary file."""
    # get_runfolder_timesteps() scans the first match of this pattern, so a match on the temporary means
    # reading a file that is empty, half-written, or already renamed away
    parquetpath = tmp_path / "estimbatch00_0000_0000.out.parquet.tmp"
    seen_midwrite: list[str] = []
    real_sink_parquet = pl.LazyFrame.sink_parquet

    def spy_sink_parquet(self: pl.LazyFrame, path: t.Any, **kwargs: t.Any) -> t.Any:
        seen_midwrite.extend(p.name for p in tmp_path.glob("estimbatch*.out.parquet*"))
        return real_sink_parquet(self, path, **kwargs)

    with mock.patch.object(pl.LazyFrame, "sink_parquet", spy_sink_parquet):
        at.misc.write_parquet_atomic(pl.DataFrame({"timestep": [0, 1]}), parquetpath)

    assert not seen_midwrite, f"a concurrent reader would have globbed the in-flight temporary file {seen_midwrite}"
    assert pl.read_parquet(parquetpath)["timestep"].to_list() == [0, 1]


def test_write_parquet_atomic_is_readable_in_a_shared_directory(tmp_path: Path) -> None:
    """A cache written into a group-shared model directory must not inherit mkstemp's private 0600 mode."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)  # chmod, not mkdir(mode=...), which the process umask would mask off
    parquetpath = shared / "out.parquet"

    at.misc.write_parquet_atomic(pl.DataFrame({"a": [1]}), parquetpath)
    assert parquetpath.stat().st_mode & 0o777 == 0o664

    # a private directory must stay private, and rewriting keeps whatever mode the destination already had
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o700)
    privatepath = private / "out.parquet"
    at.misc.write_parquet_atomic(pl.DataFrame({"a": [1]}), privatepath)
    assert privatepath.stat().st_mode & 0o777 == 0o600

    privatepath.chmod(0o640)
    at.misc.write_parquet_atomic(pl.DataFrame({"a": [2]}), privatepath, replaces=at.misc.get_file_identity(privatepath))
    assert privatepath.stat().st_mode & 0o777 == 0o640
    assert pl.read_parquet(privatepath)["a"].item() == 2


def test_write_parquet_atomic_keeps_a_concurrently_written_file(tmp_path: Path) -> None:
    """A cache another process finished first is kept, because a reader may already be streaming it.

    polars opens a parquet file again by its path between reading the metadata and reading the row groups,
    so renaming a second copy of the same data onto the path makes every scan in progress read the new file
    with the offsets of the old one.
    """
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    theirs = pl.DataFrame({"a": [1, 2, 3]})
    real_sink_parquet = pl.LazyFrame.sink_parquet

    def sink_parquet_after_another_process_finished(self: pl.LazyFrame, path: t.Any, **kwargs: t.Any) -> t.Any:
        result = real_sink_parquet(self, path, **kwargs)
        # the other process finishes while this one writes, so the destination appears from nowhere.
        # DataFrame.write_parquet() would re-enter the patched method
        real_sink_parquet(theirs.lazy(), parquetpath)
        return result

    with mock.patch.object(pl.LazyFrame, "sink_parquet", sink_parquet_after_another_process_finished):
        at.misc.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath)

    assert pl.read_parquet(parquetpath)["a"].to_list() == [1, 2, 3], "the file a reader may hold was replaced"
    assert list(tmp_path.glob("*.partial*")) == []


def test_write_parquet_atomic_replaces_an_outdated_file(tmp_path: Path) -> None:
    """The file whose identity the caller passes as replaces gets replaced, since its data is out of date."""
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(parquetpath)

    at.misc.write_parquet_atomic(
        pl.DataFrame({"a": [4, 5, 6]}), parquetpath, replaces=at.misc.get_file_identity(parquetpath)
    )

    assert pl.read_parquet(parquetpath)["a"].to_list() == [4, 5, 6]
    assert list(tmp_path.glob("*.partial*")) == []

    # without replaces, an existing file is kept: this write does not claim to supersede anything
    at.misc.write_parquet_atomic(pl.DataFrame({"a": [7, 8, 9]}), parquetpath)
    assert pl.read_parquet(parquetpath)["a"].to_list() == [4, 5, 6]


def test_write_parquet_atomic_replaces_only_the_file_found_outdated(tmp_path: Path) -> None:
    """A rival that already replaced the out-of-date file is kept, whenever this writer started.

    The identity comes from the stat that showed the caller its cache was out of date. Taking it at the
    start of the write instead would snapshot the rival's fresh file as the one to replace, and rename over
    it while a reader scans it.
    """
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(parquetpath)
    outdated = at.misc.get_file_identity(parquetpath)

    # the rival reads the same inputs and finishes its replacement first
    pl.DataFrame({"a": [4, 5, 6]}).write_parquet(tmp_path / "rival")
    (tmp_path / "rival").replace(parquetpath)
    rival = at.misc.get_file_identity(parquetpath)

    at.misc.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath, replaces=outdated)

    # tmpfs does not give the freed inode of the outdated file to the next file. A comparison with that inode thus
    # passed there when the writer replaced the file of the rival
    assert at.misc.get_file_identity(parquetpath) == rival, "the rival's file must still be in place"
    assert list(tmp_path.glob("*.partial*")) == []


def test_replace_outdated_file_keeps_a_rivals_fresh_replacement(tmp_path: Path) -> None:
    """The second of two writers that found the same out-of-date file must not replace the first's file.

    Both writers pass the identity check taken before their own write, so only the re-check under the lock
    separates "still the out-of-date file" from "already the other writer's fresh replacement".
    """
    destpath = tmp_path / "cache"
    destpath.write_text("outdated", encoding="utf-8")
    outdated = at.misc.get_file_identity(destpath)

    # the first writer replaces the out-of-date file, changing its identity
    first = tmp_path / "first"
    first.write_text("first replacement", encoding="utf-8")
    fileio.replace_outdated_file(first, destpath, outdated)
    assert destpath.read_text(encoding="utf-8") == "first replacement"

    # the second writer still holds the identity of the file both of them found out of date
    second = tmp_path / "second"
    second.write_text("second replacement", encoding="utf-8")
    fileio.replace_outdated_file(second, destpath, outdated)
    assert destpath.read_text(encoding="utf-8") == "first replacement"


def test_replace_outdated_file_waits_for_the_lock_holder(tmp_path: Path) -> None:
    """A writer that finds the replacement lock taken waits, so its caller reads the holder's fresh file.

    Returning at once would let the caller open the out-of-date file in the moment before the holder's
    rename lands. The wait is the blocking flock call, so the holder's rename is simulated inside it.
    """
    destpath = tmp_path / "cache"
    destpath.write_text("outdated", encoding="utf-8")
    outdated = at.misc.get_file_identity(destpath)

    holders_file = tmp_path / "holders_replacement"
    holders_file.write_text("holders replacement", encoding="utf-8")

    def holder_finishes_first(_fd: int, _operation: int) -> None:
        holders_file.replace(destpath)

    replacement = tmp_path / "replacement"
    replacement.write_text("replacement", encoding="utf-8")
    with mock.patch("fcntl.flock", side_effect=holder_finishes_first) as mockflock:
        fileio.replace_outdated_file(replacement, destpath, outdated)

    assert mockflock.call_count == 1
    assert destpath.read_text(encoding="utf-8") == "holders replacement"


def test_replace_outdated_file_installs_at_an_empty_destination(tmp_path: Path) -> None:
    """An empty destination takes the new file, so a file system without hard links can still create it."""
    destpath = tmp_path / "cache"
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement", encoding="utf-8")

    fileio.replace_outdated_file(replacement, destpath, None)

    assert destpath.read_text(encoding="utf-8") == "replacement"

    # a file that is already there is kept when this write does not claim to replace anything
    another = tmp_path / "another"
    another.write_text("another", encoding="utf-8")
    fileio.replace_outdated_file(another, destpath, None)
    assert destpath.read_text(encoding="utf-8") == "replacement"


def test_write_parquet_atomic_applies_the_identity_rule_without_hard_links(tmp_path: Path) -> None:
    """A file system that rejects hard links gets the same locked identity rule, not a bare rename."""
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    theirs = pl.DataFrame({"a": [1, 2, 3]})
    real_sink_parquet = pl.LazyFrame.sink_parquet

    def sink_parquet_after_another_process_finished(self: pl.LazyFrame, path: t.Any, **kwargs: t.Any) -> t.Any:
        result = real_sink_parquet(self, path, **kwargs)
        real_sink_parquet(theirs.lazy(), parquetpath)
        return result

    with (
        mock.patch.object(fileio.os, "link", side_effect=OSError),
        mock.patch.object(pl.LazyFrame, "sink_parquet", sink_parquet_after_another_process_finished),
    ):
        at.misc.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath)

    assert pl.read_parquet(parquetpath)["a"].to_list() == [1, 2, 3], "the file a reader may hold was replaced"

    # the fallback still replaces the file the caller found out of date
    with mock.patch.object(fileio.os, "link", side_effect=OSError):
        at.misc.write_parquet_atomic(
            pl.DataFrame({"a": [7, 8, 9]}), parquetpath, replaces=at.misc.get_file_identity(parquetpath)
        )
    assert pl.read_parquet(parquetpath)["a"].to_list() == [7, 8, 9]


def test_replace_outdated_file_ignores_a_leftover_lock_file(tmp_path: Path) -> None:
    """A lock file that no process holds does not block: the flock, not the file, is the lock.

    The operating system releases the flock of a holder that dies, so a leftover file from an earlier
    replacement carries no lock. The file stays in place: removing it while a rival waits on it would
    hand out a second lock on a new inode.
    """
    destpath = tmp_path / "cache"
    destpath.write_text("outdated", encoding="utf-8")
    outdated = at.misc.get_file_identity(destpath)
    lockpath = tmp_path / ".cache.replace-lock"
    lockpath.touch()

    replacement = tmp_path / "replacement"
    replacement.write_text("replacement", encoding="utf-8")
    fileio.replace_outdated_file(replacement, destpath, outdated)

    assert destpath.read_text(encoding="utf-8") == "replacement"
    assert lockpath.exists()
    # a different user in a shared model directory opens the lock for writing, because NFS needs that for an
    # exclusive flock. The chmod gives that access whatever the umask
    assert lockpath.stat().st_mode & 0o666 == 0o666


def test_get_file_identity(tmp_path: Path) -> None:
    """A file is the same file only while its device and inode are unchanged."""
    filepath = tmp_path / "cache"
    assert at.misc.get_file_identity(filepath) is None

    filepath.write_text("first", encoding="utf-8")
    identity = at.misc.get_file_identity(filepath)
    assert at.misc.get_file_identity(filepath) == identity

    # rewriting in place keeps the file, but a rename onto the path makes it a different one
    filepath.write_text("second", encoding="utf-8")
    assert at.misc.get_file_identity(filepath) == identity

    replacement = tmp_path / "replacement"
    replacement.write_text("third", encoding="utf-8")
    replacement.replace(filepath)
    assert at.misc.get_file_identity(filepath) != identity


# --- general.py --------------------------------------------------------------------------------


def test_bin_index_of_nan_is_null_and_of_an_outside_value_is_outside_the_bins() -> None:
    """Polars bin_intervals() puts NaN into the last bin, and cut() gave NaN no bin."""
    x = pl.col("x")
    df = pl.DataFrame({"x": [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, None, math.nan]})

    leftclosed = df.select(get_bin_index_expr(x, [0.0, 1.0, 2.0])).to_series().to_list()
    assert leftclosed == [-1, 0, 0, 1, 2, 2, None, None]

    rightclosed = df.select(get_bin_index_expr(x, [0.0, 1.0, 2.0], right_closed=True)).to_series().to_list()
    assert rightclosed == [-1, -1, 0, 0, 1, 2, None, None]

    # plotestimators bins an integer column for -x timestep or -x modelgridindex, and the NaN test must accept it
    dfint = pl.DataFrame({"x": [0, 1, 2, None]}, schema={"x": pl.Int32})
    assert dfint.lazy().select(get_bin_index_expr(x, [0.5, 1.5])).collect().to_series().to_list() == [-1, 0, 1, None]


def test_df_filter_minmax_bracketed() -> None:
    df = pl.DataFrame({"x": list(range(11))})  # 0..10

    # both bounds: keep the interior plus the nearest exterior row on each side (for interpolation)
    bounded = at.misc.df_filter_minmax_bracketed(df, "x", 2.5, 7.5).collect()
    assert bounded["x"].to_list() == [2, 3, 4, 5, 6, 7, 8]

    # no bounds is a pass-through
    unbounded = at.misc.df_filter_minmax_bracketed(df, "x", None, None).collect()
    assert unbounded["x"].to_list() == list(range(11))

    # single-sided bounds
    minonly = at.misc.df_filter_minmax_bracketed(df, "x", 2.5, None).collect()
    assert minonly["x"].to_list() == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    maxonly = at.misc.df_filter_minmax_bracketed(df, "x", None, 7.5).collect()
    assert maxonly["x"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


# --- cliutils.py -------------------------------------------------------------------------------


def test_parse_range() -> None:
    assert list(at.misc.parse_range("3-5", {})) == [3, 4, 5]
    assert list(at.misc.parse_range("5", {})) == [5]
    assert list(at.misc.parse_range("5-3", {})) == [3, 4, 5]  # reversed range is sorted
    assert list(at.misc.parse_range("start-end", {"start": 2, "end": 4})) == [2, 3, 4]

    with pytest.raises(ValueError, match="Bad range"):
        at.misc.parse_range("1-2-3", {})

    # "last-1" means the timestep before the last to a user. A swap gave timesteps 1 to last
    with pytest.raises(ValueError, match="ends before it starts"):
        at.misc.parse_range("last-1", {"last": 99})


def test_normalize_path_list() -> None:
    assert at.misc.normalize_path_list("a/b") == [Path("a/b")]
    assert at.misc.normalize_path_list(Path("a/b")) == [Path("a/b")]
    assert at.misc.normalize_path_list([["x"], "y"]) == [Path("x"), Path("y")]
    assert at.misc.normalize_path_list([]) == [Path()]
    assert at.misc.normalize_path_list(None, default="fallback") == [Path("fallback")]


def test_resolve_outputfile(tmp_path: Path) -> None:
    # no outputfile falls back to the default filename
    assert at.misc.resolve_outputfile(None, "default.pdf") == Path("default.pdf")

    # an existing directory gets the default filename appended
    existingdir = tmp_path / "existing"
    existingdir.mkdir()
    assert at.misc.resolve_outputfile(existingdir, "default.pdf") == existingdir / "default.pdf"

    # a path with a file extension is returned unchanged
    assert at.misc.resolve_outputfile(tmp_path / "chosen.pdf", "default.pdf") == tmp_path / "chosen.pdf"

    # a suffixless path is treated as a folder, created, and the default filename appended
    newdir = tmp_path / "newfolder"
    assert at.misc.resolve_outputfile(newdir, "default.pdf") == newdir / "default.pdf"
    assert newdir.is_dir()


def test_set_args_from_dict() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-foo", type=int, default=1)
    parser.add_argument("-o", "--output", dest="outputfile", default="z")

    # defaults can be overridden by dest name ("foo") or by an option string whose dest differs ("output")
    at.misc.set_args_from_dict(parser, {"foo": 5, "output": "y"})
    args = parser.parse_args([])
    assert args.foo == 5
    assert args.outputfile == "y"

    with pytest.raises(ValueError, match="Unknown argument names"):
        at.misc.set_args_from_dict(parser, {"nonexistent": 1})


def test_remote_path_follows_the_rule_of_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A colon before the first slash makes a remote path, whether a local file of that name exists or not.

    Path removes the "./" that marks a local path, thus the text of the argument decides.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run:2").mkdir()
    assert remote.split_remote_path("user@vae26:model") == ("user@vae26", Path("~/model"))
    assert remote.split_remote_path(Path("vae26:/lustre/model")) == ("vae26", Path("/lustre/model"))
    assert remote.is_remote_path("run:2")
    assert not remote.is_remote_path("./run:2")
    assert not remote.is_remote_path("runs/run:2")
    assert remote.model_path_from_text("./run:2") == tmp_path / "run:2"
    assert at.misc.normalize_path_list(["./run:2", "vae26:model"]) == [tmp_path / "run:2", Path("vae26:~/model")]
    # rsync takes an IPv6 address in brackets, and ssh takes it with no brackets
    assert remote.split_remote_path("user@[2001:db8::1]:/model") == ("user@[2001:db8::1]", Path("/model"))
    assert remote.get_server_argv("user@[2001:db8::1]", "artistools server")[1:3] == ["--", "user@2001:db8::1"]
    # a label of a list option can have the form host:path, thus only a remote path that is clearly a folder counts
    assert remote.names_a_remote_folder("vae26:~/model")
    assert not remote.names_a_remote_folder("second:label")
    # a band plot of two hosts gives each call the Namespace with both models. Only the argument of the model counts
    commandargs = argparse.Namespace(modelpath=[Path("hosta:~/x"), Path("hostb:~/y")], stream=sys.stdout)
    host, (serverargs, _) = remote.to_server_arguments(((Path("hosta:~/x"), commandargs), {}))
    assert host == "hosta"
    assert not hasattr(serverargs[1], "stream")


def test_reply_of_the_server_cannot_call_a_function(tmp_path: Path) -> None:
    """The client refuses a reply that calls a function, because a different user can control the remote host.

    subprocess.call runs a command on the client, and subprocess is not a module of the replies.
    numpy.memmap is a class of numpy that writes a file, thus only the numpy classes of a result pass.
    """
    import pickle  # ruff:ignore[suspicious-pickle-import]

    class CommandOnClient:
        def __reduce__(self) -> tuple[t.Any, ...]:
            return (subprocess.call, (["true"],))

    with pytest.raises(pickle.UnpicklingError, match="does not accept"):
        remote.load_reply(pickle.dumps((True, CommandOnClient())))

    class FileOnClient:
        def __reduce__(self) -> tuple[t.Any, ...]:
            return (np.memmap, (str(tmp_path / "overwritten"), "float64", "w+", 0, (1,)))

    with pytest.raises(pickle.UnpicklingError, match="does not accept"):
        remote.load_reply(pickle.dumps((True, FileOnClient())))
    assert not (tmp_path / "overwritten").exists()

    # the __setstate__ of a polars frame unpickles a plan with the plain unpickler, thus a reply holds no polars object
    with pytest.raises(pickle.UnpicklingError, match="does not accept"):
        remote.load_reply(pickle.dumps((True, pl.LazyFrame({"a": [1]}))))
    succeeded, frames = remote.load_reply(
        remote.dump_message((True, {-1: pl.LazyFrame({"a": [1.0]}), 0: pl.DataFrame()}))
    )
    assert isinstance(frames[-1], pl.LazyFrame)
    assert isinstance(frames[0], pl.DataFrame)

    result = {"array": np.arange(3.0), "value": np.float64(2.5), "dtype": np.dtype("f8")}
    succeeded, reply = remote.load_reply(remote.dump_message((True, result)))
    assert succeeded
    np.testing.assert_array_equal(reply["array"], result["array"])
    assert reply["value"] == result["value"]
    assert reply["dtype"] == result["dtype"]


def test_server_command_of_a_git_install_names_its_commit() -> None:
    """A git install gives the commit for the server command, and a release gives no suggestion.

    uvx --from git+... records the commit in direct_url.json. An install of a clone gives the commit of the folder
    of the code that runs, which can differ from the folder in direct_url.json.
    """
    import json

    def mock_direct_url(directurl: str | None) -> t.Any:
        distribution = mock.Mock()
        distribution.read_text.return_value = directurl
        return mock.patch("importlib.metadata.distribution", return_value=distribution)

    vcsinstall = {"url": "https://github.com/fork/artistools", "vcs_info": {"vcs": "git", "commit_id": "abc123"}}
    with mock_direct_url(json.dumps(vcsinstall)):
        gitsource = remote.get_git_source()
    assert gitsource is not None
    assert gitsource == remote.GitSource(
        "https://github.com/fork/artistools", "abc123", ispushed=True, releasechanges=None, notes=[]
    )
    expected = (
        f"POLARS_MAX_THREADS=16 uvx --with polars=={pl.__version__}"
        ' --from "artistools @ git+https://github.com/fork/artistools@abc123" artistools server'
    )
    assert remote.get_git_command(gitsource) == expected
    suggestion = remote.get_git_server_suggestion("vae26", None, gitsource)
    assert f"export ARTISTOOLS_REMOTE_COMMAND='{expected}'" in suggestion
    assert "but the server command runs a release" in suggestion
    # a command of ARTISTOOLS_REMOTE_COMMAND can name an old commit, thus the warning tells whether it names this one
    oldcommand = "uvx --from 'artistools @ git+https://x@0ld' artistools server"
    assert "does not name this commit" in remote.get_git_server_suggestion("vae26", oldcommand, gitsource)
    assert "names this commit. The usual command" in remote.get_git_server_suggestion("vae26", expected, gitsource)

    # an install from PyPI records no direct_url.json, thus its server is the release of the same version
    with mock_direct_url(None):
        assert remote.get_git_source() is None
    command, _ = remote.choose_server_command(None, None)
    assert command.endswith(f" artistools@{importlib.metadata.version('artistools')} server")

    packagefolder = Path(remote.__file__).resolve().parents[2]
    headcommit = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        ["git", "-C", str(packagefolder), "rev-parse", "HEAD"],  # ruff:ignore[start-process-with-partial-path]
        capture_output=True,
        text=True,
        check=False,
    )
    if headcommit.returncode != 0:
        pytest.skip("artistools does not run from a git clone")
    with mock_direct_url(json.dumps({"url": "file:///somewhere/else", "dir_info": {"editable": True}})):
        gitsource = remote.get_git_source()
    assert gitsource is not None
    # the URL is the one of the remote that holds the commit, which can be a fork
    assert gitsource.commit == headcommit.stdout.strip()


def test_server_command_runs_the_code_of_this_artistools() -> None:
    """Choose a given command first, else the release if it has the code of this commit or the commit is only local.

    A release and a later commit have the same version and the same protocol, but a reader can differ, e.g. a new
    column. The server of a commit needs a build on the host, thus the release is the choice for the same code.
    """
    releasecommand = remote.get_release_command()
    changed = remote.GitSource("https://github.com/a/b", "abc123", ispushed=True, releasechanges=["x.py"], notes=[])
    gitcommand = remote.get_git_command(changed)
    assert remote.choose_server_command(None, None)[0] == releasecommand
    assert remote.choose_server_command("my command", changed)[0] == "my command"
    assert remote.choose_server_command(None, changed._replace(releasechanges=[]))[0] == releasecommand
    assert remote.choose_server_command(None, changed)[0] == gitcommand
    # a commit with no release tag to compare with, e.g. an install from git, runs from git
    assert remote.choose_server_command(None, changed._replace(releasechanges=None))[0] == gitcommand
    # a host cannot get a commit that is only on this host
    assert remote.choose_server_command(None, changed._replace(ispushed=False))[0] == releasecommand


def test_server_of_a_commit_falls_back_to_the_release() -> None:
    """A host with no Rust cannot build the commit, thus the release starts after the failed start of the commit."""
    from importlib.metadata import version

    changed = remote.GitSource("https://github.com/a/b", "abc123", ispushed=True, releasechanges=["x.py"], notes=[])
    started = mock.Mock()
    with (
        mock.patch.object(remote, "get_git_source", return_value=changed),
        mock.patch.object(
            remote, "launch_server", side_effect=[EOFError(), (started, version("artistools"), pl.__version__)]
        ) as launch,
        mock.patch("atexit.register"),
        mock.patch.dict("os.environ", {"ARTISTOOLS_REMOTE_COMMAND": ""}),
    ):
        process, _ = remote.start_server("vae26")
    assert process is started
    assert [call.args[0][-1] for call in launch.call_args_list] == [
        remote.get_git_command(changed),
        remote.get_release_command(),
    ]


def test_server_of_a_different_protocol_stops_the_start() -> None:
    """The start line of a different protocol must give an error at once.

    That server waits for a request after its start line, thus a search for the next line would wait for ever.
    """
    fakescript = (
        "import sys, time; sys.stdout.buffer.write(b'motd\\nartistools server protocol 1\\n'); sys.stdout.flush()"
    )
    fakeserver = subprocess.Popen([sys.executable, "-c", f"{fakescript}; time.sleep(60)"], stdout=subprocess.PIPE)  # ruff:ignore[subprocess-without-shell-equals-true]
    try:
        with pytest.raises(ConnectionError, match="protocol 1"):
            remote.read_server_versions(fakeserver)
    finally:
        fakeserver.kill()
        fakeserver.wait()


def test_reader_of_a_remote_model_runs_on_the_server(tmp_path: Path) -> None:
    """A reader that gets a host:path runs on the server, and it gives the same data as for the local path.

    A local server process takes the place of ssh. The filter function and the exception must go
    through pickle, and the LazyFrames come back collected. The command runs through the dispatcher,
    because its Namespace holds the parser and, with --quiet, a stream that pickle cannot send. The
    folder comes after a list option, thus the dispatcher must see that it names the model.
    """
    from artistools.__main__ import main

    modelpath = at.get_path("testartismodel").resolve()
    remotepath = Path(f"testhost:{modelpath}")
    filterfunc = at.misc.get_filterfunc(argparse.Namespace(filtersavgol=["5", "3"]))

    remote.forget_server("testhost")
    with mock.patch.object(remote, "get_server_argv", return_value=[sys.executable, "-m", "artistools", "server"]):
        process, _ = remote.get_server("testhost")
        try:
            assert at.misc.path_is_dir(remotepath)
            assert not at.misc.path_is_file(remotepath)
            remotespectra = at.spectra.get_spectra(remotepath, timestepmin=40, fluxfilterfunc=filterfunc)
            remotelightcurve = at.lightcurve.scan_lightcurve(remotepath / "light_curve.out")
            # a caller skips a model on FileNotFoundError, thus the server must give back the same type
            with pytest.raises(FileNotFoundError, match="nosuchfile"):
                at.misc.firstexisting("nosuchfile.out", folder=remotepath, search_subfolders=False)
            main(argsraw=["plotlightcurve", "-label", "mylabel", str(remotepath), "--quiet", "-o", str(tmp_path)])
            # a collect, e.g. in a script, asks the host for the rows of the filter and the columns of the query
            dfremoteestimators = (
                at
                .scan_estimators(remotepath, timestep=[40, 41], join_modeldata=True)
                .filter(pl.col("Te") > 5000.0)
                .select("timestep", "modelgridindex", "Te", "rho")
                .collect()
            )
            # the host collects the data of all the subplots of plotestimators in one call, and the client draws
            with mock.patch.object(remote, "call_on_host", wraps=remote.call_on_host) as mockcall:
                main(
                    argsraw=[
                        *("plotestimators", "Te", "-plot", "TR", "nne", "-plot", "populations", "Fe II", "Fe III"),
                        *(str(remotepath), "-timestep", "40", "-x", "velocity", "-o", str(tmp_path / "est.pdf")),
                    ]
                )
            assert [call.args[2] for call in mockcall.call_args_list].count("get_figures_data") == 1
            # an empty selection comes back from the host as its own error, and the command then names the data
            emptyargs = parse_cli_args(
                plotestimators.addargs,
                None,
                None,
                ["Te", str(remotepath), "-ts", "40", "-x", "velocity", "-xmin", "1e9"],
            )
            emptymodelpath, emptytimesteps = plotestimators.resolve_plot_args(emptyargs)
            with pytest.raises(plotestimators.NoEstimatorRowsError):
                plotestimators.get_figures_data(emptymodelpath, emptyargs, emptytimesteps)
            # a window reads the first line of an error and the last warning from the buffer of its thread, thus a quiet
            # request brings back the standard error of the host
            for tokens, expected in (
                (["nosuchvariable"], ("'nosuchvariable' is not an estimator variable", "")),
                (["-plot", "populations", "Fe II", "Zz IX"], (None, "Can't plot populations for {(-1, 9)}")),
            ):

                def draw_remote_plot(plottokens: list[str] = tokens) -> None:
                    plotargs = parse_cli_args(plotestimators.addargs, None, None, [*plottokens, str(remotepath)])
                    plotestimators.draw_plot(plotargs, mplfig.Figure())

                message, warning = run_command_step_with_warning(draw_remote_plot, quiet=True)
                assert message == expected[0]
                assert warning.startswith(expected[1])
            estimatorscore = sys.modules["artistools.estimators.core"]
            read_estimator_rows_on_host = estimatorscore.read_estimator_rows_on_host
            hostframes: list[pl.DataFrame] = []

            def read_rows_on_host(*args: t.Any) -> pl.DataFrame:
                hostframes.append(read_estimator_rows_on_host(*args))
                return hostframes[-1]

            with mock.patch.object(estimatorscore, "read_estimator_rows_on_host", side_effect=read_rows_on_host):
                dfremotecell = (
                    at
                    .scan_estimators(str(remotepath))
                    .filter(pl.col("timestep") == 41)
                    .filter(pl.col("modelgridindex") == 0)
                    .select("nne")
                    .collect()
                )
            # the host applies the filter, thus only the row of the cell comes back through ssh
            assert [frame.height for frame in hostframes] == [1]
            # the band light curves take the Namespace of the command as an argument
            main(argsraw=["plotlightcurve", str(remotepath), "-filter", "B", "--quiet", "-o", str(tmp_path)])
        finally:
            assert process.stdin is not None
            process.stdin.close()
            assert process.wait(timeout=30) == 0
            remote.forget_server("testhost")

    assert (tmp_path / "plotlightcurves.pdf").is_file()
    assert (tmp_path / "est.pdf").is_file()
    localestimators, _ = join_cell_modeldata(at.estimators.scan_estimators(modelpath, timestep=[40, 41]), modelpath)
    pltest.assert_frame_equal(
        dfremoteestimators,
        localestimators.filter(pl.col("Te") > 5000.0).select("timestep", "modelgridindex", "Te", "rho").collect(),
        abs_tol=0.0,
    )
    dflocalcell = (
        at
        .scan_estimators(modelpath)
        .filter(pl.col("timestep") == 41)
        .filter(pl.col("modelgridindex") == 0)
        .select("nne")
        .collect()
    )
    assert dflocalcell.height == 1
    pltest.assert_frame_equal(dfremotecell, dflocalcell, abs_tol=0.0)
    assert (tmp_path / "plotBlightcurves.pdf").is_file()
    localspectra = at.spectra.get_spectra(modelpath, timestepmin=40, fluxfilterfunc=filterfunc)
    # the fluxes are far below the default absolute tolerance, thus the comparison has none
    pltest.assert_frame_equal(remotespectra[-1].collect(), localspectra[-1].collect(), abs_tol=0.0)
    pltest.assert_frame_equal(
        remotelightcurve[-1].collect(),
        at.lightcurve.scan_lightcurve(modelpath / "light_curve.out")[-1].collect(),
        abs_tol=0.0,
    )


def test_get_filterfunc() -> None:
    # no filter arguments -> no filter function
    assert at.misc.get_filterfunc(argparse.Namespace()) is None

    # a moving-average filter reproduces a windowed mean (with edge padding)
    filterfunc = at.misc.get_filterfunc(argparse.Namespace(filtermovingavg=3))
    assert filterfunc is not None
    assert filterfunc([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx([4 / 3, 2.0, 3.0, 4.0, 14 / 3])

    # the Savitzky-Golay filter matches scipy.signal.savgol_filter(y, window_length=5, polyorder=3, mode="interp"),
    # which it replaced
    # each filter argument selects one filter, thus a command line that gives both is a user error
    # and not an internal fault
    with pytest.raises(ValueError, match="only one of -filtermovingavg and -filtersavgol"):
        at.misc.get_filterfunc(argparse.Namespace(filtermovingavg=3, filtersavgol=["5", "3"]))

    filterfunc = at.misc.get_filterfunc(argparse.Namespace(filtersavgol=["5", "3"]))
    assert filterfunc is not None
    yvalues = np.sin(np.linspace(0.0, 3.0, num=12)) + np.linspace(0.0, 0.5, num=12) ** 2
    expected = [
        -4.0498103264955503e-05,
        2.7158701565113680e-01,
        5.2682820534895669e-01,
        7.4815740267140873e-01,
        9.1968938266004074e-01,
        1.0298135494226284e00,
        1.0717640450420927e00,
        1.0441198836391163e00,
        9.5090999103591567e-01,
        8.0131538546057457e-01,
        6.0937710159007052e-01,
        3.9107049789170700e-01,
    ]
    assert np.allclose(filterfunc(yvalues), expected, rtol=1e-10, atol=1e-12)

    # invalid parameters are rejected
    with pytest.raises(ValueError, match="must be an odd number"):
        at.misc.savgol_filter(yvalues, window_length=4, polyorder=3)
    with pytest.raises(ValueError, match="must be at least zero and less than window_length"):
        at.misc.savgol_filter(yvalues, window_length=5, polyorder=7)
    with pytest.raises(ValueError, match="must be at least zero and less than window_length"):
        at.misc.savgol_filter(yvalues, window_length=5, polyorder=-1)
    with pytest.raises(ValueError, match="exceeds the data length"):
        at.misc.savgol_filter(yvalues[:3], window_length=5, polyorder=3)
    with pytest.raises(ValueError, match="needs a 1D array"):
        at.misc.savgol_filter(np.tile(yvalues, (2, 1)), window_length=5, polyorder=3)


def test_gaussian_filter_wrap() -> None:
    """The smoothing must match scipy.ndimage.gaussian_filter(data, sigma=1.2, mode="wrap"), which it replaced."""
    data = np.outer(np.sin(np.linspace(0.0, np.pi, 4)), np.cos(np.linspace(0.0, 2 * np.pi, 6, endpoint=False)))
    expected = np.array([
        [
            0.16333386804037386,
            0.08166693402018696,
            -0.08166693402018693,
            -0.16333386804037395,
            -0.08166693402018704,
            0.08166693402018683,
        ],
        [
            0.22987577583564325,
            0.11493788791782167,
            -0.11493788791782165,
            -0.22987577583564336,
            -0.11493788791782181,
            0.11493788791782150,
        ],
        [
            0.22987577583564328,
            0.11493788791782168,
            -0.11493788791782164,
            -0.22987577583564340,
            -0.11493788791782182,
            0.11493788791782152,
        ],
        [
            0.16333386804037386,
            0.08166693402018697,
            -0.08166693402018695,
            -0.16333386804037395,
            -0.08166693402018706,
            0.08166693402018685,
        ],
    ])
    assert np.allclose(at.misc.gaussian_filter_wrap(data, sigma=1.2), expected, rtol=1e-10, atol=1e-12)

    with pytest.raises(ValueError, match="must be greater than zero"):
        at.misc.gaussian_filter_wrap(data, sigma=0.0)
    with pytest.raises(ValueError, match="needs a 2D array"):
        at.misc.gaussian_filter_wrap(data[0], sigma=1.2)


# --- timesteps.py ------------------------------------------------------------------------------


def test_get_timestep_of_timedays(tmp_path: Path) -> None:
    write_timesteps_out(tmp_path)

    # timesteps span [100,110), [110,120), ... [140,150)
    assert at.get_timestep_of_timedays(tmp_path, 125) == 2
    assert at.get_timestep_of_timedays(tmp_path, 100) == 0
    assert at.get_timestep_of_timedays(tmp_path, 149) == 4
    assert at.get_timestep_of_timedays(tmp_path, "125d") == 2  # accepts a "<days>d" string

    # the message says the model covers up to 150 days, thus exactly 150 names the last timestep
    assert at.get_timestep_of_timedays(tmp_path, 150) == 4

    # the message names the range that the run covers, so that the user can correct the value
    with pytest.raises(ValueError, match=r"No timestep of this model covers 500 days.*100\.00 to 150\.00 days"):
        at.get_timestep_of_timedays(tmp_path, 500)


def test_get_deposition(tmp_path: Path) -> None:
    write_timesteps_out(tmp_path)
    deplines = ["#tmid_days gammadep_Lsun positrondep_Lsun total_dep_Lsun"]
    for ts in range(5):
        tmid = 105 + ts * 10
        deplines.append(f"{tmid} {ts + 1.0} {(ts + 1) * 0.1} {(ts + 1) * 1.1}")
    (tmp_path / "deposition.out").write_text("\n".join(deplines) + "\n")

    dep = at.get_deposition(tmp_path).collect()

    assert dep.height == 5
    assert {"timestep", "tmid_days", "gammadep_Lsun", "positrondep_Lsun", "total_dep_Lsun"} <= set(dep.columns)
    row = dep.filter(pl.col("timestep") == 2)
    assert row["tmid_days"].item() == pytest.approx(125.0)
    assert row["total_dep_Lsun"].item() == pytest.approx(3.3)

    # deposition times that don't line up with the timesteps are rejected
    baddir = tmp_path / "bad"
    baddir.mkdir()
    write_timesteps_out(baddir)
    badlines = ["#tmid_days gammadep_Lsun positrondep_Lsun total_dep_Lsun"]
    badlines.extend(f"{999 + ts} {ts + 1.0} {(ts + 1) * 0.1} {(ts + 1) * 1.1}" for ts in range(5))
    (baddir / "deposition.out").write_text("\n".join(badlines) + "\n")

    with pytest.raises(ValueError, match="Deposition times do not match"):
        at.get_deposition(baddir).collect()

    # a file with more rows than the model has timesteps gets the same message, not a broadcast error
    longdir = tmp_path / "long"
    longdir.mkdir()
    write_timesteps_out(longdir)
    longlines = deplines + [f"{155 + ts * 10} 1.0 0.1 1.1" for ts in range(2)]
    (longdir / "deposition.out").write_text("\n".join(longlines) + "\n")

    with pytest.raises(ValueError, match="Deposition times do not match"):
        at.get_deposition(longdir).collect()


def test_escaped_arrivalrange_takes_all_timesteps_for_a_deposition_file_of_a_different_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deposition.out whose times do not match the timesteps gives the arrival range a warning, not a stop.

    Only the energy rates need deposition.out, but the check of its times stopped every light curve plot of the model.
    """
    from artistools.misc import get_escaped_arrivalrange
    from artistools.misc.timesteps import get_escaped_arrivalrange_cached

    modelcopy = tmp_path / "model"
    modelcopy.mkdir()
    for sourcefile in (at.get_path("testdata") / "testmodel").iterdir():
        (modelcopy / sourcefile.name).symlink_to(sourcefile)
    ntimesteps = len(at.get_timestep_times(modelcopy, loc="mid"))
    deplines = ["#tmid_days gammadep_Lsun positrondep_Lsun total_dep_Lsun"]
    deplines.extend(f"{999 + ts} 1.0 0.1 1.1" for ts in range(ntimesteps))
    (modelcopy / "deposition.out").write_text("\n".join(deplines) + "\n")

    get_escaped_arrivalrange_cached.cache_clear()
    nts_last, _, _ = get_escaped_arrivalrange(modelcopy)
    assert nts_last == ntimesteps - 1
    assert "Deposition times do not match the timesteps. The plot takes every timestep as complete" in (
        capsys.readouterr().err
    )


def test_average_direction_bins_unequal_bincounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Averaging must group bins by the phi bin count, which is only distinguishable when the two counts differ."""
    nphibins = 4
    ncosthetabins = 3

    monkeypatch.setattr(dirbins, "get_viewingdirection_phibincount", lambda: nphibins)
    monkeypatch.setattr(dirbins, "get_viewingdirection_costhetabincount", lambda: ncosthetabins)

    # dirbin == costheta_index * nphibins + phi_index, and each frame carries its own dirbin as the value
    dirbindataframes = {
        dirbin: pl.DataFrame({"timestep": [0, 1], "value": [float(dirbin), float(dirbin)]})
        for dirbin in range(nphibins * ncosthetabins)
    }

    # averaging over theta collapses each phi index over all costheta rings: bins p, p + nphibins, p + 2 * nphibins
    averaged = dirbins.average_direction_bins(dirbindataframes, overangle="theta")
    assert sorted(averaged.keys()) == [0, 1, 2, 3]
    for phibin in range(nphibins):
        expected = sum(phibin + n * nphibins for n in range(ncosthetabins)) / ncosthetabins
        assert averaged[phibin].collect()["value"].to_list() == pytest.approx([expected, expected])

    # averaging over phi collapses each contiguous run of nphibins bins
    averaged_phi = dirbins.average_direction_bins(dirbindataframes, overangle="phi")
    assert sorted(averaged_phi.keys()) == [0, 4, 8]
    for start_bin in (0, 4, 8):
        expected = sum(start_bin + n for n in range(nphibins)) / nphibins
        assert averaged_phi[start_bin].collect()["value"].to_list() == pytest.approx([expected, expected])


def test_average_direction_bins_averages_every_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every column is averaged, so a column that is not linear in the bins must be derived afterwards.

    This is why scan_lightcurve() derives the magnitude only once the bins are averaged: the mean of the
    magnitudes of the bins is not the magnitude of their mean luminosity.
    """
    nphibins = 4
    ncosthetabins = 3

    monkeypatch.setattr(dirbins, "get_viewingdirection_phibincount", lambda: nphibins)
    monkeypatch.setattr(dirbins, "get_viewingdirection_costhetabincount", lambda: ncosthetabins)

    # one bin of every group is dark, so the mean of the logs is -inf while the log of the mean is finite
    logcol = (pl.col("value").log10()).alias("logvalue")
    dirbindataframes = {
        dirbin: pl.DataFrame({"timestep": [0], "value": [float(dirbin % nphibins)]}).with_columns(logcol)
        for dirbin in range(nphibins * ncosthetabins)
    }

    averaged = dirbins.average_direction_bins(dirbindataframes, overangle="phi")

    meanvalue = sum(range(nphibins)) / nphibins
    for start_bin in (0, 4, 8):
        row = averaged[start_bin].collect()
        assert row["value"].item() == pytest.approx(meanvalue)
        # the column carried through the averaging holds the mean of the logs, which the dark bin sends to -inf
        assert row["logvalue"].item() == -math.inf
        # deriving it from the averaged value instead gives the finite answer that a caller wants
        assert averaged[start_bin].with_columns(logcol).collect()["logvalue"].item() == pytest.approx(
            math.log10(meanvalue)
        )


def test_average_direction_bins_rejects_missing_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Averaging a second time must raise instead of a bare KeyError for the bins that no longer exist."""
    nphibins = 4
    ncosthetabins = 3

    monkeypatch.setattr(dirbins, "get_viewingdirection_phibincount", lambda: nphibins)
    monkeypatch.setattr(dirbins, "get_viewingdirection_costhetabincount", lambda: ncosthetabins)

    dirbindataframes = {
        dirbin: pl.DataFrame({"timestep": [0, 1], "value": [float(dirbin), float(dirbin)]})
        for dirbin in range(nphibins * ncosthetabins)
    }

    averaged = dirbins.average_direction_bins(dirbindataframes, overangle="theta")
    with pytest.raises(ValueError, match="Cannot average over phi"):
        dirbins.average_direction_bins(averaged, overangle="phi")


def test_get_time_range_timesteps_without_clamping(tmp_path: Path) -> None:
    """A timestep range gives no times in days, so the timestep bounds must be used even when not clamping."""
    write_timesteps_out(tmp_path)

    for clamp in (True, False):
        timestepmin, timestepmax, tlow, thigh = at.misc.get_time_range(
            tmp_path, timestep_range_str="1-3", clamp_to_timesteps=clamp
        )
        assert (timestepmin, timestepmax) == (1, 3)
        assert tlow == pytest.approx(at.get_timestep_times(tmp_path, loc="start")[1])
        assert thigh == pytest.approx(at.get_timestep_times(tmp_path, loc="end")[3])

    # a plot reads one range of timesteps. A list gave the range between its ends, thus the plot
    # held the timesteps that the user left out as well
    with pytest.raises(ValueError, match="names no single range"):
        at.misc.get_time_range(tmp_path, timestep_range_str="1,3")

    # the ends of a range in either order name the same timesteps
    assert at.misc.get_time_range(tmp_path, timestep_range_str="3-1")[:2] == (1, 3)


def test_check_averaging_angles() -> None:
    """Averaging over phi and theta at once must be rejected wherever the values arrive."""
    for phi, theta in ((False, False), (True, False), (False, True)):
        at.misc.check_averaging_angles(phi, theta)

    with pytest.raises(ValueError, match="both the phi and theta"):
        at.misc.check_averaging_angles(average_over_phi=True, average_over_theta=True)


def test_viewingangle_averaging_flags_are_mutually_exclusive() -> None:
    """The two averaging flags are rejected by argparse itself, for every command that defines them."""
    parser = argparse.ArgumentParser()
    at.misc.addarg_viewingangle(parser)

    assert parser.parse_args(["--average_over_phi_angle"]).average_over_phi_angle
    assert parser.parse_args(["--average_over_theta_angle"]).average_over_theta_angle

    with pytest.raises(SystemExit):
        parser.parse_args(["--average_over_phi_angle", "--average_over_theta_angle"])


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a platform with a fork start method")
def test_parallel_map_works_when_fork_is_the_default_start_method(tmp_path: Path) -> None:
    """parallel_map must run its workers under spawn even when the interpreter default is fork.

    process_map shares one progress-bar lock with its workers, and tqdm builds that lock from the default
    context, so the pool has to run in that same context. Passing an mp_context to the pool instead of setting
    the default raises "A SemLock created in a fork context is being shared with a process in a spawn context"
    wherever fork is the default, which was every Linux run before Python 3.14.

    This runs in a subprocess: the default start method is process-wide state, so setting it here would leak
    into the rest of the session, and a broken parallel_map would fork a pytest process full of threads.
    """
    script = tmp_path / "forkdefault.py"
    script.write_text(
        """
import multiprocessing as mp
import sys

import artistools as at


def square(x):
    return x * x


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    # a bar comes first, as the estimator reader and the packet reader make one. Its lock must come
    # from a spawn context, or the pool below meets a lock of the fork context
    bar = at.misc.general.get_progress_class()
    for _ in bar(range(2), desc="a bar before the pool"):
        pass

    # a bar starts no process, thus the default start method of the caller stands
    assert mp.get_start_method() == "fork", mp.get_start_method()

    assert at.misc.parallel_map(square, range(4)) == [0, 1, 4, 9]

    # the pool takes a spawn context of its own, thus the default of the process stays for the code of the user
    assert mp.get_start_method() == "fork", mp.get_start_method()
    print("OK")
""",
        encoding="utf-8",
    )

    # put the package's parent on the child's path, so this works from an editable or a plain install
    env = os.environ.copy()
    packageparent = str(at.get_path("artistools_repository"))
    env["PYTHONPATH"] = os.pathsep.join([packageparent, env["PYTHONPATH"]]) if "PYTHONPATH" in env else packageparent

    proc = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, str(script)], capture_output=True, text=True, check=False, cwd=tmp_path, env=env
    )

    assert proc.returncode == 0, f"parallel_map failed under a fork default:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_drop_trailing_null_column() -> None:
    """The all-null trailing column must go, but only when there is data to judge it by.

    is_null().all() is vacuously true over an empty column, so a file with no data rows would otherwise lose a
    real column and no longer match the schema of its sibling rank files.
    """
    # a genuine trailing null column, as a line-ending space produces
    assert at.misc.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [None, None]})).columns == ["a"]
    assert at.misc.drop_trailing_null_column(
        pl.LazyFrame({"a": [1, 2], "b": [None, None]})
    ).collect_schema().names() == ["a"]

    # a real last column, including one that is only partly null, must stay
    assert at.misc.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [3, 4]})).columns == ["a", "b"]
    assert at.misc.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [None, 4]})).columns == ["a", "b"]

    # no rows means nothing to judge, so keep every column
    emptyschema = {"a": pl.Int64, "b": pl.Float64}
    assert at.misc.drop_trailing_null_column(pl.DataFrame({"a": [], "b": []}, schema=emptyschema)).columns == ["a", "b"]
    assert at.misc.drop_trailing_null_column(
        pl.LazyFrame({"a": [], "b": []}, schema=emptyschema)
    ).collect_schema().names() == ["a", "b"]


def test_get_series_label() -> None:
    """A series is named by its -label entry, or by the fallback when the user gave none for it."""
    assert at.misc.get_series_label(["A", "B"], 1, "modelname") == "B"
    assert at.misc.get_series_label([None, "B"], 0, "modelname") == "modelname"

    # trim_or_pad sizes the list to the model paths, so a per-series index can run off the end
    assert at.misc.get_series_label(["A"], 3, "modelname") == "modelname"

    # a sentinel index such as the -1 this codebase uses for a direction bin must not wrap to the last label
    assert at.misc.get_series_label(["A", "B"], -1, "modelname") == "modelname"

    # an empty label is a series deliberately left out of the legend, not a missing one
    # the return type is str, so falsy is the empty string rather than the model name
    assert not at.misc.get_series_label([""], 0, "modelname")


def test_shorten_middle_keeps_both_ends() -> None:
    """A long run folder name keeps the model at the start and the run details at the end."""
    name = "w7_outercut_20260816_150_410d_2e9pkt_develop_3dgrid50_virgo"
    short = at.misc.modelinfo.shorten_middle(name, 50)

    assert len(short) == 50
    assert short.startswith("w7_outercut")
    assert short.endswith("virgo")
    assert "..." in short

    # a name that fits stays whole, and no maximum length leaves it alone
    assert at.misc.modelinfo.shorten_middle("testmodel", 50) == "testmodel"
    assert at.misc.modelinfo.shorten_middle(name, None) == name


def test_check_time_selection_refuses_two_ways_to_name_one_range() -> None:
    """-timestep, -timedays, and the pair -timemin/-timemax each name a time range, thus one may come.

    get_time_range reads the range of -timedays alone, thus "-timedays 250-300 -timemin 280" gave the
    range of -timedays and took no notice of the bound.
    """
    import artistools.spectra.plotspectra

    parser = at.commands.SuggestingArgumentParser()
    artistools.spectra.plotspectra.addargs(parser)

    for argsraw in (
        [".", "-timestep", "40", "-timemin", "100", "-timemax", "200"],
        [".", "-timedays", "250-300", "-timemin", "280"],
        [".", "-timestep", "40", "-timedays", "300"],
        [".", "-timedays", "250-300", "-timemax", "280"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            at.misc.check_time_selection(parser, parser.parse_args(argsraw), argsraw)
        assert excinfo.value.code == 1, argsraw

    # each way on its own is accepted
    for argsraw in ([".", "-timestep", "40"], [".", "-timedays", "300"], [".", "-timemin", "290", "-timemax", "310"]):
        at.misc.check_time_selection(parser, parser.parse_args(argsraw))


def test_check_time_selection_reads_a_flag_that_repeats_its_default() -> None:
    """A value that the user typed counts, even when it is the same as the default of the parser."""
    import artistools.plottransitions

    parser = at.commands.SuggestingArgumentParser()
    artistools.plottransitions.addargs(parser)
    default = parser.get_default("timestep")
    assert default is not None, "this test needs a command whose -timestep has a default"

    # the user names both, and the timestep happens to be the default, thus a test of the value alone
    # would miss the conflict
    argsraw = ["-timestep", str(default), "-timedays", "300"]
    with pytest.raises(SystemExit) as excinfo:
        at.misc.check_time_selection(parser, parser.parse_args(argsraw), argsraw)
    assert excinfo.value.code == 1


def test_check_time_selection_counts_a_default_as_absent() -> None:
    """Plottransitions gives -timestep a default, thus that default must not count as a second range."""
    import artistools.plottransitions

    parser = at.commands.SuggestingArgumentParser()
    artistools.plottransitions.addargs(parser)
    assert parser.get_default("timestep") is not None, "this test needs a command whose -timestep has a default"

    # the user named only -timedays, thus the default timestep must not raise
    at.misc.check_time_selection(parser, parser.parse_args(["-timedays", "300"]))


def test_nonempty_cellcounts_reads_the_rank_assignments(tmp_path: Path) -> None:
    """modelgridrankassignments.out gives the count of cells that hold matter for each rank.

    ARTIS assigns no 3D cell to a shell that holds no matter, thus the rank of such a shell writes no
    output file. That absence is normal, and it must not read as a file that went missing.
    """
    from artistools.misc.modelinfo import get_nonempty_cellcounts

    (tmp_path / "modelgridrankassignments.out").write_text("#rank nstart ndo ndo_nonempty\n0 0 1 0\n1 1 1 0\n2 2 1 3\n")

    counts = get_nonempty_cellcounts(tmp_path)
    assert counts == {0: 0, 1: 0, 2: 3}

    # a model that holds no such file gives None, thus the caller keeps its own error
    assert get_nonempty_cellcounts(at.get_path("testdata") / "testmodel") is None


def test_read_rank_outputfiles_names_an_empty_cell(tmp_path: Path) -> None:
    """A cell that holds no matter must say so, and not name a file that it never had."""
    from artistools.misc.modelinfo import read_rank_outputfiles

    (tmp_path / "modelgridrankassignments.out").write_text("#rank nstart ndo ndo_nonempty\n0 0 1 0\n")
    # a folder counts as a run folder when it holds an estimators file
    # ARTIS ends each cell of an estimator file with an empty line, and a reader takes a cell without it as cut
    (tmp_path / "estimators_0000.out").write_text("timestep 0 modelgridindex 0\n\n")
    (tmp_path / "model.txt").write_text("1\n1.0\n0 0.0 0.0 0.0 0.0\n")

    with pytest.raises(ValueError, match="Cell 0 holds no matter"):
        read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", modelgridindex=0)


def write_rank_output_model(modelpath: Path, rowsoffolder: dict[str, list[tuple[int, int, float]]]) -> None:
    """Write a model of five cells whose run folders hold one nlte file each.

    rowsoffolder names each run folder, and it gives the timestep, the cell, and the value of each
    row of the nlte file of that folder. A name of "." puts the files in the model folder itself.
    """
    modelpath.mkdir(parents=True, exist_ok=True)
    (modelpath / "model.txt").write_text(
        "5\n1.0\n" + "".join(f"{cell} {cell + 1}.0 0.0 0.0 0.0\n" for cell in range(5)), encoding="utf-8"
    )
    # get_nprocs reads the number of MPI processes from line 22 of input.txt
    (modelpath / "input.txt").write_text("\n".join(["0"] * 21 + ["1"]) + "\n", encoding="utf-8")

    for foldername, rows in rowsoffolder.items():
        folderpath = modelpath / foldername
        folderpath.mkdir(parents=True, exist_ok=True)
        # a folder counts as a run folder when it holds an estimators file. ARTIS ends each cell with an empty line
        timesteps = sorted({timestep for timestep, _, _ in rows})
        (folderpath / "estimators_0000.out").write_text(
            "".join(f"timestep {timestep} modelgridindex 0\n\n" for timestep in timesteps), encoding="utf-8"
        )
        (folderpath / "nlte_0000.out").write_text(
            "timestep modelgridindex nnlevel\n"
            + "".join(f"{timestep} {cell} {value}\n" for timestep, cell, value in rows),
            encoding="utf-8",
        )


def test_read_rank_outputfiles_takes_a_cell_list(tmp_path: Path) -> None:
    """A sequence of cells must give the same rows as the single-cell call that it replaces.

    A model of one cell gives the same rows for every selection, thus this model holds five.
    """
    from artistools.misc.modelinfo import read_rank_outputfiles

    write_rank_output_model(tmp_path, {".": [(0, cell, cell + 0.5) for cell in range(5)]})

    df_single = read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", timestep=0, modelgridindex=3)
    df_list = read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", timestep=0, modelgridindex=[3])
    pltest.assert_frame_equal(df_list, df_single)
    assert df_single.height == 1

    # a negative cell number means no filter, also inside a sequence
    df_all = read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", timestep=0, modelgridindex=[3, -1])
    assert df_all.height == 5, "a negative cell number must take every cell"
    pltest.assert_frame_equal(df_all, read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", timestep=0))


def test_read_rank_outputfiles_drops_the_repeated_timestep_of_a_restart(tmp_path: Path) -> None:
    """A restarted run writes its first timestep again, thus the rows of the earlier folder stay.

    The code concatenated every run folder, thus each row of the repeated timestep appeared two times.
    A plot of that timestep then read two values for one cell.
    """
    from artistools.misc.modelinfo import read_rank_outputfiles

    write_rank_output_model(
        tmp_path,
        {
            "job0": [(0, 0, 1.0), (1, 0, 2.0)],
            # the restart computes timestep 1 again, and it writes another value for it. It also
            # writes cell 4, which the folder before it never held
            "job1": [(1, 0, 9.0), (1, 4, 8.0), (2, 0, 3.0)],
        },
    )

    dfout = read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out")

    assert dfout.filter(modelgridindex=0)["timestep"].to_list() == [0, 1, 2], "each timestep must appear one time"
    assert dfout.filter(timestep=1, modelgridindex=0)["nnlevel"].item() == 2.0, "the rows of the earlier folder stay"

    # the earlier folder holds no row for cell 4, thus the row of the later folder stays
    assert dfout.filter(timestep=1, modelgridindex=4)["nnlevel"].item() == 8.0


def test_addarg_modelpath_positional_also_takes_the_option() -> None:
    """A command whose path is positional must also accept -modelpath.

    Some commands take -modelpath and others take a positional path. A user who learns one form must
    not meet "unrecognized arguments" with the other.
    """

    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])
        return parser

    assert build().parse_args([]).modelpath == []
    assert build().parse_args(["a", "b"]).modelpath == [Path("a"), Path("b")]
    assert build().parse_args(["-modelpath", "a"]).modelpath == [Path("a")]
    assert build().parse_args(["-modelpath", "a", "b"]).modelpath == [Path("a"), Path("b")]

    # the positional already names the paths in the help, thus the option stays out of it
    assert "-modelpath" not in build().format_help()

    # a positional that carries a default of its own must not hide the option. argparse gives that
    # default as the value of the positional, thus plotinitialabundances read "." for every -modelpath
    def buildwithdefault() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[Path()])
        return parser

    assert buildwithdefault().parse_args([]).modelpath == [Path()]
    assert buildwithdefault().parse_args(["a"]).modelpath == [Path("a")]
    assert buildwithdefault().parse_args(["-modelpath", "a"]).modelpath == [Path("a")]


def test_out_of_range_cell_names_the_cells_of_the_model() -> None:
    """A cell that the model does not hold must name the cells that it does hold.

    The check was a bare assert, thus the dispatcher reported "an internal check of artistools failed",
    which points a user away from their own argument.
    """
    from artistools.misc.modelinfo import get_mpirankofcell

    modelpath = at.get_path("testdata") / "testmodel"
    with pytest.raises(ValueError, match=r"Cell 999 is not in this model\. Its cells are 0 to 0"):
        get_mpirankofcell(999, modelpath=modelpath)

    with pytest.raises(ValueError, match="Cell -1 is not in this model"):
        get_mpirankofcell(-1, modelpath=modelpath)

    # the one cell of the test model still resolves
    assert get_mpirankofcell(0, modelpath=modelpath) >= 0


def test_the_rank_search_of_many_cells_agrees_with_the_blocks_of_each_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """One search gives the rank of every cell, in place of a lookup for each cell.

    The lookup for each cell took 2 to 10 s for the 125 000 cells of a 3D snapshot.
    """
    from artistools.misc.modelinfo import get_cellsofmpirank
    from artistools.misc.modelinfo import get_mpiranks_of_cells
    from artistools.misc.modelinfo import get_rankassignments

    # this model has modelgridrankassignments.out, thus each rank holds the cells of its own row
    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    dfrankassignments = get_rankassignments(modelpath)
    assert dfrankassignments is not None
    expected = np.full(at.misc.get_npts_model(modelpath), -1)
    for rank, nstart, ndo in dfrankassignments.select("rank", "nstart", "ndo").iter_rows():
        expected[nstart : nstart + ndo] = rank
    cells = np.arange(len(expected), dtype=np.int64)
    assert np.array_equal(get_mpiranks_of_cells(modelpath, cells), expected)

    # with no such file, the blocks of get_cellsofmpirank give the rank of each cell. No test model has fewer
    # ranks than cells, thus 7 ranks for 100 cells give blocks of 15 and 14 cells
    from artistools.misc import modelinfo

    monkeypatch.setattr(modelinfo, "get_rankassignments", mock.Mock(return_value=None))
    monkeypatch.setattr(modelinfo, "get_nprocs", mock.Mock(return_value=7))
    monkeypatch.setattr(modelinfo, "get_npts_model", mock.Mock(return_value=100))
    ranks = get_mpiranks_of_cells(modelpath, np.arange(100, dtype=np.int64)).tolist()
    assert sorted(set(ranks)) == list(range(7))
    assert all(cell in get_cellsofmpirank(rank, modelpath) for cell, rank in enumerate(ranks))


def test_check_time_selection_reads_each_spelling_as_argparse_does() -> None:
    """The test of the command line must give each string the reading that the parser gives it.

    A value joined to a flag of more than one letter counts for that flag, e.g. -ts70 names the
    timestep 70. The command took such a value for -t, thus "-ts70 -t 300" named the time range two
    times and ran without a word, on a command whose -timestep holds that value as its default.
    """

    def parse(argsraw: list[str]) -> tuple[at.commands.SuggestingArgumentParser, argparse.Namespace]:
        parser = at.commands.SuggestingArgumentParser()
        at.misc.addarg_timedays(parser, kind="str")
        at.misc.addarg_timestep(parser, default=70)
        return parser, parser.parse_args(argsraw)

    # each spelling of the timestep names the range beside -t, and the value is the default here
    for argsraw in (
        ["-t300", "-ts", "70"],
        ["-ts=70", "-t", "300"],
        ["-ts", "70", "-t", "300"],
        ["-ts70", "-t", "300"],
        ["-timestep70", "-t", "300"],
    ):
        parser, namespace = parse(argsraw)
        assert str(namespace.timestep) == str(parser.get_default("timestep")), argsraw
        with pytest.raises(SystemExit) as excinfo:
            at.misc.check_time_selection(parser, namespace, argsraw)
        assert excinfo.value.code == 1, argsraw


def test_import_optional_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional dependency must say how to install it, and a broken one must give its real cause.

    The test patched builtins.__import__, which import_module does not call, thus it passed only when
    pyvista was not yet imported.
    """
    import importlib
    import sys

    # a None entry in sys.modules makes the next import raise ModuleNotFoundError, whatever ran before
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ModuleNotFoundError, match=r"needs pyvista.*artistools\[extras\]"):
        at.misc.import_optional("pyvista")

    # an installed package that fails, e.g. on a missing system library, must not be called missing
    def broken_import(name: str) -> object:
        msg = f"{name}: libGL.so.1: cannot open shared object file"
        raise ImportError(msg)

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match=r"installed but did not import: pyvista: libGL"):
        at.misc.import_optional("pyvista")
    monkeypatch.undo()

    # an installed module comes back as the import statement gives it
    assert at.misc.import_optional("math").sqrt(4.0) == 2.0


def test_print_warning_reaches_stderr_and_survives_quiet(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A warning goes to the standard error, thus --quiet keeps it and a script reads a clean product.

    Every warning went to the standard output before, thus --quiet discarded all of them.
    """
    at.misc.print_warning("the model is on fire")
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == "WARNING: the model is on fire\n"

    # the dispatcher redirects the standard output alone, thus the warning of a --quiet run appears
    import artistools.__main__

    artistools.__main__.main(
        argsraw=[
            "plotestimators",
            "-modelpath",
            str(at.get_path("testdata") / "testmodel"),
            "--listvariables",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert "estimator variables" in captured.out

    # the warning of a command must also reach the standard error. The test model holds no
    # deposition.out, thus get_escaped_arrivalrange gives a warning. That function caches its
    # answer, thus this test clears the cache. An earlier test in this process would otherwise
    # keep the warning
    from artistools.misc.timesteps import get_escaped_arrivalrange_cached

    get_escaped_arrivalrange_cached.cache_clear()
    artistools.__main__.main(
        argsraw=[
            "plotlightcurves",
            str(at.get_path("testdata") / "testmodel"),
            "--quiet",
            "-o",
            str(tmp_path / "quietwarning.pdf"),
        ]
    )
    captured = capsys.readouterr()
    assert "WARNING: No deposition.out file found" in captured.err


def test_progress_class_takes_a_spawn_lock_and_keeps_the_start_method() -> None:
    """A bar must take a lock that a spawn pool can hold, and it must set no default start method.

    tqdm builds its shared lock from the default multiprocessing context at the first bar. On Linux the
    default was fork before Python 3.14, thus CI stopped with "A SemLock created in a fork context is
    being shared with a process in a spawn context". get_progress_class gives tqdm a lock of a spawn
    context instead, thus a bar starts no process and the default of the caller stands.
    """
    import multiprocessing as mp

    import tqdm

    from artistools.misc.general import get_progress_class

    original = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("fork", force=True)  # the Linux default before Python 3.14
        progressbar = get_progress_class()(total=1, disable=True)
        progressbar.close()

        assert mp.get_start_method() == "fork", "a bar starts no process, thus it sets no start method"
        # a SemLock records the context that made it, thus a lock of the fork default fails here
        assert getattr(tqdm.tqdm.get_lock(), "_is_fork_ctx", None) is False, "the bar must hold a spawn lock"
    finally:
        # None with force=True clears the default again, thus this test leaves no fork default in the worker
        mp.set_start_method(original, force=True)


def test_resolve_frameset_paths(tmp_path: Path) -> None:
    """The path arithmetic of a set of frames must give one answer for every command.

    Three commands wrote this by hand, and each one broke the rule of -o in its own way.
    """
    framename = "plot_{timestep:03d}.png"

    # a -o path with no file extension names a folder, which holds the frames and the product
    frameset = at.misc.resolve_frameset_paths(
        tmp_path / "frames", framecount=3, framename=framename, productname="movie.gif"
    )
    assert frameset.frametemplate == tmp_path / "frames" / framename
    assert frameset.productpath == tmp_path / "frames" / "movie.gif"
    assert (tmp_path / "frames").is_dir(), "the folder of the frames must exist"

    # a -o path that has a file extension names the product, thus the frames go beside it
    frameset = at.misc.resolve_frameset_paths(
        tmp_path / "out" / "movie.gif",
        framecount=3,
        framename=framename,
        productname="movie.gif",
        combines=True,
        gifduration=500,
    )
    assert frameset.productpath == tmp_path / "out" / "movie.gif"
    assert frameset.frametemplate == tmp_path / "out" / framename

    # the folder of the product can carry a suffix of its own
    frameset = at.misc.resolve_frameset_paths(
        tmp_path / "results.v1" / "movie.gif",
        framecount=3,
        framename=framename,
        productname="movie.gif",
        combines=True,
        gifduration=500,
    )
    assert frameset.productpath == tmp_path / "results.v1" / "movie.gif"
    assert (tmp_path / "results.v1").is_dir()

    # a merge names its own product, thus a folder gives no name to it
    frameset = at.misc.resolve_frameset_paths(tmp_path / "m", framecount=2, framename=framename, combines=True)
    assert frameset.productpath is None
    assert frameset.frametemplate == tmp_path / "m" / framename

    # a -o path that has a file extension names the merged product, and the frames go beside it
    frameset = at.misc.resolve_frameset_paths(tmp_path / "merged.pdf", framecount=2, framename=framename, combines=True)
    assert frameset.productpath == tmp_path / "merged.pdf"
    assert frameset.frametemplate == tmp_path / framename

    # a merge writes pdf data, thus a product name with a different suffix stops before the plots
    with pytest.raises(ValueError, match="a merge writes a pdf file"):
        at.misc.resolve_frameset_paths(tmp_path / "merged.png", framecount=2, framename=framename, combines=True)

    # a name that holds no field cannot take more than one frame
    with pytest.raises(ValueError, match="names one file, and this command writes 3 frames"):
        at.misc.resolve_frameset_paths(tmp_path / "one.png", framecount=3, framename=framename)

    # one frame alone may take such a name
    frameset = at.misc.resolve_frameset_paths(tmp_path / "one.png", framecount=1, framename=framename)
    assert frameset.frametemplate == tmp_path / "one.png"


def test_combine_frames_opens_the_product_alone(tmp_path: Path) -> None:
    """The frames of a run do not open one at a time, thus the product opens in their place."""
    framepaths = [tmp_path / f"frame{i}.png" for i in range(3)]
    for framepath in framepaths:
        framepath.write_bytes(b"")

    # one frame alone is the product of the run, and it takes the name that -o gave that product
    named = tmp_path / "named.pdf"
    with mock.patch("artistools.misc.fileio.open_file") as mockopen:
        product = at.misc.combine_frames(framepaths[:1], named, openfile=True)
    assert product == named
    assert named.is_file(), "the one frame must carry the name of the product"
    assert not framepaths[0].exists(), "the frame moves to that name"
    assert mockopen.call_args.args[0] == named

    # without such a name, that frame is the product as it stands
    framepaths[0].write_bytes(b"")
    with mock.patch("artistools.misc.fileio.open_file") as mockopen:
        product = at.misc.combine_frames(framepaths[:1], None, openfile=True)
    assert product == framepaths[0]
    assert mockopen.call_args.args[0] == framepaths[0]

    # a gif of one frame is still the gif that the caller asked for
    gifpath = tmp_path / "movie.gif"
    with mock.patch("artistools.misc.fileio.write_gif") as mockgif:
        product = at.misc.combine_frames(framepaths[:1], gifpath, openfile=False, gifduration=1000.0)
    assert product == gifpath
    assert mockgif.call_args.args[0] == gifpath, "one frame must still make the gif"

    # no frame gives no product
    assert at.misc.combine_frames([], None, openfile=True) is None

    # --open takes nothing when the caller does not ask for it
    with mock.patch("artistools.misc.fileio.open_file") as mockopen:
        at.misc.combine_frames(framepaths[:1], None, openfile=False)
    assert not mockopen.called


def test_a_keyword_that_the_command_does_not_take_raises() -> None:
    """A name that names no argument of the command must raise.

    A declared name that only stops the command gave argparse a dest. Thus the test for an unknown keyword
    took that name for an argument of the command, and a wrong keyword gave no error.
    """
    import artistools.timesteps

    testmodel = at.get_path("testdata") / "testmodel"
    with pytest.raises(ValueError, match="Unknown argument names"):
        artistools.timesteps.main(argsraw=[], modelpath=testmodel, timemin=5.0)

    # a real argument of the command still reaches it
    artistools.timesteps.main(argsraw=[], modelpath=testmodel, timedays=300)


def test_a_range_keeps_a_negative_number_whole() -> None:
    """A hyphen in front of a number is no separator of a range.

    "-timestep -1" split into an empty text and "1", thus it raised "invalid literal for int()".
    """
    assert at.misc.parse_range_list("40-42") == [40, 41, 42]
    assert at.misc.parse_range_list("-1") == [-1]
    assert at.misc.parse_range_list("last", dictvars={"last": 99}) == [99]
    assert at.misc.parse_range_list("40-last", dictvars={"last": 42}) == [40, 41, 42]

    with pytest.raises(ValueError, match="Bad range"):
        at.misc.parse_range_list("10-20-30")


def test_a_merge_keeps_its_own_product(tmp_path: Path) -> None:
    """Two spellings of one path name one file, thus the merge must not remove the file that it wrote."""
    import matplotlib.pyplot as plt

    framepaths = []
    for i in range(2):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [i, i])
        framepath = tmp_path / f"frame{i}.pdf"
        fig.savefig(framepath)
        plt.close(fig)
        framepaths.append(str(framepath))

    # the product carries the name of a frame, and the caller gives it as an absolute path
    product = at.misc.merge_pdf_files(framepaths, tmp_path.resolve() / "frame0.pdf")
    assert Path(product).is_file(), "the merge must keep the file that it wrote"


def test_get_runfolder_timesteps_of_a_classic_estimator_file_gives_no_timesteps() -> None:
    """A classic-mode estimator file has no timestep header line, thus the code must not index the empty result."""
    from artistools.misc.modelinfo import get_runfolder_timesteps

    runfolder = at.get_path("testdata") / "test-classicmode_1d" / "32086771.slurm"

    assert get_runfolder_timesteps(runfolder) == ()


def test_get_runfolder_timesteps_tries_every_rank_stem(tmp_path: Path) -> None:
    """The lowest rank can exist only as a sibling that zopen does not read, e.g. a .bak file.

    A later rank then still holds a readable file. A test of the lowest stem alone returned no
    timestep, thus get_runfolders dropped a folder whose data is present.
    """
    from artistools.misc.modelinfo import get_runfolder_timesteps

    (tmp_path / "estimators_0000.out.bak").write_text("junk\n", encoding="utf-8")
    (tmp_path / "estimators_0001.out").write_text(
        "timestep 7 modelgridindex 0\n\ntimestep 8 modelgridindex 0\n\n", encoding="utf-8"
    )

    # the result holds the first timestep of the folder. Only get_runfolders knows if an earlier folder holds it
    assert get_runfolder_timesteps(tmp_path) == (7, 8)


def test_gaussian_filter_wrap_passes_over_a_nan() -> None:
    """A NaN element holds no data, thus the filter must keep it in its own element and give it no weight.

    A direction bin that received no packet holds a NaN. The old filter made every bin within four
    standard deviations of it a NaN too.
    """
    data = np.outer(np.sin(np.linspace(0.0, np.pi, 4)), np.cos(np.linspace(0.0, 2 * np.pi, 6, endpoint=False)))
    withnan = data.copy()
    withnan[1, 2] = np.nan

    smoothed = at.misc.gaussian_filter_wrap(withnan, sigma=1.2)
    assert np.isfinite(smoothed).all()

    # an element that holds data keeps a value close to the smoothing of the array without the NaN
    assert np.allclose(smoothed[0, 0], at.misc.gaussian_filter_wrap(data, sigma=1.2)[0, 0], rtol=0.2)

    # an infinite element holds no data either, thus it takes the mean of its neighbours
    withinf = data.copy()
    withinf[1, 2] = np.inf
    assert np.isfinite(at.misc.gaussian_filter_wrap(withinf, sigma=1.2)).all()

    # an element that has no neighbour with data stays a NaN
    allnan = np.full_like(data, np.nan)
    assert np.isnan(at.misc.gaussian_filter_wrap(allnan, sigma=1.2)).all()


def test_get_model_name_follows_the_working_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The name of the default model path must change with the working folder.

    A cache held the relative Path("."). Thus every plot after a change of the working folder kept
    the name of the first model.
    """
    for foldername in ("modelA", "modelB"):
        (tmp_path / foldername).mkdir()

    monkeypatch.chdir(tmp_path / "modelA")
    assert at.get_model_name(Path()) == "modelA"

    monkeypatch.chdir(tmp_path / "modelB")
    assert at.get_model_name(Path()) == "modelB"


def test_phibin_rank_ascends_with_phi() -> None:
    """The rank of a phi bin must ascend with phi, because the ARTIS bin index does not.

    A colour bar that ascends with phi gave every series the label of the mirrored phi bin.
    """
    nphibins = at.misc.get_viewingdirection_phibincount()
    ranks = [at.misc.get_phibin_rank_ascending(phibin) for phibin in range(nphibins)]
    assert sorted(ranks) == list(range(nphibins))

    phi_lower, _, binlabels = at.misc.get_phi_bins(usedegrees=False)
    binsbyrank = sorted(range(nphibins), key=at.misc.get_phibin_rank_ascending)
    assert [phi_lower[phibin] for phibin in binsbyrank] == sorted(phi_lower)

    # the bins of ARTIS are half-open, and the label of each one must say which end it holds. A
    # packet that travels along +X has phi = 0, and the binning puts it in a bin that holds 0
    dfpackets = at.packets.add_packet_directions_lazypolars(pl.DataFrame({"dirx": [1.0], "diry": [0.0], "dirz": [0.0]}))
    phibin = at.packets.bin_packet_directions_polars(dfpackets).collect()["phibin"].item()
    assert binlabels[phibin].startswith("0 \u2264"), binlabels[phibin]


def test_parquet_cache_without_a_text_source_still_checks_the_version(tmp_path: Path) -> None:
    """A cache that has no text source must still match the cache format version.

    The estimator reader skipped every check for such a cache. Thus it gave a cache of an old schema,
    and the columns that the schema lacked became zero without a warning.
    """
    parquetfilepath = tmp_path / "cache.parquet"
    at.misc.write_parquet_atomic(
        pl.DataFrame({"a": [1]}), parquetfilepath, metadata={"cacheversion": "1", "textsource_mtime": "100.0"}
    )

    # no text source: the modification time gives no comparison, but the version still applies
    assert at.misc.read_parquet_cache_metadata(parquetfilepath, 1, None)[1] is None
    assert "cache format version" in str(at.misc.read_parquet_cache_metadata(parquetfilepath, 2, None)[1])

    # a text source that changed still makes the cache stale
    assert at.misc.read_parquet_cache_metadata(parquetfilepath, 1, 100.0)[1] is None
    assert "text source changed" in str(at.misc.read_parquet_cache_metadata(parquetfilepath, 1, 200.0)[1])


def test_a_cache_from_before_the_stamps_stays_current(tmp_path: Path) -> None:
    """A reader that gives accept_unstamped must keep a cache that the versions before the stamps wrote.

    Such a cache holds no cacheversion and no textsource_mtime. A rejection rebuilds every cache of
    an archived run of estimators or packets, which costs hours and reads text files that the run may
    no longer hold. A reader that does not give accept_unstamped keeps the strict rule, because a
    cache that a few seconds rebuild gives a wrong number for no gain.
    """
    from artistools.misc.fileio import mtime_matches_stamp

    # a cache that holds no stamp dates its text source in no way, thus the stamp matches no time
    assert not mtime_matches_stamp(None, 1000.0)

    # a real cache of this kind holds only the arrow schema
    legacy = tmp_path / "legacy.parquet"
    at.misc.write_parquet_atomic(pl.DataFrame({"number": [0]}), legacy)
    assert "cacheversion" not in pl.read_parquet_metadata(legacy)
    assert "textsource_mtime" not in pl.read_parquet_metadata(legacy)

    assert at.misc.read_parquet_cache_metadata(legacy, 1, 1760711077.0, accept_unstamped=True)[1] is None
    assert "no cacheversion stamp" in str(at.misc.read_parquet_cache_metadata(legacy, 1, 1760711077.0)[1])

    # a cache that holds a stamp keeps the strict comparison
    stamped = tmp_path / "stamped.parquet"
    at.misc.write_parquet_atomic(
        pl.DataFrame({"number": [0]}), stamped, metadata={"cacheversion": "1", "textsource_mtime": "1000.0"}
    )
    assert at.misc.read_parquet_cache_metadata(stamped, 1, 1000.0)[1] is None
    assert "text source changed" in str(at.misc.read_parquet_cache_metadata(stamped, 1, 2000.0)[1])


def test_split_multitable_dataframe_tables_collect_together() -> None:
    """A collect_all of filtered tables gives the same rows as a collect of each table.

    With polars 2.0.0rc2, collect_all of filtered slices of one scan gave incorrect rows. For example, the second
    direction bin of a light curve then had other times and luminosities. head and tail avoid this.
    """
    times = list(range(10))
    dfres = pl.DataFrame({"time": times * 3, "value": [*times, *(t + 100 for t in times), *(t + 200 for t in times)]})
    tables = dirbins.split_multitable_dataframe(dfres.lazy())
    plans = [table.filter(pl.col("time").is_between(3, 6)) for table in tables.values()]

    for together, table in zip(pl.collect_all(plans), plans, strict=True):
        pltest.assert_frame_equal(together, table.collect())
    assert pl.collect_all(plans)[1]["value"].to_list() == [103, 104, 105, 106]


def test_costheta_bin_labels_have_no_negative_zero() -> None:
    """The edge of the cos θ bins at the middle is 0, and its label must not read "-0.0".

    np.arange gave -1.1e-16 for that edge, and the format of one decimal wrote "-0.0 ≤ cos θ < 0.2".
    """
    from artistools.misc.dirbins import get_costheta_bins

    lowers, _, labels = get_costheta_bins(usedegrees=False)
    assert not any("-0.0" in label for label in labels), labels
    assert 0.0 in lowers


def test_remote_path_is_not_reference_data_and_needs_no_connection() -> None:
    """A remote path is an ARTIS model, and its test must not start ssh.

    The menu of the recent models tested each remote model in the window thread, and each ssh call stopped the window.
    """
    with mock.patch("artistools.misc.remote.call_on_host", side_effect=AssertionError("ssh started")):
        assert not at.misc.fileio.path_is_reference_data("nohost.invalid:/runs/model", "data/refspectra")


def test_a_windows_drive_letter_is_no_remote_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path such as C:/Users/me/model names a drive on Windows, and not a host of one letter.

    The pattern of a remote path read the drive letter as a host. Thus each absolute path on Windows went to ssh.
    """
    monkeypatch.setattr(remote, "REMOTEPATH_PATTERN", remote.make_remotepath_pattern(windows=True))
    assert not remote.is_remote_path("C:/Users/me/model")
    assert not remote.is_remote_path("C:\\Users\\me\\model")
    assert not remote.names_a_remote_folder("C:/Users/me/model")
    assert remote.split_remote_path("host:/path") == ("host", Path("/path"))
    assert remote.split_remote_path("user@host:path") == ("user@host", Path("~/path"))
    assert remote.split_remote_path("[::1]:path") == ("[::1]", Path("~/path"))


def test_a_host_alias_of_one_letter_stays_remote_outside_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ssh host alias of one letter with an absolute path must stay remote on Linux and macOS.

    The lookahead of the Windows drive applied on each system, thus "a:/lustre/model" became a local path.
    """
    monkeypatch.setattr(remote, "REMOTEPATH_PATTERN", remote.make_remotepath_pattern(windows=False))
    assert remote.split_remote_path("a:/lustre/model") == ("a", Path("/lustre/model"))
    assert remote.names_a_remote_folder("a:/lustre/model")


def test_get_file_metadata_reads_the_sidecar_of_a_link_and_a_key_with_a_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metadata file beside a symbolic link and a key of metadata.yml with a folder must each give the metadata.

    The function resolved the link before the lookups, thus it looked beside the target and keyed on the target.
    """
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "spec.txt").write_text("data", encoding="utf-8")
    linkfolder = tmp_path / "links"
    linkfolder.mkdir()
    (linkfolder / "spec.txt").symlink_to(tmp_path / "store" / "spec.txt")
    (linkfolder / "spec.txt.meta.yml").write_text("a_v: 1.5\n", encoding="utf-8")
    assert at.misc.get_file_metadata(linkfolder / "spec.txt")["a_v"] == pytest.approx(1.5)

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "obs.txt").write_text("data", encoding="utf-8")
    (tmp_path / "data" / "metadata.yml").write_text(yaml.safe_dump({"data/obs.txt": {"a_v": 2.5}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert at.misc.get_file_metadata("data/obs.txt")["a_v"] == pytest.approx(2.5)


def test_get_file_metadata_follows_the_working_folder_and_takes_an_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The metadata of a relative path must change with the working folder, and a file of comments gives no metadata.

    A cache held the relative path, thus a second file of the same name got the metadata of the first one. A file
    of comments alone gave a TypeError, because yaml gives None for it.
    """
    for foldername, a_v in (("modelA", 1.0), ("modelB", 2.0)):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "ref.txt").write_text("data", encoding="utf-8")
        (tmp_path / foldername / "ref.txt.meta.yml").write_text(f"a_v: {a_v}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path / "modelA")
    assert at.misc.get_file_metadata("ref.txt")["a_v"] == pytest.approx(1.0)
    monkeypatch.chdir(tmp_path / "modelB")
    assert at.misc.get_file_metadata("ref.txt")["a_v"] == pytest.approx(2.0)

    (tmp_path / "empty.txt").write_text("data", encoding="utf-8")
    (tmp_path / "empty.txt.meta.yml").write_text("# a comment alone\n", encoding="utf-8")
    assert at.misc.get_file_metadata(tmp_path / "empty.txt") == {}

    combineddir = tmp_path / "combined"
    combineddir.mkdir()
    (combineddir / "spec.txt").write_text("data", encoding="utf-8")
    (combineddir / "metadata.yml").write_text("# a comment alone\n", encoding="utf-8")
    assert at.misc.get_file_metadata(combineddir / "spec.txt") == {}
    # the combined file sits beside the data file, thus a key can hold the name of the file alone
    (combineddir / "metadata.yml").write_text(yaml.safe_dump({"spec.txt": {"a_v": 3.0}}), encoding="utf-8")
    at.misc.fileio.get_file_metadata_cached.cache_clear()
    assert at.misc.get_file_metadata(combineddir / "spec.txt")["a_v"] == pytest.approx(3.0)


def test_get_mpiranklist_takes_a_numpy_array_of_cells() -> None:
    """A numpy array of cells must give the same ranks as a list, and an empty array gives all the ranks.

    A comparison of a numpy array with an empty list raises, thus the function stopped before it read the cells.
    """
    modelpath = at.get_path("testdata") / "testmodel"
    assert list(at.misc.get_mpiranklist(modelpath, modelgridindex=np.array([0]))) == list(
        at.misc.get_mpiranklist(modelpath, modelgridindex=[0])
    )
    assert list(at.misc.get_mpiranklist(modelpath, modelgridindex=np.array([], dtype=np.int64))) == list(
        at.misc.get_mpiranklist(modelpath)
    )


def test_addarg_modelpath_positional_single_path_takes_the_option() -> None:
    """A command with a positional path of one model must also accept -modelpath, and the path is not required.

    The positional argument had no nargs, thus argparse made it required and the option could never stand in
    for it.
    """

    def addargs(parser: argparse.ArgumentParser) -> None:
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=False)

    assert parse_cli_args(addargs, "x", None, ["-modelpath", "mymodel"]).modelpath == Path("mymodel")
    assert parse_cli_args(addargs, "x", None, ["mymodel"]).modelpath == Path("mymodel")
    assert parse_cli_args(addargs, "x", None, []).modelpath is None


def test_the_option_form_of_the_model_path_names_the_path_that_it_replaces(capsys: pytest.CaptureFixture[str]) -> None:
    """The option form wins over the positional path in either order, and the user gets a warning.

    The option stored its paths in place of the positional paths, thus a model that the user wrote went away with no
    message. A positional path after the option replaced the option for a command of one path.
    """

    def addargs_many(parser: argparse.ArgumentParser) -> None:
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])

    def addargs_one(parser: argparse.ArgumentParser) -> None:
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=False)

    for addargs, argsraw, expected in (
        (addargs_many, ["model1", "-modelpath", "model2"], [Path("model2")]),
        (addargs_many, ["-modelpath", "model2", "--quiet", "model1"], [Path("model2")]),
        (addargs_one, ["model1", "-modelpath", "model2"], Path("model2")),
        (addargs_one, ["-modelpath", "model2", "model1"], Path("model2")),
    ):
        assert parse_cli_args(addargs, "x", None, argsraw).modelpath == expected
        assert "ignores the path 'model1'" in capsys.readouterr().err

    assert parse_cli_args(addargs_one, "x", None, ["model1", "-modelpath", "model1"]).modelpath == Path("model1")
    assert "ignores" not in capsys.readouterr().err


def test_the_option_form_names_a_positional_path_that_equals_the_default(capsys: pytest.CaptureFixture[str]) -> None:
    """A positional path that the user writes must get the warning also when it equals the default.

    The actions compared the value with the default, thus "." of writebollightcurvedata counted as no path.
    """

    def addargs(parser: argparse.ArgumentParser) -> None:
        at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[Path()])

    for argsraw in ([".", "-modelpath", "model2"], ["-modelpath", "model2", "--quiet", "."]):
        assert parse_cli_args(addargs, "x", None, argsraw).modelpath == [Path("model2")]
        assert "ignores the path '.'" in capsys.readouterr().err

    assert parse_cli_args(addargs, "x", None, ["-modelpath", "model2"]).modelpath == [Path("model2")]
    assert parse_cli_args(addargs, "x", None, []).modelpath == [Path()]
    assert "ignores" not in capsys.readouterr().err


# --- the command line: remote paths, list options, the dispatcher, and the command tree ------------------------


def test_a_label_with_a_colon_is_no_remote_folder_and_starts_no_ssh() -> None:
    """A -label value with a colon must stay a label, and the test of the trailing folders must not start ssh.

    The host of a remote path took a space, thus "Model B: Fe/Ni" named the host "Model B". A label such as
    "W7:Kasen" went to folder_is_artis_run, which asked the host W7 through ssh and stopped the command.
    """
    from artistools.lightcurve import plotlightcurve
    from artistools.spectra import plotspectra

    modelpath = at.get_path("testdata") / "testmodel"
    assert not remote.is_remote_path("Model B: Fe/Ni")
    assert not remote.names_a_remote_folder("W7: 56Ni/56Co")
    # a real remote path keeps its host
    assert remote.split_remote_path("host:/path") == ("host", Path("/path"))
    assert remote.split_remote_path("user@host:path") == ("user@host", Path("~/path"))
    assert remote.split_remote_path("[::1]:path") == ("[::1]", Path("~/path"))
    assert remote.names_a_remote_folder("user@vae26:/lustre/model")

    with mock.patch.object(remote, "call_on_host", side_effect=AssertionError("ssh started")):
        for label in ("W7: 56Ni/56Co", "W7:Kasen", "Model B: Fe/Ni"):
            args = parse_cli_args(plotspectra.addargs, None, None, ["-label", label, str(modelpath)])
            assert args.specpath == [modelpath], label
            assert args.label == [label]

            args = parse_cli_args(plotlightcurve.addargs, None, None, [str(modelpath), "-label", "A", label])
            assert args.modelpath == [modelpath], label
            assert args.label == ["A", label]

    # an empty label at the end is a label, and not the working folder
    assert at.misc.cliutils.trailing_folder_count(["A", ""]) == 0


def test_a_list_option_that_takes_a_file_or_a_folder_gives_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reference file or a model folder with no input.txt after -label must not go away with no message.

    -label took "run1" and "sn2011fe.txt" as labels, and the command then plotted the working folder.
    """
    from artistools.spectra import plotspectra

    (tmp_path / "run1").mkdir()
    (tmp_path / "sn2011fe.txt").write_text("4000 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    args = parse_cli_args(plotspectra.addargs, None, None, ["-label", "ARTIS", "SN 2011fe", "run1", "sn2011fe.txt"])
    assert args.label == ["ARTIS", "SN 2011fe", "run1", "sn2011fe.txt"]
    assert "-label read 'run1', 'sn2011fe.txt' as values" in capsys.readouterr().err

    # every value names a path, thus the option can read paths, e.g. -reflightcurves
    parse_cli_args(plotspectra.addargs, None, None, ["-label", "run1", "sn2011fe.txt"])
    assert "WARNING" not in capsys.readouterr().err


def test_the_option_form_keeps_its_path_when_a_list_option_ends_with_a_folder() -> None:
    """A -modelpath that equals the default must keep its path, thus no folder of -label replaces it.

    The take-back compared the value with the default, thus "-modelpath ." counted as no path.
    """
    modelpath = at.get_path("testdata") / "testmodel"
    parser = at.commands.SuggestingArgumentParser()
    at.misc.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[Path()])
    at.misc.addarg_seriesstyle(parser)

    args = parser.parse_args(["-modelpath", ".", "-label", "A", str(modelpath)])
    assert args.modelpath == [Path()]
    assert args.label == ["A", str(modelpath)]


def test_the_remote_path_pattern_follows_this_system() -> None:
    """The module pattern must apply the drive rule on Windows alone, with no patch of the pattern.

    The other tests replace the pattern, thus a wrong choice at the module level passed them.
    """
    windows = os.name == "nt"
    assert remote.REMOTEPATH_PATTERN.pattern == remote.make_remotepath_pattern(windows=windows).pattern
    assert remote.is_remote_path("a:/lustre/model") is not windows


def test_a_bad_cell_names_the_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """A -cell value that names no cell must give an argparse error that names the flag, and no traceback."""
    parser = at.commands.SuggestingArgumentParser(prog="demo")
    at.misc.addarg_modelgridindex(parser)
    for value in ("abc", "3-x"):
        with pytest.raises(SystemExit) as exitinfo:
            parser.parse_args(["-cell", value])
        assert exitinfo.value.code == 2
        assert f"-modelgridindex/-cell/-mgi: '{value}' names no cells" in capsys.readouterr().err


def test_resolve_frameset_paths_gives_o_to_one_frame_and_fields_to_the_frames(tmp_path: Path) -> None:
    """-o names the one frame of a run that combines nothing, and a name with fields names each frame.

    plotestimators gives a gif name for --makegif also for one figure, and -o then named a product that never came.
    A -o name with fields became the literal name of the product, and the frames kept the default names.
    """
    framename = "plot_{timestep:03d}.pdf"
    frameset = at.misc.resolve_frameset_paths(
        tmp_path / "est.pdf", framecount=1, framename=framename, productname="evo.gif", combines=False
    )
    assert frameset.frametemplate == tmp_path / "est.pdf"
    assert frameset.finish([tmp_path / "est.pdf"], argparse.Namespace()) is None

    template = tmp_path / "rf_{cell:05d}_{timestep}.pdf"
    frameset = at.misc.resolve_frameset_paths(template, framecount=2, framename=framename, combines=True)
    assert frameset.frametemplate == template
    assert frameset.productpath is None

    template = tmp_path / "frame_{timemindays:.1f}.png"
    frameset = at.misc.resolve_frameset_paths(
        template, framecount=2, framename=framename, productname="sphericalplot.gif", combines=True, gifduration=500
    )
    assert frameset.frametemplate == template
    assert frameset.productpath == tmp_path / "sphericalplot.gif"


def test_a_list_keyword_takes_the_type_of_its_argument() -> None:
    """A list keyword must give the same values as the command line, e.g. "default" for the default of a series.

    argparse converts a default of one text alone, thus label=["default", "B"] drew the label "default", and
    color=["default", "red"] stopped with "Invalid RGBA argument".
    """
    from artistools.inputmodel import plotdensity
    from artistools.spectra import plotspectra

    args = parse_cli_args(
        plotspectra.addargs, None, None, [], {"label": ["default", "B"], "color": ["default", "red"], "linewidth": "2"}
    )
    assert args.label == [None, "B"]
    assert args.color == [None, "red"]
    assert args.linewidth == [2.0]

    # a command with a default colour for each series gives that colour to an entry "default"
    args = parse_cli_args(plotdensity.addargs, None, None, [], {"color": ["default", "red"]})
    assert args.color == ["C0", "red"]

    with pytest.raises(ValueError, match="-color"):
        parse_cli_args(plotspectra.addargs, None, None, [], {"color": ["notacolour"]})


def test_keywords_and_argsraw_combine() -> None:
    """The text of argsraw must give its arguments also when the call gives a keyword. The parser dropped it then."""
    from artistools.spectra import plotspectra

    modelpath = at.get_path("testdata") / "testmodel"
    args = parse_cli_args(plotspectra.addargs, None, None, ["-t", "300"], {"specpath": [modelpath]})
    assert args.timedays == "300"
    assert args.specpath == [modelpath]


def test_a_keyword_and_its_alias_name_the_conflict() -> None:
    """A dest and an alias of one argument must give an error that names the conflict, and not an unknown name."""
    from artistools.lightcurve import plotlightcurve

    with pytest.raises(ValueError, match="The keywords timemin, xmin name one argument"):
        parse_cli_args(plotlightcurve.addargs, None, None, [], {"timemin": 250.0, "xmin": 260.0})


@pytest.mark.parametrize(
    ("error", "code", "expected"),
    [
        (KeyboardInterrupt(), 130, "stopped at a keyboard interrupt"),
        (BrokenPipeError(), 1, ""),
        (ImportError("This command needs plotly, which is installed but did not import: libGL"), 1, "needs plotly"),
        (OSError("unexpected end of file"), 1, "A compressed file can be incomplete"),
        (pl.exceptions.PanicException("PyErr { type: <class 'EOFError'> }"), 1, "ends before the end of its data"),
    ],
)
def test_the_dispatcher_reports_an_error_with_no_traceback(
    error: BaseException, code: int, expected: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each of these errors must give a short message and an exit status, and no traceback."""
    import artistools.__main__

    monkeypatch.delenv("ARTISTOOLS_TRACEBACK", raising=False)
    # the handler of a closed pipe sends the standard output to the null device, thus it must not be the one of pytest
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    with mock.patch("artistools.commands.show_version", side_effect=error), pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["version"])

    assert exitinfo.value.code == code
    assert expected in capsys.readouterr().err


def test_the_dispatcher_shows_the_context_notes_of_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A note of one line names the host or the file of an error, and a note of more lines is a traceback."""
    import artistools.__main__

    monkeypatch.delenv("ARTISTOOLS_TRACEBACK", raising=False)
    error = OSError("The artistools server on vae26 stopped during a call of get_spectra")
    error.add_note("The traceback on the server:\nTraceback (most recent call last)")
    error.add_note("The error came from get_spectra on the artistools server of vae26")
    with mock.patch("artistools.commands.show_version", side_effect=error), pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["version"])

    message = capsys.readouterr().err
    assert "stopped during a call of get_spectra" in message
    assert "The error came from get_spectra on the artistools server of vae26" in message
    assert "Traceback" not in message

    # a panic of polars that has a different cause keeps its traceback
    with (
        mock.patch("artistools.commands.show_version", side_effect=pl.exceptions.PanicException("other")),
        pytest.raises(pl.exceptions.PanicException),
    ):
        artistools.__main__.main(argsraw=["version"])


def test_getpath_and_version_keep_their_product_with_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """The text of getpath and version is the product of the command, thus --quiet must keep it."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["getpath", "--quiet"])
    assert capsys.readouterr().out.strip() == str(at.get_path("artistools_dir"))

    artistools.__main__.main(argsraw=["version", "-q"])
    assert capsys.readouterr().out.strip() == f"artistools {importlib.metadata.version('artistools')}"


def test_a_folder_that_the_command_refuses_names_the_remedy(capsys: pytest.CaptureFixture[str]) -> None:
    """The error must name the folder and not a "--" that the user never wrote, and it must name -modelpath."""
    import artistools.__main__

    modelpath = at.get_path("testdata") / "testmodel"
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["timesteps", str(modelpath)])
    message = capsys.readouterr().err
    assert f"unrecognized arguments: {modelpath}" in message
    assert f"-modelpath {modelpath}" in message


def test_a_hidden_command_stays_out_of_the_suggestions(capsys: pytest.CaptureFixture[str]) -> None:
    """The help hides server and describeinputmodel, thus a suggestion, the list, and tab completion leave them out."""
    import artistools.__main__

    assert set(at.commands.get_hidden_commands()) == {"describeinputmodel", "server"}
    for command in ("serve", "xyzzy"):
        with pytest.raises(SystemExit):
            artistools.__main__.main(argsraw=[command])
        message = capsys.readouterr().err
        assert "server" not in message
        assert "describeinputmodel" not in message

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspetcra"])
    assert "Did you mean plotspectra" in capsys.readouterr().err


def test_the_examples_name_the_full_command_in_each_help() -> None:
    """A per-command script shows the examples, and an example of a command in a group names the group."""
    spec = at.commands.CommandSpec("inputmodel.describeinputmodel", examples=(("-modelpath .", "the model"),))
    epilog = at.commands.get_command_epilog(("inputmodel", "describe"), spec)
    assert epilog is not None
    assert "artistools inputmodel describe -modelpath ." in epilog

    parser = at.commands.build_script_parser("plotartisestimators")
    assert parser is not None
    assert "artistools plotestimators Te TR . -t 300" in parser.format_help()


def test_a_window_time_is_not_the_time_of_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    """A run with --show or --interactive waits for the user, thus it reports no time of the command."""
    from artistools.__main__ import run_command

    def run_nothing(args: argparse.Namespace) -> None:
        """Do nothing."""

    with mock.patch("time.monotonic", side_effect=[0.0, 100.0, 0.0, 100.0, 0.0, 100.0]):
        run_command(run_nothing, argparse.Namespace(show=True))
        run_command(run_nothing, argparse.Namespace(interactive=True))
        assert "took" not in capsys.readouterr().err
        run_command(run_nothing, argparse.Namespace())
        assert "The command took 100.0 seconds" in capsys.readouterr().err


def test_the_server_answers_each_request_in_this_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The main function of the server must answer each request on a copy of the standard output.

    The server ran only in a subprocess of a test, thus no test of this process covered it.
    """
    modelpath = at.get_path("testdata") / "testmodel"
    requests = io.BytesIO()
    for request in (
        ("artistools.misc.fileio", "get_remote_path_kind", (modelpath,), dict[str, t.Any](), False),
        ("artistools.misc.fileio", "nosuchfunction", (), dict[str, t.Any](), False),
    ):
        remote.write_message(requests, remote.dump_message(request))
    requests.seek(0)
    monkeypatch.setattr(sys, "stdin", argparse.Namespace(buffer=requests))
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))

    resultpath = tmp_path / "results.bin"
    savedfds = os.dup(1), os.dup(2)
    try:
        with resultpath.open("wb") as resultfile, (tmp_path / "stderr.txt").open("wb") as errorfile:
            os.dup2(resultfile.fileno(), 1)
            os.dup2(errorfile.fileno(), 2)
            with pytest.raises(SystemExit) as exitinfo:
                remote.main(argsraw=[])
    finally:
        for fd, savedfd in enumerate(savedfds, start=1):
            os.dup2(savedfd, fd)
            os.close(savedfd)

    assert exitinfo.value.code == 0
    data = resultpath.read_bytes()
    assert data.startswith(remote.SERVER_START_LINE)
    replies = io.BytesIO(data[len(remote.SERVER_START_LINE) :])
    assert remote.load_reply(remote.read_message(replies))[1] == pl.__version__
    assert remote.load_reply(remote.read_message(replies))[:2] == (True, (True, False, True))
    succeeded, error, _ = remote.load_reply(remote.read_message(replies))
    assert not succeeded
    assert "nosuchfunction" in str(error)


# --- round three of the review: fileio.py, modelinfo.py, timesteps.py, dirbins.py ----------------------------------


def write_input_txt(modelpath: Path, tmin: float = 2.0, tmax: float = 300.0, nprocs: int = 4) -> None:
    """Write an input.txt with 100 logarithmic timesteps from tmin to tmax, and nprocs on the 22nd value line."""
    valuelines = ["-1", "100", "0 99", f"{tmin} {tmax}", "0.1 10", "80", "3 250", "3", *(["0"] * 13), str(nprocs)]
    (modelpath / "input.txt").write_text("\n".join(valuelines) + "\n", encoding="utf-8")


def test_a_cached_reader_takes_the_model_path_by_keyword(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader of modelpath_cache takes the path by position or by keyword, and both calls share one cache entry.

    The cache took the path by position alone, thus at.get_nprocs(modelpath=...) raised a TypeError. A reader also
    takes the name of its own first parameter, e.g. the filename of get_composition_data.
    """
    write_input_txt(tmp_path)
    (tmp_path / "model.txt").write_text("20\n", encoding="utf-8")
    (tmp_path / "compositiondata.txt").write_text("1\n0\n0\n26 2 1 2 300 1.0 56.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    at.get_nprocs.cache_clear()
    assert at.get_nprocs(modelpath=Path()) == 4
    assert at.get_nprocs(tmp_path) == 4
    assert at.get_nprocs.cache_info().hits == 1, "a relative keyword path and the absolute path share one entry"

    assert at.get_inputparams(modelpath=tmp_path)["ntstep"] == 100
    assert at.get_inputparams(modelpath=tmp_path) is at.get_inputparams(tmp_path)
    assert at.misc.get_npts_model(modelpath=tmp_path) == 20
    assert at.get_composition_data(modelpath=tmp_path)["Z"].to_list() == [26]
    # the type checkers know only the keyword modelpath, because a ParamSpec cannot rename a parameter
    getcompositiondata: t.Any = at.get_composition_data
    assert getcompositiondata(filename=tmp_path)["Z"].to_list() == [26]


def test_a_cached_reader_goes_through_pickle_and_to_the_server(tmp_path: Path) -> None:
    """The pickle of a reader of modelpath_cache holds its name, and the server calls the reader below the cache."""
    import pickle  # ruff:ignore[suspicious-pickle-import]

    assert pickle.loads(pickle.dumps(at.get_nprocs)) is at.get_nprocs  # ruff:ignore[suspicious-pickle-usage]

    write_input_txt(tmp_path)
    (tmp_path / "compositiondata.txt").write_text("1\n0\n0\n28 2 1 2 300 1.0 58.0\n", encoding="utf-8")
    servernprocs = remote.get_server_function("artistools.misc.modelinfo", "get_nprocs")
    assert not hasattr(servernprocs, "cache_clear")
    assert servernprocs(tmp_path) == 4
    servercomposition = remote.get_server_function("artistools.atomic.core", "get_composition_data")
    assert not hasattr(servercomposition, "cache_clear")
    assert servercomposition(tmp_path)["Z"].to_list() == [28]


def test_get_runfolders_gives_a_repeated_timestep_to_the_earlier_folder(tmp_path: Path) -> None:
    """A restarted run repeats the last timestep of the folder before it, and only the earlier folder keeps it.

    A folder with no earlier folder keeps its first timestep, e.g. a run whose first job folder is gone. The rule of
    a restart dropped that timestep, thus no folder held it, and a plot of that timestep stopped.
    """
    import shutil

    for foldername, timesteps in (("job0", (0, 1, 2)), ("job1", (2, 3, 4))):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "estimators_0000.out").write_text(
            "".join(f"timestep {timestep} modelgridindex 0\n\n" for timestep in timesteps), encoding="utf-8"
        )

    job0, job1 = tmp_path / "job0", tmp_path / "job1"
    assert at.misc.get_runfolders(tmp_path) == [job0, job1]
    assert at.misc.get_runfolders(tmp_path, timestep=2) == (job0,)
    assert at.misc.get_runfolders(tmp_path, timesteps=[2]) == (job0,)
    assert at.misc.get_runfolders(tmp_path, timesteps=[2, 3]) == (job0, job1)

    shutil.rmtree(job0)
    assert at.misc.get_runfolders(tmp_path, timestep=2) == (job1,)
    assert at.misc.get_runfolders(tmp_path, timesteps=[2]) == (job1,)


def test_get_runfolders_of_one_timestep_reads_no_later_folder(tmp_path: Path) -> None:
    """The earliest folder that holds a timestep keeps it, thus get_runfolders reads no folder after that one."""
    for foldername, timesteps in (("job0", (0, 1, 2)), ("job1", (2, 3, 4)), ("job2", (4, 5))):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "estimators_0000.out").write_text(
            "".join(f"timestep {timestep} modelgridindex 0\n\n" for timestep in timesteps), encoding="utf-8"
        )

    with mock.patch.object(modelinfo, "get_runfolder_timesteps", wraps=modelinfo.get_runfolder_timesteps) as mockreader:
        assert at.misc.get_runfolders(tmp_path, timestep=2) == (tmp_path / "job0",)
        assert [call.args[0] for call in mockreader.call_args_list] == [tmp_path / "job0"]

        mockreader.reset_mock()
        assert at.misc.get_runfolders(tmp_path, timestep=99) == ()
        assert len(mockreader.call_args_list) == 4

        mockreader.reset_mock()
        assert at.misc.get_runfolders(tmp_path, timesteps=[4]) == (tmp_path / "job1",)
        assert len(mockreader.call_args_list) == 4


def test_replace_outdated_file_locks_a_descriptor_that_can_write(tmp_path: Path) -> None:
    """The lock takes a descriptor that can write, because NFS refuses an exclusive flock on a read-only descriptor.

    A user who cannot write the lock file of a different user still opens it to read, which a local file system
    accepts.
    """
    import fcntl

    accessmodes: list[int] = []

    def record_access_mode(fd: int, _operation: int) -> None:
        accessmodes.append(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE)

    realopen = os.open

    def open_without_write_access(path: t.Any, flags: int, mode: int = 0o777) -> int:
        if flags & os.O_ACCMODE != os.O_RDONLY:
            raise PermissionError(path)
        return realopen(path, flags, mode)

    def replace(text: str) -> None:
        destpath.write_text("outdated", encoding="utf-8")
        replacement = tmp_path / "replacement"
        replacement.write_text(text, encoding="utf-8")
        fileio.replace_outdated_file(replacement, destpath, at.misc.get_file_identity(destpath))
        assert destpath.read_text(encoding="utf-8") == text

    destpath = tmp_path / "cache"
    with mock.patch("fcntl.flock", side_effect=record_access_mode):
        replace("first")
        # a test can run as root, and root can write a file of any mode, thus a mock refuses the write access
        with mock.patch.object(fileio.os, "open", side_effect=open_without_write_access):
            replace("second")

    assert accessmodes == [os.O_RDWR, os.O_RDONLY]


def test_read_rank_outputfiles_names_a_timestep_that_no_run_folder_holds(tmp_path: Path) -> None:
    """A timestep after the stop of a run names the timesteps of the run, and not a missing file.

    The files were there, and the message "No nlte_*.out files found" sent the user to look for them.
    """
    from artistools.misc.modelinfo import read_rank_outputfiles

    write_rank_output_model(tmp_path, {".": [(0, 0, 1.0), (1, 0, 2.0)]})

    with pytest.raises(ValueError, match=r"holds timestep 5\. .* give timesteps 0 to 1"):
        read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", timestep=5)


def test_read_rank_outputfiles_skips_an_empty_rank_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A rank file of no bytes gives no rows, and the other ranks still give theirs.

    ARTIS writes the header of a rank file at the end of the first timestep, thus a job that stopped earlier leaves
    an empty file. polars then stopped the read of all the cells with an "empty CSV" error.
    """
    from artistools.misc.modelinfo import read_rank_outputfiles

    write_rank_output_model(tmp_path, {".": [(0, cell, cell + 0.5) for cell in range(3)]})
    # two ranks share the five cells, thus rank 0 handles cells 0 to 2 and rank 1 handles cells 3 and 4
    (tmp_path / "input.txt").write_text("\n".join(["0"] * 21 + ["2"]) + "\n", encoding="utf-8")
    (tmp_path / "nlte_0001.out").write_bytes(b"")

    dfout = read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out")

    assert dfout["modelgridindex"].to_list() == [0, 1, 2]
    assert "nlte_0001.out is empty" in capsys.readouterr().err


def test_get_file_metadata_takes_a_key_with_no_value(tmp_path: Path) -> None:
    """A key of metadata.yml with no value gives no metadata, and a value that is not a mapping gives a message.

    YAML reads a key with no value as None, and the derived values then raised a TypeError.
    """
    (tmp_path / "metadata.yml").write_text("sn.txt:\n", encoding="utf-8")
    assert at.misc.get_file_metadata(tmp_path / "sn.txt") == {}

    (tmp_path / "other.txt.meta.yml").write_text("a scalar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        at.misc.get_file_metadata(tmp_path / "other.txt")


def test_one_timestep_grid_from_timesteps_out_and_from_input_txt_gives_one_range(tmp_path: Path) -> None:
    """A timestep of two runs with one grid names one range, also when only one run holds timesteps.out.

    ARTIS writes timesteps.out with 6 significant digits, and the times of input.txt have more. The test of the grids
    took a difference of 1e-4 d as a different grid, thus -timestep stopped with two equal ranges in its message.
    """
    from artistools.misc.timesteps import apply_time_range_args

    modelpaths = [tmp_path / "frominput", tmp_path / "fromfile", tmp_path / "othergrid"]
    for modelpath, tmax in zip(modelpaths, (350.0, 350.0, 360.0), strict=True):
        modelpath.mkdir()
        write_input_txt(modelpath, tmin=250.0, tmax=tmax)

    dlogt = (math.log(350.0) - math.log(250.0)) / 100
    (modelpaths[1] / "timesteps.out").write_text(
        "#timestep tstart_days tmid_days twidth_days\n"
        + "".join(
            f"{timestep} {250.0 * math.exp(timestep * dlogt):g} {250.0 * math.exp((timestep + 0.5) * dlogt):g} "
            f"{250.0 * (math.exp((timestep + 1) * dlogt) - math.exp(timestep * dlogt)):g}\n"
            for timestep in range(100)
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(timestep="90", timedays=None, timemin=None, timemax=None)
    apply_time_range_args(args, modelpaths[:2])
    assert math.isclose(args.timemin, 250.0 * math.exp(90 * dlogt), rel_tol=1e-9)

    # a grid that is different still stops the command
    with pytest.raises(SystemExit):
        apply_time_range_args(
            argparse.Namespace(timestep="90", timedays=None, timemin=None, timemax=None), modelpaths[::2]
        )


def test_the_last_costheta_bin_holds_cos_theta_of_one() -> None:
    """ARTIS puts a packet with cos θ = 1 in the last bin, thus the label of that bin includes its upper edge."""
    from artistools.misc.dirbins import get_costheta_bins

    _, _, labels = get_costheta_bins(usedegrees=False)
    assert labels[-1] == "0.8 ≤ cos θ ≤ 1.0"
    assert labels[-2] == "0.6 ≤ cos θ < 0.8"


def test_a_direction_bin_outside_the_run_gives_a_message() -> None:
    """-plotviewingangle 500 stopped with an IndexError in the labels, and now names the bins of the run."""
    with pytest.raises(ValueError, match="not one of the 100 bins 0 to 99"):
        dirbins.get_dirbin_labels([500])


def test_dirbins_give_a_message_in_place_of_a_bare_assert() -> None:
    """A direct call with a cut table or a bin that no average holds stopped with an AssertionError and no message."""
    with pytest.raises(ValueError, match="tables have different lengths"):
        dirbins.split_multitable_dataframe(pl.LazyFrame({"nu": [1.0, 2.0, 1.0], "f": [0.0, 0.0, 0.0]}))
    with pytest.raises(ValueError, match="tables have different lengths"):
        dirbins.split_multitable_dataframe(pl.LazyFrame({"nu": [], "f": []}))

    with pytest.raises(ValueError, match=r"Direction bin 1 is not the first bin of an average group"):
        dirbins.get_dirbin_labels([1], average_over_phi=True)
    with pytest.raises(ValueError, match=r"Direction bin 10 is not the first bin of an average group"):
        dirbins.get_dirbin_labels([10], average_over_theta=True)
    with pytest.raises(ValueError, match="both the phi and theta"):
        dirbins.get_dirbin_labels([0], average_over_phi=True, average_over_theta=True)

    assert dirbins.get_dirbin_labels([10], average_over_phi=True) == {
        10: dirbins.get_costheta_bins(usedegrees=False)[2][1]
    }
