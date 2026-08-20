"""Unit tests for the shared helpers in artistools.misc.

These tests use only synthetic data written under tmp_path, so they run quickly and do not require
the downloaded ARTIS test model.
"""

import argparse
import gzip
import io
import lzma
import math
import os
import subprocess
import sys
import typing as t
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import polars.testing as pltest
import pytest
import yaml

import artistools as at
from artistools.misc import dirbins


def _write_timesteps_out(modeldir: Path) -> None:
    """Write a minimal timesteps.out with 5 evenly spaced timesteps (mids 105..145)."""
    lines = ["#timestep tmid_days tstart_days twidth_days"]
    for ts in range(5):
        tstart = 100 + ts * 10
        lines.append(f"{ts} {tstart + 5} {tstart} 10")
    (modeldir / "timesteps.out").write_text("\n".join(lines) + "\n")


# --- cliutils.py -------------------------------------------------------------------------------


def test_add_cli_arg_helpers() -> None:
    """The shared argument helpers must define the standard flags, types, and defaults."""
    parser = argparse.ArgumentParser()
    at.add_modelpath_arg(parser, multiplepaths=True, default=[])
    at.add_outputfile_arg(parser, default=Path("out.pdf"))
    at.add_timestep_arg(parser)
    at.add_timedays_arg(parser)
    at.add_timeminmax_args(parser)
    at.add_axis_limit_args(parser, xlimtype=int, xmindefault=1000, xmaxdefault=2000)
    at.add_series_style_args(parser, colordefault=["C0", "C1"], include_linealpha=True)
    at.add_figscale_args(parser, figscaledefault=1.8, include_figwidthscale=True)
    at.add_filter_args(parser)
    at.add_maxpacketfiles_arg(parser)

    args = parser.parse_args([])
    assert args.modelpath == []
    assert args.outputfile == Path("out.pdf")
    assert args.timestep is None
    assert args.timedays is None
    assert args.figscale == 1.8
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


def test_add_cli_arg_helper_variants() -> None:
    """The non-default helper modes must reproduce the per-command argument shapes."""
    parser = argparse.ArgumentParser()
    at.add_modelpath_arg(parser, positional=True, multiplepaths=True, default=[])
    at.add_timestep_arg(parser, kind="int", default=70)
    at.add_timedays_arg(parser, kind="float")
    at.add_outputpath_arg(parser)
    args = parser.parse_args(["model1", "-timestep", "12", "-timedays", "45.5"])
    assert args.modelpath == [Path("model1")]
    assert args.timestep == 12
    assert args.timedays == 45.5
    assert args.outputpath == "."

    parserappend = argparse.ArgumentParser()
    at.add_timestep_arg(parserappend, kind="strappend")
    assert parserappend.parse_args(["-ts", "5", "-ts", "6"]).timestep == ["5", "6"]

    parserrequired = argparse.ArgumentParser()
    at.add_modelpath_arg(parserrequired, required=True)
    with pytest.raises(SystemExit):
        parserrequired.parse_args([])


def test_set_args_from_dict_does_not_mutate_caller() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-outputfile", "-o", type=Path)
    kwargs = {"o": "somefile.pdf"}
    at.set_args_from_dict(parser, kwargs)
    assert kwargs == {"o": "somefile.pdf"}
    assert parser.parse_args([]).outputfile == Path("somefile.pdf")

    with pytest.raises(ValueError, match="badargname"):
        at.set_args_from_dict(parser, {"badargname": 1})


# --- fileio.py (print_saved) -------------------------------------------------------------------


def test_print_saved(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """print_saved must emit a runnable macOS open command with a path relative to the working directory."""
    monkeypatch.chdir(tmp_path)

    at.print_saved(tmp_path / "subdir" / "out.pdf")
    assert capsys.readouterr().out == "open subdir/out.pdf\n"

    at.print_saved("out.pdf")
    assert capsys.readouterr().out == "open out.pdf\n"

    at.print_saved(tmp_path / "subdir" / ".." / "out.pdf")
    assert capsys.readouterr().out == "open out.pdf\n"

    at.print_saved(tmp_path / "with space.pdf")
    assert capsys.readouterr().out == "open 'with space.pdf'\n"


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
    assert at.zopenpl(tmp_path / "plain.txt") == tmp_path / "plain.txt"
    assert at.zopenpl(tmp_path / "gz.txt") == tmp_path / "gz.txt.gz"
    # ...but an opened file object for .xz
    result_xz = at.zopenpl(tmp_path / "xz.txt")
    assert not isinstance(result_xz, Path)
    with result_xz as f:
        assert f.read().decode("utf-8") == "xz contents\n"


def test_zopen_unshadowed(tmp_path: Path) -> None:
    """A stale compressed sibling never shadows a freshly written uncompressed file."""
    (tmp_path / "both.txt").write_text("fresh contents\n")
    with gzip.open(tmp_path / "both.txt.gz", "wt", encoding="utf-8") as f:
        f.write("stale contents\n")

    # zopen prefers the compressed sibling, which is why the plain-open call sites need the other helper
    with at.zopen(tmp_path / "both.txt") as f:
        assert f.read() == "stale contents\n"

    with at.zopen_unshadowed(tmp_path / "both.txt") as f:
        assert f.read() == "fresh contents\n"

    # with the plain file gone, the compressed sibling is used after all
    (tmp_path / "both.txt").unlink()
    with at.zopen_unshadowed(tmp_path / "both.txt") as f:
        assert f.read() == "stale contents\n"

    # a compressed file addressed by its own name is decompressed, not opened raw
    with at.zopen_unshadowed(tmp_path / "both.txt.gz") as f:
        assert f.read() == "stale contents\n"

    # a name with no file and no compressed sibling reports the name the caller asked for
    with pytest.raises(FileNotFoundError):
        at.zopen_unshadowed(tmp_path / "absent.txt")


def test_read_wsv(tmp_path: Path) -> None:
    """Columns aligned with variable whitespace parse correctly, with comments and blank lines removed."""
    filepath = tmp_path / "aligned.txt"
    filepath.write_text("# file header comment\n  colA   colB  colC\n1    2.5  x # inline comment\n\n 4\t5\ty\n")

    df = at.read_wsv(filepath, comment_prefix="#")
    pltest.assert_frame_equal(df, pl.DataFrame({"colA": [1, 4], "colB": [2.5, 5.0], "colC": ["x", "y"]}))

    # skip_rows applies before comment handling, and new_columns names a headerless read
    df_noheader = at.read_wsv(filepath, has_header=False, skip_rows=2, new_columns=["a", "b", "c"], comment_prefix="#")
    assert df_noheader.columns == ["a", "b", "c"]
    assert df_noheader.height == 2

    # a compressed file is read transparently
    with gzip.open(tmp_path / "data.txt.gz", "wt", encoding="utf-8") as f:
        f.write("p   q\n1   2\n")
    pltest.assert_frame_equal(at.read_wsv(tmp_path / "data.txt"), pl.DataFrame({"p": [1], "q": [2]}))

    # trailing whitespace on every line must not become a trailing null column
    (tmp_path / "trailing.txt").write_text("colA colB  \n1 2 \n3 4 \t\n", encoding="utf-8")
    pltest.assert_frame_equal(at.read_wsv(tmp_path / "trailing.txt"), pl.DataFrame({"colA": [1, 3], "colB": [2, 4]}))

    # a name-based projection parses only the requested columns, in the requested order
    dfprojected = at.read_wsv(
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


def test_read_wsv_prefers_uncompressed_file(tmp_path: Path) -> None:
    """A freshly written uncompressed file must win over a stale compressed sibling of the same name."""
    (tmp_path / "f.txt").write_text("v\n2\n", encoding="utf-8")
    with gzip.open(tmp_path / "f.txt.gz", "wt", encoding="utf-8") as f:
        f.write("v\n1\n")

    assert at.read_wsv(tmp_path / "f.txt")["v"].to_list() == [2]

    # the header comment is read through a second open, which must apply the same precedence: reading it
    # from the stale sibling would label the fresh data with the stale column names
    (tmp_path / "h.txt").write_text("# freshA freshB\n1 2\n", encoding="utf-8")
    with gzip.open(tmp_path / "h.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# staleX staleY\n9 9\n")

    dfheader = at.read_wsv(tmp_path / "h.txt", header_from_comment=True, comment_prefix="#")
    pltest.assert_frame_equal(dfheader, pl.DataFrame({"freshA": [1], "freshB": [2]}))


def test_read_wsv_all_null_inference_sample(tmp_path: Path) -> None:
    """A column whose first 10000 rows are all null tokens must still infer as numeric from the later rows."""
    filepath = tmp_path / "nullsample.txt"
    nnullrows = 12000  # more rows than the schema inference sample
    filepath.write_text("a b\n" + "".join(f"{i} nan\n" for i in range(nnullrows)) + f"{nnullrows} 3.5\n")

    df = at.read_wsv(filepath)
    assert df["b"].dtype == pl.Float64
    assert df["b"].item(-1) == pytest.approx(3.5)
    assert df["b"].null_count() == nnullrows


def test_read_wsv_late_type_change(tmp_path: Path) -> None:
    """A column that turns from integer to float beyond the schema inference sample is read as floats."""
    filepath = tmp_path / "latefloat.txt"
    nintrows = 20000  # more rows than the schema inference sample
    filepath.write_text("a b\n" + "".join(f"{i} 1\n" for i in range(nintrows)) + f"{nintrows} 2.5\n")

    df = at.read_wsv(filepath)
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
    assert at.firstexisting_or_none(["a.txt"], folder=firstdir) == firstdir / "a.txt"
    assert at.firstexisting_or_none(["nope.txt"], folder=firstdir) is None


def test_firstexisting_with_an_absolute_path(tmp_path: Path) -> None:
    """An absolute path is not below the default folder, but the message must not raise a ValueError."""
    (tmp_path / "here.txt").write_text("here")
    assert at.firstexisting(tmp_path / "here.txt") == tmp_path / "here.txt"

    missingpath = tmp_path / "notafile.txt"
    with pytest.raises(FileNotFoundError, match=str(missingpath)):
        at.firstexisting(missingpath)

    assert at.firstexisting_or_none(missingpath) is None


def test_readnoncommentline() -> None:
    stream = io.StringIO("\n# a comment\n   # indented comment\nreal data line\nsecond\n")
    assert at.readnoncommentline(stream) == "real data line\n"
    # the next call continues from where the last one stopped
    assert at.readnoncommentline(stream) == "second\n"

    # reaching EOF without a data line raises rather than looping forever
    with pytest.raises(EOFError, match="end of file"):
        at.readnoncommentline(io.StringIO(""))
    with pytest.raises(EOFError, match="end of file"):
        at.readnoncommentline(io.StringIO("\n# only comments\n   \n"))


def test_get_file_metadata(tmp_path: Path) -> None:
    # r_v is derived from a_v and e_bminusv
    (tmp_path / "rv.txt").write_text("data")
    (tmp_path / "rv.txt.meta.yml").write_text("a_v: 1.0\ne_bminusv: 0.5\n")
    assert at.get_file_metadata(tmp_path / "rv.txt")["r_v"] == pytest.approx(2.0)

    # a_v is derived from e_bminusv and r_v
    (tmp_path / "av.txt").write_text("data")
    (tmp_path / "av.txt.meta.yml").write_text("e_bminusv: 0.4\nr_v: 3.0\n")
    assert at.get_file_metadata(tmp_path / "av.txt")["a_v"] == pytest.approx(1.2)

    # e_bminusv is derived from a_v and r_v
    (tmp_path / "ebv.txt").write_text("data")
    (tmp_path / "ebv.txt.meta.yml").write_text("a_v: 2.0\nr_v: 4.0\n")
    assert at.get_file_metadata(tmp_path / "ebv.txt")["e_bminusv"] == pytest.approx(0.5)

    # metadata can also come from a combined metadata.yml keyed by the file path
    combineddir = tmp_path / "combined"
    combineddir.mkdir()
    combinedfile = combineddir / "spectrum.txt"
    combinedfile.write_text("data")
    (combineddir / "metadata.yml").write_text(yaml.safe_dump({str(combinedfile): {"a_v": 1.0, "e_bminusv": 0.25}}))
    combined_metadata = at.get_file_metadata(combinedfile)
    assert combined_metadata["r_v"] == pytest.approx(4.0)

    # no metadata file present -> empty dict
    (tmp_path / "nometa.txt").write_text("data")
    assert at.get_file_metadata(tmp_path / "nometa.txt") == {}


def test_write_parquet_atomic(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    parquetpath = tmp_path / "out.parquet"

    at.write_parquet_atomic(df, parquetpath)

    assert parquetpath.exists()
    pltest.assert_frame_equal(pl.read_parquet(parquetpath), df)
    # the temporary partial file must not be left behind
    assert list(tmp_path.glob("*.partial*")) == []


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
        at.write_parquet_atomic(pl.DataFrame({"timestep": [0, 1]}), parquetpath)

    assert not seen_midwrite, f"a concurrent reader would have globbed the in-flight temporary file {seen_midwrite}"
    assert pl.read_parquet(parquetpath)["timestep"].to_list() == [0, 1]


def test_write_parquet_atomic_is_readable_in_a_shared_directory(tmp_path: Path) -> None:
    """A cache written into a group-shared model directory must not inherit mkstemp's private 0600 mode."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)  # chmod, not mkdir(mode=...), which the process umask would mask off
    parquetpath = shared / "out.parquet"

    at.write_parquet_atomic(pl.DataFrame({"a": [1]}), parquetpath)
    assert parquetpath.stat().st_mode & 0o777 == 0o664

    # a private directory must stay private, and rewriting keeps whatever mode the destination already had
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o700)
    privatepath = private / "out.parquet"
    at.write_parquet_atomic(pl.DataFrame({"a": [1]}), privatepath)
    assert privatepath.stat().st_mode & 0o777 == 0o600

    privatepath.chmod(0o640)
    at.write_parquet_atomic(pl.DataFrame({"a": [2]}), privatepath)
    assert privatepath.stat().st_mode & 0o777 == 0o640
    assert pl.read_parquet(privatepath)["a"].item() == 2


# --- general.py --------------------------------------------------------------------------------


def test_df_filter_minmax_bounded() -> None:
    df = pl.DataFrame({"x": list(range(11))})  # 0..10

    # both bounds: keep the interior plus the nearest exterior row on each side (for interpolation)
    bounded = at.misc.df_filter_minmax_bounded(df, "x", 2.5, 7.5).collect()
    assert bounded["x"].to_list() == [2, 3, 4, 5, 6, 7, 8]

    # no bounds is a pass-through
    unbounded = at.misc.df_filter_minmax_bounded(df, "x", None, None).collect()
    assert unbounded["x"].to_list() == list(range(11))

    # single-sided bounds
    minonly = at.misc.df_filter_minmax_bounded(df, "x", 2.5, None).collect()
    assert minonly["x"].to_list() == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    maxonly = at.misc.df_filter_minmax_bounded(df, "x", None, 7.5).collect()
    assert maxonly["x"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


# --- cliutils.py -------------------------------------------------------------------------------


def test_parse_range() -> None:
    assert list(at.parse_range("3-5", {})) == [3, 4, 5]
    assert list(at.parse_range("5", {})) == [5]
    assert list(at.parse_range("5-3", {})) == [3, 4, 5]  # reversed range is sorted
    assert list(at.parse_range("start-end", {"start": 2, "end": 4})) == [2, 3, 4]

    with pytest.raises(ValueError, match="Bad range"):
        at.parse_range("1-2-3", {})


def test_normalize_path_list() -> None:
    assert at.normalize_path_list("a/b") == [Path("a/b")]
    assert at.normalize_path_list(Path("a/b")) == [Path("a/b")]
    assert at.normalize_path_list([["x"], "y"]) == [Path("x"), Path("y")]
    assert at.normalize_path_list([]) == [Path()]
    assert at.normalize_path_list(None, default="fallback") == [Path("fallback")]


def test_resolve_outputfile(tmp_path: Path) -> None:
    # no outputfile falls back to the default filename
    assert at.resolve_outputfile(None, "default.pdf") == Path("default.pdf")

    # an existing directory gets the default filename appended
    existingdir = tmp_path / "existing"
    existingdir.mkdir()
    assert at.resolve_outputfile(existingdir, "default.pdf") == existingdir / "default.pdf"

    # a path with a file extension is returned unchanged
    assert at.resolve_outputfile(tmp_path / "chosen.pdf", "default.pdf") == tmp_path / "chosen.pdf"

    # a suffixless path is treated as a folder, created, and the default filename appended
    newdir = tmp_path / "newfolder"
    assert at.resolve_outputfile(newdir, "default.pdf") == newdir / "default.pdf"
    assert newdir.is_dir()


def test_set_args_from_dict() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-foo", type=int, default=1)
    parser.add_argument("-o", "--output", dest="outputfile", default="z")

    # defaults can be overridden by dest name ("foo") or by an option string whose dest differs ("output")
    at.set_args_from_dict(parser, {"foo": 5, "output": "y"})
    args = parser.parse_args([])
    assert args.foo == 5
    assert args.outputfile == "y"

    with pytest.raises(ValueError, match="Unknown argument names"):
        at.set_args_from_dict(parser, {"nonexistent": 1})


def test_get_filterfunc() -> None:
    # no filter arguments -> no filter function
    assert at.get_filterfunc(argparse.Namespace()) is None

    # a moving-average filter reproduces a windowed mean (with edge padding)
    filterfunc = at.get_filterfunc(argparse.Namespace(filtermovingavg=3))
    assert filterfunc is not None
    assert filterfunc([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx([4 / 3, 2.0, 3.0, 4.0, 14 / 3])

    # the Savitzky-Golay filter matches scipy.signal.savgol_filter(y, window_length=5, polyorder=3, mode="interp"),
    # which it replaced
    filterfunc = at.get_filterfunc(argparse.Namespace(filtersavgol=["5", "3"]))
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
        at.savgol_filter(yvalues, window_length=4, polyorder=3)
    with pytest.raises(ValueError, match="must be at least zero and less than window_length"):
        at.savgol_filter(yvalues, window_length=5, polyorder=7)
    with pytest.raises(ValueError, match="must be at least zero and less than window_length"):
        at.savgol_filter(yvalues, window_length=5, polyorder=-1)
    with pytest.raises(ValueError, match="exceeds the data length"):
        at.savgol_filter(yvalues[:3], window_length=5, polyorder=3)
    with pytest.raises(ValueError, match="needs a 1D array"):
        at.savgol_filter(np.tile(yvalues, (2, 1)), window_length=5, polyorder=3)


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
    assert np.allclose(at.gaussian_filter_wrap(data, sigma=1.2), expected, rtol=1e-10, atol=1e-12)

    with pytest.raises(ValueError, match="must be greater than zero"):
        at.gaussian_filter_wrap(data, sigma=0.0)
    with pytest.raises(ValueError, match="needs a 2D array"):
        at.gaussian_filter_wrap(data[0], sigma=1.2)


# --- timesteps.py ------------------------------------------------------------------------------


def test_get_timestep_of_timedays(tmp_path: Path) -> None:
    _write_timesteps_out(tmp_path)

    # timesteps span [100,110), [110,120), ... [140,150)
    assert at.get_timestep_of_timedays(tmp_path, 125) == 2
    assert at.get_timestep_of_timedays(tmp_path, 100) == 0
    assert at.get_timestep_of_timedays(tmp_path, 149) == 4
    assert at.get_timestep_of_timedays(tmp_path, "125d") == 2  # accepts a "<days>d" string

    with pytest.raises(ValueError, match="Could not find timestep"):
        at.get_timestep_of_timedays(tmp_path, 500)


def test_get_deposition(tmp_path: Path) -> None:
    _write_timesteps_out(tmp_path)
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
    _write_timesteps_out(baddir)
    badlines = ["#tmid_days gammadep_Lsun positrondep_Lsun total_dep_Lsun"]
    badlines.extend(f"{999 + ts} {ts + 1.0} {(ts + 1) * 0.1} {(ts + 1) * 1.1}" for ts in range(5))
    (baddir / "deposition.out").write_text("\n".join(badlines) + "\n")

    with pytest.raises(AssertionError, match="Deposition times do not match"):
        at.get_deposition(baddir).collect()


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


def test_average_direction_bins_rebuilds_derived_cols(monkeypatch: pytest.MonkeyPatch) -> None:
    """A column derived non-linearly is rebuilt from the averaged values, not averaged itself."""
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

    averaged = dirbins.average_direction_bins(dirbindataframes, overangle="phi", derivedcols=[logcol])

    meanvalue = sum(range(nphibins)) / nphibins
    for start_bin in (0, 4, 8):
        row = averaged[start_bin].collect()
        assert row["value"].item() == pytest.approx(meanvalue)
        assert row["logvalue"].item() == pytest.approx(math.log10(meanvalue))

    # without the expressions the magnitude-like column is averaged, which the dark bin sends to -inf
    averaged_nodeclare = dirbins.average_direction_bins(dirbindataframes, overangle="phi")
    assert averaged_nodeclare[0].collect()["logvalue"].item() == -math.inf


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
    _write_timesteps_out(tmp_path)

    for clamp in (True, False):
        timestepmin, timestepmax, tlow, thigh = at.get_time_range(
            tmp_path, timestep_range_str="1-3", clamp_to_timesteps=clamp
        )
        assert (timestepmin, timestepmax) == (1, 3)
        assert tlow == pytest.approx(at.get_timestep_times(tmp_path, loc="start")[1])
        assert thigh == pytest.approx(at.get_timestep_times(tmp_path, loc="end")[3])


def test_check_averaging_angles() -> None:
    """Averaging over phi and theta at once must be rejected wherever the values arrive."""
    for phi, theta in ((False, False), (True, False), (False, True)):
        at.check_averaging_angles(phi, theta)

    with pytest.raises(ValueError, match="both the phi and theta"):
        at.check_averaging_angles(average_over_phi=True, average_over_theta=True)


def test_viewingangle_averaging_flags_are_mutually_exclusive() -> None:
    """The two averaging flags are rejected by argparse itself, for every command that defines them."""
    parser = argparse.ArgumentParser()
    at.add_viewingangle_args(parser)

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

import artistools as at


def square(x):
    return x * x


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    assert at.parallel_map(square, range(4)) == [0, 1, 4, 9]
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
    assert at.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [None, None]})).columns == ["a"]
    assert at.drop_trailing_null_column(pl.LazyFrame({"a": [1, 2], "b": [None, None]})).collect_schema().names() == [
        "a"
    ]

    # a real last column, including one that is only partly null, must stay
    assert at.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [3, 4]})).columns == ["a", "b"]
    assert at.drop_trailing_null_column(pl.DataFrame({"a": [1, 2], "b": [None, 4]})).columns == ["a", "b"]

    # no rows means nothing to judge, so keep every column
    emptyschema = {"a": pl.Int64, "b": pl.Float64}
    assert at.drop_trailing_null_column(pl.DataFrame({"a": [], "b": []}, schema=emptyschema)).columns == ["a", "b"]
    assert at.drop_trailing_null_column(
        pl.LazyFrame({"a": [], "b": []}, schema=emptyschema)
    ).collect_schema().names() == ["a", "b"]


def test_get_series_label() -> None:
    """A series is named by its -label entry, or by the fallback when the user gave none for it."""
    assert at.get_series_label(["A", "B"], 1, "modelname") == "B"
    assert at.get_series_label([None, "B"], 0, "modelname") == "modelname"

    # trim_or_pad sizes the list to the model paths, so a per-series index can run off the end
    assert at.get_series_label(["A"], 3, "modelname") == "modelname"

    # a sentinel index such as the -1 this codebase uses for a direction bin must not wrap to the last label
    assert at.get_series_label(["A", "B"], -1, "modelname") == "modelname"

    # an empty label is a series deliberately left out of the legend, not a missing one
    # the return type is str, so falsy is the empty string rather than the model name
    assert not at.get_series_label([""], 0, "modelname")
