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
from artistools.misc import fileio


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
    at.addarg_modelpath(parser, multiplepaths=True, default=[])
    at.addarg_output(parser, kind="file", default=Path("out.pdf"))
    at.addarg_timestep(parser)
    at.addarg_timedays(parser)
    at.addarg_timeminmax(parser)
    at.addarg_axislimits(parser, xlimtype=int, xmindefault=1000, xmaxdefault=2000)
    at.addarg_seriesstyle(parser, colordefault=["C0", "C1"], include_linealpha=True)
    at.addarg_figscale(parser, figscaledefault=1.8, include_figwidthscale=True)
    at.addarg_filter(parser)
    at.addarg_maxpacketfiles(parser)

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
    at.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])
    at.addarg_timestep(parser, kind="int", default=70)
    at.addarg_timedays(parser, kind="float")
    at.addarg_output(parser, kind="folder", default=Path())
    args = parser.parse_args(["model1", "-timestep", "12", "-timedays", "45.5"])
    assert args.modelpath == [Path("model1")]
    assert args.timestep == 12
    assert args.timedays == 45.5
    # one helper serves both kinds, thus the folder of a command is a Path as the file of one is
    assert args.outputfile == Path()
    assert args.outputkind == "folder"

    parserappend = argparse.ArgumentParser()
    at.addarg_timestep(parserappend, kind="strappend")
    assert parserappend.parse_args(["-ts", "5", "-ts", "6"]).timestep == ["5", "6"]

    parserrequired = argparse.ArgumentParser()
    at.addarg_modelpath(parserrequired, required=True)
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
    """print_saved must emit a runnable open command with a path relative to the working directory."""
    monkeypatch.chdir(tmp_path)

    at.print_saved(tmp_path / "subdir" / "out.pdf")
    opencommand = "open" if sys.platform == "darwin" else "xdg-open"
    assert capsys.readouterr().out == f"{opencommand} subdir/out.pdf\n"

    at.print_saved("out.pdf")
    assert capsys.readouterr().out == f"{opencommand} out.pdf\n"

    at.print_saved(tmp_path / "subdir" / ".." / "out.pdf")
    assert capsys.readouterr().out == f"{opencommand} out.pdf\n"

    at.print_saved(tmp_path / "with space.pdf")
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
        at.print_saved("out.pdf")
        assert capsys.readouterr().out == f"{line}\n"

    # a name that holds a space takes the quotation marks that the platform reads
    monkeypatch.setattr(sys, "platform", "win32")
    at.print_saved("with space.pdf")
    assert capsys.readouterr().out == 'start "" "with space.pdf"\n'

    monkeypatch.setattr(sys, "platform", "linux")
    at.print_saved("with space.pdf")
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


def test_zopen_does_not_let_a_stale_compressed_sibling_shadow_a_named_file(tmp_path: Path) -> None:
    """A stale compressed sibling never shadows a freshly written uncompressed file."""
    (tmp_path / "both.txt").write_text("fresh contents\n")
    with gzip.open(tmp_path / "both.txt.gz", "wt", encoding="utf-8") as f:
        f.write("stale contents\n")

    # the named file wins, thus re-running the simulation is enough to change what a plot shows
    with at.zopen(tmp_path / "both.txt") as f:
        assert f.read() == "fresh contents\n"

    # zopenpl applies the same precedence, so the two readers never disagree about which file to read
    assert at.zopenpl(tmp_path / "both.txt") == tmp_path / "both.txt"

    # with the plain file gone, the compressed sibling is used after all
    (tmp_path / "both.txt").unlink()
    with at.zopen(tmp_path / "both.txt") as f:
        assert f.read() == "stale contents\n"
    assert at.zopenpl(tmp_path / "both.txt") == tmp_path / "both.txt.gz"

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


def test_read_wsv_whitespace_runs(tmp_path: Path) -> None:
    """A run of ASCII whitespace between two fields acts as a single separator."""
    filepath = tmp_path / "whitespace.txt"
    # line 3 holds whitespace only, and line 5 holds a carriage return between two fields
    filepath.write_bytes(b"a\t\tb   c\r\n 1 \t 2\t\t\t3 \r\n\t \r\n4\t5     6\r\n7 8\r9\n")

    df = at.read_wsv(filepath)
    pltest.assert_frame_equal(df, pl.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}))

    # a no-break space is not ASCII whitespace, thus it must stay inside its field
    (tmp_path / "nbsp.txt").write_bytes("ion pop\nFe\u00a0II 1.0\n".encode())
    dfnbsp = at.read_wsv(tmp_path / "nbsp.txt")
    assert dfnbsp.columns == ["ion", "pop"]
    assert dfnbsp["ion"].to_list() == ["Fe\u00a0II"]

    # a compressed file gives the same result, including an xz file, which polars cannot read itself
    with lzma.open(tmp_path / "whitespace_xz.txt.xz", "wb") as f:
        f.write(filepath.read_bytes())
    pltest.assert_frame_equal(at.read_wsv(tmp_path / "whitespace_xz.txt"), df)


def test_read_wsv_invalid_utf8(tmp_path: Path) -> None:
    """A byte that is not valid UTF-8 stops the read, unless a comment holds it."""
    # a comment of a file from a different source can hold e.g. a degree sign in Latin-1
    (tmp_path / "latin1comment.txt").write_bytes(b"colA colB\n1 2   # 30\xb0C\n3 4\n")
    pltest.assert_frame_equal(
        at.read_wsv(tmp_path / "latin1comment.txt", comment_prefix="#"), pl.DataFrame({"colA": [1, 3], "colB": [2, 4]})
    )

    # the header comment holds the column names, thus it must take a bad byte as the other comments do
    (tmp_path / "latin1header.txt").write_bytes(b"# t_days mag_30\xb0C\n1.0 2.0\n")
    dfheader = at.read_wsv(tmp_path / "latin1header.txt", header_from_comment=True, comment_prefix="#")
    assert dfheader.columns == ["t_days", "mag_30\ufffdC"]
    assert dfheader["t_days"].to_list() == [1.0]

    # the same byte in the data of a column gives an error, and not a value that holds bad text
    (tmp_path / "latin1data.txt").write_bytes(b"colA colB\n1 2\n3 4\xb0\n")
    with pytest.raises(pl.exceptions.ComputeError):
        at.read_wsv(tmp_path / "latin1data.txt", comment_prefix="#")

    # a column can hold the replacement character as data, even when a comment holds a bad byte. The
    # check reads the bytes, thus it tells the two apart
    (tmp_path / "mixed.txt").write_bytes("colA name\n1 x\ufffdy  # 30".encode() + b"\xb0C\n2 z\n")
    dfmixed = at.read_wsv(tmp_path / "mixed.txt", comment_prefix="#")
    assert dfmixed["name"].to_list() == ["x\ufffdy", "z"]


def test_read_wsv_no_data(tmp_path: Path) -> None:
    """A file that holds no data raises a polars error that names the file."""
    for filename, contents in (("empty.txt", b""), ("comments.txt", b"# only a comment\n")):
        filepath = tmp_path / filename
        filepath.write_bytes(contents)

        with pytest.raises(pl.exceptions.PolarsError) as excinfo:
            at.read_wsv(filepath, comment_prefix="#")

        assert any(str(filepath) in note for note in excinfo.value.__notes__ or [])


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
    at.write_parquet_atomic(pl.DataFrame({"a": [2]}), privatepath, replaces=at.get_file_identity(privatepath))
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
        at.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath)

    assert pl.read_parquet(parquetpath)["a"].to_list() == [1, 2, 3], "the file a reader may hold was replaced"
    assert list(tmp_path.glob("*.partial*")) == []


def test_write_parquet_atomic_replaces_an_outdated_file(tmp_path: Path) -> None:
    """The file whose identity the caller passes as replaces gets replaced, since its data is out of date."""
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(parquetpath)

    at.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath, replaces=at.get_file_identity(parquetpath))

    assert pl.read_parquet(parquetpath)["a"].to_list() == [4, 5, 6]
    assert list(tmp_path.glob("*.partial*")) == []

    # without replaces, an existing file is kept: this write does not claim to supersede anything
    at.write_parquet_atomic(pl.DataFrame({"a": [7, 8, 9]}), parquetpath)
    assert pl.read_parquet(parquetpath)["a"].to_list() == [4, 5, 6]


def test_write_parquet_atomic_replaces_only_the_file_found_outdated(tmp_path: Path) -> None:
    """A rival that already replaced the out-of-date file is kept, whenever this writer started.

    The identity comes from the stat that showed the caller its cache was out of date. Taking it at the
    start of the write instead would snapshot the rival's fresh file as the one to replace, and rename over
    it while a reader scans it.
    """
    parquetpath = tmp_path / "batch.out.parquet.tmp"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(parquetpath)
    outdated = at.get_file_identity(parquetpath)

    # the rival reads the same inputs and finishes its replacement first
    pl.DataFrame({"a": [4, 5, 6]}).write_parquet(tmp_path / "rival")
    (tmp_path / "rival").replace(parquetpath)

    at.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath, replaces=outdated)

    assert at.get_file_identity(parquetpath) != outdated, "the rival's file must still be in place"
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
        at.write_parquet_atomic(pl.DataFrame({"a": [4, 5, 6]}), parquetpath)

    assert pl.read_parquet(parquetpath)["a"].to_list() == [1, 2, 3], "the file a reader may hold was replaced"

    # the fallback still replaces the file the caller found out of date
    with mock.patch.object(fileio.os, "link", side_effect=OSError):
        at.write_parquet_atomic(pl.DataFrame({"a": [7, 8, 9]}), parquetpath, replaces=at.get_file_identity(parquetpath))
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
    # a different user in a shared model directory needs to open and flock the lock, which takes only read
    # access: the lock is opened read-only and made world-readable whatever the umask
    assert lockpath.stat().st_mode & 0o444 == 0o444


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

    # the message names the range that the run covers, so that the user can correct the value
    with pytest.raises(ValueError, match=r"No timestep of this model covers 500 days.*100\.00 to 150\.00 days"):
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


def test_average_direction_bins_averages_every_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every column is averaged, so a column that is not linear in the bins must be derived afterwards.

    This is why readfile() derives the magnitude only once the bins are averaged: the mean of the
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
    at.addarg_viewingangle(parser)

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

    # the thread pool starts no process either, thus it changes no such default
    assert at.parallel_map(square, range(4), allow_multiprocessing=False) == [0, 1, 4, 9]
    assert mp.get_start_method() == "fork", mp.get_start_method()

    assert at.parallel_map(square, range(4)) == [0, 1, 4, 9]

    # a free-threading build takes the thread pool for this call as well, thus it starts no process
    if sys._is_gil_enabled():
        assert mp.get_start_method() == "spawn", mp.get_start_method()
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

    parser = argparse.ArgumentParser()
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
    import artistools.transitions

    parser = argparse.ArgumentParser()
    artistools.transitions.addargs(parser)
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
    import artistools.transitions

    parser = argparse.ArgumentParser()
    artistools.transitions.addargs(parser)
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
    (tmp_path / "estimators_0000.out").write_text("timestep 0 modelgridindex 0\n")
    (tmp_path / "model.txt").write_text("1\n1.0\n0 0.0 0.0 0.0 0.0\n")

    with pytest.raises(ValueError, match="Cell 0 holds no matter"):
        read_rank_outputfiles(tmp_path, "nlte_{mpirank:04d}.out", modelgridindex=0)


def test_addarg_modelpath_positional_also_takes_the_option() -> None:
    """A command whose path is positional must also accept -modelpath.

    Some commands take -modelpath and others take a positional path. A user who learns one form must
    not meet "unrecognized arguments" with the other.
    """

    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        at.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])
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
        at.addarg_modelpath(parser, positional=True, multiplepaths=True, default=[Path()])
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


def test_check_time_selection_reads_each_spelling_as_argparse_does() -> None:
    """The test of the command line must give each string the reading that argparse gives it.

    argparse joins a value to a flag of one letter alone, thus -t300 gives -t the value 300, and -ts70
    reads as -t with the value s70 rather than as a timestep.
    """
    import artistools.transitions

    def parse(argsraw: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
        parser = argparse.ArgumentParser()
        artistools.transitions.addargs(parser)
        return parser, parser.parse_args(argsraw)

    # the joined value of a one-letter flag counts, and the timestep here is the default of the parser
    for argsraw in (["-t300", "-ts", "70"], ["-ts=70", "-t", "300"], ["-ts", "70", "-t", "300"]):
        parser, namespace = parse(argsraw)
        with pytest.raises(SystemExit) as excinfo:
            at.misc.check_time_selection(parser, namespace, argsraw)
        assert excinfo.value.code == 1, argsraw

    # argparse reads -ts70 as -t with the value s70, thus no timestep is given and no conflict exists
    argsraw = ["-ts70", "-t", "300"]
    parser, namespace = parse(argsraw)
    assert namespace.timestep == parser.get_default("timestep")
    at.misc.check_time_selection(parser, namespace, argsraw)


def test_timedays_of_a_joined_ts_value_names_the_mistake() -> None:
    """-ts70 reads as -t s70, thus the message must say to put a space after -ts."""
    with pytest.raises(ValueError, match=r"reads as -t s70, thus put a space after -ts"):
        at.get_timestep_of_timedays(at.get_path("testdata") / "testmodel", "s70")


def test_import_optional_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional dependency must say how to install it, and not give a bare traceback."""
    import builtins

    realimport = builtins.__import__

    def failing_import(name: str, *importargs: t.Any, **importkwargs: t.Any) -> object:
        if name.startswith("pyvista"):
            raise ImportError(name)
        return realimport(name, *importargs, **importkwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ModuleNotFoundError, match=r"needs pyvista.*artistools\[extras\]"):
        at.import_optional("pyvista")

    # an installed module comes back as the import statement gives it
    assert at.import_optional("math").sqrt(4.0) == 2.0


def test_print_warning_reaches_stderr_and_survives_quiet(capsys: pytest.CaptureFixture[str]) -> None:
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
        assert tqdm.tqdm.get_lock() is not None, "the bar must hold the lock that get_progress_class gave"
    finally:
        if original is not None:
            mp.set_start_method(original, force=True)


def test_resolve_frameset_paths(tmp_path: Path) -> None:
    """The path arithmetic of a set of frames must give one answer for every command.

    Three commands wrote this by hand, and each one broke the rule of -o in its own way.
    """
    framename = "plot_{timestep:03d}.png"

    # a -o path with no file extension names a folder, which holds the frames and the product
    frametemplate, productpath = at.resolve_frameset_paths(
        tmp_path / "frames", framecount=3, framename=framename, productname="movie.gif"
    )
    assert frametemplate == tmp_path / "frames" / framename
    assert productpath == tmp_path / "frames" / "movie.gif"
    assert (tmp_path / "frames").is_dir(), "the folder of the frames must exist"

    # a -o path that has a file extension names the product, thus the frames go beside it
    frametemplate, productpath = at.resolve_frameset_paths(
        tmp_path / "out" / "movie.gif", framecount=3, framename=framename, productname="movie.gif"
    )
    assert productpath == tmp_path / "out" / "movie.gif"
    assert frametemplate == tmp_path / "out" / framename

    # the folder of the product can carry a suffix of its own
    frametemplate, productpath = at.resolve_frameset_paths(
        tmp_path / "results.v1" / "movie.gif", framecount=3, framename=framename, productname="movie.gif"
    )
    assert productpath == tmp_path / "results.v1" / "movie.gif"
    assert (tmp_path / "results.v1").is_dir()

    # a merge names its own product, thus a folder gives no name to it
    frametemplate, productpath = at.resolve_frameset_paths(
        tmp_path / "m", framecount=2, framename=framename, combines=True
    )
    assert productpath is None
    assert frametemplate == tmp_path / "m" / framename

    # a -o path that has a file extension names the merged product, and the frames go beside it
    frametemplate, productpath = at.resolve_frameset_paths(
        tmp_path / "merged.pdf", framecount=2, framename=framename, combines=True
    )
    assert productpath == tmp_path / "merged.pdf"
    assert frametemplate == tmp_path / framename

    # a name that holds no field cannot take more than one frame
    with pytest.raises(ValueError, match="names one file, and this command writes 3 frames"):
        at.resolve_frameset_paths(tmp_path / "one.png", framecount=3, framename=framename)

    # one frame alone may take such a name
    frametemplate, _ = at.resolve_frameset_paths(tmp_path / "one.png", framecount=1, framename=framename)
    assert frametemplate == tmp_path / "one.png"


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
