"""Unit tests for the shared helpers in artistools.misc.

These tests use only synthetic data written under tmp_path, so they run quickly and do not require
the downloaded ARTIS test model.
"""

import argparse
import gzip
import io
import lzma
from pathlib import Path

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
