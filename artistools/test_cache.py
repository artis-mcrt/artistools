import shutil
import typing as t
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

import artistools as at


def test_cache_delete_keeps_the_lock_files_and_other_parquet_files(tmp_path: Path) -> None:
    """Delete the caches and the leftover files of a stopped conversion, and keep each file that is not a cache.

    A deleted lock file can give the lock to two writers. A parquet file of a different name belongs to the user, also
    when it ends in .out.parquet.
    """
    from artistools.cache import main as cachemain

    deleted = [
        "model.txt.parquet",
        "linelist.out.parquet.tmp",
        "packets/packetsbatch00_0000_0003.out.parquet",
        "12411848.slurm/estimbatch00_0000_0003.out.parquet",
        "packets/.packetsbatch01_0004_0007.out.parquet.partialab12.partial",
        "12411848.slurm/.estimators_allranks.out.parquet.parts9x/ts00001_part00000.parquet",
    ]
    kept = [
        "packets/.packetsbatch00_0000_0003.out.parquet.replace-lock",
        "spectrum_data.parquet",
        "science.out.parquet",
        "analysis/.analysis.parts.backup/notes.txt",
        "model.txt",
    ]
    for name in [*deleted, *kept]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    lockfiles = [
        ".model.txt.parquet.conversion-lock",
        ".linelist.out.parquet.conversion-lock",
        "packets/.packetsbatch00_0000_0003.out.parquet.conversion-lock",
        "packets/.packetsbatch01_0004_0007.out.parquet.conversion-lock",
        "12411848.slurm/.estimbatch00_0000_0003.out.parquet.conversion-lock",
        "12411848.slurm/.estimators_allranks.out.parquet.conversion-lock",
    ]
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == sorted([
        *kept,
        *lockfiles,
    ])


def test_cache_make_gives_no_data_for_virtual_packets_without_their_files(tmp_path: Path) -> None:
    """Give no data, and not an error, for a model that has vpkt.txt but no virtual packet files.

    sn3d writes the virtual packet files only with VPKT_WRITE_CONTRIBS, thus vspecpolmodel has vpkt.txt and none of
    these files. vpktcontrib has the files, thus it gets its cache.
    """
    from artistools.cache import make_caches

    vspecpolmodel = tmp_path / "vspecpolmodel"
    shutil.copytree(at.get_path("testdata") / "vspecpolmodel", vspecpolmodel)
    assert (vspecpolmodel / "vpkt.txt").is_file()
    dfresults = make_caches(vspecpolmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets").row(0) == (
        "virtual packets",
        "no data",
        "the model holds no virtual packet files",
    )
    assert "error" not in dfresults["status"].to_list()

    # the copy leaves out the caches of earlier runs of the tests
    vpktmodel = tmp_path / "vpktcontrib"
    shutil.copytree(at.get_path("testdata") / "vpktcontrib", vpktmodel, ignore=shutil.ignore_patterns("vpackets"))
    dfresults = make_caches(vpktmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets").row(0) == (
        "virtual packets",
        "ok",
        "1 caches of the packets of 2 ranks",
    )


def test_cache_make_gives_an_error_for_incomplete_or_remote_packet_files(tmp_path: Path) -> None:
    """Give an error, and not "no data", when the file of the first rank is missing or the model is remote.

    The other files of the model show that it has data. A remote model needs a run of the command on its host.
    """
    from artistools.cache import make_caches
    from artistools.cache import make_packet_caches

    vpktmodel = tmp_path / "vpktcontrib"
    shutil.copytree(at.get_path("testdata") / "vpktcontrib", vpktmodel, ignore=shutil.ignore_patterns("vpackets"))
    (vpktmodel / "vpackets_0000.out.zst").unlink()
    dfresults = make_caches(vpktmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets")["status"].item() == "error"

    with pytest.raises(ValueError, match="on a different host"):
        make_packet_caches(Path("modelhost:/models/kilonova"), virtual=True)


def test_cache_delete_keeps_an_active_parts_folder(tmp_path: Path) -> None:
    pytest.importorskip("fcntl")
    from artistools.cache import delete_caches
    from artistools.misc import temporary_parts_folder

    cachepath = tmp_path / "estimators_allranks.out.parquet"
    with temporary_parts_folder(cachepath) as partsfolder:
        partfile = partsfolder / "part.parquet"
        partfile.write_bytes(b"part")
        cachepath.write_bytes(b"old cache")
        delete_caches(tmp_path, force=True)
        assert partfile.read_bytes() == b"part"
        assert cachepath.read_bytes() == b"old cache"
    delete_caches(tmp_path, force=True)
    assert not cachepath.exists()


def test_cache_delete_keeps_an_active_partial_file(tmp_path: Path) -> None:
    pytest.importorskip("fcntl")
    from artistools.cache import delete_caches
    from artistools.misc import write_parquet_atomic

    cachepath = tmp_path / "model.txt.parquet"
    sink_parquet = pl.LazyFrame.sink_parquet

    def sink_then_delete(frame: pl.LazyFrame, path: Path, **kwargs: t.Any) -> None:
        sink_parquet(frame, path, **kwargs)
        delete_caches(tmp_path, force=True)
        assert path.is_file()

    with mock.patch.object(pl.LazyFrame, "sink_parquet", side_effect=sink_then_delete, autospec=True):
        write_parquet_atomic(pl.DataFrame({"value": [1]}), cachepath)
    assert pl.read_parquet(cachepath)["value"].to_list() == [1]
