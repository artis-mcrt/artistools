import shutil
from pathlib import Path

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

    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == sorted(kept)


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
