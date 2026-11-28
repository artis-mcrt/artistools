import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

import artistools as at


def test_cache_delete_keeps_the_lock_files_and_other_parquet_files(tmp_path: Path) -> None:
    """Delete the caches and the leftover files of a stopped conversion, and keep each file that is not a cache.

    A deleted lock file can give the lock to two writers. A parquet file of a different name belongs to the user, also
    when it ends in .out.parquet. The reader of abundances.txt puts its cache beside the file that sn3d reads, which
    can be in artis/data/, and the command did not find that cache.
    """
    from artistools.cache import main as cachemain

    deleted = [
        "model.txt.parquet",
        "artis/data/abundances.txt.parquet",
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


def test_cache_delete_keeps_the_only_copy_of_the_packets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Keep a packets cache whose text files are absent, because the cache is then the only copy of the packets.

    The reader takes such a cache, e.g. in an archived model folder. The command deleted it with the other caches, and
    it said that the next read converts the text files again.
    """
    from unittest import mock

    from artistools.cache import main as cachemain

    sourcemodel = at.get_path("testdata") / "test-classicmode_3d"
    shutil.copy(sourcemodel / "input.txt", tmp_path)
    (tmp_path / "packets").mkdir()
    for rank in (0, 1):
        shutil.copy(sourcemodel / "packets" / f"packets00_{rank:04d}.out.zst", tmp_path / "packets")

    with mock.patch("artistools.packets.core.RANKS_PER_BATCH", 1):
        _, caches = at.packets.get_packets_batch_parquet_paths(tmp_path)
        assert len(caches) == 2
        packetcount = at.packets.scan_packets(tmp_path)[1].select(pl.len()).collect().item()
        (tmp_path / "packets" / "packets00_0001.out.zst").unlink()

        cachemain(argsraw=["delete", "-f", str(tmp_path)])
        assert not caches[0].exists()
        assert caches[1].is_file()
        assert f"{caches[1]} is the only copy of its data, because the text files of rank 1" in capsys.readouterr().err
        assert at.packets.scan_packets(tmp_path)[1].select(pl.len()).collect().item() == packetcount

        cachemain(argsraw=["delete", "-f", "--include-only-copies", str(tmp_path)])
        assert not any((tmp_path / "packets").glob("packetsbatch*"))


def test_cache_delete_keeps_a_packets_cache_of_a_run_whose_text_files_are_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], write_packets_runfolder: Callable[[Path, str, float], None]
) -> None:
    """Keep a packets cache with absent text files when a different run folder holds files of the same ranks.

    The check took the folder that the reader reads now. The files of the later run there are not the source of the
    cache, but the command deleted the cache as a copy of them. The packets of each run here have a different energy.
    """
    from artistools.cache import main as cachemain

    write_packets_runfolder(tmp_path, "100.slurm", 1e40)
    _, (cachepath,) = at.packets.get_packets_batch_parquet_paths(tmp_path)
    shutil.rmtree(tmp_path / "100.slurm")
    write_packets_runfolder(tmp_path, "200.slurm", 2e40)

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    assert pl.read_parquet(cachepath)["e_rf"].unique().to_list() == pytest.approx([1e40], rel=1e-6)
    assert "the text files of ranks 0-1 are absent from 100.slurm" in capsys.readouterr().err


def copy_model_with_estimator_caches(modelfolder: Path) -> list[Path]:
    """Copy a model with two run folders, convert its estimators, and return the cache of each run folder."""
    sourcemodel = at.get_path("testdata") / "test-classicmode_3d"
    for filename in ("input.txt", "model.txt", "modelgridrankassignments.out"):
        shutil.copy(sourcemodel / filename, modelfolder)
    # the copy leaves out the caches of earlier runs of the tests
    for runfolder in ("job0", "job1"):
        shutil.copytree(
            sourcemodel / runfolder, modelfolder / runfolder, ignore=shutil.ignore_patterns(".*", "*.parquet*")
        )
    at.estimators.scan_estimators(modelfolder).collect()
    estimatorcaches = [modelfolder / runfolder / "estimators_allranks.out.parquet" for runfolder in ("job0", "job1")]
    assert all(cachepath.is_file() for cachepath in estimatorcaches)
    return estimatorcaches


def test_cache_delete_keeps_the_only_copy_of_the_estimators_and_the_nlte_populations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep a cache of a run folder whose text files are absent. The readers then take the cache as the only copy."""
    from artistools.cache import main as cachemain

    estimatorcaches = copy_model_with_estimator_caches(tmp_path)
    (tmp_path / "job0" / "estimators_0001.out.zst").unlink()
    nltecache = tmp_path / "job1" / "nltepops_allranks.out.parquet"
    pl.DataFrame({"timestep": [0]}).write_parquet(nltecache)

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    assert estimatorcaches[0].is_file()
    assert not estimatorcaches[1].exists()
    assert nltecache.is_file()
    stderr = capsys.readouterr().err
    assert "the text files of rank 1 are absent from job0" in stderr
    assert "the text files of ranks 0-1 are absent from job1" in stderr
    assert at.estimators.scan_estimators(tmp_path).select(pl.len()).collect().item() > 0


def test_cache_delete_takes_the_ranks_of_an_estimator_cache_from_its_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep an estimator cache when the text file of one of its ranks is absent.

    A run can have more ranks than input.txt gives now. The command took the ranks of the model for each run folder,
    thus it deleted the only copy of the estimators of a rank that only the earlier run had.
    """
    from artistools.cache import main as cachemain

    estimatorcaches = copy_model_with_estimator_caches(tmp_path)
    metadata = pl.read_parquet_metadata(estimatorcaches[0])
    assert metadata["rank_max"] == "1"
    dfestimators = pl.read_parquet(estimatorcaches[0])
    # write_parquet_atomic keeps a file that exists
    estimatorcaches[0].unlink()
    at.misc.write_parquet_atomic(
        dfestimators,
        estimatorcaches[0],
        metadata={key: value for key, value in metadata.items() if not key.startswith("ARROW")} | {"rank_max": "2"},
    )

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    assert estimatorcaches[0].is_file()
    assert not estimatorcaches[1].exists()
    assert "the text files of rank 2 are absent from job0" in capsys.readouterr().err


def test_cache_delete_keeps_a_cache_with_older_text_or_with_a_link_to_an_absent_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep a cache that its text files cannot make again.

    The readers keep a cache when the text is older than the text of the cache, because older text holds fewer
    timesteps. A link to an absent file is not a text file. The command counted the names of the files, thus it
    deleted the cache in each of these cases.
    """
    import os

    from artistools.cache import main as cachemain
    from artistools.misc.modelinfo import RANK_CACHE_VERSION

    estimatorcaches = copy_model_with_estimator_caches(tmp_path)
    nltecaches = [tmp_path / runfolder / "nltepops_allranks.out.parquet" for runfolder in ("job0", "job1")]
    olderfiles = list((tmp_path / "job0").glob("estimators_????.out*"))
    for rank in (0, 1):
        olderfiles.append(tmp_path / "job0" / f"nlte_{rank:04d}.out")
        olderfiles[-1].touch()
        (tmp_path / "job1" / f"nlte_{rank:04d}.out").symlink_to(tmp_path / "absent")
    (tmp_path / "job1" / "estimators_0001.out.zst").unlink()
    (tmp_path / "job1" / "estimators_0001.out.zst").symlink_to(tmp_path / "absent")
    for nltecache in nltecaches:
        at.misc.write_parquet_atomic(
            pl.DataFrame({"timestep": [0]}),
            nltecache,
            metadata={
                "cacheversion": str(RANK_CACHE_VERSION),
                "textsource_mtime": str(nltecache.parent.stat().st_mtime),
            },
        )
    for olderfile in olderfiles:
        oldertime = olderfile.stat().st_mtime - 3600
        os.utime(olderfile, (oldertime, oldertime))

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    assert all(cachepath.is_file() for cachepath in (*estimatorcaches, *nltecaches))
    stderr = capsys.readouterr().err
    assert stderr.count("its text files are older than the text of the cache") == 2
    assert "the text files of rank 1 are absent from job1" in stderr
    assert "the text files of ranks 0-1 are absent from job1" in stderr


def test_cache_make_reads_the_input_files_that_the_readers_take(tmp_path: Path) -> None:
    """The readers take a model.txt in data/ and an abundances.txt that sn3d reads, e.g. in artis/data/.

    The cache command searched the model folder alone, thus it made no cache for such a model.
    """
    from unittest import mock

    from artistools.cache import make_inputmodel_caches

    (tmp_path / "input.txt").write_text("", encoding="utf-8")
    for filename, folder in (("model.txt", "data"), ("abundances.txt", "artis/data")):
        (tmp_path / folder).mkdir(parents=True)
        shutil.copy2(at.get_path("testdata") / "test-classicmode_1d" / filename, tmp_path / folder / filename)
    with mock.patch("artistools.cache.scan_initelemabundances") as mockscan:
        assert make_inputmodel_caches(tmp_path) is not None
    mockscan.assert_called_once()


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
