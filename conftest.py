import typing as t
from pathlib import Path

import pytest


def pytest_configure(config: t.Any) -> None:
    """Clear the test output of previous runs."""
    import shutil

    from artistools.commands import get_path

    # pytest-xdist runs this function in the controller and in each worker. A worker that starts
    # after another worker starts its tests would delete the output of those tests. Thus the
    # controller alone clears the folder
    if hasattr(config, "workerinput"):
        return

    outputpath = get_path("testoutput")
    assert isinstance(outputpath, Path)
    repopath = get_path("artistools_repository")
    assert isinstance(repopath, Path)

    if outputpath.exists():
        # a command that writes a folder of frames keeps that folder, thus this loop reads
        # every entry and not the files alone
        for entry in outputpath.iterdir():
            if repopath.resolve() not in entry.resolve().parents:
                print(
                    f"Refusing to delete {entry.resolve()} because it is not a descendant "
                    f"of the repository {repopath.resolve()}"
                )
            # dotfiles are left alone, except the temp files write_parquet_atomic names with a leading dot,
            # which only survive if a run was killed before its cleanup ran
            elif not entry.stem.startswith(".") or entry.name.endswith(".partial"):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)

    outputpath.mkdir(exist_ok=True)


@pytest.fixture
def trajectory_copy(tmp_path: Path) -> Path:
    """Return a copy of the folder of the test trajectories.

    A read of a trajectory extracts the members of its archive beside the archive. A copy keeps the extracted files
    out of tests/data, and each run then tests the extraction too.
    """
    import shutil

    from artistools.commands import get_path

    testdatapath = get_path("testdata")
    assert isinstance(testdatapath, Path)
    trajectorypath = tmp_path / "trajectories"
    # CodSpeed runs a benchmark test more than one time in one process, with the same tmp_path
    trajectorypath.mkdir(exist_ok=True)
    for filepath in (testdatapath / "kilonova" / "trajectories").iterdir():
        if filepath.is_file() and filepath.name != ".gitignore":
            shutil.copy(filepath, trajectorypath)

    return trajectorypath
