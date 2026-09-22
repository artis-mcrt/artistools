import typing as t


def pytest_configure(config: t.Any) -> None:
    """Clear the test output of previous runs."""
    import shutil
    from pathlib import Path

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
                    f"Refusing to delete {entry.resolve()} as it is not a descendant of the repository {repopath.resolve()}"
                )
            # dotfiles are left alone, except the temp files write_parquet_atomic names with a leading dot,
            # which only survive if a run was killed before its cleanup ran
            elif not entry.stem.startswith(".") or entry.name.endswith(".partial"):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)

    outputpath.mkdir(exist_ok=True)
