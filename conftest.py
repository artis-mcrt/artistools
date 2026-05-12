import typing as t


def pytest_configure(config: t.Any) -> None:  # noqa: ARG001
    """Clear the test output of previous runs."""
    import sys
    from pathlib import Path

    from artistools.commands import get_path

    outputpath = get_path("testoutput")
    assert isinstance(outputpath, Path)
    repopath = get_path("artistools_repository")
    assert isinstance(repopath, Path)

    if outputpath.exists():
        for file in outputpath.glob("*.*"):
            if repopath.resolve() not in file.resolve().parents:
                print(
                    f"Refusing to delete {file.resolve()} as it is not a descendant of the repository {repopath.resolve()}"
                )
            elif not file.stem.startswith("."):
                file.unlink(missing_ok=True)

    outputpath.mkdir(exist_ok=True)

    # remove the artistools module from sys.modules so that typeguard can be run
    sys.modules.pop("artistools")
