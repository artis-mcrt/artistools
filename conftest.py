def pytest_configure(config) -> None:  # noqa: ARG001,ANN001
    """Clear the test output of previous runs."""
    import sys
    from pathlib import Path

    from artistools.configuration import get_config

    outputpath = get_config("path_testoutput")
    assert isinstance(outputpath, Path)
    repopath = get_config("path_artistools_repository")
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
