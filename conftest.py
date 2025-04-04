def pytest_configure(config) -> None:  # noqa: ARG001
    """Clear the test output of previous runs."""
    import shutil
    import sys
    from pathlib import Path

    from artistools.configuration import get_config

    outputpath = get_config("path_testoutput")
    assert isinstance(outputpath, Path)
    repopath = get_config("path_artistools_repository")
    assert isinstance(repopath, Path)
    if outputpath.exists():
        is_descendant = repopath.resolve() in outputpath.resolve().parents
        assert is_descendant, (
            f"Refusing to delete {outputpath.resolve()} as it is not a descendant of the repository {repopath.resolve()}"
        )
        shutil.rmtree(outputpath, ignore_errors=True)
    outputpath.mkdir(exist_ok=True)

    # remove the artistools module from sys.modules so that typeguard can be run
    sys.modules.pop("artistools")
