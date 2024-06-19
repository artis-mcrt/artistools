def pytest_sessionstart(session) -> None:
    """Clear the test output of previous runs."""
    import shutil
    from pathlib import Path

    import artistools as at

    outputpath = at.get_config("path_testoutput")
    assert isinstance(outputpath, Path)
    repopath = at.get_config("path_artistools_repository")
    assert isinstance(repopath, Path)
    if outputpath.exists():
        is_descendant = repopath.resolve() in outputpath.resolve().parents
        assert (
            is_descendant
        ), f"Refusing to delete {outputpath.resolve()} as it is not a descendant of the repository {repopath.resolve()}"
        shutil.rmtree(outputpath)
    outputpath.mkdir(exist_ok=True)
