def pytest_sessionstart(session) -> None:
    """Clear the test output of previous runs."""
    import shutil

    import artistools as at

    outputpath = at.get_config()["path_testoutput"]
    if outputpath.exists():
        shutil.rmtree(outputpath)
        outputpath.mkdir(exist_ok=True)
