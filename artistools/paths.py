"""Return the paths that artistools knows by name, e.g. the package folder and the folder of the test data."""

from pathlib import Path


def get_path(key: str) -> Path:
    """Return a well-known path by name, such as the package folder or the code comparison data folder."""
    match key:
        case "codecomparisondata1path":
            return Path(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/GitHub/sn-rad-trans/data1")
        case "codecomparisonmodelartismodelpath":
            return Path(Path.home() / "Google Drive/My Drive/artis_runs/weizmann/")
        case "artistools_repository":
            return Path(__file__).absolute().parent.parent
        case "artistools_dir":
            return Path(__file__).absolute().parent  # the package path
        case "datadir":
            return Path(__file__).absolute().parent / "data"
        case "testartismodel":
            return Path(get_path("artistools_repository"), "tests", "data", "testmodel")
        case "testdata":
            return Path(get_path("artistools_repository"), "tests", "data")
        case "testoutput":
            return Path(get_path("artistools_repository"), "tests", "output")
        case _:
            msg = f"Unknown path key: {key}"
            raise KeyError(msg)
