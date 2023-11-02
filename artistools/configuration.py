import multiprocessing
import typing as t
from pathlib import Path

config: dict[str, t.Any] = {}


def setup_config() -> None:
    config["num_processes"] = multiprocessing.cpu_count()
    # config["num_processes"] = 1
    # print(f"Using {config['num_processes']} processes")

    config["figwidth"] = 5
    config["codecomparisondata1path"] = Path(
        "/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/GitHub/sn-rad-trans/data1"
    )

    config["codecomparisonmodelartismodelpath"] = Path(Path.home() / "Google Drive/My Drive/artis_runs/weizmann/")

    config["path_artistools_repository"] = Path(__file__).absolute().parent.parent
    config["path_artistools_dir"] = Path(__file__).absolute().parent  # the package path
    config["path_datadir"] = Path(__file__).absolute().parent / "data"
    config["path_testartismodel"] = Path(config["path_artistools_repository"], "tests", "data", "testmodel")
    config["path_testdata"] = Path(config["path_artistools_repository"], "tests", "data")
    config["path_testoutput"] = Path(config["path_artistools_repository"], "tests", "output")


def get_config(key: str | None = None) -> dict[str, t.Any] | t.Any:
    if not config:
        setup_config()
    return config if key is None else config[key]


def set_config(key: str, value: t.Any) -> None:
    if not config:
        setup_config()
    config[key] = value
