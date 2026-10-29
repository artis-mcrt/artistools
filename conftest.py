import typing as t
from pathlib import Path

import pytest

if t.TYPE_CHECKING:
    import io
    from collections.abc import Callable
    from collections.abc import Sequence

    import matplotlib.figure as mplfig
    import numpy as np
    import polars as pl


def pytest_configure(config: t.Any) -> None:
    """Clear the test output of previous runs."""
    import shutil

    from artistools.paths import get_path

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

    A read of a trajectory writes its parquet cache beside the archive. A copy keeps the caches out of tests/data,
    and each test session then also tests the read of the archive.
    """
    import shutil

    from artistools.paths import get_path

    testdatapath = get_path("testdata")
    assert isinstance(testdatapath, Path)
    trajectorypath = tmp_path / "trajectories"
    # CodSpeed runs a benchmark test more than one time in one process, with the same tmp_path
    trajectorypath.mkdir(exist_ok=True)
    for filepath in (testdatapath / "kilonova" / "trajectories").iterdir():
        if filepath.is_file() and filepath.name != ".gitignore":
            shutil.copy(filepath, trajectorypath)

    return trajectorypath


@pytest.fixture
def get_plot_xy() -> "Callable[[t.Any], tuple[np.ndarray, np.ndarray]]":
    """Return a function that gives the x values and the y values of a call of Axes.plot or Axes.step."""
    import numpy as np

    def get_plot_xy(callargs: t.Any) -> "tuple[np.ndarray, np.ndarray]":
        return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)

    return get_plot_xy


@pytest.fixture
def read_data_file() -> "Callable[[Path | io.StringIO], pl.DataFrame]":
    """Return a function that reads a table that --write-data wrote or that --print-data printed."""
    import polars as pl

    def read_data_file(source: "Path | io.StringIO") -> pl.DataFrame:
        return pl.read_csv(source, separator=" ", null_values="nan", comment_prefix="#")

    return read_data_file


@pytest.fixture
def copy_model_inputs() -> "Callable[[Path, Path, Sequence[str]], None]":
    """Return a function that copies the input files of a model to a folder, and each extra file, e.g. spec.out."""
    import shutil

    def copy_model_inputs(source: Path, folder: Path, extranames: "Sequence[str]") -> None:
        for name in ("model.txt", "abundances.txt", "input.txt", "compositiondata.txt", *extranames):
            shutil.copy(source / name, folder / name)

    return copy_model_inputs


@pytest.fixture
def get_cell_texts() -> "Callable[[str], list[str]]":
    """Return a function that splits the text of an estimator file into the text of each cell.

    The line of a timestep starts the text of a cell.
    """

    def get_cell_texts(estimatortext: str) -> list[str]:
        celltexts: list[str] = []
        for line in estimatortext.splitlines(keepends=True):
            if line.startswith("timestep ") or not celltexts:
                celltexts.append(line)
            else:
                celltexts[-1] += line
        return celltexts

    return get_cell_texts


@pytest.fixture
def write_zstd_frames() -> "Callable[[Path, Sequence[str]], None]":
    """Return a function that writes each text as one zstd frame.

    ARTIS writes the estimator file of all ranks in this form, one frame after the other.
    """
    from artistools.misc.fileio import get_decompress_open

    def write_zstd_frames(filepath: Path, texts: "Sequence[str]") -> None:
        frames: list[bytes] = []
        for index, text in enumerate(texts):
            framepath = filepath.with_name(f"{filepath.name}.frame{index}")
            with get_decompress_open(".zst")(framepath, "wt", encoding="utf-8") as framefile:
                framefile.write(text)
            frames.append(framepath.read_bytes())
            framepath.unlink()
        filepath.write_bytes(b"".join(frames))

    return write_zstd_frames


@pytest.fixture
def write_trajectory_archive() -> "Callable[[Path, int, dict[str, str], str], Path]":
    """Return a function that writes a small trajectory archive, e.g. {particleid}.tar.xz.

    The members of the archive are the files of its Run_rprocess folder. The suffix gives the format of the archive.
    """
    import io
    import tarfile

    from artistools.misc.fileio import get_decompress_open

    def write_trajectory_archive(traj_root: Path, particleid: int, members: dict[str, str], suffix: str) -> Path:
        traj_root.mkdir(parents=True, exist_ok=True)
        archivepath = traj_root / f"{particleid}{suffix}"
        tarbytes = io.BytesIO()
        with tarfile.open(fileobj=tarbytes, mode="w") as tarfilehandle:
            for filename, text in members.items():
                data = text.encode("utf-8")
                tarinfo = tarfile.TarInfo(name=f"./Run_rprocess/{filename}")
                tarinfo.size = len(data)
                tarfilehandle.addfile(tarinfo, io.BytesIO(data))

        if suffix == ".tar":
            archivepath.write_bytes(tarbytes.getvalue())
        else:
            with get_decompress_open(archivepath.suffix)(archivepath, "wb") as archivefile:
                archivefile.write(tarbytes.getvalue())

        return archivepath

    return write_trajectory_archive


@pytest.fixture
def get_plotestimators_figure() -> "Callable[[Sequence[str], Path], mplfig.Figure]":
    """Return a function that runs plotestimators for the tokens and gives the figure that it saves."""
    from unittest import mock

    import artistools as at
    from artistools.estimators import plotestimators

    def get_plotestimators_figure(tokens: "Sequence[str]", outputfile: Path) -> "mplfig.Figure":
        savedfigures: list[mplfig.Figure] = []

        def keep_figure(fig: "mplfig.Figure", *_: t.Any, **__: t.Any) -> None:
            savedfigures.append(fig)

        with mock.patch.object(plotestimators, "save_figure", side_effect=keep_figure):
            at.estimators.plot(argsraw=[*tokens, "-o", str(outputfile)])
        [savedfigure] = savedfigures
        return savedfigure

    return get_plotestimators_figure


@pytest.fixture
def make_viewer() -> "Callable[[str, list[str]], t.Any]":
    """Return a function that gives the viewer of a command, e.g. of "spectra", with a canvas and no window.

    The viewer has drawn its first plot.
    """
    import importlib

    import matplotlib.figure as mplfig
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from artistools.viewertools.window import draw_values

    def make_viewer(kind: str, tokens: list[str]) -> t.Any:
        module = importlib.import_module(f"artistools.{kind}.interactive")
        viewerclass = next(
            value for name, value in vars(module).items() if name.endswith("Viewer") and isinstance(value, type)
        )
        fig = mplfig.Figure()
        FigureCanvasAgg(fig)
        viewer = viewerclass(tokens, fig)
        assert draw_values(viewer) is None
        return viewer

    return make_viewer


@pytest.fixture
def toy_codecomparison_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a code comparison model of one epoch and three shells, and return its virtual path."""
    import artistools as at
    import artistools.inputmodel.core

    physdir = tmp_path / "ccdata" / "toymodel"
    physdir.mkdir(parents=True)
    (physdir / "phys_toymodel_toycode.txt").write_text(
        "#NTIMES: 1\n"
        "#TIMES[d]: 10.0\n"
        "#TIME: 10.0\n"
        "#NVEL: 3\n"
        "#vel_mid Te rho nne nntot\n"
        "1000.0 5000.0 1e-13 1e6 1e6\n"
        "2000.0 5100.0 2e-13 2e6 2e6\n"
        "3000.0 5200.0 3e-13 3e6 3e6\n"
    )

    artismodeldir = tmp_path / "ccmodel" / "toymodel"
    artismodeldir.mkdir(parents=True)
    (artismodeldir / "model.txt").write_text(
        "3\n10.0\n1 1000.0 -13.0 0.5 0.3 0.1 0.0 0.0\n2 2000.0 -13.0 0.5 0.3 0.1 0.0 0.0\n"
        "3 3000.0 -13.0 0.5 0.3 0.1 0.0 0.0\n"
    )
    abundrow = " ".join(["0.0"] * 25 + ["0.1", "0.0", "0.9", "0.0", "0.0"])
    (artismodeldir / "abundances.txt").write_text("".join(f"{cellid} {abundrow}\n" for cellid in (1, 2, 3)))

    realgetpath = at.get_path

    def fake_get_path(key: str) -> Path:
        if key == "codecomparisondata1path":
            return tmp_path / "ccdata"
        if key == "codecomparisonmodelartismodelpath":
            return tmp_path / "ccmodel"
        return realgetpath(key)

    # codecomparison.py and inputmodel/core.py each hold their own import of the name
    monkeypatch.setattr(at.codecomparison, "get_path", fake_get_path)
    monkeypatch.setattr(artistools.inputmodel.core, "get_path", fake_get_path)

    return "codecomparison/toymodel/toycode"


@pytest.fixture
def patch_scan_packets(monkeypatch: pytest.MonkeyPatch) -> "Callable[[pl.DataFrame], None]":
    """Return a function that makes the spectra code read the given packets in place of the packets of a model."""
    from artistools.spectra import core as atspectra

    def patch_scan_packets(dfpackets: "pl.DataFrame") -> None:
        def scan_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, "pl.LazyFrame"]:
            return 1, dfpackets.lazy()

        monkeypatch.setattr(atspectra, "scan_packets", scan_packets)

    return patch_scan_packets
