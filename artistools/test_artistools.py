import contextlib
import hashlib
import importlib
import inspect
import math
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"
outputpath = at.get_path("testoutput")
outputpath.mkdir(exist_ok=True, parents=True)

REPOPATH = at.get_path("artistools_repository")


def funcname() -> str:
    """Get the name of the calling function."""
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr] # pyright: ignore[reportOptionalMemberAccess]
    except AttributeError as e:
        msg = "Could not get the name of the calling function."
        raise RuntimeError(msg) from e


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


def test_commands() -> None:
    commands: dict[str, tuple[str, str]] = {}

    # just skip the test if tomllib is not available (python < 3.11)
    with contextlib.suppress(ImportError):
        import tomllib

        assert isinstance(REPOPATH, Path)
        with (REPOPATH / "pyproject.toml").open("rb") as f:
            pyproj = tomllib.load(f)
        commands = {k: tuple(v.split(":")) for k, v in pyproj["project"]["scripts"].items()}

        # ensure that the commands are pointing to valid submodule.function() targets
        for command, (submodulename, funcname) in commands.items():
            submodule = importlib.import_module(submodulename)
            assert hasattr(submodule, funcname) or (
                funcname == "main" and hasattr(importlib.import_module(f"{submodulename}.__main__"), funcname)
            ), f"{submodulename}.{funcname} not found for command {command}"

    def recursive_check(dictcmd: at.commands.CommandType) -> None:
        for cmdtarget in dictcmd.values():
            if isinstance(cmdtarget, dict):
                recursive_check(cmdtarget)
            else:
                submodulename, funcname = cmdtarget
                namestr = f"artistools.{submodulename.removeprefix('artistools.')}" if submodulename else "artistools"
                print(namestr)
                submodule = importlib.import_module(namestr, package="artistools")
                assert hasattr(submodule, funcname) or (
                    funcname == "main" and hasattr(importlib.import_module(f"{namestr}.__main__"), funcname)
                )

    recursive_check(at.commands.subcommandtree)


def test_timestep_times() -> None:
    timestartarray = at.get_timestep_times(modelpath, loc="start")
    timedeltarray = at.get_timestep_times(modelpath, loc="delta")
    timemidarray = at.get_timestep_times(modelpath, loc="mid")
    assert len(timestartarray) == 100
    assert math.isclose(timemidarray[0], 250.421, abs_tol=1e-3)
    assert math.isclose(timemidarray[-1], 349.412, abs_tol=1e-3)

    assert all(
        tstart < tmid < (tstart + tdelta)
        for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray, strict=False)
    )


def test_get_inputparams() -> None:
    inputparams = at.get_inputparams(modelpath)
    dicthash = hashlib.sha256(str(sorted(inputparams.items())).encode("utf-8")).hexdigest()
    assert dicthash == "1edcddd5d36cc2eaed94ad083dacfb95c6915b8fd4f62591e2b79ceca6885d1e", dicthash


def test_macroatom() -> None:
    at.macroatom.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=10)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(mockstep: t.Any, mockplot: t.Any) -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.radfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=funcoutpath, showbinedges=True)

    plot_calls = {
        label.strip(): call for call in mockplot.call_args_list if isinstance((label := call.kwargs.get("label")), str)
    }
    dilute_xarr, dilute_yarr = get_plot_xy(plot_calls["Dilute blackbody model"])
    assert np.isclose(dilute_xarr.min(), 1000.0, rtol=1e-4)
    assert np.isclose(dilute_xarr.max(), 20000.0, rtol=1e-4)
    assert np.isclose(dilute_yarr.mean(), 21.27744616064978, rtol=1e-4)
    assert np.isclose(dilute_yarr.std(), 26.77850448874471, rtol=1e-4)

    fitted_xarr, fitted_yarr = get_plot_xy(plot_calls["Radiation field model"])
    assert np.isclose(fitted_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(fitted_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(fitted_yarr.mean(), 48.342355990852596, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(fitted_yarr, fitted_xarr)), 489588.12007010705, rtol=1e-4)

    bandavg_xarr, bandavg_yarr = get_plot_xy(mockstep.call_args_list[0])
    assert np.isclose(bandavg_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(bandavg_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(bandavg_yarr.mean(), 43.58807185509511, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(bandavg_yarr, bandavg_xarr)), 475489.00963827176, rtol=1e-4)


@pytest.mark.benchmark
def test_plotspherical() -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=funcoutpath)


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemax=270, outputfile=outputpath)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_transitions(mockplot: t.Any) -> None:
    at.transitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)

    assert len(mockplot.call_args_list) == 7
    expected_integrals = [
        0.03762393022815368,
        266.8869480321175,
        299.25457622600254,
        8.318170397519948,
        34.5598725883166,
        0.0,
        0.0,
    ]
    expected_maxima = [
        7.054096268640787e-05,
        0.3309041740131583,
        0.9619558273061346,
        0.013829945098038332,
        0.060540167233566825,
        0.0,
        0.0,
    ]
    for callargs, expected_integral, expected_max in zip(
        mockplot.call_args_list, expected_integrals, expected_maxima, strict=True
    ):
        xarr, yarr = get_plot_xy(callargs)
        assert np.isclose(xarr[0], 3500.0, rtol=1e-4)
        assert np.isclose(xarr[-1], 7996.0, rtol=1e-4)
        assert np.isclose(np.trapezoid(yarr, xarr), expected_integral, rtol=1e-4, atol=1e-8)
        assert np.isclose(yarr.max(), expected_max, rtol=1e-4, atol=1e-8)


@pytest.mark.benchmark
def test_writecomparisondata() -> None:
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=outputpath, selected_timesteps=list(range(99))
    )


def test_get_z_a_nucname() -> None:
    assert at.get_z_a_nucname("Pb208") == (82, 208)
    assert at.get_z_a_nucname("X_Pb208") == (82, 208)
    assert at.get_z_a_nucname("nniso_Pb208") == (82, 208)
    assert at.get_z_a_nucname("Fe56") == (26, 56)
    assert at.get_z_a_nucname("Ni56") == (28, 56)
    assert at.get_z_a_nucname("Co56") == (27, 56)
    assert at.get_z_a_nucname("H1") == (1, 1)


def test_get_atomic_number_and_elsymbol() -> None:
    assert at.get_atomic_number("Fe") == 26
    assert at.get_atomic_number("Ni") == 28
    assert at.get_atomic_number("Co") == 27
    assert at.get_atomic_number("H") == 1
    assert at.get_atomic_number("He") == 2
    assert at.get_atomic_number("X_Fe") == 26
    assert at.get_atomic_number("UnknownXYZ") == -1

    assert at.get_elsymbol(26) == "Fe"
    assert at.get_elsymbol(28) == "Ni"
    assert at.get_elsymbol(1) == "H"
    assert at.get_elsymbol(2) == "He"


def test_decode_roman_numeral() -> None:
    assert at.decode_roman_numeral("I") == 1
    assert at.decode_roman_numeral("II") == 2
    assert at.decode_roman_numeral("III") == 3
    assert at.decode_roman_numeral("IV") == 4
    assert at.decode_roman_numeral("V") == 5
    assert at.decode_roman_numeral("X") == 10
    assert at.decode_roman_numeral("XX") == 20
    assert at.decode_roman_numeral("i") == 1  # case-insensitive
    assert at.decode_roman_numeral("INVALID") == -1


def test_get_ionstring() -> None:
    assert at.get_ionstring(26, 2) == "Fe II"
    assert at.get_ionstring(26, 1) == "Fe I"
    assert at.get_ionstring(28, 3) == "Ni III"
    assert at.get_ionstring(26, 2, sep="") == "FeII"
    assert at.get_ionstring(26, None) == "Fe"
    assert at.get_ionstring(26, "ALL") == "Fe"
    assert at.get_ionstring(26, 2, style="charge") == "Fe+"
    assert at.get_ionstring(26, 3, style="charge") == "Fe2+"
    assert at.get_ionstring(26, 1, style="charge") == "Fe0"


def test_get_ion_tuple() -> None:
    assert at.get_ion_tuple("nnelement_I") == 53
    assert at.get_ion_tuple("nnion_I_II") == (53, 2)
    assert at.get_ion_tuple("Fe_II") == (26, 2)
    assert at.get_ion_tuple("Fe II") == (26, 2)
    assert at.get_ion_tuple("Fe I") == (26, 1)
    assert at.get_ion_tuple("Ni III") == (28, 3)
    assert at.get_ion_tuple("Co II") == (27, 2)
    assert at.get_ion_tuple("Ni") == 28
    assert at.get_ion_tuple("26") == 26


def test_parse_range_list() -> None:
    assert at.parse_range_list("5") == [5]
    assert at.parse_range_list("3-5") == [3, 4, 5]
    assert at.parse_range_list("1,3-5,8") == [1, 3, 4, 5, 8]
    assert at.parse_range_list([3, 5, 7]) == [3, 5, 7]
    assert at.parse_range_list(42) == [42]
    assert at.parse_range_list("5-3") == [3, 4, 5]  # reversed range is sorted


def test_makelist() -> None:
    assert at.makelist(None) == []
    assert at.makelist("hello") == ["hello"]
    assert at.makelist(Path("my/folder/path")) == [Path("my/folder/path")]
    assert at.makelist([1, 2, 3]) == [1, 2, 3]
    assert at.makelist((1, 2)) == [1, 2]


def test_flatten_list() -> None:
    assert at.flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert at.flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
    assert at.flatten_list([]) == []
    assert at.flatten_list([1, 2, 3]) == [1, 2, 3]


def test_trim_or_pad() -> None:
    result = at.trim_or_pad(3, [1, 2, 3, 4], [10, 20])
    assert list(result[0]) == [1, 2, 3]
    assert list(result[1]) == [10, 20, None]

    result2 = at.trim_or_pad(2, "single_string")
    assert list(result2[0]) == ["single_string", None]


def test_vec_len() -> None:
    assert math.isclose(at.vec_len([3.0, 4.0, 0.0]), 5.0)
    assert math.isclose(at.vec_len([1.0, 0.0, 0.0]), 1.0)
    assert math.isclose(at.vec_len([0.0, 0.0, 0.0]), 0.0)
    assert math.isclose(at.vec_len([1.0, 1.0, 1.0]), math.sqrt(3.0))


def test_stripallsuffixes() -> None:
    assert at.stripallsuffixes(Path("packets00_0000.out.gz")) == Path("packets00_0000")
    assert at.stripallsuffixes(Path("model.txt.xz")) == Path("model")
    assert at.stripallsuffixes(Path("noextension")) == Path("noextension")
    assert at.stripallsuffixes(Path("single.txt")) == Path("single")


def test_match_closest_time() -> None:
    times = [100.0, 200.0, 300.0, 400.0]
    assert at.match_closest_time(250.0, times) == "200.0"
    assert at.match_closest_time(310.0, times) == "300.0"
    assert at.match_closest_time(99.0, times) == "100.0"
    assert at.match_closest_time(400.0, times) == "400.0"


def test_get_npts_model(tmp_path: Path) -> None:
    # The 3D test model has 10^3 = 1000 cells
    assert at.misc.get_npts_model(modelpath_3d) == 1000

    # Single-number format used by 1D models
    (tmp_path / "model.txt").write_text("20\n")
    assert at.misc.get_npts_model(tmp_path) == 20

    # Two-number format (Nx Ny): total cells = Nx * Ny
    two_num_dir = tmp_path / "twonum"
    two_num_dir.mkdir()
    (two_num_dir / "model.txt").write_text("10 10\n")
    assert at.misc.get_npts_model(two_num_dir) == 100


def test_get_nprocs(tmp_path: Path) -> None:
    # input.txt: line index 21 (0-indexed, 22nd line) holds nprocs
    lines = ["placeholder\n"] * 21 + ["4 #nprocs\n"]
    (tmp_path / "input.txt").write_text("".join(lines))
    assert at.get_nprocs(tmp_path) == 4


def test_get_cellsofmpirank(tmp_path: Path) -> None:
    def make_model(path: Path, npts: int, nprocs: int) -> None:
        lines = ["placeholder\n"] * 21 + [f"{nprocs} #nprocs\n"]
        (path / "input.txt").write_text("".join(lines))
        (path / "model.txt").write_text(f"{npts}\n")

    for npts, nprocs in [(20, 4), (21, 4), (7, 3)]:
        subdir = tmp_path / f"npts{npts}_nprocs{nprocs}"
        subdir.mkdir()
        make_model(subdir, npts=npts, nprocs=nprocs)

        all_cells: list[int] = []
        cells_per_rank = []
        for rank in range(nprocs):
            cells = list(at.get_cellsofmpirank(rank, subdir))
            cells_per_rank.append(cells)
            all_cells.extend(cells)

        # Every cell index appears exactly once and all cells are covered
        assert sorted(all_cells) == list(range(npts))

        # Load balancing: ranks differ by at most 1 cell
        sizes = [len(c) for c in cells_per_rank]
        assert max(sizes) - min(sizes) <= 1

        # Cells within each rank are contiguous
        for cells in cells_per_rank:
            assert cells == list(range(cells[0], cells[0] + len(cells)))

    # Verify specific assignments for evenly divisible case (npts=20, nprocs=4)
    even_dir = tmp_path / "even"
    even_dir.mkdir()
    make_model(even_dir, npts=20, nprocs=4)
    assert list(at.get_cellsofmpirank(0, even_dir)) == list(range(5))
    assert list(at.get_cellsofmpirank(3, even_dir)) == list(range(15, 20))

    # Verify specific assignments for uneven case (npts=21, nprocs=4):
    # rank 0 gets one extra cell (leftover), ranks 1-3 get the base count
    uneven_dir = tmp_path / "uneven"
    uneven_dir.mkdir()
    make_model(uneven_dir, npts=21, nprocs=4)
    assert list(at.get_cellsofmpirank(0, uneven_dir)) == list(range(6))
    assert list(at.get_cellsofmpirank(1, uneven_dir)) == list(range(6, 11))
