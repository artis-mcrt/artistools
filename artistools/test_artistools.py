import argparse
import hashlib
import importlib
import inspect
import math
import os
import subprocess
import sys
import tomllib
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")
outputpath.mkdir(exist_ok=True, parents=True)

REPOPATH = at.get_path("artistools_repository")


def funcname() -> str:
    """Get the name of the calling function."""
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr] # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]
    except AttributeError as e:
        msg = "Could not get the name of the calling function."
        raise RuntimeError(msg) from e


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


def _console_script_targets() -> list[tuple[str, str, str]]:
    """Return (command, submodulename, funcname) for every console script declared in pyproject.toml."""
    with (REPOPATH / "pyproject.toml").open("rb") as f:
        scripts: dict[str, str] = tomllib.load(f)["project"]["scripts"]

    targets = []
    for command, target in scripts.items():
        submodulename, _, targetfuncname = target.partition(":")
        targets.append((command, submodulename, targetfuncname))
    return targets


@pytest.mark.parametrize(("command", "submodulename", "targetfuncname"), _console_script_targets())
def test_console_script_target(command: str, submodulename: str, targetfuncname: str) -> None:
    """Every console script must point to an importable module with a callable target function."""
    submodule = importlib.import_module(submodulename)
    assert callable(getattr(submodule, targetfuncname, None)), (
        f"{submodulename}.{targetfuncname} not found for command {command}"
    )


def test_commands_list_matches_scripts() -> None:
    """The completion setup command list must stay in sync with the console scripts in pyproject.toml."""
    assert set(at.commands.COMMANDS) == {command for command, _, _ in _console_script_targets()}


def test_subcommandtree() -> None:
    """Every subcommand spec must name an importable module, callable functions, and non-empty help text."""

    def recursive_check(tree: at.commands.CommandTree) -> None:
        for cmdtarget in tree.values():
            if isinstance(cmdtarget, dict):
                recursive_check(cmdtarget)
            else:
                assert cmdtarget.helptext
                submodule = importlib.import_module(f"artistools.{cmdtarget.module}")
                assert callable(getattr(submodule, cmdtarget.funcname, None))
                assert callable(getattr(submodule, "addargs", None))

    recursive_check(at.commands.subcommandtree)


def test_shared_cli_args_consistent() -> None:
    """Arguments shared between commands must present the same flags and types everywhere."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    actionsbycommand: dict[str, dict[str, argparse.Action]] = {}

    def collect(parser: argparse.ArgumentParser, prefix: str) -> None:
        for action in parser._actions:  # ruff:ignore[private-member-access]
            if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
                nameparsermap: dict[str, argparse.ArgumentParser] = action._name_parser_map  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]  # ty:ignore[invalid-assignment]
                for name, subparser in nameparsermap.items():
                    collect(subparser, f"{prefix}{name} ")
            elif action.dest != "help":
                actionsbycommand.setdefault(prefix.strip(), {})[action.dest] = action

    collect(parser, "")
    assert len(actionsbycommand) > 30

    for command, actions in actionsbycommand.items():
        for dest, action in actions.items():
            flags = set(action.option_strings)
            label = f"{command}: {dest} {sorted(flags)}"
            if dest == "modelpath" and "-modelpath" in flags:
                assert action.type is Path, label
            elif dest == "timestep" and "-timestep" in flags:
                assert "-ts" in flags, label
            elif dest == "timedays" and "-timedays" in flags:
                assert {"-time", "-t"} <= flags, label
            elif dest == "maxpacketfiles":
                assert flags == {"-maxpacketfiles", "-maxpacketsfiles"}, label
                assert action.type is int, label
            elif dest == "figscale":
                assert action.type is float, label
            elif dest == "outputfile" and "-outputfile" in flags:
                assert "-o" in flags, label
            elif dest == "filtersavgol":
                assert action.nargs == 2, label
                assert "filtermovingavg" in actions, label  # the contract read by at.get_filterfunc


def test_deprecated_flag_spellings_still_work() -> None:
    """Flags renamed to the single-dash-takes-a-value convention keep their old spellings as hidden aliases."""
    parser = argparse.ArgumentParser()
    at.transitions.addargs(parser)
    assert parser.parse_args(["--atomicdatabase", "kurucz"]).atomicdatabase == "kurucz"
    assert parser.parse_args(["-atomicdatabase", "nist"]).atomicdatabase == "nist"
    assert parser.parse_args([]).atomicdatabase == "artis"

    parser = argparse.ArgumentParser()
    at.macroatom.addargs(parser)
    assert parser.parse_args(["--modelpath", "amodel"]).modelpath == Path("amodel")
    assert parser.parse_args(["-modelpath", "amodel"]).modelpath == Path("amodel")

    parser = argparse.ArgumentParser()
    at.estimators.plotestimators.addargs(parser)
    assert parser.parse_args(["-scalefigwidth", "2.5"]).figwidthscale == 2.5
    assert parser.parse_args(["-figwidthscale", "2.5"]).figwidthscale == 2.5
    assert parser.parse_args([]).figwidthscale == 1.0

    parser = argparse.ArgumentParser()
    at.viewing_angles_visualization.addargs(parser)
    for rawargs in (
        ["model.txt", "--outfile", "vis.html", "--opacity", "0.5", "-s", "10"],
        ["model.txt", "-outputfile", "vis.html", "-opacity", "0.5", "-surface_count", "10"],
    ):
        args = parser.parse_args(rawargs)
        assert args.outputfile == "vis.html"
        assert args.opacity == 0.5
        assert args.surface_count == 10


def test_lightcurve_title_arg() -> None:
    """The lc -title flag accepts custom text, while the bare (deprecated --title) form shows the model name."""
    parser = argparse.ArgumentParser()
    at.lightcurve.plotlightcurve.addargs(parser)
    assert parser.parse_args([]).title is None
    assert parser.parse_args(["--title"]).title is True
    assert parser.parse_args(["-title", "Custom title"]).title == "Custom title"


def test_hidden_duplicate_commands() -> None:
    """Cross-level duplicate command names still work but are not advertised in at --help."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    helptext = parser.format_help()
    assert "plotspectra" in helptext
    for hiddenname in ("describeinputmodel", "maptogrid", "makeartismodelfromparticlegridmap"):
        assert hiddenname not in helptext

    args = parser.parse_args(["describeinputmodel", "somemodelpath"])
    assert args.func.__module__ == "artistools.inputmodel.describeinputmodel"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    import artistools.__main__

    artistools.__main__.main(argsraw=["version"])
    assert f"artistools {at.version.version}" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["--version"])
    assert excinfo.value.code == 0
    assert at.version.version in capsys.readouterr().out


def test_cli_unknown_command() -> None:
    import artistools.__main__

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["plotspetcra"])
    assert excinfo.value.code == 2


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    import artistools.__main__

    artistools.__main__.main(argsraw=[])
    assert "plotspectra" in capsys.readouterr().out


def test_cli_missing_model_gives_short_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing input file from the at command must exit with a one-line error instead of a traceback."""
    import artistools.__main__

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["plotestimators", "-modelpath", str(tmp_path / "nomodel")])
    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "input.txt" in stderr
    assert "nomodel" in stderr


@pytest.mark.parametrize(("comp_line", "expected"), [("at plotsp", "plotspectra"), ("at spec -timed", "-timedays")])
def test_cli_tab_completion(tmp_path: Path, comp_line: str, expected: str) -> None:
    """Tab completion must offer subcommand names and the options of a subcommand."""
    outputfile = tmp_path / "completions.txt"
    env = os.environ | {
        "_ARGCOMPLETE": "1",
        "_ARGCOMPLETE_SHELL": "bash",
        "_ARGCOMPLETE_IFS": "\v",
        "_ARGCOMPLETE_SUPPRESS_SPACE": "1",
        "_ARGCOMPLETE_STDOUT_FILENAME": str(outputfile),
        "COMP_LINE": comp_line,
        "COMP_POINT": str(len(comp_line)),
    }
    subprocess.run([sys.executable, "-m", "artistools"], env=env, check=False, cwd=REPOPATH, timeout=120)
    completions = outputfile.read_text(encoding="utf-8").split("\v")
    assert expected in completions


def test_package_attrs() -> None:
    """Every re-exported attribute must resolve."""
    for name in dir(at):
        if not name.startswith("_"):
            assert getattr(at, name) is not None


def test_plotspherical_format_arg() -> None:
    parser = argparse.ArgumentParser()
    at.plotspherical.addargs(parser)
    assert parser.parse_args([]).format == "pdf"
    with pytest.raises(SystemExit):
        parser.parse_args(["-format", "svg"])


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


def test_get_ion_tuple_no_separator() -> None:
    """Two-letter symbols must not be split on their first letter, e.g. 'FeII' is Fe II and not F + 'eII'."""
    assert at.get_ion_tuple("FeII") == (26, 2)
    assert at.get_ion_tuple("CoII") == (27, 2)
    assert at.get_ion_tuple("NiIII") == (28, 3)
    assert at.get_ion_tuple("HeII") == (2, 2)
    # single-letter symbols still work, and are not shadowed by a longer symbol that starts the same way
    assert at.get_ion_tuple("FII") == (9, 2)
    assert at.get_ion_tuple("CIV") == (6, 4)
    assert at.get_ion_tuple("OI") == (8, 1)

    with pytest.raises(ValueError, match="Could not parse ionstr"):
        at.get_ion_tuple("notanion")


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
    assert at.match_closest_time(250.0, times) == 200.0
    assert at.match_closest_time(310.0, times) == 300.0
    assert at.match_closest_time(99.0, times) == 100.0
    assert at.match_closest_time(400.0, times) == 400.0
    assert at.match_closest_time(310.0, ["100", "300.5", "400"]) == 300.5


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


@mock.patch.object(mplax.Axes, "scatter", side_effect=mplax.Axes.scatter, autospec=True)
def test_radfield_line_estimators_filter_cell_zero(mockscatter: t.Any) -> None:
    """The line estimator plot must filter on cell and timestep zero, which are falsy."""
    radfielddata = pl.DataFrame({
        "bin_num": [-2, -3, -2, -3],
        "modelgridindex": [0, 0, 1, 1],
        "timestep": [0, 0, 0, 0],
        "nu_upper": [1.0e15, 2.0e15, 3.0e15, 4.0e15],
        "J_nu_avg": [1.0e-20, 2.0e-20, 3.0e-20, 4.0e-20],
    })

    fig, ax = plt.subplots()
    at.radfield.plot_line_estimators(ax, radfielddata, modelgridindex=0, timestep=0)
    plt.close(fig)

    assert mockscatter.call_count == 1
    lambdas = np.array(mockscatter.call_args_list[0][0][1], dtype=float)
    # only the two rows of cell zero, not all four rows
    assert len(lambdas) == 2
    assert np.allclose(sorted(lambdas), sorted(at.constants.c_ang_per_s / np.array([1.0e15, 2.0e15])))


def test_ejectaopacity() -> None:
    """Binned expansion opacities need the level statistical weights, not only the transition wavelengths."""
    at.ejectaopacity.main(
        argsraw=[], modelpath=modelpath, timestep=40, lambdamin=3000.0, lambdamax=4000.0, deltalambda=10.0
    )


def test_kurucz_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """gfall.dat is fixed-width, and the wavelength field is 11 characters wide, not 12.

    Reading 12 characters takes the first character of the loggf field with it, which is only harmless while that
    character happens to be a space.
    """
    line = (
        f"{715.5170:11.4f}"  # 0-10  wavelength in nm
        f"{-1.234:7.3f}"  # 11-17 log(gf)
        f"{44.00:6.2f}"  # 18-23 element code Z.(ion_stage - 1)
        f"{25000.000:12.3f}"  # 24-35 lower level energy in cm-1
        f"{4.5:5.1f}"  # 36-40 lower level J
        " a4F       "  # 41-51 configuration label
        f"{35000.000:12.3f}"  # 52-63 upper level energy in cm-1
        f"{3.5:5.1f}" + " " + " 0" * 16 + "\n"  # 64-68 upper level J
        # the parser only reads lines with at least 24 whitespace-separated fields
    )
    (tmp_path / "gfall.dat").write_text(line, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dftransitions, ionlist = at.transitions.get_kurucz_transitions()

    assert ionlist == [at.transitions.IonTuple(44, 1)]
    assert len(dftransitions) == 1
    transition = dftransitions.iloc[0]
    assert transition.lambda_angstroms == pytest.approx(7155.170)
    assert transition.lower_statweight == pytest.approx(2 * 4.5 + 1)
    assert transition.upper_statweight == pytest.approx(2 * 3.5 + 1)

    hc_in_ev_cm = 0.0001239841984332003
    assert transition.lower_energy_ev == pytest.approx(hc_in_ev_cm * 25000.0)
    assert transition.upper_energy_ev == pytest.approx(hc_in_ev_cm * 35000.0)


def test_merge_pdf_files_keeps_inputs_until_written(tmp_path: Path) -> None:
    """The input files must survive until the merged file exists."""
    pytest.importorskip("pypdf", reason="pypdf is only installed with the extras group")

    pdfpaths = []
    for i in range(2):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [i, i])
        pdfpath = tmp_path / f"page{i}.pdf"
        fig.savefig(pdfpath, format="pdf")
        plt.close(fig)
        pdfpaths.append(str(pdfpath))

    at.merge_pdf_files(pdfpaths)

    merged = tmp_path / "page0-page1.pdf"
    assert merged.is_file()
    assert merged.stat().st_size > 0
    assert not any(Path(p).exists() for p in pdfpaths)


def test_linefluxes_emfeaturesearch_parsing() -> None:
    """Emission features given on the command line must arrive as tuples of ints, not as raw strings."""
    parser = argparse.ArgumentParser()
    at.linefluxes.addargs(parser)

    args = parser.parse_args(["-emfeaturesearch", "(26, 2, 7155, 7150, 7160)", "(28, 2, 7378, 7373, 7383)"])
    assert args.emfeaturesearch == [(26, 2, 7155, 7150, 7160), (28, 2, 7378, 7373, 7383)]

    # the default must already be usable for the two-feature flux ratio plot
    assert len(parser.parse_args([]).emfeaturesearch) >= 2

    # time bins are floats, not appended lists of strings
    args = parser.parse_args(["-timebins_tstart", "200", "250", "-timebins_tend", "250", "300"])
    assert args.timebins_tstart == [200.0, 250.0]
    assert args.timebins_tend == [250.0, 300.0]

    with pytest.raises(SystemExit):
        parser.parse_args(["-emfeaturesearch", "not a tuple"])


def test_linefluxes_lineflux_ratio_plot() -> None:
    """The line flux ratio plot must run with no arguments beyond the model path."""
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.linefluxes.main(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
        outputfile=funcoutpath / "linefluxes.pdf",
    )
    assert (funcoutpath / "linefluxes.pdf").is_file()
