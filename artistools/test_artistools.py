import argparse
import hashlib
import importlib
import inspect
import itertools
import math
import os
import subprocess
import sys
import tomllib
import typing as t
from datetime import date
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.testing as pltest
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
    thisframe = inspect.currentframe()
    try:
        if thisframe is None or thisframe.f_back is None:
            msg = "Could not get the name of the calling function."
            raise RuntimeError(msg)

        return thisframe.f_back.f_code.co_name
    finally:
        # a frame held in one of its own locals is a reference cycle, so drop it as the inspect docs advise
        del thisframe


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


def test_polars_series_expr_dispatch() -> None:
    """Polars leaves most Series methods unimplemented, so check that they reach their Expr implementations.

    On CPython 3.15 polars fails to rebind them and they silently return None, which artistools/_polarscompat.py
    repairs. Sample the plain Series methods and each namespace that the repair covers.
    """
    assert pl.Series("x", [2, 1, 1, None]).unique().sort().to_list() == [None, 1, 2]
    assert pl.Series("x", [-1, 2]).abs().to_list() == [1, 2]
    assert pl.Series("x", [1, None]).drop_nulls().to_list() == [1]
    assert pl.Series("x", ["ab"]).str.to_uppercase().to_list() == ["AB"]
    assert pl.Series("x", [[1, 1, 2]]).list.unique().list.len().to_list() == [2]
    assert pl.Series("x", [[1, 2]], dtype=pl.Array(pl.Int64, 2)).arr.sum().to_list() == [3]
    assert pl.Series("x", [date(2026, 8, 8)]).dt.year().to_list() == [2026]
    assert pl.Series("x", [{"a": 7}]).struct.field("a").to_list() == [7]
    assert pl.Series("x", ["a"], dtype=pl.Categorical).cat.len_bytes().to_list() == [1]
    assert pl.Series("x", [b"ab"]).bin.size().to_list() == [2]


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
                nameparsermap: dict[str, argparse.ArgumentParser] = action._name_parser_map  # ruff:ignore[private-member-access]
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
    from importlib.metadata import version

    import artistools.__main__

    artistools.__main__.main(argsraw=["version"])
    assert f"artistools {version('artistools')}" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["--version"])
    assert excinfo.value.code == 0
    assert version("artistools") in capsys.readouterr().out


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


def test_plotspherical_gaussian_filter() -> None:
    """-gaussian_sigma must reach the smoothing helper, which lives in artistools.misc."""
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotspherical.main(argsraw=[], modelpath=modelpath, gaussian_sigma=20, outputfile=funcoutpath)


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemax=270, outputfile=outputpath)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_logfiles(mockplot: t.Any) -> None:
    """Log file timings are parsed for every stage and rank, and plotted one page per timestep."""
    logfilepaths = at.logfiles.read_logfiles(modelpath_classic_3d)
    # compressed log files must be read too, not skipped
    assert sorted(path.name for path in logfilepaths) == [
        "output_0-0.txt",
        "output_0-0.txt",
        "output_1-0.txt.zst",
        "output_1-0.txt.zst",
    ]

    timetaken = at.logfiles.read_time_taken(logfilepaths)
    assert set(timetaken) == {"update_grid", "update_packets", "write_estimators"}
    for stage, bytimestep in timetaken.items():
        assert len(bytimestep) == 30, f"expected 30 timesteps of {stage} timings"
        for byrank in bytimestep.values():
            assert set(byrank) == {0, 1}, f"expected both mpi ranks for {stage}"
    assert timetaken["update_grid"][0] == {0: 2, 1: 2}
    assert timetaken["update_packets"][0] == {0: 1, 1: 1}
    assert timetaken["write_estimators"][2] == {0: 1, 1: 1}

    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.logfiles.main(argsraw=[], modelpath=[modelpath_classic_3d], outputfile=funcoutpath / "logfiles.pdf")

    # one line per stage on each of the 30 per-timestep pages
    assert len(mockplot.call_args_list) == 3 * 30


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


def test_make_vpkt_input_default_contents() -> None:
    """The default vpkt.txt must keep the exact layout ARTIS parses, field by field."""
    expected = (
        "3\n"  # number of viewing directions
        "1 0 -1\n"  # costheta of each direction
        "0 0 0\n"  # phi of each direction
        "0 \n"  # no opacity exclusions
        "0 0.2 1.5\n"  # override_tminmax off, then the time window
        "0\n"  # no custom wavelength ranges
        "1 100\n"  # override thick cell tau, and the threshold
        "10\n"  # tau_max_vpkt
        "0\n"  # velocity grid map off
        "0.2 1.5\n"  # velocity grid map time range
        "1 3500 6000"  # one wavelength range for the velocity grid map
    )
    assert at.make_vpkt_input.format_vpkt_input(at.make_vpkt_input.VpktConfig()) == expected


def test_make_vpkt_input_optional_blocks() -> None:
    """The opacity exclusion and custom wavelength blocks must be prefixed by their own counts."""
    config = at.make_vpkt_input.VpktConfig(
        directions_costheta_phi=[(-1, 0), (0.5, 90)],
        opacityexclusions=[0, -1, 26],
        custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
        override_tminmax=True,
        vgrid_on=True,
        override_thickcell_tau=False,
        tau_max_vpkt=7.5,
    )
    contents = at.make_vpkt_input.format_vpkt_input(config).splitlines()

    assert contents[0] == "2"
    assert contents[1] == "-1 0.5"
    assert contents[2] == "0 90"
    assert contents[3] == "1 3 0 -1 26"
    assert contents[4] == "1 0.2 1.5"
    assert contents[5] == "1 2 3500 6000 10000 12000"
    assert contents[6] == "0 100"
    assert contents[7] == "7.5"
    assert contents[8] == "1"


def test_make_vpkt_input_roundtrip() -> None:
    """Parsing a written file must recover exactly the settings it was written from."""
    for config in (
        at.make_vpkt_input.VpktConfig(),
        at.make_vpkt_input.VpktConfig(
            directions_costheta_phi=[(-1, 0), (0.5, 90)],
            opacityexclusions=[0, -1, 26],
            custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
            override_tminmax=True,
            vgrid_on=True,
            override_thickcell_tau=False,
            tau_max_vpkt=7.5,
            vspec_tmin_in_days=1.25,
        ),
    ):
        assert at.make_vpkt_input.parse_vpkt_input(at.make_vpkt_input.format_vpkt_input(config)) == config


def test_make_vpkt_input_rejects_inconsistent_file() -> None:
    """A truncated or out-of-range file must be reported, not silently accepted."""
    contents = at.make_vpkt_input.format_vpkt_input(at.make_vpkt_input.VpktConfig())
    truncated = "\n".join(contents.splitlines()[:5])
    with pytest.raises(ValueError, match="ended while reading"):
        at.make_vpkt_input.parse_vpkt_input(truncated)

    # a file ARTIS would reject still loads, so it can be repaired, but reports the problem
    badcostheta = at.make_vpkt_input.parse_vpkt_input(contents.replace("1 0 -1", "1 0 -2", 1))
    assert "outside [-1, 1]" in str(at.make_vpkt_input.fatal_config_error(badcostheta))
    with pytest.raises(ValueError, match="outside"):
        at.make_vpkt_input.format_vpkt_input(badcostheta)


def test_make_vpkt_input_matches_artis_token_reader() -> None:
    """ARTIS reads vpkt.txt with fscanf, which ignores line breaks, so the parser must too."""
    config = at.make_vpkt_input.VpktConfig(
        directions_costheta_phi=[(0.5, 90)], opacityexclusions=[0, -1], custom_lambda_ranges=[(4000, 7000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    allonelongline = " ".join(contents.split())
    assert at.make_vpkt_input.parse_vpkt_input(allonelongline) == config

    onetokenperline = "\n".join(contents.split())
    assert at.make_vpkt_input.parse_vpkt_input(onetokenperline) == config


def test_make_vpkt_input_velocity_grid_ranges_roundtrip() -> None:
    """ARTIS reads as many velocity grid ranges as the count declares, so every one must be written."""
    config = at.make_vpkt_input.VpktConfig(
        vgrid_on=True, vgrid_lambda_ranges=[(3500, 6000), (6000, 9000), (9000, 10000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    assert contents.splitlines()[-1] == "3 3500 6000 6000 9000 9000 10000"
    assert at.make_vpkt_input.parse_vpkt_input(contents) == config


def test_make_vpkt_input_accepts_file_without_velocity_grid_block() -> None:
    """ARTIS only reads the velocity grid block when the map is on, so a file may legitimately omit it."""
    contents = at.make_vpkt_input.format_vpkt_input(at.make_vpkt_input.VpktConfig())
    withoutvgridblock = "\n".join(contents.splitlines()[:9])

    config = at.make_vpkt_input.parse_vpkt_input(withoutvgridblock)
    assert not config.vgrid_on
    assert config.vgrid_lambda_ranges == [(3500.0, 6000.0)]


def test_make_vpkt_input_rejects_nonzero_first_opacity_choice() -> None:
    """ARTIS asserts opacityexclusions[0] == 0, so artistools must not be able to write such a file."""
    config = at.make_vpkt_input.VpktConfig(opacityexclusions=[26, 0])
    with pytest.raises(ValueError, match="first opacity choice must be 0"):
        at.make_vpkt_input.format_vpkt_input(config)


def test_make_vpkt_input_warns_outside_compiled_limits() -> None:
    """Bounds that ARTIS asserts against compile-time constants must warn rather than fail."""
    outsidetime = at.make_vpkt_input.VpktConfig(override_tminmax=True, vspec_tmin_in_days=0.2, vspec_tmax_in_days=1.5)
    assert any("time window" in warning for warning in at.make_vpkt_input.check_config(outsidetime))

    outsidelambda = at.make_vpkt_input.VpktConfig(custom_lambda_ranges=[(1000, 2000)])
    assert any("wavelength range" in warning for warning in at.make_vpkt_input.check_config(outsidelambda))

    assert not at.make_vpkt_input.check_config(at.make_vpkt_input.VpktConfig()), "the defaults must not warn"


def test_make_vpkt_input_cli_writes_file(tmp_path: Path) -> None:
    """The subcommand must honour -directions, including a negative leading costheta, and -outputfile."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-directions=-1,0 1,0", "-o", str(outfile), "--non-interactive"])

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "2"
    assert lines[1] == "-1 1"
    assert lines[2] == "0 0"


def test_make_vpkt_input_cli_keeps_existing_settings(tmp_path: Path) -> None:
    """Rerunning on an existing file must preserve settings that were not given on the command line."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-tau-max", "7.5", "-o", str(outfile), "--non-interactive"])
    assert at.make_vpkt_input.parse_vpkt_input(outfile.read_text(encoding="utf-8")).tau_max_vpkt == 7.5

    at.make_vpkt_input.main(argsraw=["-vspec-tmax", "9.0", "-o", str(outfile), "--non-interactive"])
    config = at.make_vpkt_input.parse_vpkt_input(outfile.read_text(encoding="utf-8"))
    assert config.vspec_tmax_in_days == 9.0
    assert config.tau_max_vpkt == 7.5, "the tau-max from the first run must survive the second"


def test_make_vpkt_input_interactive_edit() -> None:
    """An empty reply keeps the current value, and an invalid reply is asked again."""
    replies = iter([
        "",  # keep the default viewing directions
        "26 -1",  # rejected: the first opacity choice must be 0, so this question repeats
        "0 26 -1",  # set opacity choices
        "maybe",  # rejected, so this question repeats
        "yes",  # override_tminmax
        "",  # keep vspec_tmin
        "3.5",  # vspec_tmax
    ])
    # pad so the reply script does not have to track how many settings there are
    replies = itertools.chain(replies, itertools.repeat(""))
    asked: list[str] = []

    def fakeprompt(text: str) -> str:
        asked.append(text)
        return next(replies)

    config = at.make_vpkt_input.edit_config_interactively(at.make_vpkt_input.VpktConfig(), promptfunc=fakeprompt)

    assert config.directions_costheta_phi == [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
    assert config.opacityexclusions == [0, 26, -1]
    assert config.override_tminmax
    assert config.vspec_tmin_in_days == 0.2
    assert config.vspec_tmax_in_days == 3.5
    # the rejected reply must have caused the same question to be asked twice
    assert sum(question.startswith("Restrict virtual packets") for question in asked) == 2
    assert "[1,0 0,0 -1,0]" in asked[0], "the prompt must show the current value"


def test_make_vpkt_input_interactive_clears_list() -> None:
    """A single '-' must clear a list-valued setting."""
    config = at.make_vpkt_input.VpktConfig(opacityexclusions=[26])
    replies = itertools.chain(iter(["", "-"]), itertools.repeat(""))
    config = at.make_vpkt_input.edit_config_interactively(config, promptfunc=lambda _: next(replies))

    assert config.opacityexclusions == []


def test_hesma_width_luminosity_roundtrip(tmp_path: Path) -> None:
    """The widthluminosity action must build a file that plotwidthluminosity can read back."""
    (tmp_path / "Bband_testmodel_viewing_angle_data.txt").write_text(
        "peakmag risetime dm15\n"
        + "".join(f"{-19 + i / 100:.4f} {17.0:.4f} {1.0 + i / 100:.4f}\n" for i in range(100)),
        encoding="utf-8",
    )

    at.hesma_scripts.main(
        argsraw=[], action="widthluminosity", band="B", modelname="testmodel", pathtofiles=tmp_path, outputpath=tmp_path
    )

    widthlumfile = tmp_path / "testmodel_width-luminosity.dat"
    assert widthlumfile.is_file()
    dfwidthlum = at.read_wsv(widthlumfile)
    assert dfwidthlum.columns == ["peakmag", "dm15", "angle_bin"]
    assert dfwidthlum.height == 100

    plotdir = tmp_path / "widthlum"
    plotdir.mkdir()
    widthlumfile.rename(plotdir / widthlumfile.name)
    plotfile = tmp_path / "widthlum.pdf"
    at.hesma_scripts.main(argsraw=["plotwidthluminosity", "-pathtofiles", str(plotdir), "-plotfile", str(plotfile)])
    assert plotfile.is_file()


def test_hesma_reports_missing_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    """An action must name the argument it needs rather than failing on a None."""
    with pytest.raises(SystemExit):
        at.hesma_scripts.main(argsraw=["vspecfiles"])
    assert "requires -modelpath" in capsys.readouterr().out


def test_opacity_condition_labels_match_artis() -> None:
    """The codes must match trace_vpkt_direction() in vpkt.cc: -2 bound-free, -3 free-free, -4 electron scattering."""
    assert not at.misc.get_opacity_condition_label(0)
    assert at.misc.get_opacity_condition_label(-1) == "no-bb"
    assert at.misc.get_opacity_condition_label(-2) == "no-bf"
    assert at.misc.get_opacity_condition_label(-3) == "no-ff"
    assert at.misc.get_opacity_condition_label(-4) == "no-es"
    assert at.misc.get_opacity_condition_label(26) == "no-Fe"


def test_make_vpkt_input_rejects_bad_arguments() -> None:
    for baddirection in ("1", "1,0,0", "north,0", "1.5,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_directions(baddirection)

    for badrange in ("3500", "3500,6000,7000", "blue,red", "6000,3500"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_lambda_range(badrange)

    for badbool in ("maybe", "", "2"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_bool(badbool)


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

    for npts, nprocs in ((20, 4), (21, 4), (7, 3)):
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
    transition = dftransitions.row(0, named=True)
    assert transition["lambda_angstroms"] == pytest.approx(7155.170)
    assert transition["lower_statweight"] == pytest.approx(2 * 4.5 + 1)
    assert transition["upper_statweight"] == pytest.approx(2 * 3.5 + 1)

    hc_in_ev_cm = 0.0001239841984332003
    assert transition["lower_energy_ev"] == pytest.approx(hc_in_ev_cm * 25000.0)
    assert transition["upper_energy_ev"] == pytest.approx(hc_in_ev_cm * 35000.0)


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

    # unset time bins stay None, so that each model falls back to its own timestep grid
    args = parser.parse_args([])
    assert args.timebins_tstart is None
    assert args.timebins_tend is None

    # the wavelengths may be fractional, but the atomic number, ion stage, and level indices must be integers
    assert parser.parse_args(["-emfeaturesearch", "(26, 2, 12570.5, 12470.5, 12670.5)"]).emfeaturesearch == [
        (26, 2, 12570.5, 12470.5, 12670.5)
    ]

    for badfeature in ("not a tuple", "(26.0, 2, 7155)", "(26, True, 7155)", "(26, 2, 7155, 7150, 7160, 1.5, 2)"):
        with pytest.raises(SystemExit):
            parser.parse_args(["-emfeaturesearch", badfeature])


def test_linefluxes_default_timebins_use_each_models_timesteps() -> None:
    """With no explicit time bins, the packet binning must fall back to the model's own timestep grid."""
    from artistools.linefluxes import get_closelines
    from artistools.linefluxes import get_line_luminosities_from_packets

    emfeatures = [get_closelines(modelpath_classic_3d, 26, 2, 7155, 7100, 7200)]

    dflcdata_default = get_line_luminosities_from_packets("trueemissiontype", emfeatures, modelpath_classic_3d)
    dflcdata_explicit = get_line_luminosities_from_packets(
        "trueemissiontype",
        emfeatures,
        modelpath_classic_3d,
        arr_tstart=at.get_timestep_times(modelpath_classic_3d, loc="start"),
        arr_tend=at.get_timestep_times(modelpath_classic_3d, loc="end"),
    )

    pltest.assert_frame_equal(dflcdata_default, dflcdata_explicit)


def test_linefluxes_rejects_lone_timebin_argument() -> None:
    """Giving only one of the two time bin edge lists must be rejected before any data is read."""
    with pytest.raises(ValueError, match="must be given together"):
        at.linefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], timebins_tstart=[200.0, 250.0])


def test_linefluxes_rejects_emittingregions_without_enough_colours() -> None:
    """More models than the default palette must be rejected up front, not crash inside the colour conversion."""
    with pytest.raises(ValueError, match="needs a colour for each"):
        at.linefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d] * 11, plotemittingregions=True)


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


def test_get_ion_tuple_rejects_missing_element() -> None:
    """A string with a separator but no element symbol must raise rather than return atomic number -1."""
    for badionstr in (" II", "_II", "notanion", "Fe "):
        with pytest.raises(ValueError, match="Could not parse ionstr"):
            at.get_ion_tuple(badionstr)


def test_default_plotitem_keeps_estimator_columns_named_like_elements() -> None:
    """Te and W are estimator names as well as element symbols, so a real column must win over the element reading."""
    from artistools.estimators.plotestimators import default_plotitem_has_data

    estimatorcolumns = ["timestep", "modelgridindex", "Te", "TR", "W", "nne", "rho", "nnelement_Fe"]

    for plotitem in (["Te"], ["W"], ["TR"], ["rho"], ["nne"], [["averageionisation", ["Fe"]]]):
        assert default_plotitem_has_data(plotitem, estimatorcolumns), plotitem

    # an element that the model does not contain is still dropped
    assert not default_plotitem_has_data([["averageionisation", ["Sr"]]], estimatorcolumns)
    assert not default_plotitem_has_data([["populations", ["Sr I", "Sr II"]]], estimatorcolumns)

    # initabundances/initmasses come from the input model file, so they must not be gated on estimator columns
    assert default_plotitem_has_data([["initabundances", ["Sr", "Ni_stable"]]], estimatorcolumns)
    assert default_plotitem_has_data([["initmasses", ["Sr", "Ni_56"]]], estimatorcolumns)


def test_write_lbol_edep(tmp_path: Path) -> None:
    """The bolometric luminosity / deposition writer must produce one line per selected timestep.

    test_writecomparisondata() uses a model with no deposition.out, so its FileNotFoundError guard skipped this
    function entirely and hid the fact that it read column names the light curve frame does not have.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    selected_timesteps = [0, 1, 2, 5]
    write_lbol_edep(modelpath_classic_3d, selected_timesteps, outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"#NTIMES: {len(selected_timesteps)}"
    datalines = [line for line in lines if not line.startswith("#")]
    assert len(datalines) == len(selected_timesteps)
    for line in datalines:
        timedays, lbol, edep = (float(x) for x in line.split())
        assert timedays > 0.0
        assert lbol > 0.0
        assert edep > 0.0


def test_write_lbol_edep_ntimes_matches_rows(tmp_path: Path) -> None:
    """The NTIMES header must count the rows actually written, not the timesteps asked for.

    A selected timestep with no light curve or deposition data is dropped by the join, so a header quoting
    len(selected_timesteps) would promise more times than the file contains and misalign every reader.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    write_lbol_edep(modelpath_classic_3d, [0, 1, 2, 5, 9999], outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    datalines = [line for line in lines if not line.startswith("#")]
    assert lines[0] == f"#NTIMES: {len(datalines)}"
    # timestep 9999 does not exist, so it must not be counted
    assert len(datalines) == 4


def test_get_series_colors_greys_then_cycle() -> None:
    """More reference series than greys must fall back to the colour cycle instead of an IndexError."""
    colors = at.plottools.get_series_colors([False, True, True, False, True, True, True, True])

    assert colors == ["C0", "0.0", "0.4", "C1", "0.6", "0.7", "C2", "C3"]


def test_get_series_colors_keeps_the_colours_of_the_user() -> None:
    """A colour of the user has priority, and no other series gets that colour."""
    assert at.plottools.get_series_colors([False, False, True], ["C1"]) == ["C1", "C0", "0.0"]
    assert at.plottools.get_series_colors([False, True], [None, "red"]) == ["C0", "red"]

    # a grey that the user asked for goes out of the sequence of the reference series
    assert at.plottools.get_series_colors([True, True], ["0.0"]) == ["0.0", "0.4"]
    assert at.plottools.get_series_colors([True, True], [None, "0.0"]) == ["0.4", "0.0"]
    assert at.plottools.get_series_colors([False, True], ["0.0"]) == ["0.0", "0.4"]

    # a colour that is not a grey does not take a grey of the sequence
    assert at.plottools.get_series_colors([True, True], ["red"]) == ["red", "0.0"]


def test_get_series_colors_knows_the_value_of_a_cycle_colour() -> None:
    """A colour value of the cycle that the user asked for must go out of the cycle, like the name CN."""
    at.set_mpl_style()
    cyclecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert at.plottools.get_series_colors([False, False], [cyclecolors[0]]) == [cyclecolors[0], "C1"]
    assert at.plottools.get_series_colors([False, False], [cyclecolors[1]]) == [cyclecolors[1], "C0"]


def test_get_series_colors_matches_a_cycle_colour_by_any_spelling() -> None:
    """A cycle colour must leave the cycle whatever name the user gave it, not only its exact hex string."""
    at.set_mpl_style()
    firstcyclecolor = mplcolors.to_hex(plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
    assert firstcyclecolor == mplcolors.to_hex("tab:blue")

    for spelling in ("tab:blue", firstcyclecolor.upper()):
        colors = at.plottools.get_series_colors([False, False], [spelling])
        assert colors[0] == spelling
        # the second series used to be handed C0, drawing both series in the same blue
        assert mplcolors.to_hex(colors[1]) != firstcyclecolor

    # a grey that the user asked for is also matched by value, so the next reference series steps over it
    assert at.plottools.get_series_colors([True, True], ["#000000"]) == ["#000000", "0.4"]


def test_set_axis_properties_log_scale_keeps_the_data_in_view() -> None:
    """A log scale must be set before the limits, so an unrequested limit does not freeze the linear view.

    set_ylim turns autoscaling off even when both sides are None, so applying it first left the log axis with
    the linearly padded limits, whose lower bound is negative and therefore off the axis entirely.
    """
    _fig, ax = plt.subplots()
    ax.plot([1.0, 10.0], [1.0, 1000.0])

    at.plottools.set_axis_properties(ax, argparse.Namespace(logscaley=True, ymin=None, ymax=None))

    ymin, ymax = ax.get_ylim()
    assert ymin > 0.0, ymin
    assert ymin < 1.0, ymin
    assert ymax > 1000.0, ymax


def test_set_axis_properties_applies_the_requested_limits() -> None:
    """A limit that the user did give must still be applied, on either side and on either axis."""
    _fig, ax = plt.subplots()
    ax.plot([1.0, 10.0], [1.0, 1000.0])

    at.plottools.set_axis_properties(ax, argparse.Namespace(ymin=None, ymax=500.0, xmin=2.0, xmax=None))

    assert np.isclose(ax.get_ylim()[1], 500.0)
    assert np.isclose(ax.get_xlim()[0], 2.0)
    # the side that was not given stays fitted to the data rather than falling back to the default view
    assert ax.get_ylim()[0] <= 1.0
    assert ax.get_xlim()[1] >= 10.0


def test_iter_axes_flattens_a_subplot_grid() -> None:
    """One axes and a grid of axes both become a flat list, so the callers need no isinstance dance."""
    _fig, singleax = plt.subplots()
    assert at.plottools.iter_axes(singleax) == [singleax]

    _fig, axes = plt.subplots(nrows=2, ncols=3)
    assert at.plottools.iter_axes(axes.flatten()) == list(axes.flatten())

    # iterating a 2D array yields its rows, which are arrays rather than axes
    assert at.plottools.iter_axes(axes) == list(axes.flatten())


def test_path_is_artis_model_accepts_a_compressed_output_file() -> None:
    """A compressed ARTIS output file is a model, and not a reference data file."""
    assert all(at.path_is_artis_model(f"light_curve.out{ext}") for ext in ("", ".zst", ".gz", ".xz"))
    assert not at.path_is_artis_model("AT2017gfo_smarttetal2017.txt")
