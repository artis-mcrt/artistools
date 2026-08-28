import argparse
import hashlib
import importlib
import inspect
import itertools
import math
import os
import re
import subprocess
import sys
import tomllib
import typing as t
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

if t.TYPE_CHECKING:
    from collections.abc import Iterable

modelpath = at.get_path("testdata") / "testmodel"
RETIRED_COMMANDS = ("describeinputmodel", "makeartismodelfromparticlegridmap", "maptogrid")
DISPATCHERTARGET = "artistools.__main__:main"
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


@pytest.mark.skipif(sys.version_info < (3, 15), reason="polars rebinds its own Series stubs below 3.15")
def test_polarscompat_is_still_necessary() -> None:
    """Fail once polars rebinds its Series methods without help, so that the repair can go.

    polars leaves most Series methods as docstring-only stubs, and it rebinds each one to the Expr
    version when it imports. It picks them by inspecting co_consts, which CPython 3.15 no longer fills
    for such a function, thus every stub returns None. artistools/_polarscompat.py repairs that.

    This test reads the state of polars alone. test_polars_series_expr_dispatch reads the state after
    the repair, thus the two together say both that the repair works and that it is still needed.
    """
    # a fresh interpreter, because importing artistools applies the repair
    code = "import polars as pl; print(pl.Series('x', [1]).unique() is None)"
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True", (
        f"polars {pl.__version__} rebinds its own Series methods on Python "
        f"{'.'.join(str(part) for part in sys.version_info[:3])}. Delete artistools/_polarscompat.py, "
        "the call to repair_series_expr_dispatch in artistools/__init__.py, and this test"
    )


def get_console_scripts() -> dict[str, str]:
    """Return the declared target of each console script in pyproject.toml."""
    with (REPOPATH / "pyproject.toml").open("rb") as f:
        scripts: dict[str, str] = tomllib.load(f)["project"]["scripts"]

    return scripts


def test_console_scripts() -> None:
    """Every console script must run the dispatcher, and must be a dispatcher or name a subcommand."""
    scripts = get_console_scripts()
    subcommands = at.commands.get_script_subcommands()
    assert set(scripts) == {*at.commands.DISPATCHERSCRIPTS, *subcommands}

    for command, target in scripts.items():
        assert target == DISPATCHERTARGET, f"console script {command} must run {DISPATCHERTARGET}"

    submodulename, _, funcname = DISPATCHERTARGET.partition(":")
    assert callable(getattr(importlib.import_module(submodulename), funcname, None))

    for scriptname, words in subcommands.items():
        spec: at.commands.CommandSpec | at.commands.CommandTree = at.commands.subcommandtree
        for word in words:
            assert isinstance(spec, dict)
            spec = spec[word]

        assert not isinstance(spec, dict), f"{scriptname} names the command group {' '.join(words)}"
        assert spec.script == scriptname


def test_console_script_runs_its_own_subcommand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A per-command console script must run its own subcommand and name itself in the usage text."""
    import artistools.__main__

    monkeypatch.setattr(sys, "argv", ["plotartisestimators", "--help"])
    with pytest.raises(SystemExit):
        artistools.__main__.main()

    helptext = capsys.readouterr().out
    # a command of many flags names them "[options]", thus the usage takes one line
    assert helptext.startswith("usage: plotartisestimators [options]")
    # the parser holds this one command, thus it neither lists nor imports the other commands
    assert "-modelpath" in helptext
    assert "plotspectra" not in helptext

    parser = at.commands.build_script_parser("plotartisestimators")
    assert parser is not None
    assert parser.parse_args([]).func.__module__ == "artistools.estimators.plotestimators"

    for dispatcher in at.commands.DISPATCHERSCRIPTS:
        assert at.commands.build_script_parser(dispatcher) is None


def test_module_entry_points_name_a_real_subcommand() -> None:
    """Each module entry point must run through the dispatcher and name a subcommand of the tree.

    A module that calls its own main function reads no --quiet, and it reports a bad argument with a
    traceback. run_subcommand gives it the path of a console script.
    """
    names: dict[Path, str] = {}
    for path in sorted(REPOPATH.glob("artistools/**/*.py")):
        for match in re.finditer(r'run_subcommand\("([^"]+)"\)', path.read_text()):
            names[path] = match.group(1)

    assert names, "no module entry point routes through the dispatcher"

    for path, subcommand in names.items():
        spec = at.commands.subcommandtree.get(subcommand)
        assert spec is not None, f"{path.name} names the unknown subcommand {subcommand}"
        assert not isinstance(spec, dict), f"{path.name} names the command group {subcommand}"

    # every command takes --quiet, thus no module may call its main function and skip run_command
    for path in sorted(REPOPATH.glob("artistools/**/*.py")):
        text = path.read_text()
        if 'if __name__ == "__main__":' not in text or path.name.startswith("test_"):
            continue
        block = text.split('if __name__ == "__main__":')[1]
        if "run_subcommand" in block or "run_module_as_subcommand" in block:
            continue

        modulename = ".".join(path.relative_to(REPOPATH).with_suffix("").parts)
        assert at.commands.get_words_of_module(modulename) is None, (
            f"{modulename} is a subcommand, thus its entry point must run through the dispatcher"
        )

    # the tree names the module of each subcommand, thus the reverse lookup finds every one of them
    def walkspecs(tree: dict[str, t.Any]) -> "Iterable[at.commands.CommandSpec]":
        for node in tree.values():
            if isinstance(node, at.commands.CommandSpec):
                yield node
            else:
                yield from walkspecs(node)

    for spec in walkspecs(at.commands.subcommandtree):
        assert at.commands.get_words_of_module(spec.module) is not None, f"no command names the module {spec.module}"


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
    # a dest can have more than one action, e.g. a deprecated hidden alias, so the flags of every action
    # for that dest are collected together
    flagsbycommand: dict[str, dict[str, set[str]]] = {}

    def collect(parser: argparse.ArgumentParser, prefix: str) -> None:
        for action in parser._actions:  # ruff:ignore[private-member-access]
            if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
                nameparsermap: dict[str, argparse.ArgumentParser] = action._name_parser_map  # ruff:ignore[private-member-access]
                for name, subparser in nameparsermap.items():
                    collect(subparser, f"{prefix}{name} ")
            elif action.dest != "help":
                command = prefix.strip()
                flagsbycommand.setdefault(command, {}).setdefault(action.dest, set()).update(action.option_strings)
                if isinstance(action, at.misc.UnsupportedArgument):
                    # the name of a flag that this command does not take is no argument of its own, thus
                    # the rules below pass it by. The flags above still hold it, because a command that
                    # takes -t must name -timestep in one way or the other
                    continue
                if action.option_strings and not all(flag.startswith("--") for flag in action.option_strings):
                    actionsbycommand.setdefault(command, {})[action.dest] = action
                else:
                    actionsbycommand.setdefault(command, {}).setdefault(action.dest, action)

    collect(parser, "")
    assert len(actionsbycommand) > 30

    for command, actions in actionsbycommand.items():
        for dest, action in actions.items():
            flags = flagsbycommand[command][dest]
            label = f"{command}: {dest} {sorted(flags)}"
            if dest == "modelpath" and "-modelpath" in flags:
                assert action.type is Path, label
            elif dest == "timestep" and "-timestep" in flags:
                assert "-ts" in flags, label
            elif dest == "timedays" and "-timedays" in flags:
                assert "-time" in flags, label
                # -t means -timedays on every command, thus a user needs no knowledge of which other
                # arguments that command takes
                assert "-t" in flags, label
                # argparse reads "-timestep 30" as "-t imestep", thus a command that declares -t must
                # also declare -timestep, as its own argument or through addarg_unsupported
                assert any("-timestep" in f for f in flagsbycommand[command].values()), label
            elif dest == "maxpacketfiles":
                assert flags == {"-maxpacketfiles", "-maxpacketsfiles"}, label
                assert action.type is int, label
            elif dest == "figscale":
                assert action.type is float, label
            elif dest == "outputfile":
                # both directions: a command that hand-rolls -o alone makes argparse read
                # "-outputfile name" as "-o utputfile" plus a stray token
                assert {"-outputfile", "-o"} <= flags, label
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
        # -o gives a Path on every command, thus the older spelling gives one as well
        assert args.outputfile == Path("vis.html")
        assert args.opacity == 0.5
        assert args.surface_count == 10


def test_lightcurve_title_arg() -> None:
    """The lc -title flag accepts custom text, while the bare (deprecated --title) form shows the model name."""
    parser = argparse.ArgumentParser()
    at.lightcurve.plotlightcurve.addargs(parser)
    assert parser.parse_args([]).title is None
    assert parser.parse_args(["--title"]).title is True
    assert parser.parse_args(["-title", "Custom title"]).title == "Custom title"


def test_retired_duplicate_commands_stay_gone() -> None:
    """The retired top-level duplicates neither parse nor appear, and their tree names work.

    describeinputmodel, makeartismodelfromparticlegridmap, and maptogrid were hidden duplicates of the
    inputmodel commands. The repository keeps no compatibility shim, thus they are deleted.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    helptext = parser.format_help()
    for retiredname in RETIRED_COMMANDS:
        assert retiredname not in helptext

    with pytest.raises(SystemExit):
        parser.parse_args(["describeinputmodel", "somemodelpath"])

    args = parser.parse_args(["inputmodel", "describe", "somemodelpath"])
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


def test_command_groups_name_every_visible_command() -> None:
    """Every command that at --help lists must belong to exactly one group of COMMANDGROUPS."""
    import artistools.__main__

    grouped = list(itertools.chain.from_iterable(at.commands.COMMANDGROUPS.values()))
    assert len(grouped) == len(set(grouped)), "a command appears in more than one group"

    # every listed command must reach a heading, thus a new command cannot fall out of the listing
    helptext = artistools.__main__.build_parser().format_help()
    for heading in at.commands.COMMANDGROUPS:
        assert f"\n{heading}:\n" in helptext
    for name in grouped:
        assert name in helptext


def test_cli_bad_argument_gives_short_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A bad argument value must exit with a one-line error on stderr instead of a traceback."""
    import artistools.__main__

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["plotspectra", str(modelpath), "-timedays", "banana"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "banana" in captured.err
    assert "Traceback" not in captured.err


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
    # nusyn_min and nusyn_max moved by 7.4e-10 in relative terms when the hardcoded MeV_in_Hz became
    # 1e6 / h_ev_s, which is the same conversion expressed with the Planck constant of constants.py
    assert dicthash == "477eb9a026a0d526499ab11b53f32ed256d48898479dde9d2109213b988c4456", dicthash


def test_macroatom() -> None:
    at.macroatom.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=10)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(mockstep: mock.MagicMock, mockplot: mock.MagicMock) -> None:
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
def test_logfiles(mockplot: mock.MagicMock) -> None:
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
def test_transitions(mockplot: mock.MagicMock) -> None:
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
def test_radfield_line_estimators_filter_cell_zero(mockscatter: mock.MagicMock) -> None:
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


def get_kilonova_lightcurve() -> npt.NDArray[np.float64]:
    """Return a light curve that rises to a peak and then decays, as a kilonova does."""
    times = np.geomspace(0.11, 76.0, 56)
    return 1e42 * np.where(times < 0.5, (times / 0.5) ** 2.0, (times / 0.5) ** -1.3)


@pytest.mark.parametrize(
    ("name", "values", "wantslog"),
    [
        ("a flat series", np.linspace(1.0, 2.0, 50), False),
        # a light curve of a kilonova covers three orders of magnitude, and a linear axis draws its
        # late times on the line of zero. A rule that reads the spread of the values keeps that axis,
        # because the values gather near the peak
        ("the light curve of a kilonova", get_kilonova_lightcurve(), True),
        ("a decay over four decades", np.geomspace(1e4, 1.0, 100), True),
        ("a decay that falls away", np.exp(-np.linspace(0.0, 10.0, 100)), True),
        # a ramp reaches each value on the way, thus the percentiles lie near the ends of one step
        # and far apart in neither scale. The linear axis shows every value of it
        ("a ramp over four decades", np.linspace(1.0, 1e4, 100), False),
        ("one point of noise near zero", np.concatenate([np.full(100, 1.0), [1e-30]]), False),
        ("a value of zero in every second place", np.concatenate([np.zeros(50), np.geomspace(1.0, 1e4, 50)]), False),
        ("a few values of zero", np.concatenate([np.zeros(3), np.geomspace(1.0, 1e4, 97)]), True),
        ("no value at all", np.empty(0), False),
        ("fewer values than a percentile needs", np.array([1.0, 10.0, 100.0]), False),
        ("every value zero", np.zeros(20), False),
    ],
)
def test_wants_log_scale_reads_the_range_of_the_values(
    name: str, values: npt.NDArray[np.float64], wantslog: bool
) -> None:
    """A log scale belongs to values that a linear axis draws on the line of zero."""
    assert at.plottools.wants_log_scale(values.astype(np.float64)) is wantslog, name


def test_range_ratio_leaves_out_the_extreme_values() -> None:
    """The percentiles give the range, thus one value far from the others does not give it."""
    # the 5th and the 95th percentile of a ramp from 1 to 101 are 6 and 96
    assert at.plottools.get_range_ratio(np.linspace(1.0, 101.0, 101)) == pytest.approx(96.0 / 6.0)

    # one value 30 orders of magnitude below the others changes nothing
    assert at.plottools.get_range_ratio(np.concatenate([np.full(100, 1.0), [1e-30]])) == pytest.approx(1.0)


def test_auto_yscale_reads_the_drawn_values() -> None:
    """-yscale auto takes a log scale from the drawn values, and the other choices stand."""
    for ydata, wantslog in ((np.geomspace(1.0, 1e4, 50), True), (np.linspace(1.0, 2.0, 50), False)):
        fig, axis = plt.subplots()
        axis.plot(np.arange(ydata.size), ydata)

        args = argparse.Namespace(yscale="auto", logscaley=False)
        at.plottools.set_auto_yscale(axis, args)
        assert args.logscaley is wantslog

        # -yscale linear and -yscale log each give the answer already, thus the values change nothing
        for yscale, logscaley in (("linear", False), ("log", True)):
            args = argparse.Namespace(yscale=yscale, logscaley=logscaley)
            at.plottools.set_auto_yscale(axis, args)
            assert args.logscaley is logscaley

        plt.close(fig)


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


def test_prune_log_ticks_drops_only_the_ticks_against_each_end() -> None:
    """A tick at the very top or bottom of a log axis goes, and a tick well inside it stays."""
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e3)

    before = [loc for loc in ax.yaxis.get_majorticklocs() if 1e-10 <= loc <= 1e3]
    assert min(before) == pytest.approx(1e-10), "the test needs a tick at the lower end"

    at.plottools.prune_log_ticks(ax.yaxis)

    after = [loc for loc in ax.yaxis.get_majorticklocs() if 1e-10 <= loc <= 1e3]
    assert min(after) > 1e-10
    assert set(after) == {loc for loc in before if loc > 1e-10}


def test_prune_log_ticks_keeps_a_sparse_axis_unchanged() -> None:
    """A log axis of few major ticks keeps them all, rather than end with too few to read."""
    from artistools.plottools import PrunedLogLocator

    locator = PrunedLogLocator(minticks=99)
    assert list(locator.tick_values(1e-30, 1e2)) == list(mplticker.LogLocator().tick_values(1e-30, 1e2))


def test_prune_log_ticks_follows_the_view() -> None:
    """The locator prunes when matplotlib draws, thus a zoom gives the ticks of the new view.

    A FixedLocator of one view would keep those ticks through a zoom, and save_figure shows the figure
    before it writes the file.
    """
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e5)
    at.plottools.prune_log_ticks(ax.yaxis)
    first = list(ax.yaxis.get_majorticklocs())

    ax.set_ylim(1e2, 1e9)
    second = list(ax.yaxis.get_majorticklocs())

    assert second != first, "the ticks must follow the new view"
    assert max(second) > max(first), f"the zoomed view needs its own high ticks: {second}"


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


@mock.patch.object(mplax.Axes, "set_ylim", side_effect=mplax.Axes.set_ylim, autospec=True)
def test_radfield_honours_the_ymin_that_it_accepts(mocksetylim: mock.MagicMock) -> None:
    """Plotradfield adds -ymin, thus the axis must start there and not at the hard-coded zero."""
    at.radfield.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40, modelgridindex=0, ymin=1e-14)

    bottoms = [callargs.kwargs["bottom"] for callargs in mocksetylim.call_args_list if "bottom" in callargs.kwargs]
    assert bottoms, "the command must set the bottom of the axis"
    assert 1e-14 in bottoms, f"the requested -ymin is missing from {bottoms}"


def test_cli_suggests_a_close_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """A mistyped subcommand must name the closest one on every Python version that CI runs.

    SuggestingArgumentParser composes this message itself, thus Python 3.13 and 3.14 give the
    same text.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspetcra"])

    message = capsys.readouterr().err
    assert "invalid choice 'plotspetcra'" in message
    # the error names the fault, and the help line that follows names the closest subcommand
    assert "help: Did you mean plotspectra" in message
    assert message.index("error: ") < message.index("help: ")


def test_firstexisting_gives_the_purpose_of_a_missing_file(tmp_path: Path) -> None:
    """A list of file names alone does not tell a user what the file holds or which command reads it."""
    with pytest.raises(FileNotFoundError, match=r"linestat\.out gives the wavelength"):
        at.misc.firstexisting("linestat.out", folder=tmp_path, purpose="linestat.out gives the wavelength")

    # a caller that gives no purpose keeps the plain message
    with pytest.raises(FileNotFoundError) as noreason:
        at.misc.firstexisting("linestat.out", folder=tmp_path)

    assert "None of these files exist" in str(noreason.value)
    assert "gives the wavelength" not in str(noreason.value)


def test_plain_label_and_saved_path_read_well_in_a_terminal() -> None:
    """A log line must carry no LaTeX, and it must give the shorter of the two forms of a path."""
    assert at.plottools.plain_label(r"TEST MODEL +300.3d ($\pm$ 0.5d)") == "TEST MODEL +300.3d (+/- 0.5d)"
    # the subscript mark goes and the underscore stays, thus the plain form reads as M_sun
    assert at.plottools.plain_label(r"M$_{\odot}$") == "M_sun"
    assert at.plottools.plain_label("no mathematics here") == "no mathematics here"


def test_print_saved_gives_the_shorter_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A file outside the working folder gave a chain of "..", which is longer than the full path."""
    faraway = tmp_path / "figure.pdf"
    faraway.touch()

    opencommand = at.misc.fileio.get_open_command()
    at.misc.print_saved(faraway)
    reported = capsys.readouterr().out.removeprefix(f"{opencommand} ").strip().strip("'")

    assert ".." not in reported
    assert Path(reported).resolve() == faraway.resolve()

    # a file below the working folder keeps its short relative name
    here = Path("localfigure.pdf")
    at.misc.print_saved(Path.cwd() / here)
    assert capsys.readouterr().out.strip() == f"{opencommand} {here}"


def test_short_time_flag_always_means_timedays() -> None:
    """-t must mean -timedays on every command, and never something else.

    A command that takes no -timestep left -t ambiguous with -timedays and -time, thus -t failed there.
    argparse also reads "-timestep 30" as "-t imestep", thus such a command declares the name it
    refuses.
    """
    import artistools.hesma_scripts

    # every parser that a command gets is this class, thus the test builds the same one
    parser = at.commands.SuggestingArgumentParser(prog="hesma")
    artistools.hesma_scripts.addargs(parser)

    assert parser.parse_args(["-t", "300"]).timedays == 300.0
    assert parser.parse_args(["-timedays", "300"]).timedays == 300.0

    # the command takes no timestep, thus it names the argument that it does take
    for flag in ("-timestep", "-ts"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "40"])


def test_unsupported_argument_names_the_replacement(capsys: pytest.CaptureFixture[str]) -> None:
    """A declared but unsupported argument must name the argument to give in its place."""
    parser = at.commands.SuggestingArgumentParser(prog="demo")
    at.misc.addarg_unsupported(parser, "-timestep", "-ts", instead="-timedays")

    with pytest.raises(SystemExit):
        parser.parse_args(["-timestep", "40"])

    captured = capsys.readouterr().err
    assert "error: -timestep is not an argument of this command" in captured
    assert "help: Give -timedays instead" in captured

    # a hidden name stays out of the help text
    assert "-timestep" not in parser.format_help()


@pytest.mark.parametrize("example", [command for command, _ in at.commands.get_examples()])
def test_help_examples_run(example: str, tmp_path: Path) -> None:
    """Every example of the help text must run, thus no example can name an argument that went away.

    The examples give a path of ".", which this test replaces with the test model.
    """
    import artistools.__main__

    argv = [str(modelpath) if word == "." else word for word in example.split()]
    artistools.__main__.main(argsraw=[*argv, "-o", str(tmp_path), "--quiet"])


def test_help_shows_the_examples() -> None:
    """The help text must carry the examples and the way to read the help of one command."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    helptext = parser.format_help()

    for command, description in at.commands.get_examples():
        assert command in helptext, command
        assert description in helptext, description

    assert 'Run "artistools <command> --help"' in helptext

    # the help of a command with examples shows them in its own epilog as well
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    spectrahelp = subactions[0].choices["plotspectra"].format_help()
    assert "artistools plotspectra . -t 300" in spectrahelp


def test_help_wraps_a_description_and_keeps_the_epilog_lines() -> None:
    """A description wraps to the terminal, and the examples of the epilog keep their own lines.

    RawDescriptionHelpFormatter would keep both, thus a long description ran past the width.
    """
    import artistools.__main__

    parser = argparse.ArgumentParser(
        prog="demo",
        formatter_class=at.commands.CustomArgHelpFormatter,
        description=" ".join(["averylongword"] * 12),
        epilog="first line\n  second line kept as it is",
    )
    helptext = parser.format_help()

    assert "first line\n  second line kept as it is" in helptext
    # the description holds no line break of its own, thus the formatter breaks it
    assert max(len(line) for line in helptext.splitlines()) < 200
    assert helptext.count("averylongword") == 12

    # ArgumentDefaultsHelpFormatter still applies: a real command gives the default of each argument.
    # The top-level parser holds only -h and --version, thus the check reads one subcommand
    subactions = [a for a in artistools.__main__.build_parser()._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    assert "(default:" in subactions[0].choices["plotestimators"].format_help()


def test_command_group_help_points_at_one_command() -> None:
    """A group lists its commands, thus it must also say how to read the help of one of them."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    groupparser = subactions[0].choices["inputmodel"]
    helptext = groupparser.format_help()

    assert "The inputmodel commands of artistools." in helptext
    assert 'Run "artistools inputmodel <command> --help"' in helptext


def test_command_note_reaches_the_help_and_not_the_listing() -> None:
    """A note gives the help of one command what the listing of one line has no room for."""
    import artistools.__main__

    spec = at.commands.subcommandtree["plotnltepops"]
    assert not isinstance(spec, dict)
    assert spec.note

    toplevel = artistools.__main__.build_parser().format_help()
    assert spec.helptext in toplevel
    # the listing shows one line for each of 36 commands, thus the note stays out of it
    assert spec.note not in toplevel


def test_deprecated_spellings_parse_and_stay_out_of_the_help() -> None:
    """A renamed flag keeps its old spelling as a hidden alias, thus an old command line still runs.

    The help shows one spelling for each concept, thus a reader meets no synonyms.
    """
    import artistools.gsinetwork.decayproducts
    import artistools.lightcurve.plotlightcurve
    import artistools.spectra.plotspectra

    cases = {
        artistools.spectra.plotspectra.addargs: (
            ["-yvar", "packetcount", "-xunits", "nm", "-dist", "1"],
            ("yvariable", "xunit", "distmpc"),
            ("-yvar", "-xunits", "-x", "-dist_mpc", "-dist", "-fluxdistmpc"),
        ),
        artistools.lightcurve.plotlightcurve.addargs: (
            ["--plot_cmf", "-timedaysmin", "260", "--title", "x"],
            ("plotcmf", "timemin", "title"),
            ("--plot_cmf", "--showcmf", "-timedaysmin", "-timedaysmax", "--title"),
        ),
        artistools.gsinetwork.decayproducts.addargs: (["-trajectoryroot", ".", "-timemin", "5"], ("tmin",), ()),
    }
    for addargs, (argsraw, dests, hidden) in cases.items():
        parser = argparse.ArgumentParser()
        addargs(parser)
        namespace = parser.parse_args(argsraw)
        for dest in dests:
            assert getattr(namespace, dest) not in {None, False}, (addargs.__module__, dest)

        helptext = parser.format_help()
        for spelling in hidden:
            assert f"{spelling} " not in helptext, spelling
            assert f"{spelling},\n" not in helptext, spelling


def test_plotspectra_x_still_gives_the_unit() -> None:
    """-x named the unit on plotspectra before -xunit, thus a script that holds it still runs.

    -x names the axis variable on plotestimators, but each parser reads its own arguments.
    """
    import artistools.spectra.plotspectra

    parser = argparse.ArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    # each spelling gives the canonical name of the unit, thus the plot code reads one name
    assert parser.parse_args([".", "-x", "nm"]).xunit == "nm"
    assert parser.parse_args([".", "-xunits", "micron"]).xunit == "micron"
    assert parser.parse_args([".", "-xunit", "Hz"]).xunit == "hz"


def test_timesteps_command_lists_the_days_of_each_timestep(capsys: pytest.CaptureFixture[str]) -> None:
    """The timesteps command gives the table that a user needs to select a -timestep value.

    Before this command, the mapping from a timestep to its days appeared only inside the error message
    for a wrong value.
    """
    at.showtimesteps.main(argsraw=["-modelpath", str(modelpath)])
    table = capsys.readouterr().out

    lines = table.splitlines()
    assert lines[0] == "TEST MODEL: 100 timesteps from 250.000 to 350.000 days"
    assert lines[1].split() == ["timestep", "start_days", "mid_days", "end_days", "width_days"]
    assert len(lines) == 103, "a header, a column line, 100 rows, and a closing hint"

    firstrow = lines[2].split()
    assert firstrow[0] == "0"
    assert np.isclose(float(firstrow[1]), 250.0)

    # the hint names the ways to select a time, and the keyword that names the final timestep
    assert "-timestep" in lines[-1]
    assert "-timedays" in lines[-1]
    assert '"last" names timestep 99' in lines[-1]


def test_help_strings_follow_one_style() -> None:
    """Every help string starts with a capital letter, ends without a period, and writes e.g. in full.

    31 of 336 strings ended with a period, 3 started with a lowercase letter, and "eg." stood beside
    "e.g.". This holds every future string to the one style.
    """
    import artistools.__main__

    def walk(parser: argparse.ArgumentParser) -> "t.Generator[tuple[str, argparse.Action]]":
        for action in parser._actions:  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
                seen = set()
                for name, subparser in action._name_parser_map.items():  # ruff:ignore[private-member-access]
                    if id(subparser) in seen:
                        continue  # an alias maps to the same parser
                    seen.add(id(subparser))
                    yield from ((f"{name} {label}", act) for label, act in walk(subparser))
            elif action.help and action.help != argparse.SUPPRESS:
                yield str(action.option_strings or [action.dest]), action

    failures = []
    for label, action in walk(artistools.__main__.build_parser()):
        # argparse writes the -h and --version texts itself, thus they keep their own style
        if action.dest == "help" or isinstance(action, argparse._VersionAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            continue
        helptext = action.help
        assert helptext is not None
        # a menu of the choices starts with the name of its first choice, e.g. "uniform: write..."
        startswithchoice = action.choices is not None and any(
            helptext.startswith(f"{choice}:") for choice in action.choices
        )
        if helptext[0].islower() and not startswithchoice:
            failures.append(f"{label}: starts lowercase: {helptext[:60]!r}")
        if helptext.endswith(".") and not helptext.endswith(("e.g.", "etc.")):
            failures.append(f"{label}: ends with a period: {helptext[-60:]!r}")
        if "eg. " in helptext:
            failures.append(f"{label}: write e.g. in full: {helptext[:60]!r}")

    assert not failures, "\n".join(failures)


def test_default_output_names_follow_one_scheme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each command names its file <command>[_series][_cell.....][_ts...][_days] with one field width.

    The widths were 02d and 03d, the days carried zero to two decimals, and the prefix of a command
    changed between its modes. A cell of 05d sorts up to the 3D grids, ts of 03d sorts past 100
    timesteps, and days of .2f keep two sub-day timesteps apart.
    """
    monkeypatch.chdir(tmp_path)
    import artistools.__main__

    runs = [
        (["plotspectra", str(modelpath), "-t", "300"], "plotspectra_299.81d-300.82d.pdf"),
        (
            ["plotestimators", "-modelpath", str(modelpath), "-p", "rho", "-ts", "40"],
            "plotestimators_ts040_286.02d-286.98d.pdf",
        ),
        (["plotlightcurves", str(modelpath)], "plotlightcurves.pdf"),
        (
            ["plotnltepops", "-modelpath", str(modelpath), "-t", "300", "-mgi", "0"],
            "plotnltepops_Fe_cell00000_ts054_300.32d.pdf",
        ),
        (["plotradfield", "-modelpath", str(modelpath), "-ts", "40", "-mgi", "0"], "plotradfield_cell00000_ts040.pdf"),
        (["plottransitions", "-modelpath", str(modelpath), "-t", "300"], "plottransitions_cell00000_ts054_300.32d.pdf"),
        (["plotmacroatom", "-modelpath", str(modelpath), "-ts", "40"], "plotmacroatom_cell00000_ts040-040.pdf"),
    ]
    for argsraw, expectedname in runs:
        artistools.__main__.main(argsraw=argsraw)
        assert (tmp_path / expectedname).is_file(), (argsraw[0], sorted(p.name for p in tmp_path.glob("*.pdf")))
        assert expectedname.startswith(argsraw[0]), "the file must carry the name of its command"


def test_help_groups_the_shared_arguments() -> None:
    """The shared arguments sit in titled groups, thus the help of a command of 77 options has a shape."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    for command in ("plotspectra", "plotlightcurves", "plotestimators"):
        helptext = subactions[0].choices[command].format_help()
        for title in ("time selection:", "appearance:", "output:"):
            assert f"\n{title}\n" in helptext, (command, title)


def test_open_flag_runs_the_platform_opener(tmp_path: Path) -> None:
    """--open opens the saved file with the default application, thus no copied command is needed."""
    with mock.patch.object(subprocess, "run") as mockrun:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=tmp_path, plotlist=[["rho"]], timestep="40", open=True
        )

    opencalls = [call.args[0] for call in mockrun.call_args_list]
    assert len(opencalls) == 1
    assert opencalls[0][0] in {"open", "xdg-open"}
    assert opencalls[0][1].endswith(".pdf")
    assert Path(opencalls[0][1]).is_file(), "the opener must receive the file that was saved"


def test_timesteps_command_answers_a_reverse_lookup(capsys: pytest.CaptureFixture[str]) -> None:
    """-timedays names the timestep that covers a time, and -timestep gives the days of one timestep."""
    at.showtimesteps.main(argsraw=["-modelpath", str(modelpath), "-t", "300"])
    assert capsys.readouterr().out.strip() == "300 days falls in timestep 54, which covers 299.812 to 300.823 days"

    at.showtimesteps.main(argsraw=["-modelpath", str(modelpath), "-ts", "last"])
    assert capsys.readouterr().out.strip() == "timestep 99 covers 348.824 to 350.000 days"


def test_quiet_short_flag_and_slow_command_timing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """-q means --quiet, and a command that runs past the threshold reports its wall time."""
    import artistools.__main__

    argsraw = ["plotestimators", "-modelpath", str(modelpath), "--listvariables", "-q"]

    # a quick run says nothing about its time
    artistools.__main__.main(argsraw=argsraw)
    captured = capsys.readouterr()
    assert "estimator variables" in captured.out, "-q must mean --quiet, and the product must stay"
    assert "seconds" not in captured.err

    # past the threshold, the time goes to the standard error beside the progress bars
    monkeypatch.setattr(artistools.__main__, "SLOW_COMMAND_SECONDS", 0.0)
    artistools.__main__.main(argsraw=[arg for arg in argsraw if arg != "-q"])
    captured = capsys.readouterr()
    assert re.search(r"The command took \d+\.\d seconds", captured.err)

    # the time reports the progress and not a fault, thus --quiet takes it away as well
    artistools.__main__.main(argsraw=argsraw)
    captured = capsys.readouterr()
    assert "estimator variables" in captured.out, "--quiet keeps the product"
    assert "seconds" not in captured.err


def test_unknown_flag_names_the_closest_one(capsys: pytest.CaptureFixture[str]) -> None:
    """A flag that no command takes must name the closest flags of the command that was run.

    The message was "unrecognized arguments: --listvaraibles" with no suggestion, and an ambiguous
    short flag listed -t because argparse reads -timeday as -t with a joined value.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotestimators", "--listvaraibles"])
    message = capsys.readouterr().err
    assert "unrecognized arguments: --listvaraibles" in message
    assert "Did you mean --listvariables" in message

    # the suggestion comes from the arguments of the subcommand, not from the top-level parser
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", ".", "-timeday", "300"])
    message = capsys.readouterr().err
    assert "Did you mean -timedays" in message

    # a per-command console script gives the same help
    scriptparser = at.commands.build_script_parser("plotartisspectrum")
    assert scriptparser is not None
    with pytest.raises(SystemExit):
        scriptparser.parse_args([".", "--emissionabsorbtion"])
    assert "Did you mean --emissionabsorption" in capsys.readouterr().err

    # a suggestion never names a hidden alias
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", ".", "--plotcmfx"])
    assert "-dist_mpc" not in capsys.readouterr().err


def test_an_error_names_the_remedy_on_a_help_line(capsys: pytest.CaptureFixture[str]) -> None:
    """An error states the fault, and a help line that follows says what to do next.

    The two parts were one sentence, thus a long remedy hid the fault that it followed.
    """
    import artistools.__main__

    at.misc.print_error("no time was given", "Give a time or a timestep, e.g. -timedays 250")
    message = capsys.readouterr().err
    assert message.splitlines() == ["error: no time was given", "help: Give a time or a timestep, e.g. -timedays 250"]

    # an error of argparse takes the same shape, and it keeps the exit status that argparse gives
    with pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["plotestimators", "--listvaraibles"])
    assert exitinfo.value.code == 2
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith(("error: ", "help: "))]
    assert lines == [
        "error: unrecognized arguments: --listvaraibles",
        "help: Did you mean --listvariables, --listvars, --listnuclides?",
    ]


def test_every_command_takes_quiet() -> None:
    """run_command alone implements --quiet, thus every command must take it.

    Six commands of 34 declared the flag, and the other 28 refused -q. No module reads args.quiet,
    thus addcommandargs adds the flag and no module declares it.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    for subcommand, subparser in subactions[0].choices.items():
        flagsofdest = {
            action.dest: action.option_strings
            for action in subparser._actions  # ruff:ignore[private-member-access]
        }
        if subparser.get_default("argparser") is None:
            continue  # a group of subcommands holds no arguments of its own

        assert flagsofdest.get("quiet") == ["--quiet", "-q"], f"{subcommand} must take --quiet"


def get_every_subcommand(parser: argparse.ArgumentParser) -> Iterator[tuple[str, argparse.ArgumentParser]]:
    """Give the name and the parser of each subcommand, at every depth of the tree."""
    for action in parser._actions:  # ruff:ignore[private-member-access]
        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            for name, subparser in action.choices.items():
                yield name, subparser
                yield from get_every_subcommand(subparser)


def test_an_output_template_takes_the_older_name_of_a_field() -> None:
    """A template of an output name keeps the field names that the commands gave before.

    -o "plot_{modelgridindex}.pdf" and -o "plot_{time_days}d.pdf" each stopped with an error, because
    the commands renamed those fields to {cell} and {timedays}. A script holds the older names.
    """
    assert at.format_frame_path("p_{modelgridindex:03d}_ts{timestep:03d}.pdf", cell=7, timestep=22) == "p_007_ts022.pdf"
    assert at.format_frame_path("p_{time_days:.0f}d.pdf", timedays=300.4) == "p_300d.pdf"

    # the new name of each field works as well, and both names give one value
    assert at.format_frame_path("p_{cell}_{timedays}.pdf", cell=7, timedays=300.4) == "p_7_300.4.pdf"

    # the message names the fields of the command, and it leaves out the older names
    with pytest.raises(ValueError, match=r"gives \{cell\}, \{timedays\}"):
        at.format_frame_path("p_{nosuch}.pdf", cell=1, timedays=2.0)


def test_every_command_reads_the_same_cell_grammar() -> None:
    """-modelgridindex takes the same text on every command, and it names the cell that it says.

    plotnltepops read args.modelgridindex[0], and that text is a string, thus "-cell 12" gave the
    first character and plotted the cell 1. Six commands gave the argument a type or an action of
    their own, thus a range reached some and not others.
    """
    import artistools.__main__

    # the text names one cell, a range of cells, or a list of them, whatever command reads it
    assert at.get_single_modelgridindex("12") == 12
    assert at.get_single_modelgridindex(None) is None
    assert at.parse_range_list("3-7") == [3, 4, 5, 6, 7]
    assert at.parse_range_list("4,5,6") == [4, 5, 6]

    # a command that reads one cell says so, in place of taking a cell that the text does not name
    with pytest.raises(ValueError, match=r"names 5 cells, and this command reads one"):
        at.get_single_modelgridindex("3-7")

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    seen: set[int] = set()
    checked = 0
    for subcommand, subparser in subactions[0].choices.items():
        if id(subparser) in seen:
            continue
        seen.add(id(subparser))
        for action in subparser._actions:  # ruff:ignore[private-member-access]
            if "-modelgridindex" not in action.option_strings or type(action).__name__ == "UnsupportedArgument":
                continue

            assert action.type is None, f"{subcommand} gives -modelgridindex a type of its own"
            assert action.nargs is None, f"{subcommand} gives -modelgridindex an nargs of its own"
            assert action.dest == "modelgridindex", f"{subcommand} gives -modelgridindex another dest"
            checked += 1

    assert checked >= 8, f"only {checked} commands take -modelgridindex"


def test_every_command_reads_the_same_timestep_grammar() -> None:
    """-timestep takes the same text on every command, thus a user carries one grammar between them.

    Six commands gave -timestep the type int, thus "-ts 40-45" and "-ts last" each stopped with
    "invalid int value" there, and worked on the commands that read a range.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    seen: set[int] = set()
    checked = 0
    for subcommand, subparser in subactions[0].choices.items():
        if id(subparser) in seen:
            continue
        seen.add(id(subparser))
        for action in subparser._actions:  # ruff:ignore[private-member-access]
            if "-timestep" not in action.option_strings or type(action).__name__ == "UnsupportedArgument":
                continue

            # no command reads the text as an int, thus each one takes "last" and a range
            assert action.type is None, f"{subcommand} gives -timestep a type of its own"
            checked += 1

    assert checked >= 10, f"only {checked} commands take -timestep"


def test_a_joined_value_takes_the_longest_flag() -> None:
    """-ts70 gives 70 to -ts, because the user names the longest flag that the token starts with.

    argparse reads the first two characters of a single-dash token, thus -ts70 gave -t the value
    "s70", and -ts kept no value. The command then plotted a time in days that the user never gave.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    for token, timestep, timedays in (
        ("-ts70", "70", None),
        ("-ts=70", "70", None),
        ("-ts40-45", "40-45", None),
        ("-timestep40", "40", None),
        ("-t70", None, "70"),  # a flag of one letter, which argparse reads without help
    ):
        namespace = parser.parse_args(["plotspectra", token])
        assert namespace.timestep == timestep, token
        assert namespace.timedays == timedays, token

    # an abbreviation of a longer flag stays whole, thus argparse still refuses -xun
    with pytest.raises(SystemExit):
        parser.parse_args(["plotspectra", "-xun", "nm"])

    # a token after -- is a positional argument, thus no split applies to it
    assert parser.parse_args(["plotspectra", "--", "-ts70"]).specpath == [Path("-ts70")]


def test_a_flag_with_letters_after_it_names_the_flag_that_the_user_means(capsys: pytest.CaptureFixture[str]) -> None:
    """-timesteps names -timestep, because a split would give "s" to -timestep and hide the number.

    The command took "-timesteps 40" as the timestep "s" and the spectrum path "40".
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    for token, flag in (("-timesteps", "-timestep"), ("-modelpaths", "-modelpath"), ("-tslast", "-ts")):
        with pytest.raises(SystemExit):
            parser.parse_args(["plotestimators", token, "40"])

        assert f"Did you mean {flag}?" in capsys.readouterr().err, token


def test_v_keeps_the_meaning_that_each_command_gave_it() -> None:
    """-v shows the detail of each step, but it keeps an older meaning where it had one.

    Four commands gave -v to -velocity or to -rhoscale, and a script holds such a command.
    Thus -v keeps that meaning there, and --verbose is the only form of the new argument.
    """
    import artistools.__main__

    olddestof = {
        "plotradfield": "velocity",
        "plotnltepops": "velocity",
        "spencerfano": "velocity",
        "makeartismodel1dslicefromcone": "rhoscale",
    }
    seen = set()
    for subcommand, subparser in get_every_subcommand(artistools.__main__.build_parser()):
        flagsofdest = {
            action.dest: action.option_strings
            for action in subparser._actions  # ruff:ignore[private-member-access]
        }
        olddest = olddestof.get(subcommand)
        if olddest is not None and olddest in flagsofdest:
            seen.add(subcommand)
            assert "-v" in flagsofdest[olddest], f"{subcommand} must keep -v for -{olddest}"
            assert flagsofdest.get("verbose", ["--verbose"]) == ["--verbose"], subcommand
            continue

        for dest, flags in flagsofdest.items():
            assert "-v" not in flags or dest == "verbose", f"{subcommand} gives -v to {dest}"

        if "verbose" in flagsofdest:
            assert flagsofdest["verbose"] == ["--verbose", "-v"], subcommand

    assert seen == set(olddestof), f"a command changed its name: {set(olddestof) - seen}"


def test_plotspherical_makes_the_output_folder(tmp_path: Path) -> None:
    """A -o path that has no file extension names a folder, which the command makes.

    The command wrote a file that had the name of the folder and no extension, and a run with
    --makegif stopped, because it built the path of each frame below a folder that did not exist.
    """
    outfolder = tmp_path / "frames"
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=str(outfolder), timemin=250, timemax=300)

    assert outfolder.is_dir(), "the command must make the folder that -o names"
    assert list(outfolder.glob("plotspherical_*.pdf")), f"no plot in {list(outfolder.iterdir())}"


def test_timesteps_command_refuses_a_timestep_outside_the_model() -> None:
    """A timestep that the model does not hold must name the range that it holds.

    The command indexed the list of times with the given value, thus 999 gave an IndexError and -1
    read the last row of the list and gave it the wrong label.
    """
    for timestep in ("999", "-1"):
        with pytest.raises(ValueError, match=r"is not in this model\. It has 100 timesteps, 0 to 99"):
            at.showtimesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])

    # the timesteps at each end of the model are in it
    for timestep in ("0", "99", "last"):
        at.showtimesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])


def test_plotspherical_gif_keeps_the_name_that_o_gives(tmp_path: Path) -> None:
    """A -o path that has a file extension names the gif, and its folder then holds the frames.

    A path such as movie.gif became a folder, thus the file that the user asked for was never written.
    """
    gifpath = tmp_path / "movie.gif"
    at.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=str(gifpath)
    )

    assert gifpath.is_file(), f"the gif must keep its name, but {list(tmp_path.iterdir())}"
    assert list(gifpath.parent.glob("plotspherical_*.png")), "the frames go in the folder of the gif"

    # a path with no file extension still names a folder that holds the gif and the frames
    outfolder = tmp_path / "movie"
    at.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=str(outfolder)
    )
    assert (outfolder / "sphericalplot.gif").is_file(), f"no gif in {list(outfolder.iterdir())}"


def test_radfield_opens_the_merged_pdf_alone(tmp_path: Path) -> None:
    """--open must open the merged pdf and not each plot that the merge takes in.

    merge_pdf_files deletes those plots, thus an application that opened one would hold nothing.
    """
    template = str(tmp_path / "rf_cell{cell:05d}_ts{timestep:03d}.pdf")
    with mock.patch("subprocess.run") as mockrun:
        at.radfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", open=True, outputfile=template)

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"one file must open, not {len(opened)}"
    assert Path(opened[0]).is_file(), "the file that opens must be the merged pdf, which still exists"


def test_radfield_opens_the_one_plot_that_holds_data(tmp_path: Path) -> None:
    """A range that holds data for one timestep alone makes one plot, and that plot is the product.

    The run took each plot for a part of a merge, thus no plot opened. No merge came, because one
    plot cannot merge, thus --open did nothing at all.
    """
    # the test model holds no radiation field data before timestep 10
    template = str(tmp_path / "rf_cell{cell:05d}_ts{timestep:03d}.pdf")
    with mock.patch("subprocess.run") as mockrun:
        at.radfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", open=True, outputfile=template)

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"the one plot must open, not {len(opened)} files"
    assert Path(opened[0]).is_file()


def test_singledashlongflags_holds_every_name_of_the_tree() -> None:
    """The table of the long flag names must hold what the commands declare.

    addarg_collidingflags reads that table, thus a name that no line of it holds gives no message when
    another command reads it as a joined value. Building the tree to collect the names would import
    every command module, which the per-command console scripts do not do.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]

    def islongsingledash(flag: str) -> bool:
        return flag.startswith("-") and not flag.startswith("--") and len(flag) > 2

    names = {
        flag
        for subparser in subactions[0].choices.values()
        for action in subparser._actions  # ruff:ignore[private-member-access]
        for flag in action.option_strings
        if islongsingledash(flag) and not isinstance(action, at.misc.UnsupportedArgument)
    }
    names |= {
        flag
        for action in parser._actions  # ruff:ignore[private-member-access]
        for flag in action.option_strings
        if islongsingledash(flag)
    }

    missing = names - at.commands.SINGLEDASHLONGFLAGS
    assert not missing, f"add these names to SINGLEDASHLONGFLAGS: {sorted(missing)}"


def test_a_flag_of_another_command_names_the_mistake(capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse joins a value to a flag of one letter, thus a long name of another command misparses.

    "plotdensity -obsspec 100" read as "-o bsspec" and wrote the plot to a file named bsspec, and
    "plotspectra -tmin 100" read as "-t min" and left 100 for a positional argument.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotdensity", "-obsspec", "100"])
    assert "-obsspec is not an argument of this command" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", str(modelpath), "-tmin", "100"])
    message = capsys.readouterr().err
    assert "-tmin is not an argument of this command" in message
    assert "-timemin" in message, "the help line must name the argument that this command takes"

    # a joined value of a flag of one letter still works, thus -t300 means -timedays 300
    artistools.__main__.main(argsraw=["timesteps", "-modelpath", str(modelpath), "-t300"])
    assert "300 days falls in timestep 54" in capsys.readouterr().out


def test_every_output_argument_records_what_the_command_writes() -> None:
    """addarg_output records a kind, thus the dispatcher keeps the promise of -o for every command.

    Two helpers held two contracts: one promised that a path with no file extension names a folder,
    which the command creates, and the other promised nothing. 17 modules never applied either rule,
    and "inputmodel to_tardis -o newfolder" stopped with FileNotFoundError.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]

    modulebycommand: dict[str, str] = {}

    def walktree(tree: dict[str, t.Any]) -> None:
        for name, node in tree.items():
            if isinstance(node, at.commands.CommandSpec):
                modulebycommand[name] = node.module
            else:
                walktree(node)

    walktree(at.commands.subcommandtree)

    withoutput = 0
    for subcommand, subparser in subactions[0].choices.items():
        if "outputfile" not in {action.dest for action in subparser._actions}:  # ruff:ignore[private-member-access]
            continue

        withoutput += 1
        kind = subparser.get_default("outputkind")
        assert kind in {"file", "folder"}, f"{subcommand} must record what it writes"

        # a command that writes one file names that file, either from the tree or in its own main
        # an alias of a command names the same module, thus the name of the command covers it
        if kind == "file" and subparser.get_default("outputdefaultname") is None and subcommand in modulebycommand:
            modulename = modulebycommand[subcommand]
            module = REPOPATH / "artistools" / Path(*modulename.split(".")).with_suffix(".py")
            text = module.read_text(encoding="utf-8")
            assert "resolve_outputfile" in text or "resolve_frameset_paths" in text, (
                f"{subcommand} writes one file, thus it must name that file"
            )

    assert withoutput > 20, "the tree must hold many commands that write output"


def test_the_command_listing_gives_one_line_to_each_command() -> None:
    """The listing of the commands must take one line for each of them.

    The name of a command with every alias took 36 columns and left 38 for the text, thus almost every
    description wrapped over two or three lines and the listing ran to 88 lines.
    """
    import artistools.__main__

    helptext = artistools.__main__.build_parser().format_help()
    listing = helptext.partition("positional arguments:")[2].partition("\noptions:")[0]

    # a line of the listing holds a command, or it is the heading of a group, or it is empty
    for line in listing.splitlines():
        if not line.strip() or not line.startswith("    "):
            continue
        assert line.split()[0][0].isalpha(), f"a description wrapped onto its own line: {line!r}"


def test_the_help_of_a_long_command_holds_no_wall_of_flags() -> None:
    """A usage line that names 77 flags over 61 lines tells a reader nothing.

    A command of more flags than MAXUSAGEFLAGS names them "[options]", and the help text below the
    usage names each one. A command of few flags keeps them in the usage line.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]

    longcommand = subactions[0].choices["plotspectra"].format_usage()
    assert "[options]" in longcommand
    assert longcommand.count("\n") == 1, f"the usage of a long command takes one line: {longcommand!r}"

    shortcommand = subactions[0].choices["timesteps"].format_usage()
    assert "[options]" not in shortcommand, "a command of few flags names them"
    assert "-modelpath" in shortcommand

    # a default that says nothing stays out of the help text
    estimatorshelp = subactions[0].choices["plotestimators"].format_help()
    assert "(default: None)" not in estimatorshelp
    assert "(default:" in estimatorshelp, "a default that carries a value still shows"


def test_a_command_that_writes_one_file_names_it_without_o(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A command that names its output file must write that file when -o names nothing.

    resolve_output_argument passed by a run that gave no -o, thus plotdensity gave None to savefig and
    stopped, and the uniform opacity file met Path(None).
    """
    import artistools.__main__

    monkeypatch.chdir(tmp_path)
    artistools.__main__.main(argsraw=["plotdensity", str(modelpath)])
    assert (tmp_path / "densityprofile.pdf").is_file(), f"no plot in {list(tmp_path.iterdir())}"

    artistools.__main__.main(argsraw=["inputmodel", "opacityfile", "uniform", "-modelpath", str(modelpath)])
    assert (tmp_path / "opacity.txt").is_file(), f"no opacity file in {list(tmp_path.iterdir())}"


def test_a_merged_pdf_keeps_the_name_that_o_gives(tmp_path: Path) -> None:
    """A run that merges its plots must take the name that -o gives the merged pdf.

    Only a gif could take such a name. A merge took the -o path for the name of one frame, thus
    "plotradfield -timestep 40-41 -o merged.pdf" stopped before it drew anything.
    """
    merged = tmp_path / "merged.pdf"
    at.radfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(merged))

    assert merged.is_file(), f"the merged pdf must keep its name, but {list(tmp_path.iterdir())}"
    assert not list(tmp_path.glob("plotradfield_*.pdf")), "the merge takes the frames away"

    # a -o path that names a folder still gives the merged pdf the name of its frames
    outfolder = tmp_path / "rf"
    at.radfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(outfolder))
    assert list(outfolder.glob("plotradfield_*-plotradfield_*.pdf")), f"no merged pdf in {list(outfolder.iterdir())}"


def test_the_product_keeps_its_name_when_one_frame_holds_data(tmp_path: Path) -> None:
    """A run that holds data for one frame must still write the product that -o named.

    combine_frames took that frame for the product and left the named path empty, thus --open opened
    the frame. A product that carries the name of a frame also went, because the merge removes its
    inputs.
    """
    # the test model holds no radiation field data before timestep 10, thus one frame comes of the two
    merged = tmp_path / "merged.pdf"
    at.radfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", outputfile=str(merged))
    assert merged.is_file(), f"the product must keep its name, but {list(tmp_path.iterdir())}"

    # the name of the product can be the name that a frame would take
    likeaframe = tmp_path / "plotradfield_cell00000_ts040.pdf"
    at.radfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(likeaframe))
    assert likeaframe.is_file(), f"the merge must not remove its own product: {list(tmp_path.iterdir())}"


def test_a_missing_optional_package_gives_no_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """import_optional names the command that installs a package, thus a traceback adds nothing.

    The handler of the dispatcher took an AssertionError, a FileNotFoundError, and a ValueError, thus
    a command that needs pypdf or imageio printed a traceback above that message.
    """
    import artistools.__main__

    def raise_missing(args: argparse.Namespace) -> None:  # ruff:ignore[unused-function-argument]
        at.import_optional("nosuchpackage")

    with mock.patch.object(at.showtimesteps, "main", raise_missing), pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["timesteps", "-modelpath", str(modelpath)])

    assert exitinfo.value.code == 1
    message = capsys.readouterr().err
    assert "This command needs nosuchpackage" in message
    assert "Traceback" not in message
