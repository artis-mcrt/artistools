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
from collections.abc import Sequence
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
from artistools import viewertools

modelpath = at.get_path("testdata") / "testmodel"
# each retired top-level name, with the module that its inputmodel command runs
RETIRED_COMMANDS = (
    ("makeartismodelfromparticlegridmap", "artistools.inputmodel.modelfromhydro"),
    ("maptogrid", "artistools.inputmodel.maptogrid"),
)
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
    result = run_fresh_python("import polars as pl; print(pl.Series('x', [1]).unique() is None)")

    assert result.stdout.strip() == "True", (
        f"polars {pl.__version__} rebinds its own Series methods on Python "
        f"{'.'.join(str(part) for part in sys.version_info[:3])}. Delete artistools/_polarscompat.py, "
        "the call to repair_series_expr_dispatch in artistools/__init__.py, and this test"
    )


def run_fresh_python(code: str, *pythonflags: str) -> subprocess.CompletedProcess[str]:
    """Run code in a new interpreter, because the test session already holds the modules that a test examines."""
    return subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, *pythonflags, "-c", code], capture_output=True, text=True, check=False
    )


# an attribute access resolves a lazy import, thus each import runs in the order of the code. With
# -X lazy_imports=all, polars itself fails with ImportCycleError when it loads first, thus that case is absent
NUMPYCASES = [
    pytest.param("", (), id="artistools_first"),
    pytest.param("import polars; polars.__name__; ", (), id="polars_first"),
    *(
        [pytest.param("", ("-X", "lazy_imports=all"), id="artistools_first_lazy_imports_all")]
        if sys.version_info >= (3, 15)
        else []
    ),
]


@pytest.mark.parametrize(("firstimport", "pythonflags"), NUMPYCASES)
def test_polars_holds_the_real_numpy(firstimport: str, pythonflags: tuple[str, ...]) -> None:
    """Check that polars reads the names of numpy itself, and not through its lazy proxy.

    On a free-threaded build, two threads that resolve that proxy at once raise with "'module' object
    does not support item assignment". gsinetworkdecayproducts did this in parallel_map. The proxy
    also stays in the modules of polars that imported it, thus the test reads one of them too.
    """
    code = (
        f"{firstimport}import artistools; artistools.__name__; import numpy, polars._dependencies, polars.series.series; "
        "print(type(polars._dependencies.numpy).__name__, polars.series.series.np.ndarray is numpy.ndarray)"
    )
    result = run_fresh_python(code, *pythonflags)

    assert result.stdout.split() == ["module", "True"], result.stderr


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports start with Python 3.15")
def test_an_import_with_no_module_name_still_works() -> None:
    """The lazy-import filter applies to the whole process, thus it must take code that has no __name__.

    Python gives such code no name of the module that does the import. jinja2 runs its templates in this way.
    """
    result = run_fresh_python('import artistools; exec("import json", {}); print("ok")')

    assert result.stdout.strip() == "ok", result.stderr


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports start with Python 3.15")
def test_matplotlib_saves_eps_with_type42_fonts() -> None:
    """backend_ps reads fontTools.ttLib, which only an import of fontTools.subset in matplotlib gives.

    A lazy import of fontTools.subset gave "module 'fontTools' has no attribute 'ttLib'".
    """
    code = (
        "import io, artistools, matplotlib.pyplot as plt; plt.rcParams['ps.fonttype'] = 42; "
        "fig, ax = plt.subplots(); ax.set_title('eps'); fig.savefig(io.BytesIO(), format='eps'); print('ok')"
    )
    result = run_fresh_python(code)

    assert result.stdout.strip() == "ok", result.stderr


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


def test_transitions_alias_of_the_partition_function_still_works() -> None:
    """The package pynonthermal reads at.transitions.get_lte_partfunc, thus the old path gives the same value and a warning."""
    dflevels = pl.DataFrame({"g": [2.0, 4.0], "energy_ev": [0.0, 1.0]})
    expected = at.atomic.get_lte_partfunc(dflevels, 5000.0)
    assert np.isclose(expected, 2.0 + 4.0 * math.exp(-1.0 / (at.constants.K_B_ev_per_K * 5000.0)))

    with pytest.warns(DeprecationWarning, match="artistools.atomic.get_lte_partfunc"):
        assert np.isclose(at.transitions.get_lte_partfunc(dflevels, 5000.0), expected)


TOPLEVEL_API: t.Final[frozenset[str]] = frozenset({
    "add_derived_cols_to_modeldata", "decode_roman_numeral", "firstexisting", "get_atomic_number", "get_deposition",
    "get_elsymbol", "get_inputparams", "get_ion_tuple", "get_ionstring", "get_model_name", "get_modeldata",
    "get_nprocs", "get_path", "get_timestep_of_timedays", "get_timestep_times", "get_z_a_nucname", "scan_estimators",
    "zopen",
})  # fmt: skip


def test_top_level_api_is_the_documented_list() -> None:
    """The top level holds the names that a user types in a script, and the README lists each of them.

    A different name stays in its package, e.g. at.misc.addarg_modelpath. To add a name to the top level,
    add it here and to the table in the README.
    """
    import types

    # getattr and not vars: on Python 3.15 an entry of vars is a lazy proxy until its first use
    public = {
        name
        for name in vars(at)
        if not name.startswith("_")
        and not isinstance(getattr(at, name), types.ModuleType)
        and getattr(getattr(at, name), "__module__", "") != "artistools._polarscompat"
    }
    assert public == TOPLEVEL_API

    readme = Path(at.__file__).parent.parent / "README.md"
    if readme.is_file():
        readmetext = readme.read_text(encoding="utf-8")
        assert not [name for name in sorted(TOPLEVEL_API) if f"`at.{name}`" not in readmetext]
        # a row of a name that left the top level tells the user to call a name that does not exist
        readmenames = {
            name
            for name in re.findall(r"`at\.(\w+)`", readmetext)
            if not isinstance(getattr(at, name, None), types.ModuleType)
        }
        assert readmenames <= TOPLEVEL_API, f"the README names {sorted(readmenames - TOPLEVEL_API)} at the top level"


def test_each_package_command_is_named_plot() -> None:
    """A package that has a plot command gives it as plot, thus a user finds it under one name."""
    for package in (at.estimators, at.gsinetwork, at.lightcurve, at.nltepops, at.nonthermal, at.packets, at.spectra):
        # the main function of a module of the package, and not a different callable with that name
        assert package.plot.__name__ == "main", package.__name__
        assert package.plot.__module__.startswith(f"{package.__name__}."), package.__name__


def test_residuals_take_the_model_at_each_observed_point() -> None:
    """The residual is model minus observed, inside the x range of the panel and of the model alone."""
    model = at.plottools.ResidualSeries(
        "model", np.array([0.0, 10.0, 20.0]), np.array([0.0, 20.0, 40.0]), "C0", isreference=False
    )
    reference = at.plottools.ResidualSeries(
        "obs", np.array([-5.0, 5.0, 12.0, 15.0, 25.0]), np.array([1.0, 11.0, 22.0, 33.0, 50.0]), "k", isreference=True
    )
    inrange, residual = at.plottools.get_residuals(reference, model, xmin=0.0, xmax=14.0)
    # the points at -5 and 25 lie outside the model, and the point at 15 lies outside the panel
    assert inrange.tolist() == [False, True, True, False, False]
    assert np.allclose(residual, [10.0 - 11.0, 24.0 - 22.0])

    # an observed NaN, e.g. a masked telluric range, stays a gap and does not count
    masked = reference._replace(y=np.array([1.0, np.nan, 22.0, 33.0, 50.0]))
    inrange, residual = at.plottools.get_residuals(masked, model, xmin=0.0, xmax=20.0)
    assert inrange.tolist() == [False, True, True, True, False]
    assert np.isnan(residual[0])

    # a model value that is not finite leaves a gap, as in the main frame, and gives no value from its neighbours
    gapmodel = model._replace(y=np.array([0.0, np.inf, 40.0]))
    _, residual = at.plottools.get_residuals(reference, gapmodel, xmin=0.0, xmax=20.0)
    assert np.isnan(residual).all()

    _fig, axis = plt.subplots()
    dfstats = at.plottools.plot_residual_panel(axis, [masked, model], 0.0, 20.0)
    assert dfstats["npoints"].item() == 2
    assert np.isclose(dfstats["rms"].item(), math.sqrt((4.0 + 9.0) / 2.0))
    assert np.isclose(dfstats["rms_relative"].item(), dfstats["rms"].item() / ((22.0 + 33.0) / 2.0))
    # the panel shows model minus reference: 24 - 22 and 30 - 33
    assert np.allclose(np.asarray(axis.lines[0].get_ydata())[1:], [2.0, -3.0])

    # a main frame with a log y axis takes model / reference: 24 / 22 and 30 / 33
    _fig, ratioaxis = plt.subplots()
    at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, ratio=True)
    assert np.allclose(np.asarray(ratioaxis.lines[0].get_ydata())[1:], [24.0 / 22.0, 30.0 / 33.0])

    # a ratio of the RMS to the mean reference value has no meaning for a magnitude
    dfmagstats = at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, ismagnitude=True)
    assert dfmagstats["rms_relative"].item() is None


@pytest.mark.parametrize(("modelfactor", "yscale"), [(2.0, "linear"), (100.0, "log"), (0.01, "log")])
def test_ratio_panel_takes_a_log_axis_for_a_large_ratio_alone(modelfactor: float, yscale: str) -> None:
    """With --logscaley the panel shows model / reference, on a log y axis only when a ratio is above 50."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    yreference = np.array([1.0, 2.0, 4.0, 8.0])
    factors = np.array([1.0, 1.0, modelfactor, modelfactor])
    series = [
        at.plottools.ResidualSeries("obs", x, yreference, "k", isreference=True),
        at.plottools.ResidualSeries("model", x, yreference * factors, "C0", isreference=False),
    ]
    args = argparse.Namespace(logscaley=True)
    _fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.plot(x, series[0].y)
    at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args)
    assert residualaxis.get_yscale() == yscale
    assert residualaxis.get_ylabel() == "model / ref"
    assert np.allclose(residualaxis.lines[0].get_ydata(), factors)


def test_frame_figure_takes_a_shorter_row() -> None:
    """A residual panel takes a part of the frame height, and the main frame keeps its size."""
    fig, axes = at.plottools.make_frame_figure(rows=2, rowheights=(1.0, 0.35))
    fig.canvas.draw()
    mainheight = axes[0][0].get_position().height
    residualheight = axes[1][0].get_position().height
    assert np.isclose(residualheight / mainheight, 0.35, rtol=1e-3)
    # row 0 is at the top
    assert axes[0][0].get_position().y0 > axes[1][0].get_position().y1
    plt.close(fig)

    figone, axesone = at.plottools.make_frame_figure()
    figone.canvas.draw()
    figtwo, axestwo = at.plottools.make_frame_figure(rows=2, rowheights=(1.0, 0.35))
    figtwo.canvas.draw()
    inchesone = axesone[0][0].get_position().height * figone.get_figheight()
    inchestwo = axestwo[0][0].get_position().height * figtwo.get_figheight()
    assert np.isclose(inchesone, inchestwo, rtol=1e-3)
    plt.close(figone)
    plt.close(figtwo)

    with pytest.raises(ValueError, match="rowheights gives"):
        at.plottools.make_frame_figure(rows=2, rowheights=(1.0,))


def test_package_modules_import_no_package_alias() -> None:
    """A package module must import each name from the module that defines it.

    An alias of the top-level package hides an import cycle until a different module comes first. Python 3.15
    binds "import artistools.spectra.core as atspectra" to the package, thus only a re-exported name resolves.
    """
    aliasimport = re.compile(r"^\s*import artistools(\.[\w.]+)? as \w+", re.MULTILINE)
    packagedir = Path(at.__file__).parent
    offenders = [
        str(path.relative_to(packagedir))
        for path in sorted(packagedir.rglob("*.py"))
        # a test can use the alias, and a name with a space is an iCloud conflict copy
        if not path.name.startswith("test_")
        and " " not in path.name
        and aliasimport.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"these package modules import a package alias: {offenders}"


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
                assert "filtermovingavg" in actions, label  # the contract read by at.misc.get_filterfunc


def test_deprecated_flag_spellings_still_work() -> None:
    """Flags renamed to the single-dash-takes-a-value convention keep their old spellings as hidden aliases."""
    parser = argparse.ArgumentParser()
    at.plottransitions.addargs(parser)
    assert parser.parse_args(["--atomicdatabase", "kurucz"]).atomicdatabase == "kurucz"
    assert parser.parse_args(["-atomicdatabase", "nist"]).atomicdatabase == "nist"
    assert parser.parse_args([]).atomicdatabase == "artis"

    parser = argparse.ArgumentParser()
    at.estimators.plotestimators.addargs(parser)
    assert parser.parse_args(["-scalefigwidth", "2.5"]).figwidthscale == 2.5
    assert parser.parse_args(["-figwidthscale", "2.5"]).figwidthscale == 2.5
    assert parser.parse_args([]).figwidthscale == 1.0

    parser = argparse.ArgumentParser()
    at.plotviewingangles.addargs(parser)
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

    makeartismodelfromparticlegridmap and maptogrid were hidden duplicates of the inputmodel
    commands. No script holds these names, thus the top level keeps none of them. The top-level
    describeinputmodel is different, and test_describeinputmodel_names covers it.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    helptext = parser.format_help()
    for retiredname, modulename in RETIRED_COMMANDS:
        assert retiredname not in helptext
        with pytest.raises(SystemExit):
            parser.parse_args([retiredname])

        assert parser.parse_args(["inputmodel", retiredname]).func.__module__ == modulename


def test_describeinputmodel_names() -> None:
    """Every spelling of the describe command runs the same module.

    One spec gives the three names: "artistools describeinputmodel", "artistools inputmodel
    describeinputmodel", and "artistools inputmodel describe". A script that holds an older
    spelling still runs.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    for words in (["describeinputmodel"], ["inputmodel", "describe"], ["inputmodel", "describeinputmodel"]):
        args = parser.parse_args([*words, "somemodelpath"])
        assert args.func.__module__ == "artistools.inputmodel.describeinputmodel"

    # the top-level name is an alias, thus the listing of the commands leaves it out
    assert "describeinputmodel" not in parser.format_help()


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


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(mockstep: mock.MagicMock, mockplot: mock.MagicMock) -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotradfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=funcoutpath, showbinedges=True)

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


def test_plotspherical_one_pass_matches_one_pass_per_time_range() -> None:
    """The direction maps of several time ranges in one pass must equal the map of each range on its own."""
    from artistools.plotspherical import bin_packets_by_direction

    nprocs_read, dfpackets = at.packets.get_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelpath=modelpath)
    tstarts = at.get_timestep_times(modelpath, loc="start")
    tends = at.get_timestep_times(modelpath, loc="end")
    # the last range repeats the first one, as every frame does when the observer never gets light from the full ejecta
    timeranges = [(tstarts[ts], tends[ts]) for ts in (60, 61, 62, 60)]
    plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc", "emvelocityoverc_sigma"]

    dfonepass, _ = bin_packets_by_direction(
        modelpath, dfpackets, nprocs_read, timeranges, 8, 4, plotvars, dfestimators=None
    )
    assert dfonepass.height == 4 * 8 * 4
    assert dfonepass["count"].sum() > 0

    for timebin, timerange in enumerate(timeranges):
        dfsingle, _ = bin_packets_by_direction(
            modelpath, dfpackets, nprocs_read, [timerange], 8, 4, plotvars, dfestimators=None
        )
        pltest.assert_frame_equal(
            dfonepass.filter(pl.col("timebin") == timebin).with_columns(timebin=pl.lit(0, dtype=pl.Int32)), dfsingle
        )


def test_plotspherical_gif() -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemax=270, outputfile=outputpath)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_logfiles(mockplot: mock.MagicMock) -> None:
    """Log file timings are parsed for every stage and rank, and plotted one page per timestep."""
    logfilepaths = at.plotlogfiles.read_logfiles(modelpath_classic_3d)
    # compressed log files must be read too, not skipped
    assert sorted(path.name for path in logfilepaths) == [
        "output_0-0.txt",
        "output_0-0.txt",
        "output_1-0.txt.zst",
        "output_1-0.txt.zst",
    ]

    timetaken = at.plotlogfiles.read_time_taken(logfilepaths)
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
    at.plotlogfiles.main(argsraw=[], modelpath=[modelpath_classic_3d], outputfile=funcoutpath / "logfiles.pdf")

    # one line per stage on each of the 30 per-timestep pages
    assert len(mockplot.call_args_list) == 3 * 30


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_transitions(mockplot: mock.MagicMock) -> None:
    at.plottransitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)

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

    # the free neutron is "n1", and a title-case lookup gave nitrogen for it
    assert at.get_atomic_number("X_n1") == 0
    assert at.get_z_a_nucname("X_n1") == (0, 1)
    assert at.get_z_a_nucname("n1") == (0, 1)
    # a symbol is not case sensitive, thus every other "n" is nitrogen
    assert at.get_atomic_number("n") == 7
    assert at.get_atomic_number("N") == 7
    assert at.get_atomic_number("X_N14") == 7
    assert at.get_z_a_nucname("X_N14") == (7, 14)
    assert at.get_z_a_nucname("n14") == (7, 14)
    assert at.get_ion_tuple("nII") == (7, 2)

    # a symbol that the caller gave in the wrong case still resolves
    assert at.get_atomic_number("fe") == 26

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
    assert at.misc.parse_range_list("5") == [5]
    assert at.misc.parse_range_list("3-5") == [3, 4, 5]
    assert at.misc.parse_range_list("1,3-5,8") == [1, 3, 4, 5, 8]
    assert at.misc.parse_range_list([3, 5, 7]) == [3, 5, 7]
    assert at.misc.parse_range_list(42) == [42]
    assert at.misc.parse_range_list("5-3") == [3, 4, 5]  # reversed range is sorted


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
        "dirbin peak_mag_polyfit risetime_polyfit deltam15_polyfit\n"
        + "".join(f"{i} {-19 + i / 100:.4f} {17.0:.4f} {1.0 + i / 100:.4f}\n" for i in range(100)),
        encoding="utf-8",
    )

    at.hesma_scripts.main(
        argsraw=[], action="widthluminosity", band="B", modelname="testmodel", pathtofiles=tmp_path, outputpath=tmp_path
    )

    widthlumfile = tmp_path / "testmodel_width-luminosity.dat"
    assert widthlumfile.is_file()
    dfwidthlum = at.misc.read_wsv(widthlumfile)
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
    assert "requires -modelpath" in capsys.readouterr().err


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
    assert at.misc.makelist(None) == []
    assert at.misc.makelist("hello") == ["hello"]
    assert at.misc.makelist(Path("my/folder/path")) == [Path("my/folder/path")]
    assert at.misc.makelist([1, 2, 3]) == [1, 2, 3]
    assert at.misc.makelist((1, 2)) == [1, 2]


def test_flatten_list() -> None:
    assert at.misc.flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert at.misc.flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
    assert at.misc.flatten_list([]) == []
    assert at.misc.flatten_list([1, 2, 3]) == [1, 2, 3]


def test_trim_or_pad() -> None:
    result = at.misc.trim_or_pad(3, [1, 2, 3, 4], [10, 20])
    assert list(result[0]) == [1, 2, 3]
    assert list(result[1]) == [10, 20, None]

    result2 = at.misc.trim_or_pad(2, "single_string")
    assert list(result2[0]) == ["single_string", None]


def test_match_closest_time() -> None:
    times = [100.0, 200.0, 300.0, 400.0]
    assert at.misc.match_closest_time(250.0, times) == 200.0
    assert at.misc.match_closest_time(310.0, times) == 300.0
    assert at.misc.match_closest_time(99.0, times) == 100.0
    assert at.misc.match_closest_time(400.0, times) == 400.0
    assert at.misc.match_closest_time(310.0, ["100", "300.5", "400"]) == 300.5


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
    """The 22nd line that holds a value gives nprocs. ARTIS skips comment and blank lines, and keeps them."""
    lines = ["# a comment of the user\n", "\n", *(["placeholder\n"] * 21), "4 #nprocs\n"]
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
            cells = list(at.misc.get_cellsofmpirank(rank, subdir))
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
    assert list(at.misc.get_cellsofmpirank(0, even_dir)) == list(range(5))
    assert list(at.misc.get_cellsofmpirank(3, even_dir)) == list(range(15, 20))

    # Verify specific assignments for uneven case (npts=21, nprocs=4):
    # rank 0 gets one extra cell (leftover), ranks 1-3 get the base count
    uneven_dir = tmp_path / "uneven"
    uneven_dir.mkdir()
    make_model(uneven_dir, npts=21, nprocs=4)
    assert list(at.misc.get_cellsofmpirank(0, uneven_dir)) == list(range(6))
    assert list(at.misc.get_cellsofmpirank(1, uneven_dir)) == list(range(6, 11))


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
    at.plotradfield.plot_line_estimators(ax, radfielddata, modelgridindex=0, timestep=0)
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


def test_expansion_opacities_keep_the_values_of_the_join_query() -> None:
    """The Rust kernel gives the values of the earlier polars query.

    The earlier query joined each cell with each line, and the reference sums come from it. Only the order
    of the additions is different, thus the values agree within the rounding error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None)
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    dfopacities = at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    assert dfopacities.height == 100
    for column, expectedsum in {
        "exopac": 1397.6901658607103,
        "linebinned": 11675.7786539805,
        "linebinned_maxone": 1652.4668052744682,
    }.items():
        assert math.isclose(dfopacities[column].sum(), expectedsum, rel_tol=1e-12), column


def test_expansion_opacities_of_a_null_population_are_zero() -> None:
    """A null population of an ion gives no opacity from that ion, and a null temperature gives no opacity.

    The kernel reads contiguous columns with no nulls, thus get_expansion_opacities() must replace each null.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None)
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )
    ionstr = opacitylines.ionstrs[0]

    def get_opacities(dfcells: pl.DataFrame) -> pl.DataFrame:
        return at.ejectaopacity.get_expansion_opacities(opacitylines, dfcells, lambda_bin_edges, time_days).select(
            at.ejectaopacity.OPACITYCOLUMNS
        )

    pltest.assert_frame_equal(
        get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
        get_opacities(dfcell.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
    )
    dfnotemperature = get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias("Te")))
    assert np.allclose(dfnotemperature.select(pl.all().abs().max()).row(0), 0.0, rtol=0.0, atol=0.0)


def test_expansion_opacities_keep_a_nan_in_each_sum() -> None:
    """A temperature of zero gives NaN level populations, and each of the three sums must then be NaN.

    f64::min(NaN, 1) is 1, and NaN.abs() >= 1e-18 is false, thus the kernel gave a finite linebinned_maxone
    and a finite exopac for such a cell.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None).with_columns(
        pl.lit(0.0, dtype=pl.Float32).alias("Te")
    )
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    dfopacities = at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    isnan = dfopacities.select(pl.col(at.ejectaopacity.OPACITYCOLUMNS).is_nan())
    assert isnan["linebinned"].any()
    for column in at.ejectaopacity.OPACITYCOLUMNS:
        assert isnan[column].equals(isnan["linebinned"]), column


def test_expansion_opacities_skip_a_line_with_a_null_constant() -> None:
    """A line with a null A value adds nothing, as a line that the atomic data does not hold.

    The kernel reads columns with no nulls, thus a null constant stopped the sum with an error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None)
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    adata = at.ejectaopacity.get_opacity_atomic_data(modelpath)
    isfirstline = pl.int_range(pl.len()) == 0

    def get_opacities(transitions: list[pl.LazyFrame]) -> pl.DataFrame:
        adatachanged = adata.with_columns(pl.Series("transitions", transitions, dtype=pl.Object))
        opacitylines = at.ejectaopacity.get_opacity_lines(adatachanged, dfcell.columns, lambda_bin_edges, time_days)
        return at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    dfnullconstant = get_opacities([
        dftransitions.lazy().with_columns(A=pl.when(~isfirstline).then(pl.col("A")))
        for dftransitions in adata["transitions"]
    ])
    dfnoline = get_opacities([dftransitions.lazy().filter(~isfirstline) for dftransitions in adata["transitions"]])

    pltest.assert_frame_equal(dfnullconstant, dfnoline)


def test_lambda_bin_edges_reject_a_range_with_no_bin() -> None:
    """A range with no bin stops with a message, not with an IndexError in get_expansion_opacities()."""
    with pytest.raises(ValueError, match="holds no bin"):
        at.ejectaopacity.get_lambda_bin_edges(5000.0, 4000.0, 10.0)


def test_lambda_bin_edges_cover_the_full_range() -> None:
    """The bins cover the full wavelength range, also when the division of the range rounds down.

    (4000 - 3000) / 0.1 is 9999.999999999998, and int() of it gave 9999 bins. Thus the last bin was missing.
    A range that does not hold a whole number of bins lost the part after the last whole bin.
    """
    edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 0.1)
    assert len(edges) == 10001
    assert math.isclose(edges[-1], 4000.0, rel_tol=1e-12)

    edges = at.ejectaopacity.get_lambda_bin_edges(1000.0, 25010.0, 20.0)
    assert len(edges) == 1202
    assert math.isclose(edges[-1], 25020.0, rel_tol=1e-12)


def test_plotopacity_weights_the_cells_by_mass() -> None:
    """The mean over the cells weights each cell by its mass, and it sums the cells of every batch.

    Cell k holds k times the ion populations and k times the mass of the test cell. The line-binned
    opacity is linear in the populations. Thus, for n cells, the mean is sum(k^2) / sum(k) = (2n + 1) / 3
    times the opacity of the test cell. The first CELLSPERBATCH cells fill one batch, and the last 8 cells
    go into a second batch.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.ejectaopacity.get_opacity_atomic_data(modelpath)
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None)
    assert dfcell.height == 1

    cellcount = at.ejectaopacity.CELLSPERBATCH + 8
    dfcells = pl.concat([
        dfcell.with_columns(
            pl.lit(k - 1, dtype=dfcell.schema["modelgridindex"]).alias("modelgridindex"),
            pl.col("mass_g") * k,
            # a population is Float32, and k times it rounds in Float32
            pl.col("^nnion_.*$").cast(pl.Float64) * k,
        )
        for k in range(1, cellcount + 1)
    ])

    def get_linebinned(dfestimators: pl.DataFrame) -> npt.NDArray[np.float64]:
        return at.plotopacity.get_massweighted_opacities(adata, time_days, dfestimators, 3000.0, 4000.0, 10.0)[
            "linebinned"
        ].to_numpy()

    linebinned_onecell = get_linebinned(dfcell)
    assert linebinned_onecell.max() > 0.0
    meanfactor = (2 * cellcount + 1) / 3
    assert np.allclose(get_linebinned(dfcells), meanfactor * linebinned_onecell, rtol=1e-10, atol=0.0)


def test_plotopacity_window_holds_the_nearest_odd_number_of_bins() -> None:
    """The window of the moving average holds the odd number of bins that is nearest to its width.

    2 * round(n / 2) + 1 gave the odd number nearest to n + 1, e.g. 5 bins for a width of 3 bins.
    """
    windowbins = {width: at.plotopacity.get_window_bins(width, 20.0) for width in (20.0, 60.0, 62.0, 140.0, 220.0)}
    assert windowbins == {20.0: 1, 60.0: 3, 62.0: 3, 140.0: 7, 220.0: 11}
    # a width of an even number of bins is one bin from two odd numbers, and it takes the larger one
    assert at.plotopacity.get_window_bins(200.0, 20.0) == 11
    assert at.plotopacity.get_window_bins(1.2, 0.2) == 7


def test_expansion_opacity_keeps_a_weak_line() -> None:
    """A line of a very small optical depth adds to the expansion opacity of its bin.

    1 - exp(-tau) is exactly zero below tau = 5.6e-17, thus a bin of weak lines had no expansion opacity.
    """
    from artistools.rustext import sum_binned_line_opacities

    taus = [1e-18, 1e-6, 0.5]
    dflevels = pl.DataFrame({"ionindex": [0], "g": [1.0], "energy_ev": [0.0]}, schema_overrides={"ionindex": pl.UInt32})
    dflines = pl.DataFrame(
        {
            "lambda_angstroms_binindex": [0, 1, 2],
            "lower": [0, 0, 0],
            "upper": [0, 0, 0],
            "sobolev_lower": taus,
            "sobolev_upper": [0.0, 0.0, 0.0],
            "lambda_angstroms": [1.0, 1.0, 1.0],
        },
        schema_overrides={"lambda_angstroms_binindex": pl.UInt32, "lower": pl.UInt32, "upper": pl.UInt32},
    )
    dfcells = pl.DataFrame({"Te": [5000.0], "nnion_0": [1.0]})
    exopac = sum_binned_line_opacities(dflevels, dflines, dfcells, ["nnion_0"], 3, at.constants.K_B_ev_per_K)["exopac"]
    assert np.allclose(exopac.to_numpy(), -np.expm1(-np.array(taus)), rtol=1e-12, atol=0.0)


def test_opacity_cell_batches_hold_fewer_cells_for_more_bins() -> None:
    """A batch has one row for each cell and bin, thus a batch of more bins must hold fewer cells.

    Each batch held 4096 cells, and the 4998 bins of the ejectaopacity defaults then took 3.6 GB for one batch.
    """
    dfcells = pl.DataFrame({"modelgridindex": range(10000)})
    for numbins in (100, 1200, 4998, 49980):
        batches = at.ejectaopacity.get_cell_batches(dfcells, numbins)
        assert sum(batch.height for batch in batches) == dfcells.height
        assert max(batch.height for batch in batches) * numbins <= at.ejectaopacity.ROWSPERBATCH, numbins


def test_plotopacity_velocity_range_takes_the_cells_of_the_range(capsys: pytest.CaptureFixture[str]) -> None:
    """-vmin and -vmax of plotopacity select the cells with a mid-point velocity in the range, and no other cell.

    Each velocity needs a unit. The count of the cells in the range and the title give each velocity in the unit
    of the user.
    """
    lzmodel, modelmeta = at.get_modeldata(modelpath_classic_3d)
    dfvelocities = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid_on_c")
        .collect()
    )
    expectedcells = set(dfvelocities.filter(pl.col("vel_r_mid_on_c").is_between(0.1, 0.2))["modelgridindex"])
    assert 0 < len(expectedcells) < dfvelocities.height

    args = at.misc.parse_cli_args(at.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "0.2c"])
    dfestimators = at.ejectaopacity.get_cell_estimators(modelpath_classic_3d, 5, None)
    capsys.readouterr()
    dfcells = at.plotopacity.select_velocity_range(dfestimators, args.vmin, args.vmax)
    assert set(dfcells["modelgridindex"]) == expectedcells
    assert (
        f"{len(expectedcells)} of {dfestimators.height} cells with estimators are in the velocity range with vmin = 0.1c and vmax = 0.2c"
        in (capsys.readouterr().out)
    )

    args = at.misc.parse_cli_args(at.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "60000km/s"])
    assert at.plotopacity.get_cells_text(None, args.vmin, args.vmax) == (
        "mass-weighted mean of the cells with vmin = 0.1c and vmax = 60000 km/s"
    )
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(at.plotopacity.addargs, None, None, ["-vmin", "0.1"])

    args = at.misc.parse_cli_args(at.plotopacity.addargs, None, None, ["-vmin", "150000km/s"])
    with pytest.raises(
        ValueError, match=re.escape("No cell with estimators is in the velocity range with vmin = 150000 km/s")
    ):
        at.plotopacity.select_velocity_range(dfestimators, args.vmin, args.vmax)


def test_cell_estimators_of_an_empty_cell_give_an_error() -> None:
    """A cell with no matter has no estimators, and the error names the cell.

    The command stopped with "cannot concat empty list" before.
    """
    lzmodel, modelmeta = at.get_modeldata(modelpath_classic_3d)
    emptycell = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") == 0.0)
        .select(pl.col("modelgridindex").min())
        .collect()
        .item()
    )
    with pytest.raises(ValueError, match="hold no values"):
        at.ejectaopacity.get_cell_estimators(modelpath_classic_3d, 5, emptycell)


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

    dftransitions, ionlist = at.plottransitions.get_kurucz_transitions()

    assert ionlist == [(44, 1)]
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

    at.misc.merge_pdf_files(pdfpaths)

    merged = tmp_path / "page0-page1.pdf"
    assert merged.is_file()
    assert merged.stat().st_size > 0
    assert not any(Path(p).exists() for p in pdfpaths)


def test_linefluxes_emfeaturesearch_parsing() -> None:
    """Emission features given on the command line must arrive as tuples of ints, not as raw strings."""
    parser = argparse.ArgumentParser()
    at.plotlinefluxes.addargs(parser)

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
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_packets

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


def test_linefluxes_timebins_keep_a_rounding_gap_and_drop_a_real_gap() -> None:
    """A packet between two bins that meet within rounding takes a bin, and a packet in a real gap takes none.

    Each bin was [tstart, tstart + twidth), and timesteps.out gives six significant figures. A packet in the
    rounding gap before the next start then had no bin, and the sums did not include it.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [0.9, 1.0, 1.999995, 2.0, 2.9999, 3.0, 3.5, 4.0, 5.0, 5.1]})
    timebins = dftimes.select(get_timebin_expr(pl.col("t"), [1.0, 2.0, 4.0], [1.99999, 3.0, 5.0]))
    assert timebins.to_series().to_list() == [None, 0, 0, 1, 1, None, None, 2, 2, None]


def test_linefluxes_from_pops_reads_the_shell_velocities() -> None:
    """The luminosity from the populations needs the inner and the outer velocity of each shell of a 1D model."""
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    # the test model has no linestat.out, thus the feature names its one Fe II transition directly
    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    # timestep 10 is the one timestep with NLTE populations in the test model
    dflcdata = get_line_luminosities_from_pops(
        emfeatures,
        modelpath,
        arr_tstart=[at.get_timestep_times(modelpath, loc="start")[10]],
        arr_tend=[at.get_timestep_times(modelpath, loc="end")[10]],
    )
    assert dflcdata.height == 1
    assert dflcdata["Fe II 1-0"].item() > 0.0


def test_linefluxes_pops_luminosity_matches_a_loop_over_the_cells() -> None:
    """The vectorised sum over the lines and the cells must give what the loop over each cell gave.

    A cell without population data gives its volume to the next cell outward that has data, and the
    outermost empty cells give their volume to no cell. The loop here is the former algorithm.
    """
    from artistools.plotlinefluxes import sum_line_luminosities

    rng = np.random.default_rng(seed=3)
    ncells = 6
    shell_volumes_at_1s = rng.uniform(1e40, 2e40, size=ncells)
    dftimes = pl.DataFrame({"timeindex": [0, 1, 2], "timestep": [4, 7, 7], "t_sec_cubed": [1e21, 8e21, 8e21]})
    dflines = pl.DataFrame({"lineindex": [0, 1], "level": [12, 9], "A": [0.3, 0.02], "delta_ergs": [2e-12, 1.5e-12]})

    # cell 0 and cell 3 hold no data in timestep 4, and the last two cells hold no data in timestep 7
    poprows = [
        (timestep, level, modelgridindex, float(rng.uniform(1e5, 1e6)))
        for timestep, emptycells in ((4, {0, 3}), (7, {4, 5}))
        for level in (12, 9)
        for modelgridindex in range(ncells)
        if modelgridindex not in emptycells
    ]
    dfnltepops = pl.DataFrame(
        poprows,
        schema={"timestep": pl.Int32, "level": pl.Int32, "modelgridindex": pl.Int32, "n_NLTE": pl.Float64},
        orient="row",
    )
    levelpop_of_key = {(timestep, level, mgi): n_nlte for timestep, level, mgi, n_nlte in poprows}

    expected = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        shell_volumes = shell_volumes_at_1s * t_sec_cubed
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            unaccounted_shellvol = 0.0
            for modelgridindex in range(ncells):
                levelpop = levelpop_of_key.get((timestep, level, modelgridindex))
                if levelpop is None:
                    unaccounted_shellvol += shell_volumes[modelgridindex]
                    continue
                expected[timeindex] += (
                    delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                )
                unaccounted_shellvol = 0.0

    lumdata = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s)

    assert lumdata.shape == expected.shape
    assert np.allclose(lumdata, expected, rtol=1e-13)


def test_linefluxes_rejects_lone_timebin_argument() -> None:
    """Giving only one of the two time bin edge lists must be rejected before any data is read."""
    with pytest.raises(ValueError, match="must be given together"):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], timebins_tstart=[200.0, 250.0])


def test_linefluxes_rejects_emittingregions_without_enough_colours() -> None:
    """More models than the default palette must be rejected up front, not crash inside the colour conversion."""
    with pytest.raises(ValueError, match="needs a colour for each"):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d] * 11, plotemittingregions=True)


def test_linefluxes_lineflux_ratio_plot() -> None:
    """The line flux ratio plot must run with no arguments beyond the model path."""
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotlinefluxes.main(
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


def get_kilonova_lightcurve() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times and the luminosities of a light curve that rises to a peak and then decays."""
    times = np.geomspace(0.11, 76.0, 56)
    return times, 1e42 * np.where(times < 0.5, (times / 0.5) ** 2.0, (times / 0.5) ** -1.3)


@pytest.mark.parametrize(
    ("name", "values", "wantslog"),
    [
        ("a flat series", np.linspace(1.0, 2.0, 50), False),
        ("a decay over four decades", np.geomspace(1e4, 1.0, 100), True),
        ("a decay that falls away", np.exp(-np.linspace(0.0, 10.0, 100)), True),
        # half of the values of a ramp are above half of the top
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


def test_auto_yscale_weights_each_value_by_its_part_of_the_x_axis() -> None:
    """A spectrum keeps a linear axis, and a light curve at geometric times takes a log axis.

    The rule of the quartiles chose a log axis for a kilonova spectrum at 3 to 5 days. The rule of commit 1598ad51
    chose a log axis for a hot spectrum at the first days. Each value covers its part of the x axis, thus the few late
    times of a light curve cover most of a linear time axis.
    """
    wavelengths = np.linspace(2500.0, 19000.0, 400)
    # 30% of the wavelengths hold the blue end, where the flux rises over three decades. Thus the quartiles are far
    # apart
    coolspectrum = np.concatenate([
        np.geomspace(1e-4, 0.1, 120),
        np.linspace(0.3, 1.0, 140),
        np.linspace(1.0, 0.3, 140),
    ])
    # a hot spectrum has its peak at the blue end, and the flux decreases to the red end
    hotspectrum = (wavelengths / 2500.0) ** -2.0
    for xvalues, yvalues, wantslog in (
        (wavelengths, coolspectrum, False),
        (wavelengths, hotspectrum, False),
        (*get_kilonova_lightcurve(), True),
    ):
        fig, axis = plt.subplots()
        axis.plot(xvalues, yvalues)
        args = argparse.Namespace(yscale="auto", logscaley=False)
        at.plottools.set_auto_yscale(axis, args)
        assert args.logscaley is wantslog
        plt.close(fig)


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


def test_set_legend_draws_no_legend_without_a_labelled_series() -> None:
    """An axes that has no series with a label gets no legend. matplotlib raised ValueError for the empty handles."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    assert at.plottools.set_legend(ax) is None
    assert ax.get_legend() is None
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
    at.plottools.set_mpl_style()
    cyclecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert at.plottools.get_series_colors([False, False], [cyclecolors[0]]) == [cyclecolors[0], "C1"]
    assert at.plottools.get_series_colors([False, False], [cyclecolors[1]]) == [cyclecolors[1], "C0"]


def test_get_series_colors_matches_a_cycle_colour_by_any_spelling() -> None:
    """A cycle colour must leave the cycle whatever name the user gave it, not only its exact hex string."""
    at.plottools.set_mpl_style()
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
    assert all(at.misc.path_is_artis_model(f"light_curve.out{ext}") for ext in ("", ".zst", ".gz", ".xz"))
    assert not at.misc.path_is_artis_model("AT2017gfo_smarttetal2017.txt")


@mock.patch.object(mplax.Axes, "set_ylim", side_effect=mplax.Axes.set_ylim, autospec=True)
def test_radfield_honours_the_ymin_that_it_accepts(mocksetylim: mock.MagicMock) -> None:
    """Plotradfield adds -ymin, thus the axis must start there and not at the hard-coded zero."""
    at.plotradfield.main(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40, modelgridindex=0, ymin=1e-14
    )

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


def test_room_for_title_keeps_the_axes_of_a_figure_with_no_frames() -> None:
    """Only a frame figure takes more height for its title, because its divider keeps the frames at the bottom.

    The axes of a different figure grew with the new height, and the title still went past the top.
    """
    fig, axis = plt.subplots(figsize=(4.0, 3.0))
    axis.set_title("line 1\nline 2\nline 3\nline 4")
    at.plottools.make_room_for_title(fig)
    assert np.isclose(fig.get_figheight(), 3.0, rtol=1e-12, atol=0.0)
    plt.close(fig)


def test_plain_label_and_saved_path_read_well_in_a_terminal() -> None:
    """A log line must carry no LaTeX, and it must give the shorter of the two forms of a path."""
    assert at.plottools.plain_label(r"TEST MODEL +300.3d ($\pm$ 0.5d)") == "TEST MODEL +300.3d (+/- 0.5d)"
    # the subscript mark goes and the underscore stays, thus the plain form reads as M_sun
    assert at.plottools.plain_label(r"M$_{\odot}$") == "M_sun"
    assert at.plottools.plain_label(r"T$_{\rm e}$ [K]") == "T_e [K]"
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
    at.timesteps.main(argsraw=["-modelpath", str(modelpath)])
    table = capsys.readouterr().out

    lines = table.splitlines()
    assert lines[0] == "TEST MODEL (folder testmodel): 100 timesteps from 250.000 to 350.000 days"
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
    at.timesteps.main(argsraw=["-modelpath", str(modelpath), "-t", "300"])
    assert capsys.readouterr().out.strip() == "300 days falls in timestep 54, which covers 299.812 to 300.823 days"

    at.timesteps.main(argsraw=["-modelpath", str(modelpath), "-ts", "last"])
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
    for subcommand, subparser in get_every_subcommand(parser):
        if subparser.get_default("argparser") is None:
            continue  # a group of subcommands holds no arguments of its own

        flagsofdest = get_flags_of_dest(subparser)
        assert flagsofdest.get("quiet") == ["--quiet", "-q"], f"{subcommand} must take --quiet"


def get_every_subcommand(parser: argparse.ArgumentParser) -> Iterator[tuple[str, argparse.ArgumentParser]]:
    """Give the name and the parser of each subcommand, at every depth of the tree."""
    for action in parser._actions:  # ruff:ignore[private-member-access]
        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            for name, subparser in action.choices.items():
                yield name, subparser
                yield from get_every_subcommand(subparser)


def get_flags_of_dest(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    """Return the option strings of each dest.

    The result merges every action that writes the same dest. A dest can hold more than one action,
    e.g. a deprecated hidden alias. A dict that keeps the last action alone loses the flags of the
    first one.
    """
    flagsofdest: dict[str, list[str]] = {}
    for action in parser._actions:  # ruff:ignore[private-member-access]
        flagsofdest.setdefault(action.dest, []).extend(action.option_strings)

    return flagsofdest


def test_an_output_template_takes_the_older_name_of_a_field() -> None:
    """A template of an output name keeps the field names that the commands gave before.

    -o "plot_{modelgridindex}.pdf" and -o "plot_{time_days}d.pdf" each stopped with an error, because
    the commands renamed those fields to {cell} and {timedays}. A script holds the older names.
    """
    assert (
        at.misc.format_frame_path("p_{modelgridindex:03d}_ts{timestep:03d}.pdf", cell=7, timestep=22)
        == "p_007_ts022.pdf"
    )
    assert at.misc.format_frame_path("p_{time_days:.0f}d.pdf", timedays=300.4) == "p_300d.pdf"

    # the new name of each field works as well, and both names give one value
    assert at.misc.format_frame_path("p_{cell}_{timedays}.pdf", cell=7, timedays=300.4) == "p_7_300.4.pdf"

    # the message names the fields of the command, and it leaves out the older names
    with pytest.raises(ValueError, match=r"gives \{cell\}, \{timedays\}"):
        at.misc.format_frame_path("p_{nosuch}.pdf", cell=1, timedays=2.0)

    # a field with no name gets a message as well, not a raw IndexError
    with pytest.raises(ValueError, match=r"field with no name.*\{cell\}, \{timedays\}"):
        at.misc.format_frame_path("p_{}.pdf", cell=1, timedays=2.0)


def test_a_wavelength_range_takes_both_spellings() -> None:
    """-xmin and -lambdamin name one argument on every command that reads a range of wavelengths.

    ejectaopacity took -lambdamin alone, thus "-xmin 100" there gave "unrecognized arguments" and a
    suggestion of -mgi, which names a cell. Three other commands take both spellings.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    seen: set[int] = set()
    checked = 0
    for subcommand, subparser in get_every_subcommand(parser):
        if id(subparser) in seen:
            continue
        seen.add(id(subparser))
        for flags in get_flags_of_dest(subparser).values():
            # a command that takes one of the two spellings takes the other one for the same value
            if "-lambdamin" in flags:
                assert "-xmin" in flags, f"{subcommand} takes -lambdamin without -xmin"
                checked += 1
            if "-lambdamax" in flags:
                assert "-xmax" in flags, f"{subcommand} takes -lambdamax without -xmax"

    assert checked >= 4, f"only {checked} commands take -lambdamin"


def test_every_command_reads_the_same_cell_grammar() -> None:
    """-modelgridindex takes the same text on every command, and it names the cell that it says.

    plotnltepops read args.modelgridindex[0], and that text is a string, thus "-cell 12" gave the
    first character and plotted the cell 1. Six commands gave the argument a type or an action of
    their own, thus a range reached some and not others.
    """
    import artistools.__main__

    # the text names one cell, a range of cells, or a list of them, whatever command reads it
    assert at.misc.get_single_modelgridindex("12") == 12
    assert at.misc.get_single_modelgridindex(None) is None
    assert at.misc.parse_range_list("3-7") == [3, 4, 5, 6, 7]
    assert at.misc.parse_range_list("4,5,6") == [4, 5, 6]

    # a command that reads one cell says so, in place of taking a cell that the text does not name
    with pytest.raises(ValueError, match=r"names 5 cells, and this command reads one"):
        at.misc.get_single_modelgridindex("3-7")

    parser = artistools.__main__.build_parser()
    seen: set[int] = set()
    checked = 0
    for subcommand, subparser in get_every_subcommand(parser):
        if id(subparser) in seen:
            continue
        seen.add(id(subparser))
        for action in subparser._actions:  # ruff:ignore[private-member-access]
            if "-modelgridindex" not in action.option_strings:
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
    seen: set[int] = set()
    checked = 0
    for subcommand, subparser in get_every_subcommand(parser):
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


def test_an_option_that_reads_a_list_gives_back_the_model_path() -> None:
    """An option that reads a list must not keep the ARTIS folder that follows its values.

    argparse gives every word that follows to such an option. Thus
    "plotspectra -label mylabel mymodel" left the model path empty, and it made "mymodel" a second
    label. normalize_path_list then gave the working folder, and the command plotted that folder
    with no message.
    """
    import artistools.__main__
    from artistools.misc import separate_trailing_folders

    parser = artistools.__main__.build_parser()

    args = parser.parse_args(["plotspectra", "-label", "mylabel", str(modelpath)])
    assert args.specpath == [modelpath]
    assert args.label == ["mylabel"]

    args = parser.parse_args(["plotlightcurves", "-label", "mylabel", str(modelpath)])
    assert args.modelpath == [modelpath]
    assert args.label == ["mylabel"]

    # the path that the user writes in front of the option still reaches the positional argument
    args = parser.parse_args(["plotspectra", str(modelpath), "-label", "mylabel"])
    assert args.specpath == [modelpath]
    assert args.label == ["mylabel"]

    # a value that names no folder stays with the option that reads it
    args = parser.parse_args(["plotspectra", "-label", "mylabel"])
    assert args.specpath == []
    assert args.label == ["mylabel"]

    # an option that converts its values reads no folder, thus the separator must come first
    args = parser.parse_args(separate_trailing_folders(["plotspectra", "-color", "red", str(modelpath)]))
    assert args.specpath == [modelpath]
    assert args.color == ["red"]

    args = parser.parse_args(separate_trailing_folders(["plotspectra", "-plotviewingangle", "0", str(modelpath)]))
    assert args.specpath == [modelpath]
    assert args.plotviewingangle == [0]

    # -1 is a value, not a flag, thus the folder after it still reaches the positional argument
    args = parser.parse_args(separate_trailing_folders(["plotspectra", "-plotviewingangle", "-1", str(modelpath)]))
    assert args.specpath == [modelpath]
    assert args.plotviewingangle == [-1]

    # every folder at the end of the command line reaches the positional argument
    args = parser.parse_args(
        separate_trailing_folders(["plotspectra", "-label", "a", str(modelpath), str(modelpath_classic_3d)])
    )
    assert args.specpath == [modelpath, modelpath_classic_3d]
    assert args.label == ["a"]

    args = parser.parse_args(separate_trailing_folders(["plotestimators", "-plotlist", "Te", str(modelpath)]))
    assert args.plotlist == [["Te"]]
    assert args.plotitems == [str(modelpath)]

    # a file is no ARTIS folder, thus it stays with the option that reads it
    reffile = modelpath / "light_curve.out"
    args = parser.parse_args(separate_trailing_folders(["plotlightcurves", "-reflightcurves", str(reffile)]))
    assert args.reflightcurves == [str(reffile)]
    assert args.modelpath == []


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
        flagsofdest = get_flags_of_dest(subparser)
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
            at.timesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])

    # the timesteps at each end of the model are in it
    for timestep in ("0", "99", "last"):
        at.timesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])


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
        at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", open=True, outputfile=template)

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
        at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", open=True, outputfile=template)

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"the one plot must open, not {len(opened)} files"
    assert Path(opened[0]).is_file()


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

    # a joined value that looks like a value, or that is a choice of the flag, reaches the flag
    parser = artistools.__main__.build_parser()
    assert parser.parse_args(["plotspectra", "-o/plots/x.pdf"]).outputfile == Path("/plots/x.pdf")
    assert parser.parse_args(["plotspectra", "-t.5"]).timedays == ".5"
    assert parser.parse_args(["plotestimators", "-fpng", "Te"]).format == "png"

    # a joined choice that starts with "-" must not reach argparse as a separate flag
    assert parser.parse_args(["inputmodel", "makeartismodel1dslicefromcone", "-axis-z"]).axis == "-z"
    # "=" gives a list option one value alone, thus a joined negative number stays separate from the flag
    assert parser.parse_args(["plotspectra", "-plotviewingangle-1", "0"]).plotviewingangle == [-1, 0]

    # argparse lets the last flag of a group of switches take a value
    args = parser.parse_args(["plotspectra", "-qo", "/plots/x.pdf"])
    assert args.quiet
    assert args.outputfile == Path("/plots/x.pdf")
    assert parser.parse_args(["timesteps", "-qt300"]).timedays == 300

    # the top level must leave a flag of a subcommand to that subcommand, although it declares -h
    assert parser.parse_args(["hesma", "plotspectrum", "-hesmafile", "x.dat"]).hesmafile == [Path("x.dat")]

    # argparse read "-hesmafile" as -h and printed the help with exit status 0. The other names read as -x _e,
    # -plot _hesma_model, and -o ~, which made a folder with the name "~"
    for argsraw in (
        ["plotspectra", "-hesmafile", "x.dat"],
        ["deposition", "-vmax", "0.3"],
        ["plotestimators", "-x_e", "0.5", "Te"],
        ["plotestimators", "-plot_hesma_model", "x.dat"],
        ["plotspectra", "-o~/plots/x.pdf"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(argsraw)
        assert excinfo.value.code == 2
        assert f"{argsraw[1]} is not an argument of this command" in capsys.readouterr().err


def test_a_name_that_starts_with_a_flag_of_one_letter_writes_nothing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A name that starts with -o must stop the command, and not read as -o with a joined value.

    argparse read "-outputfol" as "-o utputfol", and plotdensity wrote its plot to a folder of that name.
    """
    import artistools.__main__

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotdensity", str(modelpath), "-outputfol", "--quiet"])

    assert "-outputfol is not an argument of this command" in capsys.readouterr().err
    assert not list(tmp_path.iterdir()), "a command that stops must write nothing"

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotdensity", str(modelpath), "-outputfolder", "foo"])

    assert "-outputfolder is not an argument of this command" in capsys.readouterr().err

    # the user gave the start of a longer name, thus the help line names that name and not -d
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["inputmodel", "makeartismodel", "-dim", "3"])

    # a further -dim flag of that command joins the line, thus the test reads the name alone
    assert "Did you mean -dimensionreduce" in capsys.readouterr().err


def test_every_output_argument_records_what_the_command_writes() -> None:
    """addarg_output records a kind, thus the dispatcher keeps the promise of -o for every command.

    Two helpers held two contracts: one promised that a path with no file extension names a folder,
    which the command creates, and the other promised nothing. 17 modules never applied either rule,
    and "inputmodel to_tardis -o newfolder" stopped with FileNotFoundError.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()

    modulebycommand: dict[str, str] = {}

    def walktree(tree: dict[str, t.Any]) -> None:
        for name, node in tree.items():
            if isinstance(node, at.commands.CommandSpec):
                modulebycommand[name] = node.module
            else:
                walktree(node)

    walktree(at.commands.subcommandtree)

    withoutput = 0
    for subcommand, subparser in get_every_subcommand(parser):
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
    at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(merged))

    assert merged.is_file(), f"the merged pdf must keep its name, but {list(tmp_path.iterdir())}"
    assert not list(tmp_path.glob("plotradfield_*.pdf")), "the merge takes the frames away"

    # a -o path that names a folder still gives the merged pdf the name of its frames
    outfolder = tmp_path / "rf"
    at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(outfolder))
    assert list(outfolder.glob("plotradfield_*-plotradfield_*.pdf")), f"no merged pdf in {list(outfolder.iterdir())}"


def test_the_product_keeps_its_name_when_one_frame_holds_data(tmp_path: Path) -> None:
    """A run that holds data for one frame must still write the product that -o named.

    combine_frames took that frame for the product and left the named path empty, thus --open opened
    the frame. A product that carries the name of a frame also went, because the merge removes its
    inputs.
    """
    # the test model holds no radiation field data before timestep 10, thus one frame comes of the two
    merged = tmp_path / "merged.pdf"
    at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", outputfile=str(merged))
    assert merged.is_file(), f"the product must keep its name, but {list(tmp_path.iterdir())}"

    # the name of the product can be the name that a frame would take
    likeaframe = tmp_path / "plotradfield_cell00000_ts040.pdf"
    at.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(likeaframe))
    assert likeaframe.is_file(), f"the merge must not remove its own product: {list(tmp_path.iterdir())}"


def test_a_missing_optional_package_gives_no_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """import_optional names the command that installs a package, thus a traceback adds nothing.

    The handler of the dispatcher took an AssertionError, a FileNotFoundError, and a ValueError, thus
    a command that needs pypdf or imageio printed a traceback above that message.
    """
    import artistools.__main__

    def raise_missing(args: argparse.Namespace) -> None:  # ruff:ignore[unused-function-argument]
        at.misc.import_optional("nosuchpackage")

    with mock.patch.object(at.timesteps, "main", raise_missing), pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["timesteps", "-modelpath", str(modelpath)])

    assert exitinfo.value.code == 1
    message = capsys.readouterr().err
    assert "This command needs nosuchpackage" in message
    assert "Traceback" not in message


def test_writecomparisondata_rejects_an_empty_timestep_list(tmp_path: Path) -> None:
    """An empty timestep list wrote files that hold a header and no data."""
    with pytest.raises(ValueError, match="selected_timesteps"):
        at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[])


def test_ionfrac_header_counts_the_stages_from_neutral(tmp_path: Path) -> None:
    """The format labels the neutral stage 0, thus the stages below the lowest ion of an element hold zero.

    The header gave each column its ARTIS ion stage, which is 1 for the neutral stage, thus Co II took the label co2.
    """
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=list(range(10))
    )

    ionfracfiles = sorted(tmp_path.glob("ionfrac_*_artisnebular.txt"))
    assert ionfracfiles, "the run wrote no ion fraction file"

    elementlist = at.atomic.get_composition_data(modelpath)
    lowermost_of_elsymbol = {
        at.get_elsymbol(row["Z"]).lower(): row["lowermost_ion_stage"] for row in elementlist.iter_rows(named=True)
    }
    assert any(lowermost > 1 for lowermost in lowermost_of_elsymbol.values())

    for ionfracfile in ionfracfiles:
        elsymbol = ionfracfile.name.split("_")[1]
        lines = ionfracfile.read_text(encoding="utf-8").splitlines()
        nstages = int(next(line for line in lines if line.startswith("#NSTAGES:")).split()[1])
        headers = [line for line in lines if line.startswith("#vel_mid")]
        assert headers
        for header in headers:
            assert header.split()[1:] == [f"{elsymbol}{stage}" for stage in range(nstages)]

        datarows = [line.split()[1:] for line in lines if not line.startswith("#")]
        assert datarows
        for row in datarows:
            assert len(row) == nstages
            assert all(float(value) == 0.0 for value in row[: lowermost_of_elsymbol[elsymbol] - 1])


def test_writecomparisondata_keeps_a_tiny_ion_fraction(tmp_path: Path) -> None:
    """An ion fraction below 1e-38 must keep its digits.

    The estimator cache stores a population as Float32. A Float32 ratio below 1e-38 is subnormal, thus it lost
    digits or became zero. The old reader divided Python floats, and the files of the two readers differed.
    """
    from artistools.writecomparisondata import write_ionfracts

    dfestimators = pl.DataFrame(
        {"timestep": [0], "vel_r_mid": [1e9], "nnelement_Fe": [1e8], "nnion_Fe_II": [1e-37]},
        schema={"timestep": pl.Int32, "vel_r_mid": pl.Float64, "nnelement_Fe": pl.Float32, "nnion_Fe_II": pl.Float32},
    )
    write_ionfracts(modelpath, "tiny", [0], dfestimators, tmp_path)

    datalines = [
        line
        for line in (tmp_path / "ionfrac_fe_tiny_artisnebular.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    expected = float(np.float32(1e-37)) / float(np.float32(1e8))
    # the columns are the velocity, then the stages from the neutral stage, thus Fe II is the third value
    assert float(datalines[0].split()[2]) == pytest.approx(expected, rel=1e-4, abs=0.0)


def test_completions_writes_the_code_to_a_redirect(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--quiet and a redirect must get the code, and a terminal must get the instructions.

    --quiet sent the code to the null device, thus "completions zsh --quiet > file" wrote an empty file. An old
    script runs "completions > file" and sources that file, thus a redirect with no shell gets the code too.
    """
    import artistools.__main__
    from artistools.completions import get_completion_code

    monkeypatch.setenv("SHELL", "/bin/zsh")
    expected = get_completion_code("zsh")

    artistools.__main__.main(argsraw=["completions", "zsh", "--quiet"])
    assert capsys.readouterr().out.strip() == expected.strip()

    # the captured standard output is no terminal, as a redirect to a file is not
    artistools.__main__.main(argsraw=["completions"])
    assert capsys.readouterr().out.strip() == expected.strip()

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    artistools.__main__.main(argsraw=["completions"])
    captured = capsys.readouterr()
    assert not captured.out
    assert "To enable tab completion in zsh" in captured.err


def test_writecomparisondata_edep_takes_the_next_timestep(tmp_path: Path) -> None:
    """ARTIS writes the deposition of timestep n into the estimators of timestep n + 1.

    The edep file took the row of timestep n, thus it gave the deposition of the timestep before, and it
    disagreed with the lbol_edep file of the same export by about 5 % at 200 days.
    """
    at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[50])

    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    nextrow = at.scan_estimators(modelpath=modelpath, timestep=(51,)).select("total_dep").collect()
    assert float(datalines[0].split()[1]) == pytest.approx(nextrow.item(), rel=1e-4, abs=0.0)

    # the run wrote no timestep after the last one, thus the file gives zero there and the export goes on
    lasttimestep = len(at.get_timestep_times(modelpath)) - 1
    at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[lasttimestep])
    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    assert float(datalines[0].split()[1]) == 0.0


def test_linefluxes_refuse_overlapping_time_bins() -> None:
    """Overlapping bins gave each shared interval to the last bin, and each bin still divided by its full width.

    Thus the bins [100, 200) and [150, 250) gave the first bin the packets of 100 to 150 days alone.
    """
    with pytest.raises(ValueError, match="time bins overlap"):
        at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 150.0], [200.0, 250.0])

    # bins that meet at an edge are no overlap, in either order
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 200.0], [200.0, 300.0])
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [200.0, 100.0], [300.0, 200.0])


def test_linefluxes_emitting_regions_give_one_file_for_each_time_bin(tmp_path: Path) -> None:
    """Each time bin of the emitting regions has its own figure, thus two bins can overlap.

    The command wrote each figure to the one name that -o gives, thus only the last one stayed.
    """
    # the test data holds no floers_te_nne.json, thus one reference point stands in for it
    refdata = (["5"], np.array([5.0]), [{"ne": [5.0], "temp": [5000.0]}])
    with mock.patch.object(at.plotlinefluxes, "read_te_nne_refdata", return_value=refdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            use_lastemissiontype=True,
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.pdf",
        )

    assert sorted(path.name for path in tmp_path.glob("*.pdf")) == ["emreg_5.0d.pdf", "emreg_6.0d.pdf"]


def test_viewer_status_line_gives_the_error() -> None:
    """The status line gives the error of argparse, and not the usage line that argparse prints before it."""
    stderr = "usage: artistools [options] [specpath ...]\nerror: argument -xmin: invalid float value: 'abc'\nhelp: -h"
    assert viewertools.get_first_line(stderr) == "argument -xmin: invalid float value: 'abc'"
    assert viewertools.get_first_line("A file is missing\nThe second line") == "A file is missing"


def test_viewer_queue_moves_a_clamped_control_back() -> None:
    """A handler that clamps a control to the values of the plot gives unchanged values, and the control must move back.

    The queue returned before it showed the values, thus a slider stayed at a position that the plot did not show.
    """
    # PySide6 is an optional dependency. CI does not install it for each Python version, and a CI machine with no
    # libEGL.so.1 raises an ImportError that is not a ModuleNotFoundError
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    viewer = mock.Mock(values=5)
    showvalues = mock.Mock()
    queue = viewertools.DrawQueue(mock.Mock(), viewer, mock.Mock(), showvalues, mock.Mock())
    queue.apply(5)
    showvalues.assert_called_once_with()
    assert queue.requestedvalues is None, "unchanged values must draw no plot"


def test_viewer_open_model_gives_the_reason_of_the_new_window() -> None:
    """The status line of Open Model must give the error of the first plot of the new window.

    open_window returned only a bool, thus the message took the first line of the traceback on stderr.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

    def open_window(tokens: Sequence[str], windows: Sequence[object]) -> str:
        sys.stderr.write("Traceback (most recent call last):\n")
        return f"ComputeError: the query failed for {tokens[0]} and {len(windows)} window"

    with mock.patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="mymodel"):
        message = viewertools.open_model_window(mock.Mock(), open_window, [mock.Mock()])
    assert message == "The viewer cannot open mymodel: ComputeError: the query failed for mymodel and 1 window"


def test_viewer_typed_centre_gives_back_the_range() -> None:
    """The centre that the time field shows gives back the same range of timesteps, also for an even count.

    The viewer took the timestep that holds the centre as the middle, and an even range then moved one timestep.
    """
    tmids = at.get_timestep_times(at.get_path("testdata") / "testmodel", loc="mid")
    for count in (1, 2, 3, 4):
        for start in range(len(tmids) - count + 1):
            centre = float(f"{(tmids[start] + tmids[start + count - 1]) / 2.0:.4g}")
            assert viewertools.get_nearest_range_start(tmids, centre, count) == start, (count, start)


def test_viewer_option_rows_split_a_group_of_switches() -> None:
    """Argparse reads -qv as -q and -v, and the option table must give each switch its own row.

    The table read -qv as the flag -q with the value "v", and the command then held a stray positional argument.
    """
    parser = viewertools.make_parser(at.estimators.addargs)
    rows, othertokens = viewertools.split_option_rows(parser, ["Te", "mymodel", "-qv", "-qt300", "-xmin", "5"])
    assert rows == (("--quiet", ()), ("--verbose", ()), ("--quiet", ()), ("-timedays", ("300",)), ("-xmin", ("5",)))
    assert othertokens == ["Te", "mymodel"]
