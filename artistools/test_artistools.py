import argparse
import contextlib
import dataclasses as dc
import hashlib
import importlib
import inspect
import io
import itertools
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import tomllib
import typing as t
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.legend as mpllegend
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.container import ErrorbarContainer

import artistools as at
from artistools.misc.remote import model_path_from_text
from artistools.viewertools import application as viewerapplication
from artistools.viewertools import core as viewercore
from artistools.viewertools import menus as viewermenus
from artistools.viewertools import sections as viewersections
from artistools.viewertools import widgets as viewerwidgets
from artistools.viewertools import window as viewerwindow

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
    """The old path at.transitions.get_lte_partfunc gives the same value and a warning. pynonthermal reads it."""
    dflevels = pl.DataFrame({"g": [2.0, 4.0], "energy_ev": [0.0, 1.0]})
    expected = at.atomic.get_lte_partfunc(dflevels, 5000.0)
    assert np.isclose(expected, 2.0 + 4.0 * math.exp(-1.0 / (at.constants.K_B_ev_per_K * 5000.0)))

    with pytest.warns(DeprecationWarning, match="artistools.atomic.get_lte_partfunc"):
        assert np.isclose(at.transitions.get_lte_partfunc(dflevels, 5000.0), expected)


def test_external_packages_find_their_names() -> None:
    """The packages pynonthermal and artisatomic read these names, thus a release that moves one stops them."""
    import importlib

    externalnames = {
        "artistools": ["get_composition_data", "get_ionstring"],  # artisatomic, pynonthermal
        "artistools.atomic": ["get_levels"],  # pynonthermal
        "artistools.transitions": ["get_lte_partfunc"],  # pynonthermal
    }
    for modulename, names in externalnames.items():
        module = importlib.import_module(modulename)
        assert not [name for name in names if not callable(getattr(module, name, None))], modulename

    assert at.get_composition_data is at.atomic.get_composition_data


TOPLEVEL_API: t.Final[frozenset[str]] = frozenset({
    "add_derived_cols_to_modeldata", "decode_roman_numeral", "firstexisting", "get_atomic_number",
    "get_composition_data", "get_deposition", "get_elsymbol", "get_inputparams", "get_ion_tuple", "get_ionstring",
    "get_model_name", "get_modeldata", "get_nprocs", "get_path", "get_timestep_of_timedays", "get_timestep_times",
    "get_z_a_nucname", "scan_estimators", "zopen",
})  # fmt: skip


def test_top_level_api_is_the_documented_list() -> None:
    """The top level holds the names that a user types in a script, and the README lists each of them.

    A different name stays in its package, e.g. at.misc.addarg_modelpath. To add a name to the top level,
    add it here and to the table in the README.
    """
    import types

    # getattr and not vars: on Python 3.15 an entry of vars is a lazy proxy until its first use
    public = {
        name for name in vars(at) if not name.startswith("_") and not isinstance(getattr(at, name), types.ModuleType)
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


def test_residuals_take_the_reference_at_each_model_point() -> None:
    """Keep the model points inside the panel and reference ranges, with gaps for non-finite values."""
    model = at.plottools.ResidualSeries(
        "model", np.array([0.0, 5.0, 10.0, 12.0, 15.0, 20.0]), np.array([0.0, 10.0, 20.0, 24.0, 30.0, 40.0]), "C0"
    )
    reference = at.plottools.ResidualSeries(
        "obs",
        np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        np.array([1.0, 1.0, 11.0, 22.0, 33.0, 42.0, 50.0]),
        "k",
    )
    inrange, residual, yreference = at.plottools.get_residuals(reference, model, xmin=0.0, xmax=14.0)
    assert inrange.tolist() == [True, True, True, True, False, False]
    assert np.allclose(yreference, [1.0, 11.0, 22.0, 26.4])
    assert np.allclose(residual, [-1.0, -1.0, -2.0, -2.4])

    masked = reference._replace(y=np.array([1.0, 1.0, np.nan, 22.0, 33.0, 42.0, 50.0]))
    inrange, residual, _ = at.plottools.get_residuals(masked, model, xmin=0.0, xmax=20.0)
    assert inrange.all()
    assert np.isnan(residual[1])

    gapmodel = model._replace(y=np.array([0.0, 10.0, np.inf, 24.0, 30.0, 40.0]))
    _, residual, _ = at.plottools.get_residuals(reference, gapmodel, xmin=0.0, xmax=20.0)
    assert np.isnan(residual[2])
    assert np.isfinite(residual[[0, 1, 3, 4, 5]]).all()

    fig, axis = plt.subplots()
    dfstats = at.plottools.plot_residual_panel(axis, [masked, model], 0.0, 20.0)
    expected = np.array([-1.0, -2.0, -2.4, -3.0, -2.0])
    assert dfstats["npoints"].item() == 5
    assert np.isclose(dfstats["rms"].item(), np.sqrt(np.mean(expected**2)))
    assert np.isclose(dfstats["rms_relative"].item(), dfstats["rms"].item() / np.mean([1.0, 22.0, 26.4, 33.0, 42.0]))
    assert np.allclose(np.asarray(axis.lines[0].get_ydata())[[0, 2, 3, 4, 5]], expected)
    plt.close(fig)

    fig, ratioaxis = plt.subplots()
    at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, residualtype="relative")
    assert np.allclose(
        np.asarray(ratioaxis.lines[0].get_ydata())[[0, 2, 3, 4, 5]],
        [0.0, 20.0 / 22.0, 24.0 / 26.4, 30.0 / 33.0, 40.0 / 42.0],
    )
    dfmagstats = at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, ismagnitude=True)
    assert dfmagstats["rms_relative"].item() is None
    plt.close(fig)


@pytest.mark.parametrize("referencex", [[], [np.nan, np.inf, -np.inf], [2.0], [np.nan, 2.0, np.inf]])
@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_accepts_exact_matches_to_a_single_baseline_point(referencex: list[float], ratio: bool) -> None:
    """Keep exact matches to one finite baseline point and exclude other x values."""
    x = np.array([1.0, np.nextafter(2.0, 0.0), 2.0, 2.0, np.nextafter(2.0, 3.0), 3.0, np.nan])
    model = at.plottools.ResidualSeries("model", x, np.array([1.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0]), "C0")
    reference = at.plottools.ResidualSeries(
        "baseline", np.asarray(referencex, dtype=np.float64), np.full(len(referencex), 4.0), "k"
    )
    inrange, residual, yreference = at.plottools.get_residuals(reference, model, xmin=1.0, xmax=3.0)
    fig, axis = plt.subplots()
    stats = at.plottools.plot_residual_panel(
        axis, [reference, model], 1.0, 3.0, residualtype="relative" if ratio else "absolute"
    )
    if np.isfinite(reference.x).any():
        assert inrange.tolist() == [False, False, True, True, False, False, False]
        assert np.allclose(residual, [2.0, 4.0])
        assert np.allclose(yreference, [4.0, 4.0])
        assert stats["npoints"].item() == 2
        assert np.isclose(stats["rms"].item(), np.sqrt(10.0))
        assert np.isclose(stats["rms_relative"].item(), np.sqrt(10.0) / 4.0)
        assert np.allclose(axis.lines[0].get_xdata(), [2.0, 2.0])
        assert np.allclose(axis.lines[0].get_ydata(), [1.5, 2.0] if ratio else [2.0, 4.0])
        outside, outside_residual, _ = at.plottools.get_residuals(reference, model, xmin=2.1, xmax=3.0)
        assert not outside.any()
        assert outside_residual.size == 0
    else:
        assert not inrange.any()
        assert residual.size == yreference.size == 0
        assert stats.is_empty()
    plt.close(fig)


@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_keeps_the_resolution_and_line_properties(ratio: bool) -> None:
    """A sparse reference keeps each comparison point and the line properties of the main frame."""
    x = np.linspace(-1.0, 9.0, 1001)
    yreference = np.interp(x, [0.0, 4.0, 8.0], [2.0, 6.0, 18.0])
    y = yreference + np.sin(x)
    fig, (mainaxis, residualaxis) = plt.subplots(2)
    (mainline,) = mainaxis.plot(
        x,
        y,
        color="purple",
        linestyle=(1, (2, 3)),
        linewidth=2.5,
        marker="s",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=2,
        alpha=0.4,
        drawstyle="steps-mid",
    )
    reference = at.plottools.ResidualSeries("baseline", np.array([8.0, 0.0, 4.0]), np.array([18.0, 2.0, 6.0]), "k")
    model = at.plottools.ResidualSeries("comparison", x, y, mainline.get_color(), line=mainline)
    stats = at.plottools.plot_residual_panel(
        residualaxis, [reference, model], 1.0, 7.0, residualtype="relative" if ratio else "absolute"
    )
    inrange = (x >= 1.0) & (x <= 7.0)
    line = residualaxis.lines[0]
    assert np.allclose(line.get_xdata(), x[inrange])
    assert np.allclose(line.get_ydata(), (y / yreference if ratio else y - yreference)[inrange])
    assert stats["npoints"].item() == int(inrange.sum())
    assert line.get_transform() == residualaxis.transData
    assert line.get_transform() != mainline.get_transform()
    assert np.allclose(line.get_clip_box().bounds, residualaxis.bbox.bounds)
    assert line.get_color() == mainline.get_color()
    assert line.get_linestyle() == mainline.get_linestyle()
    assert np.isclose(line.get_linewidth(), mainline.get_linewidth())
    assert line.get_marker() == mainline.get_marker()
    assert np.isclose(line.get_markersize(), mainline.get_markersize())
    assert line.get_markerfacecolor() == mainline.get_markerfacecolor()
    assert np.isclose(line.get_markeredgewidth(), mainline.get_markeredgewidth())
    assert np.isclose(line.get_alpha(), mainline.get_alpha())
    assert line.get_drawstyle() == mainline.get_drawstyle()
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize("baselineindex", [0, 1, 2])
@pytest.mark.parametrize("selectedindices", [None, [], [0, 2], [2, 2], [1], [2, 0]])
@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_uses_the_selected_baseline(
    baselineindex: int, selectedindices: list[int] | None, ratio: bool
) -> None:
    """Each other series uses the selected baseline, its points, and its own colour."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 8.0])
    factors = [1.0, 2.0, 4.0]
    series = [
        at.plottools.ResidualSeries(f"series {index}", x, y * factor, f"C{index}")
        for index, factor in enumerate(factors)
    ]
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(argparse.Namespace(logscaley=ratio))
    mainaxis.set_xlim(1.0, 3.0)
    dfstats = at.plottools.draw_residual_panel(
        residualaxis,
        mainaxis,
        series,
        argparse.Namespace(
            residualbaselineseries=baselineindex,
            residuals=selectedindices,
            logscaley=ratio,
            residualtype="relative" if ratio else "absolute",
        ),
    )
    otherindices = [
        index
        for index in range(len(series))
        if index != baselineindex and (not selectedindices or index in selectedindices)
    ]
    assert dfstats["model"].to_list() == [series[index].label for index in otherindices]
    assert dfstats["reference"].to_list() == [series[baselineindex].label] * len(otherindices)
    assert len(residualaxis.lines) == len(otherindices) + 1
    for line, index in zip(residualaxis.lines[:-1], otherindices, strict=True):
        expected = (
            np.full_like(y, factors[index] / factors[baselineindex])
            if ratio
            else y * (factors[index] - factors[baselineindex])
        )
        assert np.allclose(line.get_ydata(), expected)
        assert line.get_color() == series[index].color
    expectedrms = [
        float(np.sqrt(np.mean((y * (factors[index] - factors[baselineindex])) ** 2))) for index in otherindices
    ]
    assert np.allclose(dfstats["rms"].to_numpy(), expectedrms)
    plt.close(fig)


@pytest.mark.parametrize(("seriescount", "baselineindex"), [(0, 0), (1, 0), (2, -1), (2, 2)])
def test_residual_panel_rejects_an_invalid_baseline(seriescount: int, baselineindex: int) -> None:
    """The panel needs two series and an index inside their range."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(index), x, x, "k") for index in range(seriescount)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, baselineindex=baselineindex)
    plt.close(fig)


@pytest.mark.parametrize("addargs", [at.spectra.plotspectra.addargs, at.lightcurve.plotlightcurve.addargs])
@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([], None),
        (["-residualbaselineseries"], 0),
        (["-residualbaselineseries", "2"], 2),
        (["-residualbaselineseries=2"], 2),
        (["-residuals"], 0),
        (["-residuals", "0"], 0),
        (["-residuals", "2"], 2),
        (["-res"], 0),
        (["-res", "2"], 2),
        (["--residuals"], 0),
        (["--residuals", "1"], 0),
    ],
)
def test_residual_option_takes_an_optional_index(
    addargs: Callable[[argparse.ArgumentParser], None], tokens: list[str], expected: int | None
) -> None:
    """Both commands take an optional baseline index and keep the old spelling as a hidden alias."""
    parser = argparse.ArgumentParser()
    addargs(parser)
    assert parser.parse_args(tokens).residualbaselineseries == expected
    assert "-residualbaselineseries [INDEX]" in parser.format_help()
    assert "-residuals [INDEX]" not in parser.format_help()
    assert "--residuals" not in parser.format_help()
    assert "-res [INDEX]" not in parser.format_help()
    action = viewercore.get_actions_by_flag(parser)["-residualbaselineseries"]
    assert viewercore.get_default_tokens(action) == ()
    assert viewercore.get_option_kind(action) == "text"


@pytest.mark.parametrize("command", ["plotspectra", "plotlightcurves"])
@pytest.mark.parametrize(
    ("flag", "indexed"),
    [
        ("-residualbaselineseries", False),
        ("-residualbaselineseries", True),
        ("-residuals", False),
        ("-residuals", True),
        ("-res", False),
        ("-res", True),
        ("--residuals", False),
    ],
)
def test_residual_option_keeps_the_next_positional_path(command: str, flag: str, indexed: bool) -> None:
    """A bare residual flag keeps the next path positional, and an integer selects the baseline."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    args = parser.parse_args([command, flag, *(["1"] if indexed else []), "model1", "model2", "--quiet"])
    assert args.residualbaselineseries == (1 if indexed else 0)
    paths = args.specpath if command == "plotspectra" else args.modelpath
    assert [str(path) for path in paths] == ["model1", "model2"]


@pytest.mark.parametrize("command", ["plotspectra", "plotlightcurves"])
@pytest.mark.parametrize("indices", [[], [1], [2, 0]])
def test_residual_selection_keeps_the_next_positional_paths(command: str, indices: list[int]) -> None:
    """The residual option takes only indices and keeps the paths in their original order."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    args = parser.parse_args([command, "model0", "-residual", *map(str, indices), "model1", "model2", "--quiet"])
    assert args.residuals == indices
    assert args.residualbaselineseries is None
    paths = args.specpath if command == "plotspectra" else args.modelpath
    assert [str(path) for path in paths] == ["model0", "model1", "model2"]
    subparser = argparse.ArgumentParser()
    addargs = at.spectra.plotspectra.addargs if command == "plotspectra" else at.lightcurve.plotlightcurve.addargs
    addargs(subparser)
    action = viewercore.get_actions_by_flag(subparser)["-residual"]
    assert viewercore.get_option_kind(action) == "list"
    assert viewercore.get_default_tokens(action) == ()


@pytest.mark.parametrize("command", ["plotspectra", "plotlightcurves"])
@pytest.mark.parametrize("selection", [["-residual"], ["-residual", "1"], ["-residual=1"]])
@pytest.mark.parametrize("earlieroptions", [[], ["-t", "300"]])
@pytest.mark.parametrize("pathkind", ["folder", "file"])
def test_residual_selection_joins_paths_before_options(
    command: str, selection: list[str], earlieroptions: list[str], pathkind: str, tmp_path: Path
) -> None:
    """Residual paths join the first positional group, with a separator or a joined selection."""
    import artistools.__main__

    paths = [tmp_path / "model0", tmp_path / "model1"]
    for path in paths:
        if pathkind == "folder":
            path.mkdir()
            (path / "input.txt").touch()
            (path / "model.txt").touch()
        else:
            path.touch()
    tokens = [command, str(paths[0]), *earlieroptions, *selection, str(paths[1])]
    tokens = at.misc.separate_trailing_folders(tokens)
    if pathkind == "folder" and selection == ["-residual", "1"]:
        assert "--" in tokens
    args = artistools.__main__.build_parser().parse_args(tokens)
    assert args.residuals == ([] if selection == ["-residual"] else [1])
    assert args.residualtype == "relative"
    assert list(map(str, args.specpath if command == "plotspectra" else args.modelpath)) == list(map(str, paths))


@pytest.mark.parametrize("index", [-1, 2])
def test_residual_panel_rejects_an_invalid_selection(index: int) -> None:
    """The panel rejects a selected index outside the range of the series."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(i), x, x, "k") for i in range(2)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, selectedindices=[index])
    assert not axis.lines
    plt.close(fig)


def test_residual_path_rewrite_keeps_an_ambiguous_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    """The path rewrite leaves an ambiguous prefix for argparse to reject."""
    parser = at.commands.SuggestingArgumentParser()
    at.misc.addarg_residuals(parser)
    parser.add_argument("-reset", action="store_true")
    parser.add_argument("paths", nargs="*")
    assert parser.split_joined_flags(["-resi", "model1", "model2"]) == ["-resi", "model1", "model2"]
    with pytest.raises(SystemExit):
        parser.parse_args(["-resi", "model1", "model2"])
    assert "ambiguous option: -resi" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["plotspectra", "plotlightcurves"])
def test_old_residual_flag_keeps_a_numeric_model_path(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old flag takes no value, so a numeric model path stays positional."""
    import artistools.__main__

    (tmp_path / "1").mkdir()
    monkeypatch.chdir(tmp_path)
    args = artistools.__main__.build_parser().parse_args([command, "--residuals", "1"])
    assert args.residualbaselineseries == 0
    paths = args.specpath if command == "plotspectra" else args.modelpath
    assert [str(path) for path in paths] == ["1"]


@pytest.mark.parametrize("modelfactor", [2.0, 100.0, 0.01])
@pytest.mark.parametrize("logscaley", [False, True])
@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_sets_the_calculation_and_scale(modelfactor: float, logscaley: bool, residualtype: str) -> None:
    """The residual type sets the values and the y scale independently of the main frame."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    yreference = np.array([1.0, 2.0, 4.0, 8.0])
    factors = np.array([1.0, 1.0, modelfactor, modelfactor])
    series = [
        at.plottools.ResidualSeries("obs", x, yreference, "k"),
        at.plottools.ResidualSeries("model", x, yreference * factors, "C0"),
    ]
    args = argparse.Namespace(logscaley=logscaley, residualbaselineseries=0, residuals=None, residualtype=residualtype)
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.plot(x, series[0].y)
    mainaxis.set_yscale("log" if logscaley else "linear")
    mainaxis.set_ylabel("Flux [erg]")
    stats = at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args)
    expected = yreference * (factors - 1.0) if residualtype == "absolute" else factors
    assert residualaxis.get_yscale() == ("log" if residualtype == "relativelog" else "linear")
    assert residualaxis.get_ylabel() == (
        "series $-$ baseline\n[erg]" if residualtype == "absolute" else "series / baseline"
    )
    assert np.allclose(residualaxis.lines[0].get_ydata(), expected)
    assert np.allclose(residualaxis.lines[-1].get_ydata(), 0.0 if residualtype == "absolute" else 1.0)
    assert np.isclose(stats["rms"].item(), np.sqrt(np.mean((series[1].y - yreference) ** 2)))
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
@pytest.mark.parametrize("zeropoint", [0.0, 30.0])
def test_magnitude_residuals_keep_differences(residualtype: str, zeropoint: float) -> None:
    """Magnitude residuals keep their differences, scale, and error bars when the zero point changes."""
    x = np.array([1.0, 2.0])
    ybaseline = np.array([-10.0, -8.0]) + zeropoint
    yerr = np.array([[0.3, 0.4], [0.5, 0.6]])
    series = [
        at.plottools.ResidualSeries("baseline", x, ybaseline, "k"),
        at.plottools.ResidualSeries("model", x, ybaseline + 2.0, "r", yerr=yerr),
    ]
    args = argparse.Namespace(logscaley=False, residualbaselineseries=0, residuals=None, residualtype=residualtype)
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.set_xlim(1.0, 2.0)
    mainaxis.set_ylabel("Magnitude [mag]")
    with mock.patch.object(residualaxis, "errorbar", wraps=residualaxis.errorbar) as mockerrorbar:
        stats = at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args, ismagnitude=True)
    assert np.allclose(residualaxis.lines[0].get_ydata(), 2.0)
    assert np.allclose(residualaxis.lines[-1].get_ydata(), 0.0)
    assert np.allclose(mockerrorbar.call_args.kwargs["yerr"], yerr)
    assert residualaxis.get_yscale() == "linear"
    assert residualaxis.yaxis_inverted()
    assert residualaxis.get_ylabel() == "series $-$ baseline\n[mag]"
    assert np.isclose(stats["rms"].item(), 2.0)
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_keeps_gaps_for_undefined_values(residualtype: str) -> None:
    """Zero baselines give gaps in ratios, and a logarithmic axis also excludes non-positive ratios."""
    x = np.arange(5, dtype=np.float64)
    baseline = at.plottools.ResidualSeries("baseline", x, np.array([1.0, -2.0, 0.0, 2.0, np.nan]), "k")
    model = at.plottools.ResidualSeries("model", x, np.array([2.0, 4.0, 3.0, 0.0, 2.0]), "C0")
    expected = {
        "absolute": [1.0, 6.0, 3.0, -2.0, np.nan],
        "relative": [2.0, -2.0, np.nan, 0.0, np.nan],
        "relativelog": [2.0, np.nan, np.nan, np.nan, np.nan],
    }
    fig, axis = plt.subplots()
    stats = at.plottools.plot_residual_panel(axis, [baseline, model], 0.0, 4.0, residualtype=residualtype)
    assert np.allclose(axis.lines[0].get_ydata(), expected[residualtype], equal_nan=True)
    assert stats["npoints"].item() == 4
    assert np.isclose(stats["rms"].item(), np.sqrt(np.mean(np.array([1.0, 6.0, 3.0, -2.0]) ** 2)))
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_scales_asymmetric_error_bars(residualtype: str) -> None:
    """Ratio error bars use the baseline magnitude and exchange their sides for a negative baseline."""
    x = np.array([1.0, 2.0])
    baseline = at.plottools.ResidualSeries("baseline", x, np.array([2.0, -4.0]), "k")
    model = at.plottools.ResidualSeries(
        "model", x, np.array([4.0, -8.0]), "C0", yerr=np.array([[0.4, 0.8], [0.8, 1.6]])
    )
    fig, axis = plt.subplots()
    at.plottools.plot_residual_panel(axis, [baseline, model], 1.0, 2.0, residualtype=residualtype)
    bars = axis.containers[0]
    assert isinstance(bars, ErrorbarContainer)
    segments = bars.lines[2][0].get_segments()
    expected = [[1.6, 2.8], [-4.8, -2.4]] if residualtype == "absolute" else [[1.8, 2.4], [1.6, 2.2]]
    assert np.allclose([segment[:, 1] for segment in segments], expected)
    plt.close(fig)


def test_residual_panel_rejects_an_unknown_type() -> None:
    """The panel rejects a residual type that it cannot calculate."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(index), x, x, "k") for index in range(2)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, residualtype="unknown")
    assert not axis.lines
    plt.close(fig)


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
                # a model path is a Path, and a remote path follows the rule of rsync
                assert action.type is model_path_from_text, label
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
    # nine bins of this cell have no fit (W < 0). Their model takes the full-spectrum fit, as ARTIS radfield() does,
    # and their band average takes J. Both were zero before
    assert np.isclose(fitted_yarr.mean(), 45.14912462715829, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(fitted_yarr, fitted_xarr)), 492291.22990819113, rtol=1e-4)

    bandavg_xarr, bandavg_yarr = get_plot_xy(mockstep.call_args_list[0])
    assert np.isclose(bandavg_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(bandavg_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(bandavg_yarr.mean(), 43.854670073915386, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(bandavg_yarr, bandavg_xarr)), 476000.8181544015, rtol=1e-4)


def test_radfield_bin_with_no_fit_takes_the_full_spectrum_fit() -> None:
    """A bin with a negative W takes the full-spectrum dilute blackbody, as radfield() of ARTIS does.

    The plot drew zero for such a bin, and the band average drew zero although the bin holds its J.
    """
    from artistools.plotradfield import get_binaverage_field
    from artistools.plotradfield import get_fitted_field
    from artistools.plotradfield import j_nu_dbb

    radfielddata = pl.DataFrame({
        "timestep": [5, 5, 5],
        "modelgridindex": [0, 0, 0],
        "bin_num": [-1, 0, 1],
        "nu_lower": [0.0, 5.0e14, 6.0e14],
        "nu_upper": [0.0, 6.0e14, 7.0e14],
        "J": [1.0, 40.0, 30.0],
        "T_R": [6000.0, -1.0, 7000.0],
        "W": [0.5, -1.0, 0.1],
    })

    arr_lambda, j_lambda = get_fitted_field(radfielddata, modelgridindex=0, timestep=5)
    arr_nu = at.constants.c_ang_per_s / np.array(arr_lambda)
    j_nu_fullspec = np.array(j_nu_dbb(arr_nu, 0.5, 6000.0))
    j_nu_bins = np.array(j_nu_dbb(arr_nu, 0.1, 7000.0))
    j_nu = np.array(j_lambda) * np.array(arr_lambda) / arr_nu
    assert np.allclose(j_nu[:200], j_nu_fullspec[:200], rtol=1e-10)
    assert np.allclose(j_nu[200:], j_nu_bins[200:], rtol=1e-10)

    # before FIRST_NLTE_RADFIELD_TIMESTEP, ARTIS takes the full-spectrum fit for every bin
    _, j_lambda_fullspec = get_fitted_field(radfielddata, modelgridindex=0, timestep=5, usebinfits=False)
    assert np.allclose(np.array(j_lambda_fullspec) * np.array(arr_lambda) / arr_nu, j_nu_fullspec, rtol=1e-10)

    _, bandaverage = get_binaverage_field(radfielddata, modelgridindex=0, timestep=5)
    dlambda = at.constants.c_ang_per_s * (1 / 5.0e14 - 1 / 6.0e14)
    assert np.isclose(bandaverage[1], 40.0 / dlambda, rtol=1e-10)


def test_radfield_takes_the_last_timestep_with_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With no timestep, the plot takes the last timestep that holds data, and a run with no plot gives an error.

    The command took the last timestep of the time grid. For a run that stopped early, it wrote no file and ended
    with no error.
    """
    radfielddata = at.plotradfield.read_radfield(modelpath, modelgridindex=[0])
    with mock.patch.object(
        at.plotradfield, "read_radfield", return_value=radfielddata.filter(pl.col("timestep") <= 59)
    ):
        at.plotradfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=tmp_path)
        assert [path.name for path in tmp_path.glob("*.pdf")] == ["plotradfield_cell00000_ts059.pdf"]

        with pytest.raises(SystemExit):
            at.plotradfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, timestep=5, outputfile=tmp_path)
    assert "no radiation field data" in capsys.readouterr().err


def test_plot_commands_write_the_format_of_the_suffix(tmp_path: Path) -> None:
    """The suffix of -o sets the format of the file, e.g. -o rf.png gives a PNG file.

    The commands gave format="pdf" to savefig, thus a file named rf.png held a PDF document.
    """
    pngmagic = b"\x89PNG"
    at.plotradfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, timestep=40, outputfile=tmp_path / "rf.png")
    assert (tmp_path / "rf.png").read_bytes().startswith(pngmagic)

    at.plottransitions.main(argsraw=[], modelpath=modelpath, timedays=300, outputfile=tmp_path / "trans.png")
    assert (tmp_path / "trans.png").read_bytes().startswith(pngmagic)

    at.plotspherical.main(argsraw=[], modelpath=modelpath, timestep=40, outputfile=tmp_path / "map.png")
    assert (tmp_path / "map.png").read_bytes().startswith(pngmagic)


def test_save_figure_takes_the_format_of_the_suffix(tmp_path: Path) -> None:
    """A caller gave format="pdf" to save_figure, thus a file named x.png held a PDF document."""
    for filename, magic in (("x.png", b"\x89PNG"), ("x.pdf", b"%PDF"), ("x", b"%PDF")):
        fig = plt.figure()
        at.plottools.save_figure(fig, tmp_path / filename, format="pdf")
        assert (tmp_path / filename).read_bytes().startswith(magic)


@pytest.mark.benchmark
def test_plotspherical(tmp_path: Path) -> None:
    at.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=tmp_path)

    assert [path.name for path in tmp_path.glob("plotspherical_*.pdf")] == ["plotspherical_256.67-333.82d.pdf"]


def test_plotspherical_gaussian_filter(tmp_path: Path) -> None:
    """-gaussian_sigma must reach smooth_direction_map, which takes the width in phi bins and not in degrees."""
    from artistools.plotspherical import smooth_direction_map

    with mock.patch.object(at.plotspherical, "smooth_direction_map", side_effect=smooth_direction_map) as mocksmooth:
        at.plotspherical.main(argsraw=[], modelpath=modelpath, gaussian_sigma=20, outputfile=tmp_path)

    # one map for each of the three default plot variables, on the default grid of 64 phi bins
    assert mocksmooth.call_count == 3
    for callargs in mocksmooth.call_args_list:
        _, sigma_bins, nphibins = callargs.args
        assert nphibins == 64
        assert np.isclose(sigma_bins, 20 / 360 * 64)


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


def test_plotspherical_smoothing_wraps_phi_and_crosses_each_pole() -> None:
    """The smoothing must blend no pole with the opposite pole.

    The filter wrapped the cos theta axis as it wraps phi. Thus the row at the south pole took the light of
    the row at the north pole.
    """
    from artistools.plotspherical import smooth_direction_map

    ncosthetabins, nphibins = 8, 16
    data = np.zeros((ncosthetabins, nphibins))
    data[-1, 0] = 1.0  # the north polar cap, at phi bin 0

    smoothed = smooth_direction_map(data, 1.0, nphibins)

    assert np.allclose(smoothed[0], 0.0), "the south polar cap must stay dark"
    # phi wraps, thus the last phi bin is a neighbour of phi bin 0
    assert smoothed[-1, -1] > 0.0
    # the bin on the other side of the north pole is at phi + pi in the same row
    assert smoothed[-1, nphibins // 2] > smoothed[-1, nphibins // 4]


@mock.patch.object(mplax.Axes, "pcolormesh", side_effect=mplax.Axes.pcolormesh, autospec=True)
def test_plotspherical_puts_phi_zero_at_the_centre(mockpcolormesh: mock.MagicMock) -> None:
    """The direction phi = 0 (+X) must be at the longitude zero, which is the centre of the map.

    The longitude of the map started at -pi with phi bin 0, thus +X was at the edge and -X at the centre.
    """
    from artistools.plotspherical import plot_spherical

    ncosthetabins, nphibins = 4, 8
    isphibinzero = [phibin == 0 for _ in range(ncosthetabins) for phibin in range(nphibins)]
    dirbins = pl.DataFrame({"count": [1] * len(isphibinzero), "luminosity": [float(x) for x in isphibinzero]})

    fig, _ = plot_spherical(dirbins, ["luminosity"], nphibins=nphibins, ncosthetabins=ncosthetabins)
    plt.close(fig)

    # the colour bar draws a mesh of its own after the map
    _, meshgrid_phi, _, data = mockpcolormesh.call_args_list[0].args
    (litcolumn,) = np.flatnonzero(np.asarray(data)[0])
    assert np.allclose(meshgrid_phi[0, litcolumn : litcolumn + 2], [0.0, 2 * np.pi / nphibins])


@pytest.mark.parametrize("nphibins", [1, 5, 8])
def test_plotspherical_map_columns_cover_each_longitude_with_the_correct_phi_bin(nphibins: int) -> None:
    """Each column of the map must show the phi bin that holds its longitude, from -pi to +pi.

    For an odd -nphibins, the map ran from -0.8 pi to 1.2 pi for five bins, thus a strip at -pi stayed blank.
    """
    from artistools.plotspherical import get_map_columns

    longitude_edges, column_phibins = get_map_columns(nphibins)

    assert len(longitude_edges) == len(column_phibins) + 1
    assert np.isclose(longitude_edges[0], -np.pi)
    assert np.isclose(longitude_edges[-1], np.pi)
    assert np.all(np.diff(longitude_edges) > 0.0)
    assert np.any(np.isclose(longitude_edges, 0.0)), "phi = 0 must be an edge at the centre of the map"
    column_centre_phi = np.mod(0.5 * (longitude_edges[:-1] + longitude_edges[1:]), 2 * np.pi)
    assert np.array_equal(np.floor(column_centre_phi / (2 * np.pi) * nphibins).astype(int), column_phibins)


def test_plotspherical_smoothing_with_odd_phi_bins_crosses_the_pole_at_phi_plus_pi() -> None:
    """For an odd -nphibins, the row beyond a pole must take the two bins on each side of phi + pi."""
    from artistools.plotspherical import get_rows_across_pole

    rows = np.arange(5, dtype=np.float64).reshape((1, 5))

    # the centre of bin 0 is at 0.2 pi, thus phi + pi is at 1.2 pi, which is the edge of bins 2 and 3
    assert np.allclose(get_rows_across_pole(rows, 5)[0], [2.5, 3.5, 2.0, 0.5, 1.5])


def test_plotspherical_gif(tmp_path: Path) -> None:
    """The gif holds one frame for each timestep in the time range, in the order of time.

    The frames and the gif go in a folder of this test alone. A pytest run clears the test output folder when it
    starts. Thus a second run deleted the frames of this test before the gif read them.
    """
    from PIL import Image

    # the timesteps 0, 1, and 2 end before 253 d. Three frames cover the code of the gif
    at.plotspherical.main(argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=tmp_path)

    framenames = sorted(path.name for path in tmp_path.glob("plotspherical_*.png"))
    assert framenames == [
        "plotspherical_250.00-250.84d.png",
        "plotspherical_250.84-251.69d.png",
        "plotspherical_251.69-252.54d.png",
    ]
    with Image.open(tmp_path / "sphericalplot.gif") as gif:
        assert getattr(gif, "n_frames", 1) == 3


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
    assert set(timetaken) == {"update_grid", "update_packets", "communicate_estimators"}
    for stage, bytimestep in timetaken.items():
        assert len(bytimestep) == 30, f"expected 30 timesteps of {stage} timings"
        for byrank in bytimestep.values():
            assert set(byrank) == {0, 1}, f"expected both mpi ranks for {stage}"
    assert timetaken["update_grid"][0] == {0: 2, 1: 2}
    assert timetaken["update_packets"][0] == {0: 1, 1: 1}
    assert timetaken["communicate_estimators"][2] == {0: 1, 1: 1}

    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.plotlogfiles.main(argsraw=[], modelpath=[modelpath_classic_3d], outputfile=funcoutpath / "logfiles.pdf")

    # one line per stage on each of the 30 per-timestep pages
    assert len(mockplot.call_args_list) == 3 * 30


def test_logfiles_read_the_old_and_the_current_log_format(tmp_path: Path) -> None:
    """The timings of the current ARTIS log lines and of the old log lines must both be read.

    The patterns held the Unix time and the whole seconds of the old lines. The current lines hold no Unix time,
    the grid line says "on all processes", and the seconds have decimals. Thus the command found no timing data.
    """
    (tmp_path / "output_0-0.txt").write_text(
        "2026-10-04T12:00:00Z timestep 3: time after update grid on all processes"
        " (rank 0 took 1.2s, waited 0.3s, total 1.5s)\n"
        "2026-10-04T12:00:05Z timestep 3: time after update packets for all processes"
        " (rank 0 took 3.5s, waited 0.1s, total 3.6s)\n"
        "2026-10-04T12:00:06Z timestep 3: time after estimators have been communicated (took 0.4 seconds)\n"
        "2023-11-14T15:01:59Z timestep 2: time after update grid for all processes 1699974119"
        " (rank 0 took 2s, waited 0s, total 2s)\n"
        "2023-11-14T15:02:09Z timestep 2: time after update packets for all processes 1699974129"
        " (rank 0 took 7s, waited 0s, total 7s)\n"
        "2023-11-14T15:02:10Z timestep 2: time after estimators have been communicated 1699974130 (took 1 seconds)\n",
        encoding="utf-8",
    )

    timetaken = at.plotlogfiles.read_time_taken([tmp_path / "output_0-0.txt"])

    assert timetaken["update_grid"] == {3: {0: pytest.approx(1.2)}, 2: {0: pytest.approx(2.0)}}
    assert timetaken["update_packets"] == {3: {0: pytest.approx(3.5)}, 2: {0: pytest.approx(7.0)}}
    assert timetaken["communicate_estimators"] == {3: {0: pytest.approx(0.4)}, 2: {0: pytest.approx(1.0)}}


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_transitions_with_a_model_also_draws_the_lte_temperatures(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """With a model path, each -T gives an LTE series beside the NLTE series.

    The command drew the NLTE series alone and ignored -T with no message.
    """
    at.plottransitions.main(argsraw=[], modelpath=modelpath, outputfile=tmp_path, timedays=300, T=[5000.0, 8000.0])

    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    # one series of each type in each panel of the seven ions
    assert labels.count("NLTE") == 7
    assert labels.count("LTE T1 = 5000 K") == 7
    assert labels.count("LTE T2 = 8000 K") == 7


def test_plotspherical_element_arguments_give_messages(capsys: pytest.CaptureFixture[str]) -> None:
    """-elem with -atomic_number, or an -elem that is no element symbol, gives a message.

    The first stopped with a bare AssertionError, and the second plotted the packets of "Z=-1".
    """
    with pytest.raises(SystemExit):
        at.plotspherical.main(argsraw=[], modelpath=modelpath, elem="Fe", atomic_number=26)
    assert "give one of -elem and -atomic_number" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        at.plotspherical.main(argsraw=[], modelpath=modelpath, elem="Xx")
    assert "-elem Xx is not an element symbol" in capsys.readouterr().err


def test_plotviewingangles_writes_an_animation_of_each_direction_bin(tmp_path: Path) -> None:
    """The html file holds one frame of the animation for each of the 100 direction bins."""
    pytest.importorskip("plotly")
    outputfile = tmp_path / "viewingangles.html"
    at.plotviewingangles.main(argsraw=[str(modelpath_3d), "-o", str(outputfile)])

    framenames = set(re.findall(r'"name":"(\d\d)"', outputfile.read_text(encoding="utf-8")))
    assert framenames == {f"{dirbin:02d}" for dirbin in range(100)}


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
def test_writecomparisondata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The files take the name of the model folder, and the spectra keep the luminosity of spec.out.

    The default model path "." gave an empty model name, e.g. spectra__artisnebular.txt.
    """
    monkeypatch.chdir(modelpath)
    at.writecomparisondata.main(argsraw=[], modelpath=Path(), outputpath=tmp_path, selected_timesteps=list(range(99)))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "edep_testmodel_artisnebular.txt",
        "ionfrac_co_testmodel_artisnebular.txt",
        "ionfrac_fe_testmodel_artisnebular.txt",
        "phys_testmodel_artisnebular.txt",
        "spectra_testmodel_artisnebular.txt",
    ]

    from artistools.spectra.core import read_spec

    # the spectra give L_lambda, and spec.out gives F_nu at 1 Mpc
    dfspecout = read_spec(modelpath).collect()
    arr_specout_nu = dfspecout["nu"].to_numpy()
    arr_written = np.loadtxt(tmp_path / "spectra_testmodel_artisnebular.txt", comments="#")
    assert arr_written.shape == (len(arr_specout_nu), 1 + 99)
    area = 4.0 * math.pi * at.constants.megaparsec_to_cm**2
    for timestep in (10, 50):
        lum_written = np.trapezoid(arr_written[:, 1 + timestep], arr_written[:, 0])
        lum_specout = abs(np.trapezoid(dfspecout[:, 1 + timestep].to_numpy(), arr_specout_nu)) * area
        assert np.isclose(lum_written, lum_specout, rtol=1e-4)

    # the ion fractions of an element sum to one, as the format of the workshop requires
    arr_ionfrac_fe = np.loadtxt(tmp_path / "ionfrac_fe_testmodel_artisnebular.txt", comments="#")
    assert np.allclose(arr_ionfrac_fe[:, 1:].sum(axis=1), 1.0, rtol=1e-3)


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


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"opacityexclusions": [0, -5]}, "below -4"),
        ({"custom_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
        ({"tau_max_vpkt": 0.0}, "tau_max_vpkt 0 must be more than zero"),
        ({"cell_is_optically_thick_vpkt": -1.0}, "thick cell optical depth -1"),
        ({"override_tminmax": True, "vspec_tmin_in_days": 4.0, "vspec_tmax_in_days": 4.0}, "time window"),
        ({"vgrid_on": True, "tmin_vgrid_in_days": 2.0, "tmax_vgrid_in_days": 1.0}, "velocity grid map time range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": []}, "at least one wavelength range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
    ],
)
def test_make_vpkt_input_refuses_each_fatal_setting_of_artis(changes: dict[str, t.Any], reason: str) -> None:
    """Each setting that read_vpktparameterfile() in vpkt.cc stops on must give an error, not a file.

    The command wrote a vpkt.txt with such a setting, and ARTIS then stopped at the start of the run.
    """
    config = dc.replace(at.make_vpkt_input.VpktConfig(), **changes)
    assert reason in str(at.make_vpkt_input.fatal_config_error(config))
    with pytest.raises(ValueError, match=re.escape(reason)):
        at.make_vpkt_input.format_vpkt_input(config)

    # ARTIS reads the ranges of the velocity grid map only when the map is on
    assert (
        at.make_vpkt_input.fatal_config_error(dc.replace(at.make_vpkt_input.VpktConfig(), vgrid_lambda_ranges=[]))
        is None
    )
    assert (
        at.make_vpkt_input.fatal_config_error(
            dc.replace(at.make_vpkt_input.VpktConfig(), vgrid_lambda_ranges=[(0.0, 6000.0)])
        )
        is None
    )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_hesma_plots_take_the_times_and_the_tables_of_each_file(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The action plotresspec takes the closest time and each table, and plotspectrum takes each file.

    The action plotresspec read the column 11.7935 and the tables 0 to 4 of every file. The action plotspectrum
    read the first file alone.
    """
    resfile = tmp_path / "model_vspec_res.dat"
    tables = [
        f"0 10.0 12.0\n3000.0 {dirbin + 1.0} {dirbin + 2.0}\n4000.0 {dirbin + 3.0} {dirbin + 4.0}\n"
        for dirbin in range(2)
    ]
    resfile.write_text("# a comment line of the file\n" + "".join(tables), encoding="utf-8")
    at.hesma_scripts.main(
        argsraw=["plotresspec", "-hesmafile", str(resfile), "-timedays", "11.5", "-plotfile", str(tmp_path / "a.pdf")]
    )
    assert len(mockplot.call_args_list) == 2
    for dirbin, callargs in enumerate(mockplot.call_args_list):
        xarr, yarr = get_plot_xy(callargs)
        assert np.allclose(xarr, [3000.0, 4000.0])
        assert np.allclose(yarr, np.array([dirbin + 2.0, dirbin + 4.0]) * 1e-10)

    mockplot.reset_mock()
    specfiles = [tmp_path / "a_spec.dat", tmp_path / "b_spec.dat"]
    for specfile in specfiles:
        specfile.write_text("0.0000 11.7935\n3000.0 1.0\n4000.0 3.0\n", encoding="utf-8")
    at.hesma_scripts.main(
        argsraw=[
            "plotspectrum",
            "-hesmafile",
            *map(str, specfiles),
            "-timedays",
            "12",
            "-plotfile",
            str(tmp_path / "b.pdf"),
        ]
    )
    assert [callargs.kwargs["label"] for callargs in mockplot.call_args_list] == ["HESMA a_spec", "HESMA b_spec"]


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


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_hesma_spectrum_takes_the_column_names_of_the_file(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The header of a HESMA spectrum gives the times with four decimals, and the lookup must find them.

    The lookup made the name of the column again with two decimals, thus it did not find "11.7935" or "0.0000".
    """
    hesmafile = tmp_path / "model_spec.dat"
    hesmafile.write_text("0.0000 11.7935 12.3456\n3000.0 1.0 2.0\n4000.0 3.0 4.0\n", encoding="utf-8")
    fig, ax = plt.subplots()

    at.hesma_scripts.plot_hesma_spectrum(12.0, [ax], hesmafile)
    plt.close(fig)

    xarr, yarr = get_plot_xy(mockplot.call_args)
    assert np.allclose(xarr, [3000.0, 4000.0])
    assert np.allclose(yarr, np.array([1.0, 3.0]) * 1e-10)


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


def test_ejectaopacity(capsys: pytest.CaptureFixture[str]) -> None:
    """The global Planck mean opacity of the binned expansion opacities agrees with the reference value.

    The opacities need the statistical weights of the levels. A sum without them gives 13.90 cm^2/g.
    """
    at.ejectaopacity.main(
        argsraw=[], modelpath=modelpath, timestep=40, lambdamin=3000.0, lambdamax=4000.0, deltalambda=10.0
    )

    (planckmeanline,) = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Global Planck mean opacity:")
    ]
    assert np.isclose(float(planckmeanline.split(":")[1].split()[0]), 13.21, rtol=1e-3)


@mock.patch.object(mplax.Axes, "axhline", side_effect=mplax.Axes.axhline, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_ratios_and_the_planck_mean(
    mockplot: mock.MagicMock, mockaxhline: mock.MagicMock, tmp_path: Path
) -> None:
    """The plot has the bins and a moving average of each opacity, a dashed capped opacity, ratios, and the Planck mean.

    The Planck mean takes the bins of the x range, with the Planck function at the temperature of the cell. Only
    --showplanckmean draws it.
    """
    at.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "--showplanckmean",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = [call.args[0] for call in mockplot.call_args_list]
    assert len(set(axes)) == 2, "the opacities and the ratios need two frames"
    mainplots = [call for call in mockplot.call_args_list if call.args[0] is axes[0]]
    assert len(mainplots) == 6, "each opacity needs its bins and its moving average"
    assert [call.kwargs["linestyle"] for call in mainplots if call.kwargs.get("label")] == ["-", "--", "-"]

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    dfbins = at.ejectaopacity.get_expansion_opacities(lines, dfcell, edges, time_days)
    lambda_cm = dfbins["lambda_angstroms_bin_mid"].to_numpy() * 1e-8
    temperature = dfcell["T_exc"].item()
    planck = lambda_cm**-5 / np.expm1(
        at.constants.h_erg_s * at.constants.C_cm_per_s / lambda_cm / temperature / at.constants.K_B_erg_per_K
    )
    expected = float(np.sum(planck * dfbins["exopac"].to_numpy()) / np.sum(planck))
    assert mockaxhline.call_count == 1
    assert np.isclose(mockaxhline.call_args.args[1], expected, rtol=1e-9, atol=0.0)

    # the line and its calculation need --showplanckmean
    at.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-o",
            str(tmp_path / "noplanck.pdf"),
        ]
    )
    assert mockaxhline.call_count == 1


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_each_cap_and_the_line_count(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """-taucaps draws an opacity for each cap, and --showlinecount adds a panel with the number of lines in each bin.

    The panel counts each line that the opacities sum. A cap that is given two times gives one opacity. The :g format
    gave 1.0000001 the column name of 1, thus the kernel stopped with an error.
    """
    at.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-taucaps",
            "0.1",
            "1",
            "10",
            "1",
            "1.0000001",
            "--showlinecount",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    assert len(axes) == 3, "the opacities, the ratios, and the number of lines need three panels"
    labels = [call.kwargs["label"] for call in mockplot.call_args_list if call.kwargs.get("label")]
    assert labels == [
        "Expansion opacity",
        *(rf"Line-binned, $\tau_\mathrm{{l,max}}$ = {taucap}" for taucap in ("0.1", "1", "10", "1.0000001")),
        "Line-binned",
    ]
    ratioplots = [call for call in mockplot.call_args_list if call.args[0] is axes[1]]
    assert len(ratioplots) == 5, "each line-binned opacity needs a ratio"

    (linecountplot,) = [call for call in mockplot.call_args_list if call.args[0] is axes[2]]
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    # each bin draws its count at its lower edge and at its upper edge
    assert np.nansum(linecountplot.args[2]) == 2 * lines.dflines.height


def test_plotopacity_average_cell_takes_the_mean_composition(capsys: pytest.CaptureFixture[str]) -> None:
    """--averagecell takes one cell with the mass-weighted mean of n_ion / rho, of rho, and of T_exc, and logs T_exc.

    A model of one cell gives the same cell. A cell with no temperature does not count in the mean temperature.
    """
    dfone = at.ejectaopacity.get_cell_estimators(modelpath, 40, None, "Te")
    pltest.assert_frame_equal(
        at.plotopacity.get_average_cell(dfone),
        dfone.select(at.plotopacity.get_average_cell(dfone).columns),
        check_dtypes=False,
        rel_tol=1e-12,
    )

    dfcells = pl.DataFrame({
        "modelgridindex": [0, 1, 2],
        "timestep": [5, 5, 5],
        "T_exc": [1000.0, 3000.0, None],
        "rho": [1.0, 2.0, 4.0],
        "mass_g": [1.0, 3.0, 4.0],
        "nnion_Fe_II": [2.0, 4.0, 8.0],
    })
    capsys.readouterr()
    dfmean = at.plotopacity.get_average_cell(dfcells)
    assert np.isclose(dfmean["T_exc"].item(), (1000.0 + 3 * 3000.0) / 4, rtol=1e-12, atol=0.0)
    meanrho = (1.0 + 3 * 2.0 + 4 * 4.0) / 8
    assert np.isclose(dfmean["rho"].item(), meanrho, rtol=1e-12, atol=0.0)
    assert np.isclose(dfmean["nnion_Fe_II"].item(), meanrho * (2.0 + 3 * 2.0 + 4 * 2.0) / 8, rtol=1e-12, atol=0.0)
    assert "T_exc = 2500 K" in capsys.readouterr().out
    assert (
        at.plotopacity.get_cells_text(None, None, None, "TJ = 2500 K") == "mean composition of all cells at TJ = 2500 K"
    )


def test_excitation_temperature_is_the_temperature_that_artis_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-exctemperature auto takes TJ or Te from LTEPOP_EXCITATION_USE_TJ of the run, and the log names it.

    The kernel took Te before. A classic run sets LTEPOP_EXCITATION_USE_TJ = true, and after the LTE timesteps its Te
    differs from TJ, thus the level populations did not match ARTIS. A commented line does not count.
    """
    get_column = at.ejectaopacity.get_excitation_temperature_column
    capsys.readouterr()
    assert get_column(tmp_path, "auto") == "Te"
    assert "gives no LTEPOP_EXCITATION_USE_TJ" in capsys.readouterr().err

    (tmp_path / "artis").mkdir()
    optionspath = tmp_path / "artis" / "artisoptions.h"
    for value, expectedcolumn in (("true", "TJ"), ("false", "Te")):
        optionspath.write_text(
            f"// constexpr bool LTEPOP_EXCITATION_USE_TJ = {'false' if value == 'true' else 'true'};\n"
            f"constexpr bool LTEPOP_EXCITATION_USE_TJ = {value};\n",
            encoding="utf-8",
        )
        assert get_column(tmp_path, "auto") == expectedcolumn
        assert f"T_exc = {expectedcolumn}," in capsys.readouterr().out
    assert get_column(tmp_path, "TJ") == "TJ"
    assert "-exctemperature TJ" in capsys.readouterr().out


def test_expansion_opacities_keep_the_values_of_the_join_query() -> None:
    """The Rust kernel gives the values of the earlier polars query.

    The earlier query joined each cell with each line, and the reference sums come from it. Only the order
    of the additions is different, thus the values agree within the rounding error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    dfopacities = at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    assert dfopacities.height == 100
    for column, expectedsum in {
        "exopac": 1397.6901658607103,
        "linebinned": 11675.7786539805,
        "linebinned_cap1": 1652.4668052744682,
    }.items():
        assert math.isclose(dfopacities[column].sum(), expectedsum, rel_tol=1e-12), column


def test_expansion_opacities_give_each_cap_its_own_column() -> None:
    """The kernel gives the sum of each cap to the column of that cap. The order of the caps has no effect.

    A cap of 1 gives the value of the regression test above. A larger cap gives a larger sum, and a cap above each
    tau_sobolev gives the sum with no cap.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (10.0, 1e300, 0.1, 1.0)
    dfopacities = at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    sums = {taucap: dfopacities[at.ejectaopacity.get_capped_column(taucap)].sum() for taucap in sorted(taucaps)}
    assert at.ejectaopacity.get_capped_column(1234567.0) != at.ejectaopacity.get_capped_column(1234568.0)
    assert math.isclose(sums[1.0], 1652.4668052744682, rel_tol=1e-12)
    assert math.isclose(sums[1e300], dfopacities["linebinned"].sum(), rel_tol=1e-12)
    assert list(sums.values()) == sorted(sums.values())
    assert len(set(sums.values())) == len(sums)


def test_expansion_opacities_of_a_null_population_are_zero() -> None:
    """A null population of an ion gives no opacity from that ion, and a null temperature gives no opacity.

    The kernel reads contiguous columns with no nulls, thus get_expansion_opacities() must replace each null.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )
    ionstr = opacitylines.ionstrs[0]

    def get_opacities(dfcells: pl.DataFrame) -> pl.DataFrame:
        return at.ejectaopacity.get_expansion_opacities(opacitylines, dfcells, lambda_bin_edges, time_days).select(
            at.ejectaopacity.get_opacity_columns(at.ejectaopacity.DEFAULT_TAUCAPS)
        )

    pltest.assert_frame_equal(
        get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
        get_opacities(dfcell.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
    )
    dfnotemperature = get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias("T_exc")))
    assert np.allclose(dfnotemperature.select(pl.all().abs().max()).row(0), 0.0, rtol=0.0, atol=0.0)


def test_expansion_opacities_keep_a_nan_in_each_sum() -> None:
    """A NaN temperature gives NaN level populations, and each of the sums must then be NaN.

    f64::min(NaN, 1) is 1, and NaN.abs() >= 1e-18 is false, thus the kernel gave a finite capped opacity
    and a finite exopac for such a cell.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te").with_columns(
        pl.lit(float("nan"), dtype=pl.Float32).alias("T_exc")
    )
    lambda_bin_edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (0.1, 1.0)
    dfopacities = at.ejectaopacity.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    opacitycolumns = at.ejectaopacity.get_opacity_columns(taucaps)
    isnan = dfopacities.select(pl.col(opacitycolumns).is_nan())
    assert isnan["linebinned"].any()
    for column in opacitycolumns:
        assert isnan[column].equals(isnan["linebinned"]), column


def test_expansion_opacities_skip_a_line_with_a_null_constant() -> None:
    """A line with a null A value adds nothing, as a line that the atomic data does not hold.

    The kernel reads columns with no nulls, thus a null constant stopped the sum with an error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
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
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
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
        dfopacities, _ = at.plotopacity.get_massweighted_opacities(
            adata, time_days, dfestimators, at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
        )
        return dfopacities["linebinned"].to_numpy()

    linebinned_onecell = get_linebinned(dfcell)
    assert linebinned_onecell.max() > 0.0
    meanfactor = (2 * cellcount + 1) / 3
    assert np.allclose(get_linebinned(dfcells), meanfactor * linebinned_onecell, rtol=1e-10, atol=0.0)


def test_plotopacity_calculates_only_the_bins_of_the_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The bins of the plot range give the same plotted data as a calculation of the full grid of rpkt.h.

    The range also holds one bin more than half the window of the moving average at each end. The width of a bin
    comes from rpkt.h, and the command prints the range and the number of the bins.
    """
    (tmp_path / "artis").mkdir()
    (tmp_path / "artis" / "rpkt.h").write_text(
        "constexpr double expopac_lambdamin = 3000.;\n"
        "constexpr double expopac_lambdamax = 4000.;\n"
        "constexpr double expopac_deltalambda = 10.;\n",
        encoding="utf-8",
    )
    assert at.ejectaopacity.get_expopac_grid(tmp_path) == (3000.0, 4000.0, 10.0)
    assert at.ejectaopacity.get_expopac_grid(modelpath) is None

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.ejectaopacity.get_opacity_atomic_data(modelpath)
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    dffull, _ = at.plotopacity.get_massweighted_opacities(
        adata, time_days, dfcell, at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    )
    xmin, xmax, width = 3213.0, 3517.0, 50.0
    capsys.readouterr()
    edges, deltalambda = at.plotopacity.get_computed_bin_edges(tmp_path, xmin, xmax, None, width)
    assert deltalambda == pytest.approx(10.0, rel=1e-12, abs=0.0)
    assert len(edges) - 1 < 50 < dffull.height
    assert (
        f"{len(edges) - 1} wavelength bins of 10 Angstroms from {edges[0]:g} to {edges[-1]:g} Angstroms of the 100 bins"
        " of rpkt.h from 3000 to 4000 Angstroms"
    ) in capsys.readouterr().out
    dfpart, _ = at.plotopacity.get_massweighted_opacities(adata, time_days, dfcell, edges)

    windowbins = at.plotopacity.get_window_bins(width, deltalambda)
    plotted = [
        (
            df.filter(pl.col("lambda_angstroms_upper") > xmin, pl.col("lambda_angstroms_lower") < xmax),
            at.misc.df_filter_minmax_bracketed(
                at.plotopacity.get_moving_averages(df, windowbins, ["exopac", "linebinned", "linebinned_cap1"]),
                "lambda_angstroms_bin_mid",
                xmin,
                xmax,
            ).collect(),
        )
        for df in (dffull, dfpart)
    ]
    pltest.assert_frame_equal(plotted[0][0], plotted[1][0], rel_tol=1e-9)
    pltest.assert_frame_equal(plotted[0][1], plotted[1][1], rel_tol=1e-9)


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
    dfcells = pl.DataFrame({"T_exc": [5000.0], "nnion_0": [1.0]})
    exopac = sum_binned_line_opacities(dflevels, dflines, dfcells, ["nnion_0"], [], 3, at.constants.K_B_ev_per_K)[
        "exopac"
    ]
    assert np.allclose(exopac.to_numpy(), -np.expm1(-np.array(taus)), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("edges", [[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 2.0, 3.0]])
def test_rust_bin_sums_match_numpy_histogram(edges: list[float]) -> None:
    """The Rust kernel gives the bins of np.histogram, for uniform and for other edges.

    A bin is [lower, upper), the last bin also holds its upper edge, and a value outside the edges or NaN is in no
    bin. The uniform edges take a guess that the exact edges correct, thus a value on an inner edge tests it.
    """
    from artistools.rustext import sum_weights_in_bins

    values = [-0.1, 0.0, 0.5, 0.9999999, 1.0, 2.0, 2.5, 3.0, 3.0000001, math.nan]
    df = pl.DataFrame({"x": values, "e": [float(2**index) for index in range(len(values))]})
    # 3.0000001 is 3.0 in 32 bits, thus each type of value has its own reference
    for dtype in (pl.Float64, pl.Float32):
        dftyped = df.with_columns(pl.col("x").cast(dtype))
        typedvalues = dftyped["x"].to_numpy()
        isnumber = ~np.isnan(typedvalues)
        refcounts, _ = np.histogram(typedvalues[isnumber], bins=edges)
        refsums, _ = np.histogram(typedvalues[isnumber], bins=edges, weights=df["e"].to_numpy()[isnumber])
        refsquaresums, _ = np.histogram(typedvalues[isnumber], bins=edges, weights=df["e"].to_numpy()[isnumber] ** 2)
        sums = sum_weights_in_bins(dftyped, "x", "e", edges, sumsquares=True)
        assert np.allclose(sums["sum"].to_numpy(), refsums, rtol=1e-12, atol=0.0), dtype
        assert np.allclose(sums["sumsquares"].to_numpy(), refsquaresums, rtol=1e-12, atol=0.0), dtype
        assert sums["count"].to_list() == refcounts.tolist(), dtype

    # each group has its own bins, in the order [group][bin]
    rng = np.random.default_rng(seed=1)
    dfgroups = pl.DataFrame({
        "x": rng.uniform(-0.5, 3.5, 100_000),
        "e": rng.uniform(0.0, 1.0, 100_000),
        "group": rng.integers(0, 3, 100_000, dtype=np.int32),
    })
    grouped = sum_weights_in_bins(dfgroups, "x", "e", edges, "group", 3)
    for group in range(3):
        dfgroup = dfgroups.filter(pl.col("group") == group)
        expectedcounts, _ = np.histogram(dfgroup["x"].to_numpy(), bins=edges)
        expectedsums, _ = np.histogram(dfgroup["x"].to_numpy(), bins=edges, weights=dfgroup["e"].to_numpy())
        rows = slice(group * (len(edges) - 1), (group + 1) * (len(edges) - 1))
        assert np.allclose(grouped["sum"].to_numpy()[rows], expectedsums, rtol=1e-12, atol=0.0)
        assert grouped["count"].to_numpy()[rows].tolist() == expectedcounts.tolist()
    # pyo3-polars raises its own ComputeError, which is not the class of the polars package
    with pytest.raises(Exception, match="a group is outside"):
        sum_weights_in_bins(dfgroups, "x", "e", edges, "group", 2)


def test_rust_bin_indices_match_the_bins_of_the_sums() -> None:
    """get_bin_indices gives the bin of each value with the rules of sum_weights_in_bins, and -1 outside the edges."""
    from artistools.rustext import get_bin_indices
    from artistools.rustext import sum_weights_in_bins

    rng = np.random.default_rng(seed=2)
    edges = [0.0, 0.5, 2.0, 3.0]
    values = np.concatenate([rng.uniform(-0.5, 3.5, 10_000), edges, [math.nan]])
    df = pl.DataFrame({"x": values, "e": np.ones(len(values))})
    bins = get_bin_indices(df, "x", edges)["binindex"].to_numpy()
    counts = sum_weights_in_bins(df, "x", "e", edges)["count"].to_numpy()
    assert np.array_equal(np.bincount(bins[bins >= 0], minlength=len(edges) - 1), counts)
    # each edge starts its bin, the last edge is in the last bin, and NaN is in no bin
    assert bins[-5:].tolist() == [0, 1, 2, 2, -1]


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
    dfestimators = at.ejectaopacity.get_cell_estimators(modelpath_classic_3d, 5, None, "Te")
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
    """A cell with no matter has no estimators, and the error names the cell and the next cell with estimators.

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
    estimatorcells = at.scan_estimators(modelpath_classic_3d, timestep=5).select("modelgridindex").collect()
    nextcell = estimatorcells.filter(pl.col("modelgridindex") > emptycell)["modelgridindex"].min()
    with pytest.raises(ValueError, match=rf"hold no values for cell {emptycell} .* is cell {nextcell}$"):
        at.ejectaopacity.get_cell_estimators(modelpath_classic_3d, 5, emptycell, "Te")


def test_kurucz_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """gfall.dat is fixed-width, and the wavelength field is 11 characters wide, not 12.

    Reading 12 characters takes the first character of the loggf field with it, which is only harmless while that
    character happens to be a space. The second line gives the upper level first, and a negative energy for a
    predicted level. The reader took the first level as the lower level and kept the sign of the energy.
    """

    def get_gfall_line(energy1: float, j1: float, energy2: float, j2: float) -> str:
        return (
            f"{715.5170:11.4f}"  # 0-10  wavelength in nm
            f"{-1.234:7.3f}"  # 11-17 log(gf)
            f"{44.00:6.2f}"  # 18-23 element code Z.(ion_stage - 1)
            f"{energy1:12.3f}"  # 24-35 energy of the first level in cm-1
            f"{j1:5.1f}"  # 36-40 J of the first level
            " a4F       "  # 41-51 configuration label
            f"{energy2:12.3f}"  # 52-63 energy of the second level in cm-1
            f"{j2:5.1f}" + " " + " 0" * 16 + "\n"  # 64-68 J of the second level
            # the parser only reads lines with at least 24 whitespace-separated fields
        )

    (tmp_path / "gfall.dat").write_text(
        get_gfall_line(25000.0, 4.5, 35000.0, 3.5) + get_gfall_line(-35000.0, 3.5, 25000.0, 4.5), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    dftransitions, ionlist = at.plottransitions.get_kurucz_transitions()

    assert ionlist == [(44, 1)]
    assert len(dftransitions) == 2
    hc_in_ev_cm = 0.0001239841984332003
    expected_values = {
        "lambda_angstroms": 7155.170,
        "lower_statweight": 2 * 4.5 + 1,
        "upper_statweight": 2 * 3.5 + 1,
        "lower_energy_ev": hc_in_ev_cm * 25000.0,
        "upper_energy_ev": hc_in_ev_cm * 35000.0,
    }
    for colname, expected in expected_values.items():
        assert dftransitions[colname].to_list() == pytest.approx([expected, expected]), colname

    # the two lines give the same transition, thus the same A value
    assert np.isclose(dftransitions["A"][0], dftransitions["A"][1], rtol=1e-12)


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


def test_linefluxes_from_pops_reads_the_cell_volumes() -> None:
    """The luminosity from the populations needs the volume of each cell of the model."""
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


def test_linefluxes_from_pops_takes_the_cell_volumes_of_a_3d_model(tmp_path: Path) -> None:
    """A 3D model has no shell velocities, thus the luminosity takes the volume of each cell.

    The function selected vel_r_min_kmps, which only a 1D model has, and a 3D model stopped with ColumnNotFoundError.
    """
    import itertools

    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    # a 3D grid of 2^3 cells with the files of the 1D test model. Only cell 0 has NLTE populations
    model3dpath = tmp_path / "model3d"
    model3dpath.mkdir()
    for filepath in modelpath.iterdir():
        if filepath.is_file() and filepath.name != "model.txt" and not filepath.name.startswith("."):
            (model3dpath / filepath.name).symlink_to(filepath)
    t_model_days = 0.00115740740741
    vmax = 8.0e8
    cellwidth = vmax * t_model_days * 86400.0
    cellrows = "".join(
        f"{cellid} {x} {y} {z} {10**-0.18}\n1.0 0.9 0.0 0.0 0.0\n"
        for cellid, (z, y, x) in enumerate(itertools.product((-cellwidth, 0.0), repeat=3), start=1)
    )
    (model3dpath / "model.txt").write_text(f"8\n{t_model_days}\n{vmax}\n{cellrows}", encoding="utf-8")

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    arr_tstart = [at.get_timestep_times(modelpath, loc="start")[10]]
    arr_tend = [at.get_timestep_times(modelpath, loc="end")[10]]
    lum1d = get_line_luminosities_from_pops(emfeatures, modelpath, arr_tstart, arr_tend)["Fe II 1-0"].item()
    lum3d = get_line_luminosities_from_pops(emfeatures, model3dpath, arr_tstart, arr_tend)["Fe II 1-0"].item()

    # the cell of the 3D model is a cube of side vmax, and the shell of the 1D model is a sphere of radius vmax
    assert np.isclose(lum3d / lum1d, 3.0 / (4.0 * math.pi), rtol=1e-5)


def test_linefluxes_from_pops_names_a_level_or_a_time_with_no_populations() -> None:
    """A line with no NLTE population at a time gives a message, not a bare AssertionError.

    The test model writes NLTE populations only at even timesteps from 10.
    """
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    with pytest.raises(ValueError, match="hold no value of the upper level 1 at timestep 11"):
        get_line_luminosities_from_pops(
            emfeatures,
            modelpath,
            arr_tstart=[at.get_timestep_times(modelpath, loc="start")[11]],
            arr_tend=[at.get_timestep_times(modelpath, loc="end")[11]],
        )

    with pytest.raises(ValueError, match="holds no Fe II line for the feature 7155"):
        get_closelines(modelpath_classic_3d, 26, 2, 7155, 7155.0, 7155.01)


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

    lumdata = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=True)

    assert lumdata.shape == expected.shape
    assert np.allclose(lumdata, expected, rtol=1e-13)

    # the cells of a 2D or 3D model have no radial order, thus an empty cell gives its volume to no other cell
    expected_cells = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            for modelgridindex in range(ncells):
                if (levelpop := levelpop_of_key.get((timestep, level, modelgridindex))) is not None:
                    expected_cells[timeindex] += (
                        delta_ergs * A_val * levelpop * shell_volumes_at_1s[modelgridindex] * t_sec_cubed
                    )
    lumdata_cells = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=False)
    assert np.allclose(lumdata_cells, expected_cells, rtol=1e-13)
    assert not np.allclose(lumdata_cells, expected, rtol=1e-6)


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
    from artistools.estimators.plotestimators import get_default_plotitem_skip_reason

    estimatorcolumns = ["timestep", "modelgridindex", "Te", "TR", "W", "nne", "rho", "nnelement_Fe"]

    for plotitem in (["Te"], ["W"], ["TR"], ["rho"], ["nne"], [["averageionisation", ["Fe"]]]):
        assert get_default_plotitem_skip_reason(plotitem, estimatorcolumns) is None, plotitem

    # an element that the model does not contain is still dropped
    assert get_default_plotitem_skip_reason([["averageionisation", ["Sr"]]], estimatorcolumns) == (
        "the estimators have no Sr"
    )
    assert get_default_plotitem_skip_reason([["populations", ["Sr I", "Sr II"]]], estimatorcolumns) is not None

    # initabundances/initmasses come from the input model file, so they must not be gated on estimator columns
    assert get_default_plotitem_skip_reason([["initabundances", ["Sr", "Ni_stable"]]], estimatorcolumns) is None
    assert get_default_plotitem_skip_reason([["initmasses", ["Sr", "Ni_56"]]], estimatorcolumns) is None


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


@pytest.mark.parametrize("yscale", ["linear", "log", "inverted"])
def test_set_legend_gives_the_legend_room_clear_of_the_data(yscale: str) -> None:
    """The legend of set_legend does not cover the data, on a linear, a log, and an inverted axis.

    Each command gave its legend room with a factor of its own, e.g. a top of 1.2 times the tallest peak.
    """
    fig = mplfig.Figure(figsize=(5.0, 3.5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    xvalues = np.linspace(0.0, 10.0, 50)
    ax.plot(xvalues, 10.0 ** (1.0 + xvalues / 10.0), label="rising")
    # a line of two points crosses the whole legend, and only the ends of its segment are data points
    ax.plot([0.0, 10.0], [90.0, 90.0], label="flat")
    if yscale == "log":
        ax.set_yscale("log")
    at.plottools.set_legend(ax, loc="upper right")
    # a limit that comes after set_legend still leaves the legend its room, because the room follows at the draw
    ax.set_ylim(5.0, 100.0)
    if yscale == "inverted":
        ax.invert_yaxis()

    canvas.draw()
    renderer = canvas.get_renderer()
    legend = ax.get_legend()
    assert legend is not None
    frame = at.plottools.get_legend_frame(legend, renderer)
    yrange = at.plottools.get_data_fraction_range(ax, frame.x0, frame.x1)
    assert yrange is not None
    # the fractions of the axes run upwards also on an inverted axis, thus the data stays below the legend
    assert yrange[1] < frame.y0
    ylimits = ax.get_ylim()
    canvas.draw()
    assert np.allclose(ax.get_ylim(), ylimits, rtol=1e-12, atol=0.0), "a second draw must keep the limits"


def test_set_legend_room_leaves_out_full_width_spans() -> None:
    """An axhspan covers each x, e.g. the span of a Shift-drag of a viewer, and it must not move the top.

    The top of the axis moved during a Shift-drag near the legend, thus the drag moved under the pointer.
    """
    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([0.0, 10.0], [0.0, 1.0], label="rising")
    ax.set_ylim(0.0, 2.0)
    at.plottools.set_legend(ax, loc="upper right")
    ax.axhspan(1.7, 1.9, color="0.5")
    ax.axhline(1.95)
    canvas.draw()
    assert np.allclose(ax.get_ylim(), (0.0, 2.0), rtol=1e-12, atol=0.0)


def test_set_legend_keeps_a_top_of_the_user() -> None:
    """A -ymax of the user stays, although the legend then covers the data."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 1.0], label="flat")
    ax.set_ylim(0.0, 1.02)
    at.plottools.set_legend(ax, argparse.Namespace(ymax=1.02), loc="upper right")
    fig.canvas.draw()
    assert np.isclose(ax.get_ylim()[1], 1.02, rtol=1e-12, atol=0.0)
    plt.close(fig)


@pytest.mark.parametrize(("labelwidth", "legendcols"), [(4, None), (60, None), (4, 1)])
def test_set_legend_takes_columns_for_a_long_legend(labelwidth: int, legendcols: int | None) -> None:
    """A long legend takes more columns, until it has half the frame height or it would be wider than the frame.

    Only one legend goes on the axes, thus a trial legend of the rule must stay off them. -legendcols overrides the
    rule.
    """
    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    _, axesgrid = at.plottools.make_frame_figure(fig=fig)
    ax = axesgrid[0][0]
    for index in range(16):
        ax.plot([0.0, 1.0], [index, index], label=f"{index}".ljust(labelwidth, "x"))
    # an ncol of None asks for the rule, as plotspectra and plotestimators give for a plot of one column
    legend = at.plottools.set_legend(ax, argparse.Namespace(legendcols=legendcols), loc="upper right", ncol=None)
    assert legend is not None
    assert ax.get_legend() is legend
    assert [child for child in ax.get_children() if isinstance(child, mpllegend.Legend)] == [legend]
    # the draw gives each text of the legend its position
    canvas.draw()
    renderer = canvas.get_renderer()
    frame = at.plottools.get_legend_frame(legend, renderer)
    # each column of the legend starts its labels at one x position
    ncols = len({round(text.get_window_extent(renderer).x0) for text in legend.get_texts()})
    if legendcols is not None:
        assert ncols == legendcols
    elif labelwidth > 10:
        # a wide label leaves no room for a second column
        assert frame.width <= 1.0
        assert ncols == 1
    else:
        assert frame.height <= at.plottools.MAX_LEGEND_HEIGHT_FRACTION
        assert ncols > 1


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
    """A tick at the very top or bottom of a log axis goes, and every decade inside it stays."""
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e3)
    at.plottools.prune_log_ticks(ax.yaxis)

    after = [loc for loc in ax.yaxis.get_majorticklocs() if 1e-10 <= loc <= 1e3]
    assert after == pytest.approx([10.0**exponent for exponent in range(-9, 3)])


def test_log_ticks_every_decade_on_a_short_axis() -> None:
    """A short log axis of many decades labels each power of ten, and it has minor ticks between them.

    The default locator counted the decades that fit the axis length, thus the opacity frame labelled
    every second decade and the ratio panel below it had no minor ticks.
    """
    fig, ax = plt.subplots(figsize=(3.0, 0.7))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 2e8)
    fig.canvas.draw()
    assert len([loc for loc in ax.yaxis.get_majorticklocs() if 0.4 <= loc <= 2e8]) < 9

    at.plottools.set_log_ticks_every_decade(ax.yaxis)
    fig.canvas.draw()
    majors = [loc for loc in ax.yaxis.get_majorticklocs() if 0.4 <= loc <= 2e8]
    assert majors == pytest.approx([10.0**exponent for exponent in range(9)])
    minors = [loc for loc in ax.yaxis.get_minorticklocs() if 1.0 <= loc <= 10.0]
    assert minors == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    # a range below one decade holds no power of ten, thus the minor ticks carry the labels
    ax.set_ylim(1.2, 8.0)
    fig.canvas.draw()
    assert [label.get_text() for label in ax.yaxis.get_minorticklabels() if label.get_text()]


def test_prune_log_ticks_keeps_a_sparse_axis_unchanged() -> None:
    """A log axis of few major ticks keeps them all, rather than end with too few to read."""
    from artistools.plottools import PrunedLogLocator

    locator = PrunedLogLocator(minticks=99, numticks=999)
    assert list(locator.tick_values(1e-30, 1e2)) == list(mplticker.LogLocator(numticks=999).tick_values(1e-30, 1e2))


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


def test_room_for_title_predicts_the_place_of_the_title_of_the_draw() -> None:
    """The height for the title comes from a prediction with no draw, and a draw must then put the title there.

    A draw took 83 ms of a plot of 250 ms. The draw moves a title that overlaps the offset text of the y axis, thus a
    title with and without an offset text must end 0.05 inches below the top of the figure. The text of the draw
    can end one pixel from the prediction.
    """
    for ymax in (2e-5, 20.0):
        fig, axes = at.plottools.make_frame_figure(argparse.Namespace(figwidthscale=0.3, figscale=1.0))
        axis = axes[0, 0]
        axis.plot([0.0, 1.0], [ymax / 2, ymax])
        axis.set_title("A model name\nTimestep 40 (10.00d-11.00d)")
        at.plottools.make_room_for_title(fig)
        fig.canvas.draw()
        assert bool(axis.yaxis.offsetText.get_text()) == (ymax < 1.0)
        gap = fig.get_figheight() - axis.title.get_window_extent().y1 / fig.dpi
        assert np.isclose(gap, 0.05, rtol=0.0, atol=0.01), (ymax, gap)
        plt.close(fig)


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
    assert at.misc.get_single_modelgridindex([12]) == 12
    assert at.misc.get_single_modelgridindex(None) is None
    assert at.misc.parse_range_list("3-7") == [3, 4, 5, 6, 7]
    assert at.misc.parse_range_list("4,5,6") == [4, 5, 6]

    # a command that reads one cell says so, in place of taking a cell that the text does not name
    with pytest.raises(ValueError, match=r"'3-7' names 5 cells, and this command reads one"):
        at.misc.get_single_modelgridindex([3, 4, 5, 6, 7])

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
            assert isinstance(action, at.misc.cliutils.CellListAction), f"{subcommand} gives -modelgridindex a text"
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


def test_writecomparisondata_of_a_run_with_no_estimators_gives_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run with no estimator files stopped with a raw ColumnNotFoundError for Te."""
    runpath = tmp_path / "noestimators"
    runpath.mkdir()
    for sourcefile in modelpath.iterdir():
        if sourcefile.is_file() and not sourcefile.name.startswith("estimators"):
            (runpath / sourcefile.name).symlink_to(sourcefile.resolve())

    with pytest.raises(SystemExit) as excinfo:
        at.writecomparisondata.main(argsraw=[], modelpath=runpath, outputpath=tmp_path / "out", selected_timesteps=[50])

    assert excinfo.value.code == 1
    assert "holds no estimator files" in capsys.readouterr().err


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


def test_linefluxes_timebins_in_reverse_order_take_the_order_in_time() -> None:
    """The bins in reverse order must give each time the same bin as the bins in the order of time.

    The upper edge went to the last bin of the list, and a rounding gap closed only between neighbours of the list.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [100.0, 150.0, 200.0, 250.0, 300.0]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [200.0, 100.0], [300.0, 200.0])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        0,
    ]

    dftimes = pl.DataFrame({"t": [1.0, 1.999995, 2.0, 3.0, 3.1]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [2.0, 1.0], [3.0, 1.99999])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        None,
    ]


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


def test_linefluxes_emitting_regions_take_the_timestep_of_the_thermal_emission() -> None:
    """The cell and the timestep of the emitting regions must come from the same event.

    The cell came from the last thermal emission and the timestep from the last interaction, thus a
    packet that scattered in a later timestep read the estimators of the wrong time.
    """
    dfpackets = pl.LazyFrame({
        "em_timestep": [52, 52, 3],
        "emtrue_timestep": [50, -1, 3],
        "emtrue_modelgridindex": [40, 40, 2],
    })

    assert at.plotlinefluxes.get_emission_columns(dfpackets, "trueemissiontype") == (
        "emtrue_timestep",
        "emtrue_modelgridindex",
    )

    # the last interaction keeps its own timestep and cell
    assert at.plotlinefluxes.get_emission_columns(dfpackets, "emissiontype") == ("em_timestep", "em_modelgridindex")

    # an old packets file has no trueem_time, thus the cell and the timestep both come from the last interaction
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_timestep"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )

    # a 3D packets file with trueem_time and no trueem_posx gives no cell of the thermal emission. The join on
    # emtrue_modelgridindex then stopped with ColumnNotFoundError
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_modelgridindex"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )


def test_linefluxes_emitting_regions_without_the_reference_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The package holds no floers_te_nne.json, thus the message names --norefdata, which plots the model alone.

    Each figure of a time bin has its own legend, and a label that holds LaTeX braces stays as it is.
    str.format read the braces of such a label as a field and raised KeyError, and only the last figure had a legend.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], plotemittingregions=True)
    assert "--norefdata" in capsys.readouterr().err

    # the packets give no position of the thermal emission, thus the cell comes from the last interaction
    with (
        mock.patch.object(at.plotlinefluxes, "set_legend") as mocklegend,
        mock.patch.object(
            at.plotlinefluxes, "plot_nne_te_points", side_effect=at.plotlinefluxes.plot_nne_te_points
        ) as mockpoints,
    ):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            norefdata=True,
            label=["$M_{\\rm ej}$ = 1 at {timeavg:.0f}d"],
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.png",
        )

    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["emreg_5.0d.png", "emreg_6.0d.png"]
    assert mocklegend.call_count == 2, "each figure needs a legend"
    assert [callargs.args[1] for callargs in mockpoints.call_args_list] == [
        "$M_{\\rm ej}$ = 1 at 5d",
        "$M_{\\rm ej}$ = 1 at 6d",
    ]


def test_linefluxes_label_fields_keep_latex_braces() -> None:
    """Only the fields timeavg and modeltag change, thus a LaTeX brace of a label stays."""
    from artistools.plotlinefluxes import format_label_fields

    assert (
        format_label_fields("$M_{\\rm ej}$ {modeltag} {timeavg:.0f}d", timeavg=250.4, modeltag="w7")
        == "$M_{\\rm ej}$ w7 250d"
    )


def test_viewer_status_line_gives_the_error() -> None:
    """The status line gives the error of argparse, and not the usage line that argparse prints before it."""
    stderr = "usage: artistools [options] [specpath ...]\nerror: argument -xmin: invalid float value: 'abc'\nhelp: -h"
    assert viewercore.get_first_line(stderr) == "argument -xmin: invalid float value: 'abc'"
    assert viewercore.get_first_line("A file is missing\nThe second line") == "A file is missing"


def test_viewer_queue_moves_a_clamped_control_back() -> None:
    """A handler that clamps a control to the values of the plot gives unchanged values, and the control must move back.

    The queue returned before it showed the values, thus a slider stayed at a position that the plot did not show.
    """
    # PySide6 is an optional dependency. CI does not install it for each Python version, and a CI machine with no
    # libEGL.so.1 gives an ImportError that is not a ModuleNotFoundError. The spinner and the banner of the queue
    # need QtWidgets, which needs libEGL.so.1, thus each queue test skips without QtWidgets
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    qtcore = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    viewer = mock.Mock(values=5)
    showvalues = mock.Mock()
    queue = viewerwindow.DrawQueue(qtcore.QObject(), viewer, mock.Mock(), showvalues, mock.Mock(), render=mock.Mock())
    queue.apply(5)
    showvalues.assert_called_once_with()
    assert queue.requestedvalues is None, "unchanged values must draw no plot"
    queue.close()


def test_viewer_undo_reverts_a_drag_in_one_step_and_skips_a_rejected_change() -> None:
    """One step of Undo reverts all the changes of a drag, and Undo skips a change that the command rejected.

    A rejected change leaves the old values, thus its step of Undo holds the current values and changes nothing.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    qtcore = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    viewer = mock.Mock(values=1)
    queue = viewerwindow.DrawQueue(qtcore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=mock.Mock())
    # the test needs no plot, and a timer of a plot that stays in the process crashed a later test of this worker
    with mock.patch.object(queue, "redraw"):
        # a drag gives a new value at each movement of the mouse
        for values in (2, 3, 4):
            queue.apply(values)
        # the user stops for a time that is longer than the merge time
        queue.lastchangetime -= 10.0 * viewercore.UNDO_MERGE_SECONDS
        queue.apply(5)
        # the command rejected 5, thus the viewer kept the values of the last plot
        viewer.values = 4

        queue.undo()
        assert viewer.values == 1
        queue.redo()
        assert viewer.values == 4
        queue.apply(7)
        assert not queue.can_redo(), "a new change must remove the steps of Redo"
        queue.apply(8, undoable=False)
        queue.undo()
        assert viewer.values == 4, "a change of the window, e.g. a step of Play, must give no step of Undo"
    queue.close()


def test_viewer_cancel_keeps_the_history_of_the_plot_on_the_screen() -> None:
    """Cancel Plot after Undo keeps the step of Undo, because the plot on the screen did not change.

    The cancel returned to the values of the plot but kept the lists of Undo and Redo of the cancelled step. Thus
    Undo lost the step back to the earlier values.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    viewer = mock.Mock(values=1, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=mock.Mock())
    with mock.patch.object(queue, "redraw"), mock.patch.object(viewerwindow, "set_plot_busy"):
        queue.apply(2)
        # the plot of 2 is on the screen, then Undo asks for a plot of 1, which waits
        queue.drawnvalues = 2
        queue.undo()
        assert viewer.values == 1
        queue.requestedvalues = 1
        queue.cancel()
        assert viewer.values == 2
        assert queue.can_undo()
        assert not queue.can_redo()
        queue.undo()
        assert viewer.values == 1
    queue.close()


def test_viewer_undo_ignores_the_parts_that_the_window_sets() -> None:
    """A step of Undo that differs only in a part that the window sets, e.g. the width of the figure, is no step.

    keep_on_undo gives such a part its current value. Undo compared the entries before it applied keep_on_undo.
    Thus a resize of the window after a rejected change enabled Undo, and Undo only drew the same plot again.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    @dc.dataclass(frozen=True, kw_only=True)
    class Values:
        time: int
        figwidthscale: float
        dpi: int | None = None

    viewer = mock.Mock(values=Values(time=1, figwidthscale=1.0))
    queue = viewerwindow.DrawQueue[Values](
        QtCore.QObject(),
        viewer,
        mock.Mock(),
        mock.Mock(),
        mock.Mock(),
        render=mock.Mock(),
        keep_on_undo=viewercore.keep_figwidthscale,
    )
    with mock.patch.object(queue, "redraw") as mockredraw:
        queue.apply(Values(time=2, figwidthscale=1.0))
        # the command rejected the change, thus the viewer kept the values of the plot
        viewer.values = Values(time=1, figwidthscale=1.0)
        queue.apply(Values(time=1, figwidthscale=1.3), undoable=False)
        assert not queue.can_undo()
        mockredraw.reset_mock()
        queue.undo()
        assert viewer.values == Values(time=1, figwidthscale=1.3)
        mockredraw.assert_not_called()
    queue.close()


def test_viewer_rejection_marks_the_field_of_its_own_plot() -> None:
    """A rejection marks the text field that gave its own change, and a field that Qt deleted gets no mark.

    The queue kept one field for all the plots, thus a second edit during a plot took the mark of the first plot. A
    new card of a subplot replaces its fields, and a mark of the deleted field raised an error, so the window then
    showed the rejected values.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def render(values: int) -> Callable[[], str | None]:
        time.sleep(0.2)
        return lambda: f"rejected {values}" if values >= 2 else None

    fields = [mock.Mock(name=f"field{index}") for index in range(4)]
    deletedfield = fields[3]
    viewer = mock.Mock(values=0, warning="")
    showvalues = mock.Mock()
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), showvalues, mock.Mock(), render=render)

    def wait_for_plots() -> None:
        deadline = time.perf_counter() + 10.0
        while (queue.rendering is not None or queue.requestedvalues is not None) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    def is_live(field: object) -> bool:
        return field is not deletedfield

    with (
        mock.patch.object(viewerwindow, "get_edited_field", side_effect=fields),
        mock.patch.object(viewerwindow, "is_live", side_effect=is_live),
        mock.patch.object(viewerwindow, "mark_field_error") as mockmark,
        mock.patch.object(viewerwindow, "clear_field_error"),
        mock.patch.object(viewerwindow, "set_plot_busy"),
        mock.patch.object(viewerwindow, "show_plot_banner"),
    ):
        queue.apply(1)
        app.processEvents()
        # the plot of 1 is in progress, and the command accepts it. The command rejects the next change
        queue.apply(2)
        wait_for_plots()
        mockmark.assert_called_once_with(fields[1], "rejected 2")
        queue.apply(3)
        wait_for_plots()
        assert mockmark.call_args == mock.call(fields[2], "rejected 3")
        showvalues.reset_mock()
        queue.apply(4)
        wait_for_plots()
        assert mockmark.call_count == 2, "a field that Qt deleted must get no mark"
        assert showvalues.called, "the window must show the values of the plot after a rejection"
    queue.close()


def test_viewer_dark_colours_keep_the_colours_of_the_series() -> None:
    """In Dark Mode, a black line and a black text take the colour of the text, and a coloured series keeps its colour.

    A black line on the dark background of the window cannot show, and a series needs its colour for the legend. A
    dark colour of an element, e.g. of O, Si, or S, is a colour of a series too.
    """
    import matplotlib.colors as mcolors
    import matplotlib.figure as mplfig

    fig = mplfig.Figure()
    axis = fig.add_subplot()
    blackline = axis.plot([0, 1], [0, 1], color="black", label="model")[0]
    blueline = axis.plot([0, 1], [1, 0], color="tab:blue", label="reference")[0]
    # the colour of sulphur is dark, and it made the series grey
    sulphurline = axis.plot([0, 1], [0.5, 0.5], color="#7d0200", label="S")[0]
    axis.set_xlabel("velocity")
    legend = axis.legend()

    viewermenus.apply_dark_colours(fig, "#1e1e1e", "#dddddd")

    assert mcolors.same_color(blackline.get_color(), "#dddddd")
    assert mcolors.same_color(blueline.get_color(), "tab:blue")
    assert mcolors.same_color(sulphurline.get_color(), "#7d0200")
    assert mcolors.same_color(axis.xaxis.label.get_color(), "#dddddd")
    assert mcolors.same_color(axis.get_facecolor(), "#1e1e1e")
    # the frame of the legend keeps its transparency
    assert mcolors.to_hex(legend.get_frame().get_facecolor()) == "#1e1e1e"
    assert all(mcolors.same_color(text.get_color(), "#dddddd") for text in legend.get_texts())


def test_viewer_default_options_fill_only_the_options_that_the_command_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Settings window gives a new window -figscale and -labelfontsize, and an option of the command has priority.

    plotspectra has no -labelfontsize, thus its window gets no such option.
    """
    settings = {"default-figscale": 1.5, "default-labelfontsize": 12.0}

    def get_float_setting(key: str, default: float) -> float:
        return settings.get(key, default)

    monkeypatch.setattr(viewermenus, "get_float_setting", get_float_setting)
    estimatorparser = viewercore.make_parser(at.estimators.addargs)
    assert viewermenus.add_default_options(estimatorparser, ["Te", "-figscale", "2"]) == [
        "-labelfontsize",
        "12",
        "Te",
        "-figscale",
        "2",
    ]
    spectraparser = viewercore.make_parser(at.spectra.plotspectra.addargs)
    assert viewermenus.add_default_options(spectraparser, ["mymodel"]) == ["-figscale", "1.5", "mymodel"]


def test_viewer_thread_output_keeps_the_output_of_each_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker thread hides the output of its plot, and the window thread still prints to the terminal.

    contextlib.redirect_stdout changed the stream of each thread, thus a print of the window went to the plot.
    """
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stdout", viewercore.ThreadOutput(terminal))
    monkeypatch.setattr(sys, "stderr", viewercore.ThreadOutput(io.StringIO()))
    plotoutput = io.StringIO()
    inside, printed = threading.Event(), threading.Event()

    def plot() -> None:
        with viewercore.send_output(plotoutput, io.StringIO()):
            print("a line of the plot")
            inside.set()
            printed.wait(timeout=10)

    worker = threading.Thread(target=plot)
    worker.start()
    assert inside.wait(timeout=10)
    print("a line of the window")
    printed.set()
    worker.join()
    assert plotoutput.getvalue() == "a line of the plot\n"
    assert terminal.getvalue() == "a line of the window\n"


def test_viewer_cancel_discards_the_plot_in_progress() -> None:
    """Cancel Plot keeps the plot on the screen, and the plot in progress does not replace it when it ends.

    A change after Cancel Plot waits for the end of the discarded plot, and then the queue draws it.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    shown: list[int] = []

    def render(values: int) -> "Callable[[], str | None]":
        time.sleep(0.3)

        def show() -> str | None:
            shown.append(values)
            return None

        return show

    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=render)

    def wait_for_plots() -> None:
        deadline = time.perf_counter() + 10.0
        while (queue.rendering is not None or queue.requestedvalues is not None) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    queue.apply(1)
    app.processEvents()
    assert queue.is_busy()
    queue.cancel()
    assert viewer.values == 0, "the controls must show the values of the plot on the screen"
    assert not queue.is_busy()
    wait_for_plots()
    assert not shown, "the discarded plot must not replace the plot on the screen"
    queue.apply(2)
    app.processEvents()
    queue.cancel()
    queue.apply(3)
    wait_for_plots()
    assert shown == [3]
    assert queue.drawnvalues == viewer.values == 3
    queue.close()


def test_viewer_queue_draws_in_a_worker_thread() -> None:
    """The window stays free during a plot, and a drag during a plot gives a plot of the last values alone.

    The window thread drew each plot, thus a drag of the time slider stopped until the plot ended.
    """
    # the queue needs the timers of Qt alone, and a QCoreApplication loads no plugin of a display
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    renderthreads: list[str] = []
    showthreads: list[str] = []
    rendered: list[int] = []
    # the plot waits until the test releases it. A limit of the wall time failed on a busy machine
    releaseplot = threading.Event()

    def render(values: int) -> Callable[[], str | None]:
        renderthreads.append(threading.current_thread().name)
        rendered.append(values)
        assert releaseplot.wait(timeout=10.0), "the test did not release the plot"

        def show() -> str | None:
            showthreads.append(threading.current_thread().name)
            return "rejected" if values < 0 else None

        return show

    viewer = mock.Mock(values=0)
    afterdraw = mock.Mock()
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), afterdraw, render=render)

    def wait_until(condition: Callable[[], bool]) -> None:
        deadline = time.perf_counter() + 10.0
        while not condition() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    def plots_ended() -> bool:
        return queue.rendering is None and queue.requestedvalues is None

    queue.apply(1)
    wait_until(lambda: rendered == [1])
    # the plot of 1 waits for the release, thus the window thread runs these changes while the plot is in progress
    queue.apply(2)
    queue.apply(3)
    app.processEvents()
    assert rendered == [1], "a change during a plot must wait for the end of that plot"
    assert viewer.values == 3, "the window must show the values of a change at once"
    releaseplot.set()
    wait_until(plots_ended)
    assert rendered == [1, 3]
    assert "MainThread" not in renderthreads
    assert showthreads == ["MainThread", "MainThread"]
    assert viewer.values == queue.drawnvalues == 3

    # a rejection keeps the values of the last plot
    queue.apply(-1)
    wait_until(plots_ended)
    assert viewer.values == queue.drawnvalues == 3
    afterdraw.assert_called_with("rejected")
    queue.close()


def test_viewer_open_model_gives_the_reason_of_the_new_window() -> None:
    """The status line of Open Model must give the error of the first plot of the new window.

    open_window returned only a bool, thus the message took the first line of the traceback on stderr.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

    def open_window(tokens: Sequence[str], windows: Sequence[object]) -> str:
        sys.stderr.write("Traceback (most recent call last):\n")
        return f"ComputeError: the query failed for {tokens[0]} and {len(windows)} window"

    with mock.patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="mymodel"):
        message = viewermenus.open_model_window(mock.Mock(), open_window, [mock.Mock()])
    assert message == "The viewer cannot open mymodel: ComputeError: the query failed for mymodel and 1 window"


def test_viewer_status_line_reads_text_in_colour() -> None:
    """The library rich colours its text under FORCE_COLOR also in a capture, and the status line must read it.

    A colour code in front of "error: " hid the error, thus the status line showed the usage line of argparse.
    """
    warning = "\x1b[1;33mWARNING: every Te value is below the requested minimum\x1b[0m\n"
    assert viewercore.get_last_warning(warning) == "every Te value is below the requested minimum"
    error = "\x1b[1;34musage: \x1b[0martistools [options]\n\x1b[1;31merror: \x1b[0m'q=1' is not a plane or a line\n"
    assert viewercore.get_first_line(error) == "'q=1' is not a plane or a line"


def test_viewer_shift_drag_selects_a_y_range_in_one_frame() -> None:
    """The embedded canvas receives no key events, thus the Shift key comes from the modifiers of the mouse event.

    A release in a different frame gives a y value of a different scale, thus it selects nothing.
    """
    from matplotlib.backend_bases import MouseButton
    from matplotlib.backend_bases import MouseEvent

    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    frames = fig.subplots(2, 1)
    frames[0].set_ylim(0.0, 10.0)
    frames[1].set_ylim(1e5, 1e9)
    frames[1].set_yscale("log")
    canvas.draw()
    xselections: list[tuple[float, float]] = []
    yselections: list[tuple[int, float, float]] = []
    viewerwindow.connect_plot_mouse(
        canvas,
        get_frames=lambda: list(frames),
        get_readout=lambda _event, _frame: "",
        readoutlabel=mock.MagicMock(),
        on_select=lambda low, high: xselections.append((low, high)),
        on_reset=lambda: None,
        can_select=lambda: True,
        on_select_y=lambda index, low, high: yselections.append((index, low, high)),
    )

    def drag(startframe: int, starty: float, endframe: int, endy: float) -> None:
        x0, y0 = frames[startframe].transData.transform((0.5, starty))
        x1, y1 = frames[endframe].transData.transform((0.5, endy))
        for name, x, y in (
            ("button_press_event", x0, y0),
            ("motion_notify_event", x1, y1),
            ("button_release_event", x1, y1),
        ):
            canvas.callbacks.process(
                name, MouseEvent(name, canvas, x, y, button=MouseButton.LEFT, modifiers=frozenset({"shift"}))
            )

    drag(0, 2.0, 0, 8.0)
    assert len(yselections) == 1
    assert yselections[0][0] == 0
    assert np.allclose(yselections[0][1:], (2.0, 8.0), rtol=1e-6, atol=0.0)
    drag(0, 6.0, 1, 1e7)
    assert len(yselections) == 1
    assert not xselections


def test_viewer_save_gives_the_resolution_of_the_command(tmp_path: Path) -> None:
    """Each format of file takes the resolution of the Figure section.

    The estimator viewer dropped -dpi, thus a PDF file lost the resolution of its colour image.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    dpi = 300
    savedtokens: list[list[str]] = []

    def commandmain(argsraw: Sequence[str]) -> None:
        savedtokens.append(list(argsraw))
        Path(argsraw[-1]).write_text("figure", encoding="utf-8")

    from artistools.commands import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    parser.add_argument("-dpi", type=int, default=250)
    parser.add_argument("-figscale", type=float, default=1.0)
    statusbar = mock.Mock()
    for suffix in ("png", "pdf"):
        filename = str(tmp_path / "plot")

        with (
            mock.patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(filename, "")),
            mock.patch.object(viewermenus, "show_wait_cursor", contextlib.nullcontext),
        ):
            viewermenus.save_figure_of_command(
                mock.Mock(), statusbar, commandmain, "plotspectra", ["-xmin", "5"], parser, (suffix, dpi)
            )
        # a name with no suffix takes the suffix of the selected format
        assert savedtokens[-1] == ["-xmin", "5", "-dpi", "300", "-o", f"{filename}.{suffix}"]
        statusbar.message.setText.assert_called_with(f"Saved {filename}.{suffix}")


def test_viewer_copy_gives_the_file_of_the_selected_format() -> None:
    """Copy Figure runs the command for a file of the selected format, and puts the file on the clipboard.

    Qt gave only a TIFF image to the clipboard of macOS, thus a copy could not give a PDF or an SVG file.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from artistools.commands import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    parser.add_argument("-dpi", type=int, default=250)
    parser.add_argument("-figscale", type=float, default=1.0)
    commands: list[list[str]] = []

    def commandmain(argsraw: Sequence[str]) -> None:
        commands.append(list(argsraw))
        Path(argsraw[-1]).write_text("<svg/>", encoding="utf-8")

    def run_task(task: Callable[[], str | None], _statustext: str, on_done: Callable[[str | None], None]) -> bool:
        on_done(task())
        return True

    queue = mock.Mock(run_task=run_task)
    statusbar = mock.Mock()
    with mock.patch.object(viewermenus, "put_file_on_clipboard", return_value=None) as mockclipboard:
        viewermenus.copy_figure_of_command(
            queue, statusbar, commandmain, parser, ["-xmin", "5", "-dpi", "300"], ("svg", 150)
        )
    assert commands[0][:4] == ["-xmin", "5", "-dpi", "150"]
    assert commands[0][-1].endswith("figure.svg")
    mockclipboard.assert_called_once_with(b"<svg/>", "svg")
    statusbar.message.setText.assert_called_with("Copied the figure as SVG")


def test_viewer_row_wraps_its_groups() -> None:
    """A group of a row goes to a new line when the line is full, and a hidden group takes no place and no gap."""
    assert viewerwidgets.get_wrapped_lines([100, 100, 100], 250, 12) == [[0, 1], [2]]
    assert viewerwidgets.get_wrapped_lines([100, 0, 100], 212, 12) == [[0, 1, 2]]
    # a group wider than the line has a line of its own
    assert viewerwidgets.get_wrapped_lines([50, 400, 50], 300, 12) == [[0], [1], [2]]


def test_viewer_queue_runs_a_task_between_plots() -> None:
    """A task of the worker thread, e.g. Reload Data, waits for the plot in progress, and a new plot waits for it.

    A plot that the user asked for during a reload took the time of the reload as its plot time. Play then made no
    pause.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    events: list[str] = []

    def render(values: int) -> Callable[[], str | None]:
        events.append(f"plot {values}")
        time.sleep(0.05)
        return lambda: None

    def task() -> str | None:
        events.append("task")
        time.sleep(0.5)
        return None

    statusbar = mock.Mock()
    ondone = mock.Mock()
    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, statusbar, mock.Mock(), mock.Mock(), render=render)
    queue.apply(1)
    app.processEvents()
    assert queue.run_task(task, "Reload in progress...", ondone)
    assert not queue.run_task(task, "Reload in progress...", ondone), "a second task must wait for the first"
    queue.apply(2)
    deadline = time.perf_counter() + 10.0
    while (
        queue.rendering is not None or queue.task is not None or queue.requestedvalues is not None
    ) and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert events == ["plot 1", "task", "plot 2"]
    ondone.assert_called_once_with(None)
    assert queue.plotseconds < 0.4, "the time of the task is not the time of a plot"
    assert "Reload in progress..." in [call.args[0] for call in statusbar.drawtime.setText.call_args_list]
    queue.close()


def test_viewer_typed_centre_gives_back_the_range() -> None:
    """The centre that the time field shows gives back the same range of timesteps, also for an even count.

    The viewer took the timestep that holds the centre as the middle, and an even range then moved one timestep.
    """
    tmids = at.get_timestep_times(at.get_path("testdata") / "testmodel", loc="mid")
    for count in (1, 2, 3, 4):
        for start in range(len(tmids) - count + 1):
            centre = float(f"{(tmids[start] + tmids[start + count - 1]) / 2.0:.4g}")
            assert viewercore.get_nearest_range_start(tmids, centre, count) == start, (count, start)


def test_viewer_option_rows_split_a_group_of_switches() -> None:
    """Argparse reads -qv as -q and -v, and the option table must give each switch its own row.

    The table read -qv as the flag -q with the value "v", and the command then held a stray positional argument.
    """
    parser = viewercore.make_parser(at.estimators.addargs)
    rows, othertokens = viewercore.split_option_rows(parser, ["Te", "mymodel", "-qv", "-qt300", "-xmin", "5"])
    assert rows == (("--quiet", ()), ("--verbose", ()), ("--quiet", ()), ("-timedays", ("300",)), ("-xmin", ("5",)))
    assert othertokens == ["Te", "mymodel"]


def test_viewer_removes_an_option_of_two_values_with_its_values_alone() -> None:
    """An option of nargs 2 takes two values, thus the tokens after them stay in the command.

    remove_options took each token up to the next flag, as for nargs "*", thus the command lost a path after the values.
    """
    parser = viewercore.make_parser(at.spectra.plotspectra.addargs)
    tokens = ["-emissionvelocityrange", "1000", "2000", "mymodel", "-xmin", "5"]
    assert viewercore.remove_options(parser, tokens, {"emissionvelocityrange"}) == ["mymodel", "-xmin", "5"]
    rows, othertokens = viewercore.split_option_rows(parser, tokens)
    assert rows == (("-emissionvelocityrange", ("1000", "2000")), ("-xmin", ("5",)))
    assert othertokens == ["mymodel"]


# each viewer with the arguments of a plot that sets several of its controls
VIEWER_CASES: t.Final = (
    ("spectra", [str(modelpath), "-t", "300", "-xmin", "3000", "-label", "model", "--interactive"]),
    (
        "lightcurve",
        [
            str(modelpath_classic_3d),
            "-deposition",
            "gamma",
            "-thermalisation",
            "gamma",
            "--showbarnes",
            "--interactive",
        ],
    ),
    ("estimators", ["Te", str(modelpath), "-timestep", "50", "--interactive"]),
)


def make_viewer(kind: str, tokens: "Sequence[str]") -> t.Any:
    """Return the viewer of a command, with a canvas and no window, after its first plot."""
    module = importlib.import_module(f"artistools.{kind}.interactive")
    viewerclass = next(
        value for name, value in vars(module).items() if name.endswith("Viewer") and isinstance(value, type)
    )
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = viewerclass(tokens, fig)
    assert viewer.draw() is None
    return viewer


@pytest.mark.parametrize(("kind", "timetokens"), [("spectra", ["-t", "300"]), ("lightcurve", [])])
def test_viewer_takes_the_model_of_the_path_option(
    kind: str, timetokens: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A viewer must plot the model of -modelpath, and not read the working folder, which is not an ARTIS run."""
    monkeypatch.chdir(tmp_path)
    viewer = make_viewer(kind, ["-modelpath", str(modelpath), *timetokens, "--interactive"])
    seriespaths = viewer.values.spectra if kind == "spectra" else viewer.values.lightcurves
    assert seriespaths == (str(modelpath),)
    assert "-modelpath" not in viewer.get_command()


@pytest.mark.parametrize(("kind", "tokens"), VIEWER_CASES)
def test_viewer_command_opens_the_same_plot(kind: str, tokens: list[str]) -> None:
    """The command that a viewer shows must open a viewer with the same command.

    A user copies the command of a window and runs it later, e.g. in a script. An option that the window gives but
    does not read back, or reads in a different way, then gives a different plot.
    """
    viewer = make_viewer(kind, tokens)
    command = viewer.get_command()
    again = make_viewer(kind, [*shlex.split(command)[2:], "--interactive"])
    assert again.get_command() == command


@pytest.mark.parametrize(("kind", "tokens"), VIEWER_CASES)
def test_viewer_keeps_its_plot_after_a_rejected_change(kind: str, tokens: list[str]) -> None:
    """A change that the command rejects must keep the values and the figure of the last plot, and give the reason."""
    viewer = make_viewer(kind, tokens)
    oldvalues, oldfig = viewer.values, viewer.fig
    badvalues = dc.replace(oldvalues, otheroptions=(*oldvalues.otheroptions, ("-figscale", ("abc",))))
    message = viewer.change(badvalues)
    assert message is not None
    assert "-figscale" in message
    assert viewer.values == oldvalues
    assert viewer.fig is oldfig


def test_figure_shows_data_only_inside_the_axis_limits() -> None:
    """The window shows the note of an empty plot when no point of a line is inside the limits of its axes."""
    fig = mplfig.Figure()
    axis = fig.add_subplot()
    axis.plot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert viewerwidgets.figure_shows_data(fig)
    axis.set_xlim(10.0, 20.0)
    assert not viewerwidgets.figure_shows_data(fig)
    # an inverted axis has its limits in the other order, and a point inside them shows
    axis.set_xlim(3.5, 0.5)
    assert viewerwidgets.figure_shows_data(fig)
    axis.set_ylim(5.0, 6.0)
    assert not viewerwidgets.figure_shows_data(fig)
    axis.imshow([[1.0, 2.0]])
    assert viewerwidgets.figure_shows_data(fig)
    # a segment crosses the frame between two points, e.g. a time range between two timesteps
    crossfig = mplfig.Figure()
    crossaxis = crossfig.add_subplot()
    crossaxis.plot([0.0, 10.0], [0.0, 100.0])
    crossaxis.set_xlim(4.0, 6.0)
    assert viewerwidgets.figure_shows_data(crossfig)
    crossaxis.set_ylim(70.0, 90.0)
    assert not viewerwidgets.figure_shows_data(crossfig)


def test_direction_list_maps_all_directions_to_no_direction_option() -> None:
    """All the directions alone give no direction option, and the virtual packets never combine with the average."""
    choose = viewersections.get_new_direction_choice
    nooption = viewersections.DirectionChoice(kind="", bins=(), usedegrees=False)
    assert choose("bin", (-1,), (-1,), usedegrees=False, onebin=False) == nooption
    assert choose("phi", (-1, 3), (3,), usedegrees=False, onebin=False).bins == (-1, 3)
    # the virtual packets have no average, thus all the directions give the plot of the real packets
    assert choose("vpkt", (-1, 2), (-1,), usedegrees=False, onebin=False) == nooption
    assert choose("vpkt", (-1, 2), (2,), usedegrees=False, onebin=False).bins == (2,)
    # the average with no observer of the virtual packets is all the directions, and not a list with no bin
    assert choose("vpkt", (-1,), (), usedegrees=False, onebin=False) == nooption
    # a plot of one bin takes the bin of the click
    assert choose("bin", (0, 5), (5,), usedegrees=False, onebin=True).bins == (5,)
    assert choose("bin", (-1, 5), (-1,), usedegrees=False, onebin=True) == nooption


def test_mathtext_lock_wraps_the_parser_one_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lock of the mathtext parser must install on each Python version and keep the parse of matplotlib.

    The wrapper had annotations that Python 3.13 evaluates at once, and MathTextParser takes no subscript, thus no
    viewer opened on Python 3.13.
    """
    import matplotlib.mathtext as mplmathtext

    # monkeypatch restores the parse of matplotlib after the test
    monkeypatch.setattr(mplmathtext.MathTextParser, "parse", mplmathtext.MathTextParser.parse)
    viewerapplication.serialise_mathtext_parser()
    wrapped = mplmathtext.MathTextParser.parse
    viewerapplication.serialise_mathtext_parser()
    assert mplmathtext.MathTextParser.parse is wrapped
    width, height, *_ = mplmathtext.MathTextParser("path").parse(r"$10^{-3}$", dpi=72)
    assert width > 0
    assert height > 0


def test_direction_data_follows_the_files_and_the_data_source() -> None:
    """A direction bin needs a *_res.out file or the packets with --frompackets, and an observer needs vpkt.txt.

    The commands read the packets for a direction bin only with --frompackets. The viewer enabled the bins of a run
    with only the packets, and the plot then showed all the directions. It disabled the observers of a run with only
    the vspecpol files.
    """
    has_data = viewercore.run_has_direction_data
    datafolder = at.get_path("testdata")
    lightcurveres = ("light_curve_res.out",)
    # testmodel has the packets files and no light_curve_res.out
    assert not has_data(datafolder / "testmodel", "bin", lightcurveres, "", frompackets=False)
    assert has_data(datafolder / "testmodel", "bin", lightcurveres, "", frompackets=True)
    # vspecpolmodel has vpkt.txt and the vspecpol files, and no packets
    assert has_data(datafolder / "vspecpolmodel", "vpkt", ("spec_res.out",), "vspecpol*", frompackets=False)
    assert not has_data(datafolder / "vspecpolmodel", "vpkt", lightcurveres, "", frompackets=True)
    # vpktcontrib has the virtual packets
    assert has_data(datafolder / "vpktcontrib", "vpkt", lightcurveres, "", frompackets=True)
    assert not has_data(datafolder / "vpktcontrib", "vpkt", lightcurveres, "", frompackets=False)


def test_series_styles_keep_the_values_after_the_paths() -> None:
    """A style value of a series that the command adds after the paths, e.g. -obsspec, must stay on that series."""
    rows = (("-label", ("A", "B", "R")),)
    # a new order of the two models keeps R on the third series
    assert viewercore.move_series_styles(rows, ("a", "b"), ("b", "a")) == (("-label", ("B", "A", "R")),)
    # a new model goes before the series of the command
    assert viewercore.move_series_styles(rows, ("a", "b"), ("a", "b", "c")) == (("-label", ("A", "B", "default", "R")),)
    assert viewercore.set_series_rows(rows, ("a", "b"), "a", {"-label": "N"}) == (("-label", ("N", "B", "R")),)


def test_viewer_queue_polls_a_task_that_after_draw_starts() -> None:
    """A task that after_draw starts must end, and a later plot must start after it.

    show_rendered stopped the timer after after_draw. A task that after_draw started then had no timer, and the queue
    refused each later plot.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    ondone = mock.Mock()
    rendered: list[int] = []

    def render(values: int) -> Callable[[], str | None]:
        rendered.append(values)
        return lambda: None

    def after_draw(_message: str | None) -> None:
        if not ondone.called and queue.task is None:
            queue.run_task(lambda: None, "Task in progress...", ondone)

    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), after_draw, render=render)

    def wait_for_queue() -> None:
        deadline = time.perf_counter() + 10.0
        while (
            queue.rendering is not None or queue.task is not None or queue.requestedvalues is not None
        ) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    with mock.patch.object(viewerwindow, "set_plot_busy"), mock.patch.object(viewerwindow, "show_plot_banner"):
        queue.apply(1)
        wait_for_queue()
        ondone.assert_called_once_with(None)
        queue.apply(2)
        wait_for_queue()
    assert rendered == [1, 2]
    queue.close()


def test_viewer_cancel_keeps_the_status_of_a_task() -> None:
    """Cancel Plot during a task, e.g. Reload Data, keeps the spinner and the status text of the task.

    The cancel hid the spinner and showed "Plot cancelled" while the task still ran in the worker thread.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    statusbar = mock.Mock()
    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, statusbar, mock.Mock(), mock.Mock(), render=mock.Mock())
    with mock.patch.object(viewerwindow, "set_plot_busy") as mockbusy:
        queue.task = lambda: None
        # a change of the user during the task waits for the end of the task
        queue.requestedvalues = 1
        queue.cancel()
        assert queue.requestedvalues is None
        mockbusy.assert_not_called()
        assert mock.call("Plot cancelled") not in statusbar.drawtime.setText.call_args_list
        queue.task = None
        queue.requestedvalues = 1
        queue.cancel()
        mockbusy.assert_called_once()
        statusbar.drawtime.setText.assert_called_with("Plot cancelled")
    queue.close()


def test_viewer_session_keeps_an_empty_token() -> None:
    """An empty value, e.g. of -label "", stays empty in the command of the next start.

    Path("") is the working folder, and it exists. Thus the empty label became the path of the working folder.
    """
    tokens = viewerapplication.get_absolute_tokens([str(modelpath), "-label", "", "-title", ""])
    assert tokens == [str(modelpath.absolute()), "-label", "", "-title", ""]


def test_viewer_starts_on_a_qt_platform_with_no_display() -> None:
    """A Qt platform that needs no display, e.g. offscreen or vnc, starts with no DISPLAY.

    The viewer stopped whenever DISPLAY and WAYLAND_DISPLAY were not set, and QT_QPA_PLATFORM=offscreen then failed.
    """
    needs_missing_display = viewerapplication.needs_missing_display
    assert needs_missing_display({})
    assert needs_missing_display({"QT_QPA_PLATFORM": "xcb"})
    assert needs_missing_display({"QT_QPA_PLATFORM": "wayland;xcb"})
    assert not needs_missing_display({"DISPLAY": ":0"})
    assert not needs_missing_display({"WAYLAND_DISPLAY": "wayland-0"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "offscreen"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "vnc:size=1280x800"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "wayland;eglfs"})


def test_viewer_bundle_keeps_a_copy_of_the_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy of the Python executable in the macOS bundle stays at the next start if the executable is the same.

    A copy on a different volume is not the same file as the executable. Thus each start copied the executable again.
    """
    baseexecutable = tmp_path / "python3"
    baseexecutable.write_bytes(b"executable")
    monkeypatch.setattr(sys, "executable", str(baseexecutable))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(viewerapplication, "LSREGISTER", tmp_path / "nolsregister")

    def refuse_hardlink(_path: Path, _target: Path) -> None:
        msg = "a hard link must be on the same volume"
        raise OSError(msg)

    monkeypatch.setattr(Path, "hardlink_to", refuse_hardlink)
    executable = viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",))
    assert executable.read_bytes() == b"executable"
    inode = executable.stat().st_ino
    assert viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",)) == executable
    assert executable.stat().st_ino == inode, "the same executable must not give a new copy"

    # a new Python executable replaces the copy
    baseexecutable.write_bytes(b"new executable")
    viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",))
    assert executable.read_bytes() == b"new executable"


def test_viewer_reload_clears_the_frequency_grid(tmp_path: Path) -> None:
    """Reload Data clears the cache of the frequency grid of spec.out, as it clears the cache of the spectra.

    exspec can write spec.out again with a different grid. The viewer then read the new spectra with the old grid.
    """

    def write_spec(nus: Sequence[float]) -> None:
        lines = ["0 1.0 2.0", *(f"{nu:g} 0.0 0.0" for nu in nus)]
        (tmp_path / "spec.out").write_text("\n".join(lines) + "\n")

    write_spec([1e14, 2e14])
    assert len(at.misc.get_nu_grid(tmp_path)) == 2
    write_spec([1e14, 2e14, 3e14, 4e14])
    viewerwindow.clear_output_caches()
    assert len(at.misc.get_nu_grid(tmp_path)) == 4


def test_viewer_command_tokens_read_the_words_of_the_dispatcher() -> None:
    """A call of the dispatcher from Python code gives the words of that call, without the name of the subcommand."""
    args = argparse.Namespace(dispatcherargsraw=["plotspectra", "mymodel", "--interactive"])
    assert viewercore.get_command_tokens(args, None, {}, fromdispatcher=True) == ["mymodel", "--interactive"]
    assert viewercore.get_command_tokens(args, ["other"], {}, fromdispatcher=True) == ["other"]


def test_viewer_finds_each_path_option() -> None:
    """Each option form of the paths of the positional argument gives a series of the viewer, e.g. -specpath."""
    for module, flags in (
        (at.spectra.plotspectra, {"-specpath", "-modelpath"}),
        (at.lightcurve.plotlightcurve, {"-modelpath"}),
    ):
        assert viewercore.get_path_option_flags(viewercore.make_parser(module.addargs)) == flags
    viewertokens = viewercore.parse_viewer_tokens(
        at.spectra.plotspectra.addargs, ["-specpath", str(modelpath), "-t", "300"], set()
    )
    assert viewertokens.paths == [str(modelpath)]


def test_viewer_python_code_gives_the_changed_arguments() -> None:
    """The Python code gives each argument that differs from its default, and a rejected command gives a comment."""
    parser = viewercore.make_parser(at.lightcurve.plotlightcurve.addargs)
    code = viewerwidgets.get_python_code(parser, ["mymodel", "--notitle"], "plotlightcurves", "at.lightcurve.plot")
    assert code.startswith("import artistools as at\n\nat.lightcurve.plot(\n")
    assert "notitle=True" in code
    rejected = viewerwidgets.get_python_code(parser, ["-xmin", "abc"], "plotlightcurves", "at.lightcurve.plot")
    assert rejected == "# plotlightcurves rejects the command"
