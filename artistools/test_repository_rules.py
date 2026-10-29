import importlib.util
import os
import re
import shutil
import subprocess
import sys
import typing as t
from pathlib import Path

import pytest

import artistools as at


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
        f"{firstimport}import artistools; artistools.__name__; "
        "import numpy, polars._dependencies, polars.series.series; "
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


def test_external_packages_find_their_names() -> None:
    """The packages pynonthermal and artisatomic read these names, thus a release that moves one stops them."""
    import importlib

    externalnames = {
        "artistools": ["get_composition_data", "get_elsymbol", "get_ionstring"],  # artisatomic, pynonthermal
        "artistools.atomic": ["get_levels"],  # pynonthermal
    }
    for modulename, names in externalnames.items():
        module = importlib.import_module(modulename)
        assert not [name for name in names if not callable(getattr(module, name, None))], modulename

    assert at.get_composition_data is at.atomic.get_composition_data


TOPLEVEL_API: t.Final[frozenset[str]] = frozenset({
    "add_derived_cols_to_modeldata", "decode_roman_numeral", "firstexisting", "get_atomic_number",
    "get_composition_data", "scan_deposition", "get_elsymbol", "get_inputparams", "get_ion_tuple", "get_ionstring",
    "get_model_name", "scan_modeldata", "get_nprocs", "get_path", "get_timestep_of_timedays", "get_timestep_times",
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


def test_reader_prefix_tells_lazy_from_eager(package_modules: list[Path]) -> None:
    """A reader with the prefix scan_ returns a LazyFrame, and a reader with the prefix read_ returns eager data.

    The README gives this rule to the users. read_spec returned a LazyFrame, and get_spectra returned a dictionary of
    them, thus the name did not tell a user when the reader read the data. The test read only the core modules and
    misc, thus it did not find scan_cell_estimators in writecomparisondata, which returned a DataFrame.
    """
    import ast

    packagefolder = Path(at.__file__).parent
    wrongnames = []
    for module in package_modules:
        for node in ast.parse(module.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and node.returns is not None:
                islazy = "LazyFrame" in ast.unparse(node.returns)
                if (node.name.startswith("read_") and islazy) or (node.name.startswith("scan_") and not islazy):
                    wrongnames.append(f"{module.relative_to(packagefolder)}:{node.name}")

    assert not wrongnames


def test_each_package_command_is_named_plot() -> None:
    """A package that has a plot command gives it as plot, thus a user finds it under one name."""
    for package in (
        at.estimators,
        at.gsinetwork,
        at.lightcurve,
        at.logs,
        at.nltepops,
        at.nonthermal,
        at.packets,
        at.radfield,
        at.spectra,
    ):
        # the main function of a module of the package, and not a different callable with that name
        assert package.plot.__name__ == "main", package.__name__
        assert package.plot.__module__.startswith(f"{package.__name__}."), package.__name__


def test_package_modules_import_no_package_alias(package_modules: list[Path]) -> None:
    """A package module must import each name from the module that defines it.

    An alias of the top-level package hides an import cycle until a different module comes first. Python 3.15
    binds "import artistools.spectra.core as atspectra" to the package, thus only a re-exported name resolves.
    """
    aliasimport = re.compile(r"^\s*import artistools(\.[\w.]+)? as \w+", re.MULTILINE)
    packagedir = Path(at.__file__).parent
    # a test can use the alias
    offenders = [
        str(path.relative_to(packagedir))
        for path in package_modules
        if aliasimport.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"these package modules import a package alias: {offenders}"


def test_codex_finds_the_claude_setup() -> None:
    """Codex reads the skill, the hooks, and the reviewer of Claude Code through links and one small file.

    Codex rejects a key of hooks.json other than description and hooks, e.g. the attribution of Claude Code, and it
    then runs no hook.
    """
    import json
    import shutil
    import subprocess
    import tomllib

    repo = Path(at.__file__).resolve().parent.parent
    assert (repo / ".agents" / "skills" / "artis-data").resolve() == at.get_path(
        "artistools_dir"
    ) / "data" / "skills" / "artis-data"
    assert (repo / ".codex" / "hooks.json").resolve() == (repo / ".claude" / "settings.json").resolve()
    settings = json.loads((repo / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert set(settings) <= {"description", "hooks"}
    # Codex sets no CLAUDE_PROJECT_DIR, thus each command finds the root of the checkout with git
    commands = [hook["command"] for groups in settings["hooks"].values() for group in groups for hook in group["hooks"]]
    assert all(command.startswith('"${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"/') for command in commands)

    agentrole = tomllib.loads((repo / ".codex" / "agents" / "ste-style-reviewer.toml").read_text(encoding="utf-8"))
    agentmd = (repo / ".claude" / "agents" / "ste-style-reviewer.md").read_text(encoding="utf-8")
    descriptionline = next(line for line in agentmd.splitlines() if line.startswith("description: "))
    assert agentrole["description"] == descriptionline.removeprefix("description: ")

    # Codex edits with apply_patch, and the hooks read the files from the text of the patch
    if shutil.which("jq") is None:
        pytest.skip("the hooks need jq")
    hookinput = json.dumps({
        "cwd": "/repo",
        "tool_input": {
            "command": "*** Begin Patch\n*** Update File: a.py\n*** Update File: b.py\n*** Move to: c.py\n"
            "*** Delete File: uv.lock\n"
        },
    })
    # the command is a script of this repository with a fixed input, and no shell
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [repo / ".claude" / "hooks" / "get-edited-files.sh"],
        input=hookinput,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == ["/repo/a.py", "/repo/b.py", "/repo/c.py", "/repo/uv.lock"]

    # Codex reads a JSON object from the stdout of a Stop hook, and an empty stdout is not one
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [repo / ".claude" / "hooks" / "verify-tree.sh"],
        input=json.dumps({"stop_hook_active": True}),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {}


def test_stop_hook_checks_a_failed_tree_again(tmp_path: Path) -> None:
    """The hook wrote the state of a tree before a failed check, thus the next stop found no change and passed."""
    gitpath = shutil.which("git")
    if gitpath is None:
        pytest.skip("the hook needs git")
    repo = Path(at.__file__).resolve().parent.parent
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "module.py").write_text("x = 1\n", encoding="utf-8")
    # the stand-in for uv makes each check fail fast, and it changes no file
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "uv").write_text("#!/bin/sh\necho 'an error'\nexit 1\n", encoding="utf-8")
    (bindir / "uv").chmod(0o755)
    env = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": str(checkout),
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
    }
    # the commands are git and a script of this repository with fixed arguments, and no shell
    for gitargs in (
        ["init", "-q"],
        ["add", "module.py"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a"],
    ):
        subprocess.run([gitpath, *gitargs], cwd=checkout, check=True, env=env)  # ruff:ignore[subprocess-without-shell-equals-true]

    def run_stop_hook() -> int:
        completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            [repo / ".claude" / "hooks" / "verify-tree.sh"],
            input="{}",
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        return completed.returncode

    assert run_stop_hook() == 2
    assert run_stop_hook() == 2


# the code of each order of import, which checks the names that broke after PySide6 loaded
LAZY_IMPORT_CHECK_CODE: t.Final = """
import importlib.util
import sys
import types

qtfirst = sys.argv[1] == "qtfirst"
hasqt = importlib.util.find_spec("PySide6") is not None
if qtfirst:
    from PySide6 import QtCore
import artistools
if not qtfirst:
    # a search loads no Qt package, thus the imports of artistools stay lazy
    importlib.util.find_spec("PySide6")
    assert sys.get_lazy_imports() == "all"
    if hasqt:
        # the imports of a script are eager, thus PySide6 runs one time and keeps each of its modules
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PySide6 import QtCore
        from PySide6 import QtWidgets
        import PySide6
        assert PySide6.QtCore is QtCore and PySide6.QtWidgets is QtWidgets
    else:
        # the loader of a Qt package runs this function, and the test calls it in place of that load
        artistools.lazyimports.resolve_lazy_imports()
pending = [
    f"{module.__name__}.{name}"
    for module in list(sys.modules.values())
    if module.__name__.startswith("artistools")
    for name, value in list(vars(module).items())
    if isinstance(value, getattr(types, "LazyImportType", ()))
]
assert not pending, pending
from artistools.misc import print_error
assert callable(print_error)
if hasqt:
    import kiwisolver
    assert isinstance(kiwisolver.__version__, str)
"""


@pytest.mark.skipif(not hasattr(sys, "set_lazy_imports"), reason="only Python 3.15 and later have lazy imports")
@pytest.mark.parametrize("order", ["qtfirst", "artistoolsfirst"])
def test_lazy_imports_survive_the_qt_bindings(order: str) -> None:
    """No lazy import of artistools stays pending when PySide6 loads, in either order of import.

    After PySide6 loads, the attribute of a module gave a pending lazy import as it was, e.g. print_error of artistools
    and the version of kiwisolver. A window then stopped with "'lazy_import' object is not callable". The guard then
    resolved the imports inside the search of PySide6. A lazy import of PySide6 in a script thus ran the package two
    times, and a search alone made the imports eager. The test runs in a new interpreter, because the order of the
    imports decides the result.
    """
    if order == "qtfirst" and importlib.util.find_spec("PySide6") is None:
        pytest.skip("the import of PySide6 before artistools needs PySide6")
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", LAZY_IMPORT_CHECK_CODE, order], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
