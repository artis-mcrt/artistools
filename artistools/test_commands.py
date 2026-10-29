import argparse
import importlib
import itertools
import os
import re
import subprocess
import sys
import tomllib
import typing as t
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

import artistools as at
from artistools.misc import model_path_from_text
from artistools.viewertools import core as viewercore

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
REPOPATH = at.get_path("artistools_repository")


# each retired top-level name, with the module that its inputmodel command runs
RETIRED_COMMANDS = (
    ("makeartismodelfromparticlegridmap", "artistools.inputmodel.modelfromhydro"),
    ("maptogrid", "artistools.inputmodel.maptogrid"),
)


DISPATCHERTARGET = "artistools.__main__:main"


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
    assert helptext.startswith("Usage: plotartisestimators [options]")
    # the parser holds this one command, thus it neither lists nor imports the other commands
    assert "-modelpath" in helptext
    assert "plotspectra" not in helptext

    parser = at.commands.build_script_parser("plotartisestimators")
    assert parser is not None
    assert parser.parse_args([]).func.__module__ == "artistools.estimators.plotestimators"

    for dispatcher in at.commands.DISPATCHERSCRIPTS:
        assert at.commands.build_script_parser(dispatcher) is None


@pytest.mark.parametrize(
    "addargs",
    [at.spectra.plotspectra.addargs, at.lightcurve.plotlightcurve.addargs],
    ids=["plotspectra", "plotlightcurves"],
)
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
@pytest.mark.parametrize("optionsafterselection", [False, True])
@pytest.mark.parametrize("pathkind", ["folder", "file"])
def test_residual_selection_joins_paths_before_options(
    command: str,
    selection: list[str],
    earlieroptions: list[str],
    optionsafterselection: bool,
    pathkind: str,
    tmp_path: Path,
) -> None:
    """Residual paths keep their order across options, separators, and joined selections."""
    import artistools.__main__

    paths = [tmp_path / "model0", tmp_path / "model1"]
    for path in paths:
        if pathkind == "folder":
            path.mkdir()
            (path / "input.txt").touch()
            (path / "model.txt").touch()
        else:
            path.touch()
    options = [*selection, *earlieroptions] if optionsafterselection else [*earlieroptions, *selection]
    tokens = [command, str(paths[0]), *options, str(paths[1])]
    tokens = at.misc.separate_trailing_folders(tokens)
    if pathkind == "folder" and selection == ["-residual", "1"]:
        assert "--" in tokens
    args = artistools.__main__.build_parser().parse_args(tokens)
    assert args.residuals == ([] if selection == ["-residual"] else [1])
    assert args.residualtype == "relative"
    assert list(map(str, args.specpath if command == "plotspectra" else args.modelpath)) == list(map(str, paths))


def test_residual_path_rewrite_keeps_an_ambiguous_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    """The path rewrite leaves an ambiguous prefix for argparse to reject."""
    parser = at.misc.argparser.SuggestingArgumentParser()
    at.misc.addarg_residuals(parser)
    parser.add_argument("-reset", action="store_true")
    parser.add_argument("paths", nargs="*")
    assert parser.split_joined_flags(["-resi", "model1", "model2"]) == ["-resi", "model1", "model2"]
    with pytest.raises(SystemExit):
        parser.parse_args(["-resi", "model1", "model2"])
    assert "Ambiguous option: -resi" in capsys.readouterr().err


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
                assert flags == {"-maxpacketfiles", "-maxpacketsfiles", "--maxpacketfiles", "--maxpacketsfiles"}, label
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
    from artistools.misc import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    at.plottransitions.addargs(parser)
    parser.add_standard_flags()
    assert parser.parse_args(["--atomicdatabase", "kurucz"]).atomicdatabase == "kurucz"
    assert parser.parse_args(["-atomicdatabase", "nist"]).atomicdatabase == "nist"
    assert parser.parse_args([]).atomicdatabase == "artis"

    parser = argparse.ArgumentParser()
    at.estimators.plotestimators.addargs(parser)
    assert parser.parse_args(["-scalefigwidth", "2.5"]).figwidthscale == pytest.approx(2.5)
    assert parser.parse_args(["-figwidthscale", "2.5"]).figwidthscale == pytest.approx(2.5)
    assert parser.parse_args([]).figwidthscale == 1

    parser = SuggestingArgumentParser()
    at.plotviewingangles.addargs(parser)
    parser.add_standard_flags()
    for rawargs in (
        ["model.txt", "--outfile", "vis.html", "--opacity", "0.5", "-s", "10"],
        ["model.txt", "-outputfile", "vis.html", "-opacity", "0.5", "-surface_count", "10"],
    ):
        args = parser.parse_args(rawargs)
        # -o gives a Path on every command, thus the older spelling gives one as well
        assert args.outputfile == Path("vis.html")
        assert args.opacity == pytest.approx(0.5)
        assert args.surface_count == 10


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
    """A bad argument value must exit with a one-line error on stderr instead of a traceback.

    The type of -timedays reads the time, thus argparse refuses the value with its exit status 2 before any read.
    """
    import artistools.__main__

    with pytest.raises(SystemExit) as excinfo:
        artistools.__main__.main(argsraw=["plotspectra", str(modelpath), "-timedays", "banana"])
    assert excinfo.value.code == 2
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


@pytest.mark.parametrize(
    ("comp_line", "expected"),
    [("at plotsp", "plotspectra"), ("at spec --timed", "--timedays"), ("at spec --print", "--print-data")],
)
def test_cli_tab_completion(tmp_path: Path, comp_line: str, expected: str) -> None:
    """Tab completion must offer subcommand names and the standard spelling of the options of a subcommand."""
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
    # the help hides the older spellings, e.g. --print_data, thus tab completion does not offer them
    assert "--print_data" not in completions


def test_every_command_that_saves_a_plot_has_darkmode_show_and_open(package_modules: list[Path]) -> None:
    """--darkmode, --show, and --open apply to each plot file, thus each command that saves a plot must accept them.

    Only the commands with --show had --darkmode, thus e.g. plotlogfiles wrote a light PDF file. A command can save
    the plot in a module that it imports, e.g. makeartismodel with --downscaleplot, thus the check follows the imports.
    """
    import inspect
    import re

    # matplotlib, plotly, and pyvista each have a function that writes a plot file
    savesplot = re.compile(r"save_figure\(|savefig\(|write_image\(|write_html\(|screenshot=")
    packagefolder = Path(at.__file__).parent
    # plottools defines save_figure, and each module that draws a plot imports it
    writermodules = {
        "artistools." + ".".join(path.relative_to(packagefolder).with_suffix("").parts)
        for path in package_modules
        if path.name not in {"plottools.py", "__init__.py"} and savesplot.search(path.read_text(encoding="utf-8"))
    }

    def walktree(tree: at.commands.CommandTree) -> list[at.commands.CommandSpec]:
        return [spec for node in tree.values() for spec in (walktree(node) if isinstance(node, dict) else [node])]

    plotmodules = 0
    missingflags: list[str] = []
    for spec in walktree(at.commands.subcommandtree):
        module = importlib.import_module(f"artistools.{spec.module}")
        importedmodules = set(re.findall(r"^from (artistools[\w.]*) import", inspect.getsource(module), re.MULTILINE))
        if module.__name__ not in writermodules and not importedmodules & writermodules:
            continue
        plotmodules += 1
        parser = argparse.ArgumentParser()
        module.addargs(parser)
        actions = parser._actions  # ruff:ignore[private-member-access]
        flags = {flag for action in actions for flag in action.option_strings}
        assert "--darkmode" in flags, f"{spec.module} saves a plot but has no --darkmode"
        missingflags.extend(f"{spec.module} {flag}" for flag in ("--show", "--open") if flag not in flags)
    # save_figure reads --show and --open, thus each command that saves a plot also gives them
    assert not missingflags
    assert plotmodules >= 22


def test_cli_suggests_a_close_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """A mistyped subcommand must name the closest one on every Python version that CI runs.

    SuggestingArgumentParser composes this message itself, thus Python 3.13 and 3.14 give the
    same text.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspetcra"])

    message = capsys.readouterr().err
    assert "'plotspetcra' is not a command of" in message
    # the error names the fault, and the help line that follows names the closest subcommand
    assert "help: Did you mean plotspectra" in message
    assert message.index("error: ") < message.index("help: ")

    # with no close name, the help line names --help. The list of every command filled the line
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["xyzzy"])
    message = capsys.readouterr().err
    assert "help: Run `" in message
    assert "plotspectra" not in message


def test_help_command_gives_the_help_of_one_command(capsys: pytest.CaptureFixture[str]) -> None:
    """The help command gives the text of --help of a command, as `uv help run` gives `uv run --help`."""
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["inputmodel", "describe", "--help"])
    flaghelp = capsys.readouterr().out

    artistools.__main__.main(argsraw=["help", "inputmodel", "describe"])
    assert capsys.readouterr().out == flaghelp

    # a word after a command that has no subcommands is a mistake, and not a word that the command ignores
    with pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["help", "timesteps", "extra"])
    assert exitinfo.value.code == 2
    assert "error: The command timesteps has no subcommand extra" in capsys.readouterr().err


def test_short_time_flag_always_means_timedays() -> None:
    """-t must mean -timedays on every command, and never something else.

    A command that takes no -timestep left -t ambiguous with -timedays and -time, thus -t failed there.
    argparse also reads "-timestep 30" as "-t imestep", thus such a command declares the name it
    refuses.
    """
    import artistools.hesma_scripts

    # every parser that a command gets is this class, thus the test builds the same one
    parser = at.misc.argparser.SuggestingArgumentParser(prog="hesma")
    artistools.hesma_scripts.addargs(parser)

    assert parser.parse_args(["-t", "300"]).timedays == 300
    assert parser.parse_args(["-timedays", "300"]).timedays == 300

    # the command takes no timestep, thus it names the argument that it does take
    for flag in ("-timestep", "-ts"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "40"])


def test_unsupported_argument_names_the_replacement(capsys: pytest.CaptureFixture[str]) -> None:
    """A declared but unsupported argument must name the argument to give in its place."""
    parser = at.misc.argparser.SuggestingArgumentParser(prog="demo")
    at.misc.addarg_unsupported(parser, "-timestep", "-ts", instead="-timedays")

    with pytest.raises(SystemExit):
        parser.parse_args(["-timestep", "40"])

    captured = capsys.readouterr().err
    assert "error: -timestep is not an argument of this command" in captured
    assert "help: Give -timedays instead" in captured

    # a hidden name stays out of the help text
    assert "-timestep" not in parser.format_help()


@pytest.mark.parametrize(
    ("example", "testfolder", "needs"),
    [
        (commandline, spec.exampletestfolder, spec.exampleneeds)
        for commandline, _, spec in at.commands.get_example_specs()
    ],
)
def test_help_examples_run(
    example: str, testfolder: str | None, needs: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every example of the help text must run, thus no example can name an argument that went away.

    The examples give a path of ".", which this test replaces with the test folder of the command. An example
    that needs data that the test data does not have must still parse. A command can write into the working
    folder, thus the test runs in tmp_path.
    """
    import artistools.__main__

    folder = at.get_path("testdata") / (testfolder or "testmodel")
    argv = [str(folder) if word == "." else word for word in example.split()]
    args = artistools.__main__.build_parser().parse_args(at.misc.separate_trailing_folders(argv))
    if testfolder is None:
        return
    # a CI job can have no optional packages, thus such an example only parses there
    for package in needs:
        pytest.importorskip(package)

    monkeypatch.chdir(tmp_path)
    outputargs = ["-o", str(tmp_path)] if hasattr(args, "outputkind") else []
    artistools.__main__.main(argsraw=[*argv, *outputargs, "--quiet"])


def test_help_shows_the_examples() -> None:
    """The help text must carry the examples and the way to read the help of one command."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    helptext = parser.format_help()

    for command, description in at.commands.get_examples():
        assert command in helptext, command
        assert description in helptext, description

    assert "Run `artistools <command> --help`" in helptext

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
        formatter_class=at.misc.argparser.CustomArgHelpFormatter,
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
    assert "Run `artistools inputmodel <command> --help`" in helptext


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
            ("--plot_cmf", "--showcmf", "-timedaysmin", "-timedaysmax", "-title", "-xmin", "-xmax"),
        ),
        artistools.gsinetwork.decayproducts.addargs: (["-trajectoryroot", ".", "-timemin", "5"], ("timemin",), ()),
    }
    from artistools.misc import SuggestingArgumentParser

    for addargs, (argsraw, dests, hidden) in cases.items():
        parser = SuggestingArgumentParser()
        addargs(parser)
        parser.add_standard_flags()
        namespace = parser.parse_args(argsraw)
        for dest in dests:
            assert getattr(namespace, dest) not in {None, False}, (addargs.__module__, dest)

        helptext = parser.format_help()
        for spelling in hidden:
            # --xmin holds the text -xmin, thus the spelling must not follow a dash
            assert not re.search(rf"(?<![\w-]){re.escape(spelling)}[ ,]", helptext), spelling


def test_short_help_gives_the_first_sentence_of_each_option(capsys: pytest.CaptureFixture[str]) -> None:
    """-h gives a summary of the options, and --help gives each detail and the examples, as uv does.

    The help of plotspectra had 464 lines, which no reader takes in to find one option.
    """
    import artistools.__main__

    helptexts = {}
    for flag in ("-h", "--help"):
        with pytest.raises(SystemExit) as exitinfo:
            artistools.__main__.main(argsraw=["plotspectra", flag])
        assert exitinfo.value.code == 0
        helptexts[flag] = capsys.readouterr().out
    assert len(helptexts["-h"].splitlines()) < len(helptexts["--help"].splitlines()) / 2
    assert "plotspectra --help` for the full help and the examples" in helptexts["-h"]
    assert "Examples" not in helptexts["-h"]
    assert "Examples" in helptexts["--help"]
    # the first sentence of a help text stays, and a later sentence goes, also after an abbreviation such as e.g.
    assert at.misc.argparser.get_first_sentence("Give a time, e.g. 300. A unit is optional") == "Give a time, e.g. 300"


def test_each_short_flag_has_one_meaning_in_the_help() -> None:
    """-v is --verbose and -f is --force in each command, as in ruff and uv. An older meaning stays a hidden alias.

    -v also named --velocity and --rhoscale, and -f also named --format, thus a short flag did a different thing in a
    different command.
    """
    import artistools.__main__
    from artistools.misc import get_shown_flags

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    commands = subactions[0].choices
    for command in ("plotnltepops", "plotestimators", "plotspherical", "cache", "deposition"):
        for action in commands[command]._actions:  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            if action.help == argparse.SUPPRESS:
                continue
            shown = get_shown_flags(action)
            assert "-v" not in shown or action.dest == "verbose", (command, action.dest)
            assert "-f" not in shown or action.dest == "force", (command, action.dest)

    # a script that holds an older meaning still runs
    assert commands["plotnltepops"].parse_args(["-v", "5000"]).velocity_kmps == [5000.0]
    assert commands["plotestimators"].parse_args(["-f", "png"]).format == "png"


def test_viewer_and_completion_know_the_standard_flag_spellings() -> None:
    """A viewer reads the standard spelling of a flag, and tab completion offers that spelling alone.

    The parser of a viewer had no standard spelling, thus "--timedays 300" became two paths of the series. argcomplete
    offered --print_data beside --print-data.
    """
    import artistools.__main__
    from artistools.commands import get_older_flag_spellings
    from artistools.spectra import plotspectra
    from artistools.viewertools.core import parse_viewer_tokens

    viewertokens = parse_viewer_tokens(
        plotspectra.addargs, ["--timedays", "300", "--print-data", str(modelpath)], {"timedays"}
    )
    assert viewertokens.paths == [str(modelpath)]

    older = set(get_older_flag_spellings(artistools.__main__.build_parser()))
    assert {"--print_data", "-timedays", "-modelpath"} <= older
    assert not {"--print-data", "--timedays", "-t", "--modelpath"} & older


def test_validation_errors_give_the_standard_flag_spelling(capsys: pytest.CaptureFixture[str]) -> None:
    """A check of artistools names the flags as the help gives them, also for a flag that the user gave as -timestep."""
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", str(modelpath), "-timestep", "3", "-timedays", "300"])
    assert "error: --timestep, --timedays name the time range in more than one way" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotlightcurves", str(modelpath), "-timemin", "300", "-timemax", "250"])
    assert "error: --timemin 300 is above --timemax 250" in capsys.readouterr().err


def test_help_gives_the_standard_flag_spellings(capsys: pytest.CaptureFixture[str]) -> None:
    """The help gives each long flag two dashes and hyphens, as ruff does. The parser still accepts the older spellings.

    The parser keeps -timedays and --print_data as hidden aliases, thus a script that holds them still runs.
    """
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    helptext = subactions[0].choices["plotspectra"].format_help()
    assert "--timedays, --time, -t" in helptext
    assert "--print-data" in helptext
    assert not re.search(r"(?<![\w-])-(timedays|modelpath|xmin)\b", helptext)
    assert "--print_data" not in helptext

    for argsraw in (["-timedays", "300", "--print_data"], ["--timedays", "300", "--print-data"]):
        args = parser.parse_args(["plotspectra", *argsraw])
        assert str(args.timedays) == "300"
        assert args.print_data

    # an error of argparse names the spellings that the help gives
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", "-xmin", "abc"])
    assert "error: Argument --xmin/--lambdamin: invalid float value" in capsys.readouterr().err


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


def test_help_groups_the_shared_arguments() -> None:
    """The shared arguments sit in titled groups, thus the help of a command of 77 options has a shape."""
    import artistools.__main__

    parser = artistools.__main__.build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    for command in ("plotspectra", "plotlightcurves", "plotestimators"):
        helptext = subactions[0].choices[command].format_help()
        for title in ("Time selection:", "Appearance:", "Output:"):
            assert f"\n{title}\n" in helptext, (command, title)

        # each command gives the groups in one order, thus a reader finds a group at the same place
        headings = [line for line in helptext.splitlines() if line.endswith(":") and not line.startswith(" ")]
        assert headings[:2] == ["Positional arguments:", "Options:"], command
        sharedorder = ["Time selection:", "Viewing direction:", "Appearance:", "Output:"]
        assert [heading for heading in headings if heading in sharedorder] == [
            heading for heading in sharedorder if heading in headings
        ], command
        assert headings[-2] == "Output:", command

    # the 57 general options of plotspectra were one list. A topic group, e.g. "Packets", divides them
    helptext = subactions[0].choices["plotspectra"].format_help()
    assert "\nPackets:\n" in helptext
    assert "\nEmission and absorption:\n" in helptext


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

    The message was "Unrecognised arguments: --listvaraibles" with no suggestion, and an ambiguous
    short flag listed -t because argparse reads -timeday as -t with a joined value.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotestimators", "--listvaraibles"])
    message = capsys.readouterr().err
    assert "Unrecognised arguments: --listvaraibles" in message
    assert "Did you mean --listvariables" in message

    # the suggestion comes from the arguments of the subcommand, not from the top-level parser
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", ".", "-timeday", "300"])
    message = capsys.readouterr().err
    assert "Did you mean --timedays" in message

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
        "error: Unrecognised arguments: --listvaraibles",
        "help: Did you mean --listvariables, --listvars?",
    ]

    # difflib compares the names without their dashes, thus a flag that shares only the dashes is no suggestion
    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", "--bogus"])
    assert "help: Run `" in capsys.readouterr().err


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


def test_every_time_and_wavelength_option_takes_a_unit() -> None:
    """Each option of a time or a wavelength reads its value with a unit, thus "-t 12h" works on every command.

    plotspectra reads -lambdamin as a number in the unit of -xunit, thus it alone takes a plain float.
    """
    import artistools.__main__
    from artistools.misc import time_days_arg
    from artistools.misc import time_days_range_arg
    from artistools.misc import wavelength_angstroms_arg

    unittypes = {
        "-timemin": {time_days_arg},
        "-timemax": {time_days_arg},
        "-timedays": {time_days_arg, time_days_range_arg},
        "-lambdamin": {wavelength_angstroms_arg},
        "-lambdamax": {wavelength_angstroms_arg},
    }
    checked = 0
    for subcommand, subparser in get_every_subcommand(artistools.__main__.build_parser()):
        for action in subparser._actions:  # ruff:ignore[private-member-access]
            if type(action).__name__ == "UnsupportedArgument" or subcommand in {"plotspectra", "spec"}:
                continue
            for flag in set(unittypes).intersection(action.option_strings):
                assert action.type in unittypes[flag], f"{subcommand} {flag} reads no unit"
                checked += 1
    assert checked >= 20


def test_every_command_that_reads_a_model_takes_explain() -> None:
    """--explain shows the selection before a long read, thus each command with a model path must take it."""
    import artistools.__main__
    from artistools.misc import MODELPATH_DESTS

    for subcommand, subparser in get_every_subcommand(artistools.__main__.build_parser()):
        flagsofdest = get_flags_of_dest(subparser)
        if any(dest in flagsofdest for dest in MODELPATH_DESTS):
            assert flagsofdest.get("explain") == ["--explain"], f"{subcommand} must take --explain"


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
        "plotradfield": "velocity_kmps",
        "plotnltepops": "velocity_kmps",
        "spencerfano": "velocity_kmps",
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
            assert any(
                name in text for name in ("resolve_outputfile", "resolve_frameset_paths", "resolve_table_outputfile")
            ), f"{subcommand} writes one file, thus it must name that file"

    assert withoutput > 20, "the tree must hold many commands that write output"


def test_the_command_listing_gives_one_line_to_each_command() -> None:
    """The listing of the commands must take one line for each of them.

    The name of a command with every alias took 36 columns and left 38 for the text, thus almost every
    description wrapped over two or three lines and the listing ran to 88 lines.
    """
    import artistools.__main__

    helptext = artistools.__main__.build_parser().format_help()
    listing = helptext.partition("Plot commands:")[2].partition("\nOptions:")[0]
    assert "Positional arguments:" not in helptext, "the headings of the groups replace the heading of argparse"
    assert "\nOther commands:\n  catalog " in listing

    # a line of the listing holds a command, or it is the heading of a group, or it is empty
    for line in listing.splitlines():
        if not line.strip() or not line.startswith("  "):
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
    assert "modelpath" in shortcommand

    # a default that says nothing stays out of the help text
    estimatorshelp = subactions[0].choices["plotestimators"].format_help()
    assert "(default: None)" not in estimatorshelp
    assert "(default:" in estimatorshelp, "a default that carries a value still shows"


def test_a_missing_optional_package_gives_no_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """import_optional names the command that installs a package, thus a traceback adds nothing.

    The handler of the dispatcher took an AssertionError, a FileNotFoundError, and a ValueError, thus
    a command that needs pypdf or imageio printed a traceback above that message.
    """
    import artistools.__main__

    def raise_missing(args: argparse.Namespace) -> None:  # ruff:ignore[unused-function-argument]
        at.misc.import_optional("nosuchpackage")

    with mock.patch.object(at.listtimesteps, "main", raise_missing), pytest.raises(SystemExit) as exitinfo:
        artistools.__main__.main(argsraw=["timesteps", "-modelpath", str(modelpath)])

    assert exitinfo.value.code == 1
    message = capsys.readouterr().err
    assert "This command needs nosuchpackage" in message
    assert "Traceback" not in message


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
