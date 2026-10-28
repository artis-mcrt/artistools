import functools
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_installskill_replaces_a_link_and_keeps_the_skill_of_the_package(tmp_path: Path) -> None:
    """A copy must replace the link itself, because rmtree through the link deletes the skill of the package."""
    from artistools.installskill import get_skill_folder
    from artistools.installskill import install_skill

    skillsdir = tmp_path / "skills"
    packageskill = get_skill_folder() / "SKILL.md"
    assert install_skill(skillsdir, copy=False, force=False)
    assert not install_skill(skillsdir, copy=False, force=False)

    with pytest.raises(SystemExit):
        install_skill(skillsdir, copy=True, force=False)

    assert install_skill(skillsdir, copy=True, force=True)
    destination = skillsdir / "artis-data"
    assert not destination.is_symlink()
    assert packageskill.is_file()
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == packageskill.read_text(encoding="utf-8")


def test_installskill_names_each_file_that_rmtree_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The warning named only SKILL.md, but rmtree then also deleted the other files of a real skill folder."""
    from artistools.installskill import install_skill

    skillsdir = tmp_path / "skills"
    (skillsdir / "artis-data" / "references").mkdir(parents=True)
    (skillsdir / "artis-data" / "SKILL.md").write_text("an older copy", encoding="utf-8")
    usernotes = skillsdir / "artis-data" / "references" / "notes.md"
    usernotes.write_text("notes of the user", encoding="utf-8")

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        install_skill(skillsdir, copy=True, force=False)
    assert usernotes.is_file()
    assert "notes.md" in capsys.readouterr().err

    assert install_skill(skillsdir, copy=True, force=True)
    assert "notes.md" in capsys.readouterr().err


def test_installskill_refuses_a_destination_in_the_skill_of_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comparison of the path text let rmtree delete the skill of the package, or let the command write into it."""
    from artistools.installskill import install_skill

    packageskill = tmp_path / "package" / "skills" / "artis-data"
    packageskill.mkdir(parents=True)
    (packageskill / "SKILL.md").write_text("the skill of the package", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: packageskill)

    # on a case-insensitive file system, e.g. the macOS default, SKILLS is a different spelling of the same folder
    for skillsdir in (packageskill, packageskill.parent, packageskill.parent.with_name("SKILLS")):
        if skillsdir.exists():
            with pytest.raises(SystemExit):
                install_skill(skillsdir, copy=True, force=True)
    assert [path.name for path in packageskill.iterdir()] == ["SKILL.md"]

    # the destination tmp/artis-data holds the package, and rmtree of the destination deleted the skill of the package
    nestedskill = tmp_path / "artis-data" / "artistools" / "data" / "skills" / "artis-data"
    nestedskill.mkdir(parents=True)
    (nestedskill / "SKILL.md").write_text("the skill of the package", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: nestedskill)
    with pytest.raises(SystemExit):
        install_skill(tmp_path, copy=True, force=True)
    assert (nestedskill / "SKILL.md").is_file()


def test_installskill_makes_no_link_into_the_uv_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With uvx, artistools runs from the uv cache.

    When uv deletes the cache, the link breaks and no message tells the user.
    """
    from artistools.installskill import install_skill

    cachedskill = tmp_path / "uv" / "archive-v0" / "abc123" / "artistools" / "data" / "skills" / "artis-data"
    cachedskill.mkdir(parents=True)
    (cachedskill / "SKILL.md").write_text("the skill in the cache", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: cachedskill)

    with pytest.raises(SystemExit):
        install_skill(tmp_path / "skills", copy=False, force=False)
    assert install_skill(tmp_path / "skills", copy=True, force=False)
    assert not (tmp_path / "skills" / "artis-data").is_symlink()


def test_artis_data_skill_names_each_function_that_exists() -> None:
    """A rename needs no shim in artistools, thus a stale name in the skill gives an agent an error and no warning."""
    from artistools.installskill import get_skill_folder

    skilltext = "\n".join(path.read_text(encoding="utf-8") for path in sorted(get_skill_folder().glob("*.md")))
    names = sorted(set(re.findall(r"\bat\.([a-z_]+(?:\.[a-z_]+)*)\(", skilltext)))
    assert len(names) > 10

    stalenames = []
    for name in names:
        try:
            function = functools.reduce(getattr, name.split("."), at)
        except AttributeError:
            stalenames.append(name)
        else:
            if not callable(function):
                stalenames.append(name)
    assert not stalenames, f"SKILL.md names functions that artistools does not have: {stalenames}"


def test_artis_data_skill_summary_examples_match_the_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The skill shows an example of each summary line, and a change of the format made the example incorrect.

    A number of an example matches any number, "mymodel" matches any model name, and "..." matches any text.
    """
    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    section = skilltext.split("one summary line for each drawn series.", 1)[1].split("A line that starts with", 1)[0]
    examples = [" ".join(span.split()) for span in re.findall(r"`([^`]+)`", section) if not span.startswith("-")]
    assert len(examples) == 5, examples

    numberpattern = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"

    def get_example_pattern(example: str) -> str:
        wordpatterns = []
        for word in example.split():
            if word.startswith("..."):
                wordpatterns.append(".*" + re.escape(word.removeprefix("...")))
            elif word.startswith("mymodel"):
                wordpatterns.append(".+" + re.escape(word.removeprefix("mymodel")))
            else:
                parts = re.split(f"({numberpattern})", word)
                wordpatterns.append(
                    "".join(numberpattern if index % 2 else re.escape(part) for index, part in enumerate(parts))
                )
        return r"\s+".join(wordpatterns)

    at.estimators.plot(argsraw=[], modelpath=modelpath_classic_3d, outputfile=tmp_path, timedays=5, plotlist=[["Te"]])
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], filter=["B"], timemin=250, timemax=300, outputfile=tmp_path)
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], frompackets=True, outputfile=tmp_path)
    at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", outputfile=tmp_path / "spec.pdf")
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        timedays="5",
        showemission=True,
        showabsorption=True,
        outputfile=tmp_path / "em.pdf",
    )
    output = capsys.readouterr().out
    unmatched = [example for example in examples if re.search(get_example_pattern(example), output) is None]
    assert not unmatched, f"SKILL.md gives summary lines that the commands do not print: {unmatched}"


def test_artis_data_skill_python_ignores_a_module_in_the_model_folder(tmp_path: Path) -> None:
    """The command of the skill runs Python in a model folder, and "python -" imported an artistools.py of that folder.

    A model from a different person can hold such a file, thus the agent ran it with the permission to read data.
    """
    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    command = next(block for block in re.findall(r"```sh\n(.*?)```", skilltext, re.DOTALL) if "ATPY=" in block)
    command = command.replace("...", "print(at.__file__)")
    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    marker = tmp_path / "ran.txt"
    (modelfolder / "artistools.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")

    # the command of pip install --user is in ~/.local/bin, and that folder holds no python. Thus the command of this
    # test is in a folder of its own. The editable installation of a checkout can need PYTHONPATH, and -P keeps it
    commandfolder = tmp_path / "bin"
    commandfolder.mkdir()
    # a console script does not put the working folder on sys.path, and -m does without -P
    (commandfolder / "artistools").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -P -m artistools "$@"\n', encoding="utf-8"
    )
    (commandfolder / "artistools").chmod(0o755)
    env = os.environ | {
        "PATH": os.pathsep.join([str(commandfolder), os.environ.get("PATH", "")]),
        "PYTHONPATH": str(Path(at.__file__).resolve().parent.parent),
    }
    bashpath = shutil.which("bash")
    assert bashpath is not None
    # the command is the fixed text of the skill, which the agent runs in a shell
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [bashpath, "-c", command], cwd=modelfolder, env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert Path(completed.stdout.strip()).resolve() == Path(at.__file__).resolve()


def test_artis_data_skill_examples_run_on_the_test_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each example of the skill must run, because a change of a reader can stop it with no other failed test.

    The examples use 10, 12, and 50 days, but the test model ends at 7.9 days. Thus the test uses 5 days and 6 days.
    """
    import runpy

    from artistools.installskill import get_skill_folder

    examples = [
        (path.name, skilltext, example)
        for path in sorted(get_skill_folder().glob("*.md"))
        for skilltext in [path.read_text(encoding="utf-8")]
        for example in re.finditer(r"```python\n(.*?)```", skilltext, re.DOTALL)
    ]
    assert len(examples) > 3

    monkeypatch.chdir(modelpath_classic_3d)
    for index, (filename, skilltext, example) in enumerate(examples):
        code = (
            example
            .group(1)
            .replace("timedays(modelpath, 10)", "timedays(modelpath, 5)")
            .replace("timedays(modelpath, 50)", "timedays(modelpath, 5)")
            .replace("timemin=10, timemax=12", "timemin=5, timemax=6")
        )
        scriptpath = tmp_path / f"example{index}.py"
        scriptpath.write_text(code, encoding="utf-8")
        try:
            runpy.run_path(str(scriptpath), run_name="__main__")
        except Exception as exc:
            exc.add_note(f"in the example at line {skilltext.count(chr(10), 0, example.start()) + 1} of {filename}")
            raise


def test_artis_data_skill_compares_ionisation_processes_on_a_model_with_both_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 3D test model has no non-thermal rates, thus the test of all the examples runs only one branch."""
    import runpy

    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    example = next(block for block in re.findall(r"```python\n(.*?)```", skilltext, re.DOTALL) if "gamma_NT" in block)
    scriptpath = tmp_path / "ionisationprocesses.py"
    scriptpath.write_text(example.replace("timedays(modelpath, 50)", "timedays(modelpath, 300)"), encoding="utf-8")

    monkeypatch.chdir(modelpath)
    runpy.run_path(str(scriptpath), run_name="__main__")
    output = capsys.readouterr().out
    # non-thermal ionisation gives most ionisations of Fe II at 300 days, and photoionisation most of Fe I
    rows = {line.split()[1]: line.split() for line in output.splitlines() if line.startswith("│ Fe_")}
    assert float(rows["Fe_II"][-2]) > 0.9
    assert float(rows["Fe_I"][-2]) < 0.1


def test_artis_data_skill_balances_ionisation_and_recombination_on_a_model_with_both_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Alpha_R*nne_<ion> is the recombination to the ion below, thus an incorrect pair of ions gives no balance.

    The 3D test model has no recombination rates, thus the test of all the examples runs only the branch that prints
    this.
    """
    import runpy

    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "recipes.md").read_text(encoding="utf-8")
    example = next(block for block in re.findall(r"```python\n(.*?)```", skilltext, re.DOTALL) if "Alpha_R" in block)
    scriptpath = tmp_path / "ionisationbalance.py"
    scriptpath.write_text(example.replace("timedays(modelpath, 50)", "timedays(modelpath, 300)"), encoding="utf-8")

    monkeypatch.chdir(modelpath)
    runpy.run_path(str(scriptpath), run_name="__main__")
    output = capsys.readouterr().out
    ratios = {line.split()[1]: float(line.split()[-2]) for line in output.splitlines() if line.startswith("│ Fe_")}
    assert set(ratios) == {"Fe_I", "Fe_II", "Fe_III", "Fe_IV"}
    assert all(ratio == pytest.approx(1.0, rel=0.05) for ratio in ratios.values())
