#!/usr/bin/env python3
"""Ask a Claude Code or a Codex agent questions about the test models, and compare each answer with artistools.

The script measures how well the artis-data skill lets an agent answer a question about a model. artistools
calculates the expected answer of each question when the script runs. The agent must end its reply with a line
"ANSWER: <value>". The script copies the test models into a temporary folder, thus the agent cannot change the files
of the repository. Each question starts a new session of `claude -p` or `codex exec`. Each needs a login
(`claude auth login` or `codex login`) and costs tokens.

Run it from the repository root after `source ./setuptestdata.sh` in tests/data:

    uv run -- python eval_artis_data_skill.py
    uv run -- python eval_artis_data_skill.py -questions peaktime fe2fraction -model sonnet
    uv run -- python eval_artis_data_skill.py -agent codex
"""

import argparse
import json
import os
import re
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import tempfile
import typing as t
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import megaparsec_to_cm
from artistools.installskill import get_skill_folder

MODEL3D = "test-classicmode_3d"
MODEL1D = "testmodel"
ANSWERINSTRUCTION = (
    "Use the artis-data skill. End your reply with one line in the form 'ANSWER: <value>', with only the value and no"
    " unit."
)
RE_ANSWER = re.compile(r"^\s*\**ANSWER:\**\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
RE_THOUSANDSCOMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
RE_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class Question(t.NamedTuple):
    """A question, the model folder that it is about, and a function that gives the expected answer."""

    name: str
    modelname: str
    text: str
    get_expected: Callable[[Path], float | str]
    rtol: float = 0.02


def get_peak_time_days(modelpath: Path) -> float:
    """Return the time of the maximum of the observer-frame luminosity, for the average over all directions."""
    dflc = at.lightcurve.scan_lightcurve(at.lightcurve.find_lightcurve_file(modelpath))[-1].collect()
    return float(dflc.select(pl.col("time_days").sort_by("luminosity_erg/s", descending=True).first()).item())


def get_deposition_6d(modelpath: Path) -> float:
    """Return the total deposition rate of deposition.out at 6 days in erg/s."""
    timestep = at.get_timestep_of_timedays(modelpath, 6)
    dfdep = at.get_deposition(modelpath).filter(pl.col("timestep") == timestep).collect()
    return float(dfdep["total_dep_Lsun"].item()) * Lsun_to_erg_per_s


def get_temperature_limits(modelpath: Path) -> list[float]:
    """Return the values of MINTEMP and MAXTEMP that artisoptions.h gives."""
    return [
        float(value)
        for name in ("MINTEMP", "MAXTEMP")
        if (value := at.misc.get_artis_option(modelpath, name)) is not None
    ]


def get_maxtemp_cellcount_6d(modelpath: Path) -> float:
    """Return the number of cells with Te at MAXTEMP at 6 days."""
    maxtemp = max(get_temperature_limits(modelpath))
    timestep = at.get_timestep_of_timedays(modelpath, 6)
    return float(at.scan_estimators(modelpath, timestep=timestep).filter(pl.col("Te") == maxtemp).collect().height)


def get_main_cooling_process_6d(modelpath: Path) -> str:
    """Return the cooling column with the largest sum of rate times volume, for the cells with a thermal balance.

    The test of the balance is the test of recipes.md: no LTE, no temperature limit, and heating and cooling that agree
    to 1 %.
    """
    timestep = at.get_timestep_of_timedays(modelpath, 6)
    lfcells = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
    columns = lfcells.collect_schema().names()
    heatingcols = [col for col in columns if re.fullmatch(r"heating_[a-z]+", col)]
    coolingcols = [col for col in columns if re.fullmatch(r"cooling_[a-z]+", col)]
    islte = (pl.col("W") == 1.0) & (pl.col("Te") == pl.col("TR")) & (pl.col("Te") == pl.col("TJ"))
    isunbalanced = ((pl.sum_horizontal(heatingcols) / pl.sum_horizontal(coolingcols)) - 1.0).abs() > 0.01
    dfsums = (
        lfcells
        .filter(~islte, ~pl.col("Te").is_in(get_temperature_limits(modelpath)), ~isunbalanced)
        .select((pl.col(col).fill_null(0.0) * pl.col("volume")).sum() for col in coolingcols)
        .collect()
    )
    return coolingcols[int(np.argmax(dfsums.row(0)))]


def get_main_ion_of_blue_packets(modelpath: Path) -> str:
    """Return the ion whose line emission gives the most energy of the packets from 3500 to 4500 Å, 5 to 7 days."""
    _, _, daysmin, daysmax = at.misc.get_time_range(modelpath, timemin=5, timemax=7)
    _, lfpackets = at.packets.get_packets(modelpath, escape_type="TYPE_RPKT")
    dflines = at.atomic.get_linelist_pldf(modelpath).collect()
    dfions = (
        lfpackets
        .filter(pl.col("t_arrive_d").is_between(daysmin, daysmax, closed="left"))
        .filter((at.constants.c_ang_per_s / pl.col("nu_rf")).is_between(3500.0, 4500.0))
        .filter(pl.col("emissiontype") >= 0)
        .select("e_rf", "emissiontype")
        .collect()
        .join(dflines, left_on="emissiontype", right_on="lineindex", how="inner")
        .group_by("atomic_number", "ion_stage")
        .agg(pl.col("e_rf").sum())
        .sort("e_rf", descending=True)
    )
    return at.get_ionstring(dfions["atomic_number"][0], dfions["ion_stage"][0])


def get_main_ionisation_process_fe2(modelpath: Path) -> str:
    """Return the process that gives the most ionisations of Fe II per second at 300 days."""
    timestep = at.get_timestep_of_timedays(modelpath, 300)
    dftotals = (
        at
        .scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
        .select(
            (pl.col(f"{process}_Fe_II") * pl.col("nnion_Fe_II") * pl.col("volume")).sum().alias(process)
            for process in ("gamma_R", "gamma_NT")
        )
        .collect()
    )
    return "non-thermal" if dftotals["gamma_NT"].item() > dftotals["gamma_R"].item() else "photoionisation"


def get_fe2_fraction_300d(modelpath: Path) -> float:
    """Return the fraction of all the Fe atoms that are Fe II at 300 days."""
    timestep = at.get_timestep_of_timedays(modelpath, 300)
    return float(
        at
        .scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
        .select((pl.col("nnion_Fe_II") * pl.col("volume")).sum() / (pl.col("nnelement_Fe") * pl.col("volume")).sum())
        .collect()
        .item()
    )


def get_spectrum_luminosity_300d(modelpath: Path) -> float:
    """Return 4 pi (1 Mpc)^2 times the integral of f_lambda of spec.out at 300 days, from 3500 to 7000 Å."""
    timestep = at.get_timestep_of_timedays(modelpath, 300)
    dfspec = at.spectra.get_spectra(modelpath, timestepmin=timestep, timestepmax=timestep)[-1].collect()
    dfspec = dfspec.sort("lambda_angstroms")
    lambdas, flambdas = dfspec["lambda_angstroms"].to_numpy(), dfspec["f_lambda"].to_numpy()
    isinside = (lambdas > 3500.0) & (lambdas < 7000.0)
    xpoints = np.concatenate([[3500.0], lambdas[isinside], [7000.0]])
    return float(np.trapezoid(np.interp(xpoints, lambdas, flambdas), x=xpoints)) * 4 * np.pi * megaparsec_to_cm**2


def get_packet_noise_300d(modelpath: Path) -> float:
    """Return sqrt(sum(e^2)) / sum(e) of the escaped r-packets of the time step at 300 days, from 2500 to 19000 Å."""
    timestep = at.get_timestep_of_timedays(modelpath, 300)
    tstart = at.get_timestep_times(modelpath, loc="start")[timestep]
    tend = at.get_timestep_times(modelpath, loc="end")[timestep]
    _, lfpackets = at.packets.get_packets(modelpath, escape_type="TYPE_RPKT")
    return float(
        lfpackets
        .filter(pl.col("t_arrive_d").is_between(tstart, tend, closed="left"))
        .filter((at.constants.c_ang_per_s / pl.col("nu_rf")).is_between(2500.0, 19000.0))
        .select(pl.col("e_rf").pow(2).sum().sqrt() / pl.col("e_rf").sum())
        .collect()
        .item()
    )


def get_mean_te_5d(modelpath: Path) -> float:
    """Return the mean of Te at 5 days with the volume of each cell as the weight."""
    timestep = at.get_timestep_of_timedays(modelpath, 5)
    return float(
        at
        .scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
        .select((pl.col("Te") * pl.col("volume")).sum() / pl.col("volume").sum())
        .collect()
        .item()
    )


QUESTIONS = (
    Question(
        "peaktime",
        MODEL3D,
        f"At which time in days does the bolometric light curve of the model {MODEL3D} have its maximum? Use the"
        " average over all directions.",
        get_peak_time_days,
    ),
    Question(
        "deposition",
        MODEL3D,
        f"What is the total energy deposition rate in erg/s of the model {MODEL3D} at 6 days?",
        get_deposition_6d,
    ),
    Question(
        "maxtempcells",
        MODEL3D,
        f"How many cells of the model {MODEL3D} have an electron temperature at the upper limit of ARTIS at 6 days?",
        get_maxtemp_cellcount_6d,
        rtol=0.0,
    ),
    Question(
        "cooling",
        MODEL3D,
        f"Which process cools the electrons most in the model {MODEL3D} at 6 days, in the cells that have a thermal"
        " balance? Give the estimator column, e.g. cooling_ff.",
        get_main_cooling_process_6d,
    ),
    Question(
        "lineion",
        MODEL3D,
        f"In the model {MODEL3D}, which ion emits the most energy by line emission in the packets that escape with"
        " wavelengths from 3500 to 4500 Å and arrive from 5 to 7 days? Give the ion in the form 'Fe II'.",
        get_main_ion_of_blue_packets,
    ),
    Question(
        "fe2ionisation",
        MODEL1D,
        f"In the model {MODEL1D} at 300 days, which process gives more ionisations of Fe II per second in all the"
        " ejecta: photoionisation or non-thermal ionisation? Answer 'photoionisation' or 'non-thermal'.",
        get_main_ionisation_process_fe2,
    ),
    Question(
        "fe2fraction",
        MODEL1D,
        f"In the model {MODEL1D} at 300 days, what fraction of all the iron atoms in the ejecta is Fe II?",
        get_fe2_fraction_300d,
    ),
    Question(
        "spectrumluminosity",
        MODEL1D,
        f"What is the luminosity in erg/s of the spectrum of the model {MODEL1D} at 300 days, from 3500 to 7000 Å?",
        get_spectrum_luminosity_300d,
    ),
    Question(
        "packetnoise",
        MODEL1D,
        f"What is the relative Monte Carlo noise of the luminosity of the packet spectrum of the model {MODEL1D} at"
        " 300 days, from 2500 to 19000 Å? Give it as a fraction, e.g. 0.05 for 5 %.",
        get_packet_noise_300d,
        rtol=0.05,
    ),
    Question(
        "meante",
        MODEL3D,
        f"What is the mean electron temperature in K of the model {MODEL3D} at 5 days, with the volume of each cell as"
        " the weight?",
        get_mean_te_5d,
    ),
)


def answer_is_correct(answer: str, expected: float | str, rtol: float) -> bool:
    """Return True when the answer is the expected name, or its one number agrees with the expected number.

    A name must be the whole answer, because an answer such as "non-thermal, not photoionisation" names both
    choices. Quotes, backticks, and punctuation at the ends do not count.
    """
    if isinstance(expected, str):
        return answer.strip().strip("`'\".,;:").strip().lower() == expected.lower()
    # a comma before a group of three digits, e.g. 1,234, is part of the number. Another comma separates two numbers,
    # thus "5,7" and "5, not 7" give two numbers
    numbers = RE_NUMBER.findall(RE_THOUSANDSCOMMA.sub("", answer))
    return len(numbers) == 1 and bool(np.isclose(float(numbers[0]), expected, rtol=rtol, atol=0.0))


def ask_codex(question: Question, workfolder: Path, model: str | None, env: dict[str, str]) -> dict[str, t.Any]:
    """Run one session of `codex exec` in the work folder and return a result in the form of ask_claude.

    codex exec writes its last message to the file of -o, and the JSON events give the number of tokens. The sandbox
    lets the agent write in the work folder alone, e.g. a cache of a model.
    """
    lastmessagefile = workfolder / ".codex_last_message.txt"
    command = [
        *("codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--sandbox", "workspace-write"),
        *("--output-last-message", str(lastmessagefile)),
        *(("--model", model) if model else ()),
        f"{question.text} {ANSWERINSTRUCTION}",
    ]
    # the command is the codex CLI with the fixed text of a question of this script, and no shell. codex exec reads
    # stdin when it is open, thus the command gets none, or it waits
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        command, cwd=workfolder, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
    )
    events = []
    for line in completed.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    reply = lastmessagefile.read_text(encoding="utf-8") if lastmessagefile.is_file() else ""
    lastmessagefile.unlink(missing_ok=True)
    usage = [event.get("usage", {}) for event in events if event.get("type") == "turn.completed"]
    return {
        "is_error": completed.returncode != 0 or not reply,
        "result": reply or completed.stdout[-2000:] + completed.stderr[-2000:],
        "num_turns": sum(1 for event in events if event.get("type") == "item.completed"),
        "total_cost_usd": None,
        "tokens": sum(int(turn.get("input_tokens", 0)) + int(turn.get("output_tokens", 0)) for turn in usage),
    }


def ask_claude(question: Question, workfolder: Path, model: str | None, env: dict[str, str]) -> dict[str, t.Any]:
    """Run one session of `claude -p` in the work folder and return its JSON result."""
    command = [
        *("claude", "-p", f"{question.text} {ANSWERINSTRUCTION}"),
        *("--output-format", "json", "--no-session-persistence", "--permission-mode", "dontAsk"),
        *("--allowedTools", "Bash", "Read", "Glob", "Grep", "Skill"),
        *(("--model", model) if model else ()),
    ]
    # the command is the claude CLI with the fixed text of a question of this script, and no shell
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        command, cwd=workfolder, env=env, capture_output=True, text=True, check=False
    )
    try:
        return t.cast("dict[str, t.Any]", json.loads(completed.stdout))
    except json.JSONDecodeError:
        return {"is_error": True, "result": completed.stdout + completed.stderr}


def make_workfolder(folder: Path, modelnames: set[str]) -> None:
    """Copy the test models and give the folder a link to the skill of this checkout for each agent."""
    for modelname in sorted(modelnames):
        modelpath = at.get_path("testdata") / modelname
        if not modelpath.is_dir():
            msg = f"{modelpath} does not exist. Run setuptestdata.sh in tests/data"
            raise FileNotFoundError(msg)
        shutil.copytree(modelpath, folder / modelname)
    # Claude Code reads .claude/skills, and Codex reads .agents/skills
    for skillsdir in (folder / ".claude" / "skills", folder / ".agents" / "skills"):
        skillsdir.mkdir(parents=True)
        (skillsdir / "artis-data").symlink_to(get_skill_folder().resolve(), target_is_directory=True)


def get_agent_environment() -> dict[str, str]:
    """Return the environment of the agent, with the artistools command and the package of this script.

    The agent runs Python in the work folder. The parent folder of the imported package goes first on PYTHONPATH, thus
    the agent reads the models with the same artistools as this script.
    """
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["PYTHONPATH"] = os.pathsep.join([
        str(Path(at.__file__).resolve().parent.parent),
        *filter(None, [env.get("PYTHONPATH")]),
    ])
    return env


def main() -> None:
    """Ask each selected question, and print a table of the expected answers and the answers of the agent."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-questions",
        nargs="+",
        choices=[question.name for question in QUESTIONS],
        help="The names of the questions to ask. The default is all the questions",
    )
    parser.add_argument(
        "-agent", choices=["claude", "codex"], default="claude", help="The agent that answers the questions"
    )
    parser.add_argument(
        "-model", help="The model of the agent, e.g. sonnet. The default is the model of the claude or codex command"
    )
    args = parser.parse_args()
    questions = [question for question in QUESTIONS if args.questions is None or question.name in args.questions]

    rows = []
    with tempfile.TemporaryDirectory(prefix="eval_artis_data_skill_") as tmpdir:
        workfolder = Path(tmpdir)
        make_workfolder(workfolder, {question.modelname for question in questions})
        env = get_agent_environment()
        for question in questions:
            expected = question.get_expected(workfolder / question.modelname)
            askagent = ask_codex if args.agent == "codex" else ask_claude
            result = askagent(question, workfolder, args.model, env)
            reply = str(result.get("result", ""))
            answers = RE_ANSWER.findall(reply)
            answer = answers[-1] if answers else ""
            correct = not result.get("is_error") and answer_is_correct(answer, expected, question.rtol)
            print(f"{question.name}: {'correct' if correct else 'WRONG'}. Expected {expected}, answer {answer!r}")
            if not correct:
                print(reply[-2000:])
            rows.append({
                "question": question.name,
                "correct": correct,
                "expected": str(expected),
                "answer": answer,
                "turns": result.get("num_turns"),
                "cost_usd": result.get("total_cost_usd"),
                "tokens": result.get("tokens"),
            })

    dfresults = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
        print(dfresults)
    print(f"{dfresults['correct'].sum()} of {dfresults.height} answers are correct")


if __name__ == "__main__":
    main()
