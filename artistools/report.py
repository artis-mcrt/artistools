"""Make one HTML page that is the first page to read about an ARTIS model.

The page gives the provenance, the health check, the model summary, and the log summary of rank 0. It also gives the
bolometric light curve, and the spectrum and the electron temperature at three timesteps. Each figure comes from a call
of its plot command, and the page gives that command and the lines that it printed. A figure whose command fails gives
a line with the error, and the page continues. The figures read light_curve.out, spec.out, and the estimators. They
read no packets file.
"""

import argparse
import contextlib
import html
import io
import re
import shlex
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_model_folder
from artistools.misc import get_provenance_lines
from artistools.misc import get_timestep_times
from artistools.misc import is_remote_path
from artistools.misc import on_model_host
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_saved
from artistools.misc import resolve_modelpath
from artistools.misc import without_compressed_extension
from artistools.misc import write_text_file_atomic

if t.TYPE_CHECKING:
    import polars as pl

LIGHTCURVEFILENAME = "lightcurve.png"
SPECTRUMFILENAME = "spectrum_timestep{timestep:03d}.png"
TEMPERATUREFILENAME = "Te_timestep{timestep:03d}.png"

# print_series_summary writes "N points" or "no point" in each summary line of the three plot commands, and
# print_warning and print_error start their lines with these prefixes
RE_SUMMARYLINE = re.compile(r"^\s*(?:warning|error|help):|\b(?:\d+ points?|no point)\b")

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 70rem; margin: 0 auto; padding: 1rem; line-height: 1.4;
  color: #111; background: #fff; }
pre { background: #f4f4f4; padding: 0.5rem; white-space: pre-wrap; overflow-wrap: anywhere; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #bbb; padding: 0.2rem 0.4rem; text-align: left; vertical-align: top; }
tr.status-warning td { background: #fff3cd; }
tr.status-error td { background: #f8d7da; }
img { max-width: 100%; height: auto; }
.error { color: #a00000; font-weight: bold; }
"""


def run_captured[T](function: Callable[[], T]) -> tuple[T | None, str | None, list[str]]:
    """Run a function, and return its result, the text of its error, and the lines that it printed.

    The error text is None if the function did not fail. A command that stops with exit_with_error prints its error
    before it raises SystemExit, thus the error text is then that printed line.
    """
    captured = io.StringIO()
    result: T | None = None
    errortext: str | None = None
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        try:
            result = function()
        except SystemExit as exc:
            errortext = f"the command stopped with the exit status {exc.code}"
        # the page names the error of a part, and the next part runs
        except Exception as exc:  # ruff:ignore[blind-except]
            errortext = f"{type(exc).__name__}: {exc}"
    lines = captured.getvalue().splitlines()
    if errortext is not None and (errorlines := [line for line in lines if line.startswith(("error:", "help:"))]):
        errortext = " ".join(errorlines)
    return result, errortext, lines


def get_lines_html(lines: Sequence[str]) -> str:
    """Return lines of text as a preformatted HTML block."""
    return f"<pre>{html.escape(chr(10).join(lines))}</pre>\n"


def get_output_html(lines: Sequence[str], errortext: str | None) -> str:
    """Return the error of a part, its summary lines and warnings, and all its printed lines in a collapsed block."""
    parts: list[str] = []
    if errortext is not None:
        parts.append(f'<p class="error">This part of the report failed: {html.escape(errortext)}</p>\n')
    if summarylines := [line.strip() for line in lines if RE_SUMMARYLINE.search(line)]:
        parts.append(get_lines_html(summarylines))
    if lines:
        linecount = "1 line" if len(lines) == 1 else f"{len(lines)} lines"
        parts.append(
            f"<details><summary>All the lines that the command printed ({linecount})</summary>\n"
            f"{get_lines_html(lines)}</details>\n"
        )
    return "".join(parts)


def get_table_html(dftable: "pl.DataFrame", statuscolumn: str | None = None) -> str:
    """Return a dataframe as an HTML table. Each row gets a class from the value of statuscolumn, e.g. status-error."""
    header = "".join(f"<th>{html.escape(column)}</th>" for column in dftable.columns)
    rows: list[str] = []
    for row in dftable.iter_rows(named=True):
        cells = "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row.values())
        rowclass = (
            f' class="status-{html.escape(str(row[statuscolumn]).replace(" ", "-"))}"'
            if statuscolumn is not None
            else ""
        )
        rows.append(f"<tr{rowclass}>{cells}</tr>")
    return f"<table>\n<thead><tr>{header}</tr></thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n"


def get_provenance_html(args: argparse.Namespace, modelfolder: Path) -> str:
    """Return the section that tells how artistools made the page: the version, the time, the folders, the command."""
    lines = [*get_provenance_lines(args), f"model folder: {resolve_modelpath(modelfolder)}"]
    return '<h2 id="provenance">Provenance</h2>\n' + get_lines_html(lines)


def get_check_results_table(modelpath: Path) -> "pl.DataFrame":
    """Return the table of the checks of artistools check. The checks read no packets file."""
    from artistools.check import get_check_results

    return get_check_results(modelpath, frompackets=False)


def get_health_check_html(modelfolder: Path) -> str:
    """Return the section with one row for each check of the model and its status."""
    dfchecks, errortext, lines = run_captured(lambda: get_check_results_table(modelfolder))
    parts = [
        '<h2 id="health-check">Health check</h2>\n',
        "<p>The checks of <code>artistools check</code>, with no check of the packets files.</p>\n",
    ]
    if dfchecks is not None:
        parts.append(get_table_html(dfchecks, statuscolumn="status"))
    parts.append(get_output_html(lines, errortext))
    return "".join(parts)


def get_model_summary_html(modelfolder: Path) -> str:
    """Return the section with the summary of artistools describe."""
    from artistools.describemodel import get_model_summary

    summarylines, errortext, lines = run_captured(lambda: get_model_summary(modelfolder))
    parts = ['<h2 id="model-summary">Model summary</h2>\n', "<p>The output of <code>artistools describe</code>.</p>\n"]
    if summarylines is not None:
        parts.append(get_lines_html(summarylines))
    parts.append(get_output_html(lines, errortext))
    return "".join(parts)


@on_model_host
def get_log_summary_table(modelfolder: Path) -> "tuple[list[str], pl.DataFrame | None]":
    """Return the names of the logs of rank 0 in the model folder, and the table of the kinds of their messages.

    The table is None if the model has no log of rank 0. The host of a remote model gives the names, because it
    expands "~" in the model folder and in the paths of the logs.
    """
    from artistools.logsummary import get_log_summary
    from artistools.logsummary import get_logfiles

    logfiles = get_logfiles(modelfolder, [0])
    lognames = [str(logfile.relative_to(modelfolder)) for logfile in logfiles]
    return lognames, get_log_summary(logfiles) if logfiles else None


def get_log_summary_html(modelfolder: Path) -> str:
    """Return the section with the table of the kinds of the warnings and the errors in the logs of rank 0."""
    logresult, errortext, lines = run_captured(lambda: get_log_summary_table(modelfolder))
    parts = ['<h2 id="log-summary">Log summary of rank 0</h2>\n']
    if logresult is not None:
        lognames, dfsummary = logresult
        if dfsummary is None:
            parts.append("<p>The model has no log of rank 0.</p>\n")
        else:
            lognamestext = ", ".join(html.escape(logname) for logname in lognames)
            methodtext = (
                f"<p>The output of <code>artistools logsummary</code>. The table counts the warnings and the errors"
                f" in {len(lognames)} logs of rank 0, one log for each run folder: {lognamestext}. A kind of message is"
                " its text with each number replaced by N.</p>\n"
            )
            parts.extend((
                methodtext,
                get_table_html(dfsummary)
                if not dfsummary.is_empty()
                else "<p>The logs hold no warning and no error.</p>\n",
            ))
    parts.append(get_output_html(lines, errortext))
    return "".join(parts)


def get_report_timesteps(modelfolder: Path) -> tuple[list[int], str]:
    """Return three timesteps spread over the valid time range of the model, and a sentence that gives the method.

    The timesteps are the first, the middle, and the last timestep that start and end inside the range of
    get_escaped_arrivalrange. plotspectra rejects a timestep outside that range. A model with no such timestep gives
    timesteps from all its timesteps.
    """
    tstarts = get_timestep_times(modelfolder, loc="start")
    tends = get_timestep_times(modelfolder, loc="end")
    alltimesteps = list(range(len(tstarts)))
    try:
        _, validstart, validend = get_escaped_arrivalrange(modelfolder)
    except FileNotFoundError as exc:
        validstart, validend = None, None
        reason = f"The model has no valid time range ({exc})."
    else:
        reason = (
            f"Light from the whole model reaches the observer from {validstart:.4g} to {validend:.4g} days."
            if validstart is not None and validend is not None
            else "The model has no valid time range."
        )
    candidates = alltimesteps
    if validstart is not None and validend is not None:
        candidates = [
            timestep for timestep in alltimesteps if tstarts[timestep] >= validstart and tends[timestep] <= validend
        ]
        reason += f" {len(candidates)} timesteps start and end in that range."
    if not candidates:
        candidates = alltimesteps
        reason += " Thus the report takes all the timesteps."
    picks = sorted({candidates[round(index)] for index in np.linspace(0, len(candidates) - 1, 3)})
    method = (
        f"{reason} The report takes the first, the middle, and the last of these {len(candidates)} timesteps:"
        f" {', '.join(str(timestep) for timestep in picks)}."
    )
    return picks, method


def get_timestep_text(modelfolder: Path, timestep: int) -> str:
    """Return the name of a timestep with its start and end times, e.g. "timestep 8 (3.90 to 4.03 days)"."""
    tstart = get_timestep_times(modelfolder, loc="start")[timestep]
    tend = get_timestep_times(modelfolder, loc="end")[timestep]
    return f"timestep {timestep} ({tstart:.2f} to {tend:.2f} days)"


def get_figure_html(
    reportfolder: Path,
    filename: str,
    alttext: str,
    command: "tuple[str, Callable[..., None]]",
    arguments: Sequence[str],
) -> str:
    """Run a plot command that writes filename in the report folder, and return the figure and the printed lines.

    command is the name of the command and its main function, and arguments are the arguments of the command before
    the -o argument of the report.
    """
    commandname, commandmain = command
    outputpath = reportfolder / filename
    argsraw = [*arguments, "-o", str(outputpath)]

    def plot() -> None:
        try:
            commandmain(argsraw=argsraw)
        finally:
            plt.close("all")

    # a file of an earlier report must not hide a failure of the command
    outputpath.unlink(missing_ok=True)
    _, errortext, lines = run_captured(plot)
    if errortext is None and not outputpath.is_file():
        errortext = f"the command wrote no file {filename}"
    print_detail(f"{filename}: {'ok' if errortext is None else errortext}")
    commandline = shlex.join(["artistools", commandname, *argsraw])
    parts = ["<figure>\n"]
    if errortext is None:
        parts.append(f'<img src="{html.escape(filename)}" alt="{html.escape(alttext)}">\n')
    parts.extend((
        f"<figcaption>The command: <code>{html.escape(commandline)}</code></figcaption>\n</figure>\n",
        get_output_html(lines, errortext),
    ))
    return "".join(parts)


def get_light_curve_html(modelfolder: Path, reportfolder: Path) -> str:
    """Return the section with the bolometric light curve of light_curve.out."""
    from artistools.lightcurve.plotlightcurve import main as plotlightcurves

    return '<h2 id="light-curve">Bolometric light curve</h2>\n' + get_figure_html(
        reportfolder,
        LIGHTCURVEFILENAME,
        f"The bolometric luminosity of the model {resolve_modelpath(modelfolder).name} against time,"
        " from light_curve.out",
        ("plotlightcurves", plotlightcurves),
        [str(modelfolder)],
    )


def get_spectra_html(modelfolder: Path, reportfolder: Path, timesteps: Sequence[int], method: str) -> str:
    """Return the section with the spectrum of spec.out at each of the timesteps."""
    from artistools.spectra.plotspectra import main as plotspectra

    parts = ['<h2 id="spectra">Spectra</h2>\n', f"<p>{html.escape(method)}</p>\n"]
    for timestep in timesteps:
        timesteptext = get_timestep_text(modelfolder, timestep)
        parts.extend((
            f"<h3>Spectrum at {html.escape(timesteptext)}</h3>\n",
            get_figure_html(
                reportfolder,
                SPECTRUMFILENAME.format(timestep=timestep),
                f"The flux of the model {resolve_modelpath(modelfolder).name} against wavelength at {timesteptext},"
                " from spec.out",
                ("plotspectra", plotspectra),
                [str(modelfolder), "-timestep", str(timestep)],
            ),
        ))
    return "".join(parts)


def get_temperature_html(modelfolder: Path, reportfolder: Path, timesteps: Sequence[int], method: str) -> str:
    """Return the section with the electron temperature against velocity at each of the timesteps."""
    from artistools.estimators.plotestimators import main as plotestimators

    parts = ['<h2 id="temperature">Electron temperature against velocity</h2>\n', f"<p>{html.escape(method)}</p>\n"]
    for timestep in timesteps:
        timesteptext = get_timestep_text(modelfolder, timestep)
        parts.extend((
            f"<h3>Electron temperature at {html.escape(timesteptext)}</h3>\n",
            get_figure_html(
                reportfolder,
                TEMPERATUREFILENAME.format(timestep=timestep),
                f"The electron temperature Te of the model {resolve_modelpath(modelfolder).name} against velocity at"
                f" {timesteptext}, from the estimators",
                ("plotestimators", plotestimators),
                ["Te", str(modelfolder), "-timestep", str(timestep)],
            ),
        ))
    return "".join(parts)


def get_page_html(modelname: str, sections: Sequence[str]) -> str:
    """Return the full HTML page with the sections in the given order."""
    title = f"Report of the ARTIS model {modelname}"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n" + "".join(sections) + "</body>\n</html>\n"
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    addarg_output(
        parser,
        kind="folder",
        helptext="The folder of the report. The default is <model name>_report in the working folder",
    )
    addarg_force(parser)


def get_report_files_in_folder(reportfolder: Path) -> list[str]:
    """Return the names of the files of an earlier report in the folder, which a new report can replace.

    The timesteps of the figures come from the model, and the model is not read yet. Thus each figure of a report
    counts, also a figure of a different timestep.
    """
    if not reportfolder.is_dir():
        return []
    patterns = [
        "index.html",
        LIGHTCURVEFILENAME,
        SPECTRUMFILENAME.replace("{timestep:03d}", "*"),
        TEMPERATUREFILENAME.replace("{timestep:03d}", "*"),
    ]
    return sorted({path.name for pattern in patterns for path in reportfolder.glob(pattern)})


def get_remote_model_folder_from_text(modelpath: Path) -> Path:
    """Return the model folder of a remote model path from its text alone, with no question to the host.

    A model path can name a file of the model folder, e.g. host:~/models/foo/light_curve.out. An output file of ARTIS
    ends with .out, possibly with a compressed extension, and an input file of a model ends with .txt.
    """
    name = without_compressed_extension(modelpath).name
    return modelpath.parent if name.endswith((".out", ".txt")) else modelpath


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make an HTML page with the summary, the health check, the log summary, and standard figures of a model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    # the refusal must come before the work starts, e.g. before a remote model starts its server. The test of a file
    # of a remote model asks its host, thus the default name of the folder takes the text of a remote model path
    givenpath = Path(args.modelpath)
    namedfolder = (
        get_remote_model_folder_from_text(givenpath) if is_remote_path(givenpath) else get_model_folder(givenpath)
    )
    reportfolder = Path(args.outputfile or f"{resolve_modelpath(namedfolder).name}_report")
    oldreportfiles = get_report_files_in_folder(reportfolder)
    confirm_overwrite([reportfolder], oldreportfiles, force=args.force)

    modelfolder = get_model_folder(args.modelpath)
    modelname = resolve_modelpath(modelfolder).name
    timesteps, method = get_report_timesteps(modelfolder)
    print_detail(method)
    reportfolder.mkdir(parents=True, exist_ok=True)
    # the new report can take other timesteps, and a figure of the earlier report must not seem to belong to it. The
    # user agreed to replace these files, and unlink removes a link and not its target
    for filename in oldreportfiles:
        (reportfolder / filename).unlink(missing_ok=True)

    sections = [
        get_provenance_html(args, modelfolder),
        get_health_check_html(modelfolder),
        get_model_summary_html(modelfolder),
        get_log_summary_html(modelfolder),
        get_light_curve_html(modelfolder, reportfolder),
        get_spectra_html(modelfolder, reportfolder, timesteps, method),
        get_temperature_html(modelfolder, reportfolder, timesteps, method),
    ]
    pagepath = reportfolder / "index.html"
    pagehtml = get_page_html(modelname, sections)
    # a link at the path of the page gets replaced, and its target stays as it was
    write_text_file_atomic(pagepath, lambda stream: stream.write(pagehtml))
    print_saved(pagepath)
