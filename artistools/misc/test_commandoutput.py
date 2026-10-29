import csv
import json
import subprocess
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"


def test_plot_commands_write_the_format_of_the_suffix(tmp_path: Path) -> None:
    """The suffix of -o sets the format of the file, e.g. -o rf.png gives a PNG file.

    The commands gave format="pdf" to savefig, thus a file named rf.png held a PDF document.
    """
    pngmagic = b"\x89PNG"
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, modelgridindex=0, timestep=40, outputfile=tmp_path / "rf.png"
    )
    assert (tmp_path / "rf.png").read_bytes().startswith(pngmagic)

    at.plottransitions.main(argsraw=[], modelpath=modelpath, timedays=300, outputfile=tmp_path / "trans.png")
    assert (tmp_path / "trans.png").read_bytes().startswith(pngmagic)

    at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, timestep=40, outputfile=tmp_path / "map.png")
    assert (tmp_path / "map.png").read_bytes().startswith(pngmagic)


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


def test_print_saved_gives_the_shorter_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A file outside the working folder gave a chain of "..", which is longer than the full path."""
    faraway = tmp_path / "figure.pdf"
    faraway.touch()

    opencommand = at.misc.commandoutput.get_open_command()
    at.misc.print_saved(faraway)
    reported = capsys.readouterr().out.removeprefix(f"{opencommand} ").strip().strip("'")

    assert ".." not in reported
    assert Path(reported).resolve() == faraway.resolve()

    # a file below the working folder keeps its short relative name
    here = Path("localfigure.pdf")
    at.misc.print_saved(Path.cwd() / here)
    assert capsys.readouterr().out.strip() == f"{opencommand} {here}"


def test_progress_goes_to_the_standard_error_and_the_product_to_the_standard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pipe gets the table of --print-data alone, as for ruff and uv. The progress messages go to the standard error.

    Before this change, "Reading .../light_curve.out" and the model name came in front of the table, unless the user
    gave --quiet.
    """
    import artistools.__main__

    artistools.__main__.main(
        argsraw=["plotlightcurves", str(modelpath), "--print-data", "-o", str(tmp_path / "lightcurve.pdf")]
    )
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0].split()[:2] == ["model", "modelpath"]
    assert "Reading" in captured.err
    assert "Reading" not in captured.out


def test_output_format_gives_json_and_csv(capsys: pytest.CaptureFixture[str]) -> None:
    """--output-format gives a table as JSON or CSV on the standard output, as --output-format of ruff does."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["timesteps", str(modelpath), "--output-format", "csv"])
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert rows[0] == ["timestep", "start_days", "mid_days", "end_days", "width_days"]
    assert np.isclose(float(rows[1][1]), 250.0)

    artistools.__main__.main(argsraw=["check", str(modelpath), "--output-format", "json"])
    checks = json.loads(capsys.readouterr().out)["rows"]
    assert {"check", "status", "detail"} <= set(checks[0])

    # catalog printed its "#" method lines in front of the JSON object, and json.loads refused the output
    artistools.__main__.main(argsraw=["catalog", str(at.get_path("testdata")), "--output-format", "json"])
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["comments"]
    assert catalog["rows"]


def test_output_format_of_plotted_data_gives_one_json_document_and_the_file_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plot with a main table and a summary table prints one JSON document, which json.loads reads.

    Before this change, the command printed one JSON document for each table, and a parser of JSON refused the output.
    --write-data with --output-format csv wrote a text file with spaces and the suffix .txt.
    """
    import artistools.__main__

    outputfile = tmp_path / "lightcurve.pdf"
    artistools.__main__.main(
        argsraw=["plotlightcurves", str(modelpath), "--print-data", "--output-format", "json", "-o", str(outputfile)]
    )
    document = json.loads(capsys.readouterr().out)
    assert "data" in document
    assert "model" in document["data"][0]

    # a reader of CSV takes one header, thus csv prints the main table alone and names the summary table
    artistools.__main__.main(
        argsraw=["plotlightcurves", str(modelpath), "--print-data", "--output-format", "csv", "-o", str(outputfile)]
    )
    captured = capsys.readouterr()
    rows = list(csv.reader(captured.out.splitlines()))
    assert rows[0][:2] == ["model", "modelpath"]
    assert len({len(row) for row in rows}) == 1
    assert "the other tables stay out: summary" in captured.err

    artistools.__main__.main(
        argsraw=["plotlightcurves", str(modelpath), "--write-data", "--output-format", "csv", "-o", str(outputfile)]
    )
    datalines = [line for line in (tmp_path / "lightcurve.csv").read_text().splitlines() if not line.startswith("#")]
    assert datalines[0].startswith("model,modelpath,")


def test_figures_of_one_command_print_one_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A command with one figure for each element prints the data of all the figures as one CSV or JSON document.

    plotnltepops printed a header and rows for each figure, thus a reader of CSV took the second header as data.
    """
    import artistools.__main__

    argsraw = ["plotnltepops", "Fe", "Co", str(modelpath), "--timestep", "40", "--print-data", "-o", str(tmp_path)]
    artistools.__main__.main(argsraw=[*argsraw, "--output-format", "csv"])
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert [row for row in rows if row == rows[0]] == [rows[0]]
    ioncolumn = rows[0].index("ion")
    assert {row[ioncolumn].split()[0] for row in rows[1:]} == {"Fe", "Co"}

    # the table of a figure holds no column of the element or the timestep, thus the column figure names its file
    figurecolumn = rows[0].index("figure")
    assert {row[figurecolumn].split("_")[1] for row in rows[1:]} == {"Fe", "Co"}

    artistools.__main__.main(argsraw=[*argsraw, "--output-format", "json"])
    jsonrows = json.loads(capsys.readouterr().out)["data"]
    assert {row["ion"].split()[0] for row in jsonrows} == {"Fe", "Co"}
    assert all(row["figure"].startswith(f"plotnltepops_{row['ion'].split()[0]}_") for row in jsonrows)


def test_json_rows_give_null_for_nan_and_infinity() -> None:
    """A strict parser of JSON refuses NaN and Infinity, thus the JSON output gives null for each of them."""
    from artistools.misc.commandoutput import get_json_rows

    dfdata = pl.DataFrame({"x": [1.5, float("nan"), float("inf"), float("-inf"), None], "n": [1, 2, 3, 4, 5]})
    rows = get_json_rows(dfdata)
    assert [row["x"] for row in rows] == [1.5, None, None, None, None]
    assert [row["n"] for row in rows] == [1, 2, 3, 4, 5]
    json.loads(json.dumps(rows, allow_nan=False))


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
    assert "Did you mean --dimensionreduce" in capsys.readouterr().err


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
    at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(merged))

    assert merged.is_file(), f"the merged pdf must keep its name, but {list(tmp_path.iterdir())}"
    assert not list(tmp_path.glob("plotradfield_*.pdf")), "the merge takes the frames away"

    # a -o path that names a folder still gives the merged pdf the name of its frames
    outfolder = tmp_path / "rf"
    at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(outfolder))
    assert list(outfolder.glob("plotradfield_*-plotradfield_*.pdf")), f"no merged pdf in {list(outfolder.iterdir())}"


def test_the_product_keeps_its_name_when_one_frame_holds_data(tmp_path: Path) -> None:
    """A run that holds data for one frame must still write the product that -o named.

    combine_frames took that frame for the product and left the named path empty, thus --open opened
    the frame. A product that carries the name of a frame also went, because the merge removes its
    inputs.
    """
    # the test model holds no radiation field data before timestep 10, thus one frame comes of the two
    merged = tmp_path / "merged.pdf"
    at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", outputfile=str(merged))
    assert merged.is_file(), f"the product must keep its name, but {list(tmp_path.iterdir())}"

    # the name of the product can be the name that a frame would take
    likeaframe = tmp_path / "plotradfield_cell00000_ts040.pdf"
    at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", outputfile=str(likeaframe))
    assert likeaframe.is_file(), f"the merge must not remove its own product: {list(tmp_path.iterdir())}"
