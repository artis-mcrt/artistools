import argparse
import re
import shutil
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

import artistools as at

modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_report_holds_each_part_and_its_figures(tmp_path: Path) -> None:
    """The page gives each part in order, an image for each figure, and the summary lines of each figure."""
    from artistools.report import main

    dfchecks = pl.DataFrame({
        "check": ["light curve", "estimators"],
        "status": ["ok", "warning"],
        "detail": ["light_curve.out has 30 timesteps", "121 cells <at MAXTEMP>"],
    })
    reportfolder = tmp_path / "report"
    with mock.patch("artistools.report.get_check_results_table", return_value=dfchecks) as mockcheck:
        main(argsraw=[], modelpath=modelpath_classic_3d, outputfile=reportfolder)
    mockcheck.assert_called_once()

    page = (reportfolder / "index.html").read_text(encoding="utf-8")
    headings = re.findall(r"<h2 id=\"[^\"]+\">([^<]+)</h2>", page)
    assert headings == [
        "Provenance",
        "Health check",
        "Model summary",
        "Log summary of rank 0",
        "Bolometric light curve",
        "Spectra",
        "Electron temperature against velocity",
    ]
    # the page escapes the text of a check, and the row of a warning has its class
    assert '<tr class="status-warning"><td>estimators</td><td>warning</td><td>121 cells &lt;at MAXTEMP&gt;</td>' in page
    assert "model.txt: 3D, 1000 cells, 695 cells with matter" in page
    assert "call_T_e_finder: heating bigger than cooling" in page

    imagenames = re.findall(r'<img src="([^"]+)" alt="[^"]+">', page)
    assert imagenames == [
        "lightcurve.png",
        "spectrum_timestep005.png",
        "spectrum_timestep014.png",
        "spectrum_timestep023.png",
        "Te_timestep005.png",
        "Te_timestep014.png",
        "Te_timestep023.png",
    ]
    assert all((reportfolder / imagename).is_file() for imagename in imagenames)
    assert "<pre>test-classicmode_3d: 19 points from 3.591 to 6.468 d, peak 1.206e+43 erg/s at 5.864 d" in page
    assert "warning: Te: 260 points are at MAXTEMP" in page
    assert "<script" not in page
    assert "http" not in page

    # a second run asks before it overwrites the report, and stdin of pytest is no terminal
    with pytest.raises(SystemExit):
        main(argsraw=[], modelpath=modelpath_classic_3d, outputfile=reportfolder)


def test_report_asks_about_the_overwrite_before_it_reads_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report read the timesteps of the model before the question, e.g. through ssh for a remote model."""
    from artistools import report

    reportfolder = tmp_path / "myreport"
    reportfolder.mkdir()
    (reportfolder / "spectrum_timestep007.png").write_bytes(b"")
    with (
        mock.patch.object(report, "get_report_timesteps", side_effect=AssertionError("the model was read")),
        # the test of a file of a remote model asks its host through ssh
        mock.patch.object(report, "get_model_folder", side_effect=AssertionError("the model path was resolved")),
        mock.patch("sys.stdin.isatty", return_value=False),
        pytest.raises(SystemExit),
    ):
        report.main(argsraw=["testhost:~/mymodel", "-o", str(reportfolder)])
    stderr = capsys.readouterr().err
    assert "spectrum_timestep007.png exists" in stderr
    assert "did not overwrite the files" in stderr


def test_report_replaces_a_link_at_the_page_and_leaves_its_target(tmp_path: Path) -> None:
    """write_text followed a link at index.html, thus --force wrote the page into a file of the user."""
    from artistools import report

    target = tmp_path / "notes.txt"
    target.write_text("keep\n", encoding="utf-8")
    reportfolder = tmp_path / "myreport"
    reportfolder.mkdir()
    (reportfolder / "index.html").symlink_to(target)
    sectionnames = [
        "get_provenance_html",
        "get_health_check_html",
        "get_model_summary_html",
        "get_log_summary_html",
        "get_light_curve_html",
        "get_spectra_html",
        "get_temperature_html",
    ]
    with (
        mock.patch.object(report, "get_report_timesteps", return_value=([], "no timesteps")),
        mock.patch.multiple(report, **dict.fromkeys(sectionnames, mock.DEFAULT)) as mocksections,
    ):
        for mocksection in mocksections.values():
            mocksection.return_value = ""
        report.main(argsraw=[str(modelpath_classic_3d), "-o", str(reportfolder), "--force"])

    assert target.read_text(encoding="utf-8") == "keep\n"
    assert not (reportfolder / "index.html").is_symlink()
    assert "<html" in (reportfolder / "index.html").read_text(encoding="utf-8")

    # a figure of a timestep of the earlier report stayed and seemed to belong to the new report
    (reportfolder / "spectrum_timestep007.png").write_bytes(b"")
    with (
        mock.patch.object(report, "get_report_timesteps", return_value=([], "no timesteps")),
        mock.patch.multiple(report, **dict.fromkeys(sectionnames, mock.DEFAULT)) as mocksections,
    ):
        for mocksection in mocksections.values():
            mocksection.return_value = ""
        report.main(argsraw=[str(modelpath_classic_3d), "-o", str(reportfolder), "--force"])
    assert not (reportfolder / "spectrum_timestep007.png").exists()


def test_report_takes_the_model_folder_of_a_remote_file_path() -> None:
    """A remote model path can name a file of the model, and the default report folder took the name of the file."""
    from artistools.report import get_remote_model_folder_from_text

    assert get_remote_model_folder_from_text(Path("host:~/models/foo/light_curve.out")).name == "foo"
    assert get_remote_model_folder_from_text(Path("host:~/models/foo/spec.out.zst")).name == "foo"
    assert get_remote_model_folder_from_text(Path("host:~/models/foo")).name == "foo"


def test_report_provenance_keeps_a_remote_model_folder() -> None:
    """A local resolve put the working folder in front of host:path, which named a folder that does not exist."""
    from artistools.report import get_provenance_html

    provenance = get_provenance_html(argparse.Namespace(), Path("testhost:~/mymodel"))

    assert "model folder: testhost:~/mymodel" in provenance


def test_report_of_model_with_no_estimators_names_each_error(tmp_path: Path) -> None:
    """A part that fails gives one error line, and the page continues to its end."""
    from artistools.report import main

    modelfolder = tmp_path / "noestimators"
    modelfolder.mkdir()
    for filename in (
        "input.txt",
        "model.txt",
        "abundances.txt",
        "compositiondata.txt",
        "light_curve.out",
        "spec.out",
        "deposition.out",
    ):
        shutil.copy(modelpath_classic_3d / filename, modelfolder / filename)
    reportfolder = tmp_path / "report"
    with mock.patch("artistools.report.get_check_results_table", side_effect=ValueError("no check")):
        main(argsraw=[], modelpath=modelfolder, outputfile=reportfolder)

    page = (reportfolder / "index.html").read_text(encoding="utf-8")
    assert '<p class="error">This part of the report failed: ValueError: no check</p>' in page
    assert "The model has no log of rank 0." in page
    assert page.count('<img src="spectrum_timestep') == 3
    assert '<img src="Te_timestep' not in page
    assert page.count('<p class="error">This part of the report failed:') == 4
    assert page.rstrip().endswith("</html>")
