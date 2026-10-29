import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import pytest

import artistools as at


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_hesma_plots_take_the_times_and_the_tables_of_each_file(
    mockplot: mock.MagicMock, tmp_path: Path, get_plot_xy: Callable[[t.Any], tuple[np.ndarray, np.ndarray]]
) -> None:
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
        assert np.allclose(yarr, np.array([dirbin + 2.0, dirbin + 4.0]) * 1e-10, rtol=1e-6, atol=0.0)

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
def test_hesma_spectrum_takes_the_column_names_of_the_file(
    mockplot: mock.MagicMock, tmp_path: Path, get_plot_xy: Callable[[t.Any], tuple[np.ndarray, np.ndarray]]
) -> None:
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
    assert np.allclose(yarr, np.array([1.0, 3.0]) * 1e-10, rtol=1e-6, atol=0.0)


def test_hesma_reports_missing_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    """An action must name the argument it needs rather than failing on a None."""
    with pytest.raises(SystemExit):
        at.hesma_scripts.main(argsraw=["vspecfiles"])
    assert "requires --modelpath" in capsys.readouterr().err
