import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


def read_data_file(path: Path) -> pl.DataFrame:
    """Read a table that --write_data wrote."""
    return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")


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


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plottransitions_writes_the_drawn_values(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """--write_data gives one row for each ion, each series, and each wavelength, and the ion names the panel."""
    at.plottransitions.main(
        argsraw=[], modelpath=modelpath, timedays=300, T=[5000.0], outputfile=tmp_path / "trans.pdf", write_data=True
    )
    dftable = read_data_file(tmp_path / "trans.txt")

    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    ionlabels = list(dict.fromkeys(dftable["ion"]))
    assert len(ionlabels) == len(axes) == 7
    assert dftable.height == sum(len(call.args[1]) for call in mockplot.call_args_list)
    for call in mockplot.call_args_list:
        axis, drawnx, drawny = call.args
        dfseries = dftable.filter(
            pl.col("ion") == ionlabels[axes.index(axis)], pl.col("series") == call.kwargs["label"]
        )
        assert np.allclose(dfseries["lambda_angstroms"].to_numpy(), drawnx, rtol=1e-12, atol=0.0)
        assert np.allclose(dfseries["f_lambda_arbitrary"].to_numpy(), drawny, rtol=1e-12, atol=0.0)
