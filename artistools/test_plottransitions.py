import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest

import artistools as at
from artistools.constants import K_B_ev_per_K

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")


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
def test_transitions(mockplot: mock.MagicMock, get_plot_xy: Callable[[t.Any], tuple[np.ndarray, np.ndarray]]) -> None:
    at.plottransitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)

    assert len(mockplot.call_args_list) == 7
    expected_integrals = [
        0.05309512674525126,
        376.57154726748985,
        422.3699041724463,
        11.76194204379188,
        48.772050822952906,
        0.0,
        0.0,
    ]
    expected_maxima = [
        8.724265104752106e-05,
        0.40791643763306906,
        1.079855927437704,
        0.016257895914721012,
        0.062324908090780305,
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
    character happens to be a space. Thus the loggf of the test fills its field. The second line gives the upper level
    first, and a negative energy for a predicted level. The reader took the first level as the lower level and kept
    the sign of the energy.
    """

    def get_gfall_line(energy1: float, j1: float, energy2: float, j2: float) -> str:
        return (
            f"{715.5170:11.4f}"  # 0-10  wavelength in nm
            f"{-10.234:7.3f}"  # 11-17 log(gf)
            f"{44.00:6.2f}"  # 18-23 element code Z.(ion_stage - 1)
            f"{energy1:12.3f}"  # 24-35 energy of the first level in cm-1
            f"{j1:5.1f}"  # 36-40 J of the first level
            " a4F       "  # 41-51 configuration label
            f"{energy2:12.3f}"  # 52-63 energy of the second level in cm-1
            f"{j2:5.1f}" + " " + " 0" * 17 + "\n"  # 64-68 J of the second level
            # the parser only reads lines with at least 24 whitespace-separated fields, and a loggf that fills its
            # field joins the wavelength in one field
        )

    (tmp_path / "gfall.dat").write_text(
        get_gfall_line(25000.0, 4.5, 35000.0, 3.5) + get_gfall_line(-35000.0, 3.5, 25000.0, 4.5), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    dftransitions, ionlist = at.plottransitions.get_kurucz_transitions()

    assert ionlist == [(44, 1)]
    assert len(dftransitions) == 2
    expected_values = {
        "lambda_angstroms": 7155.170,
        "lower_statweight": 2 * 4.5 + 1,
        "upper_statweight": 2 * 3.5 + 1,
        "lower_energy_ev": at.constants.hc_in_ev_cm * 25000.0,
        "upper_energy_ev": at.constants.hc_in_ev_cm * 35000.0,
    }
    for colname, expected in expected_values.items():
        assert dftransitions[colname].to_list() == pytest.approx([expected, expected]), colname

    # the two lines give the same transition, thus the same A value
    assert np.isclose(dftransitions["A"][0], dftransitions["A"][1], rtol=1e-12)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plottransitions_writes_the_drawn_values(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
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


def test_normalised_spectrum_of_an_ion_with_no_line_stays_zero(tmp_path: Path) -> None:
    """An ion with no line in the x range has a spectrum of zeros, and --normalised divided it by zero."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        at.plottransitions.main(
            argsraw=[],
            modelpath=modelpath,
            timedays=300,
            normalised=True,
            xmin=7140,
            xmax=7170,
            outputfile=tmp_path / "trans.pdf",
        )


@pytest.mark.parametrize("temperature", [3000.0, 8000.0])
def test_transitions_distribute_the_superlevel_population(temperature: float) -> None:
    levels = pl.DataFrame({"levelindex": [0, 1, 2, 3], "g": [1.0, 3.0, 5.0, 7.0], "energy_ev": [0.0, 1.0, 2.0, 3.0]})
    nltepops = pl.DataFrame({
        "modelgridindex": [2, 2, 2],
        "Z": [26, 26, 26],
        "ion_stage": [2, 2, 2],
        "level": [0, 1, -1],
        "n_NLTE": [10.0, 20.0, 30.0],
    })
    conditions = at.plottransitions.PlotConditions(
        ionpopdict={(26, 2): 60.0},
        temperature_list=["NOTEMPNLTE"],
        vardict={"Te": 5000.0, "T_exc": temperature},
        figure_title="",
        dfnltepops=nltepops,
    )
    transitions = pl.DataFrame({"upper": [1, 2, 3], "flux_factor": [2.0, 3.0, 4.0], "upper_pop_Te": [1.0, 1.0, 1.0]})
    result = at.plottransitions.add_nlte_pop(transitions, levels, conditions, (26, 2))
    weights = np.array([5.0, 7.0]) * np.exp(-np.array([0.0, 1.0]) / K_B_ev_per_K / temperature)
    expected = np.concatenate(([20.0], 30.0 * weights / weights.sum()))
    assert result["upper_pop_nlte"].to_numpy() == pytest.approx(expected)
    assert result["flux_factor_nlte"].to_numpy() == pytest.approx(expected * np.array([2.0, 3.0, 4.0]))
    assert result["upper_departure"].to_numpy() == pytest.approx(expected)


def test_transition_superlevel_uses_only_the_kept_levels(tmp_path: Path) -> None:
    levels = pl.DataFrame({"levelindex": [0, 1, 2], "g": [1.0, 3.0, 5.0], "energy_ev": [0.0, 1.0, 2.0]})
    atomicdata = pl.DataFrame({"Z": [26], "ion_stage": [2], "levels": pl.Series([levels], dtype=pl.Object)})
    with (
        mock.patch.object(at.lines.core, "get_levels", return_value=atomicdata),
        mock.patch.object(at.lines.core, "get_kept_level_counts", return_value={26: 2}),
        mock.patch.object(at.lines.core, "get_single_level_top_ion_stages", return_value={}),
    ):
        result = at.plottransitions.get_transition_atomic_data(tmp_path, [(26, 2)], from_model=True)
    keptlevels = result["levels"][0]
    nltepops = pl.DataFrame({
        "modelgridindex": [0, 0],
        "Z": [26, 26],
        "ion_stage": [2, 2],
        "level": [0, -1],
        "n_NLTE": [10.0, 30.0],
    })
    conditions = at.plottransitions.PlotConditions(
        ionpopdict={(26, 2): 40.0},
        temperature_list=["NOTEMPNLTE"],
        vardict={"Te": 5000.0, "T_exc": 5000.0},
        figure_title="",
        dfnltepops=nltepops,
    )
    assert keptlevels.height == 2
    transitions = pl.DataFrame({"upper": [1], "flux_factor": [1.0], "upper_pop_Te": [1.0]})
    populations = at.plottransitions.add_nlte_pop(transitions, keptlevels, conditions, (26, 2))
    assert populations["upper_pop_nlte"].sum() == pytest.approx(30.0)


@pytest.mark.parametrize(("option", "expected"), [("true", 8000.0), ("false", 5000.0)])
def test_transitions_take_the_model_excitation_temperature(tmp_path: Path, option: str, expected: float) -> None:
    (tmp_path / "artisoptions.h").write_text(f"constexpr bool LTEPOP_EXCITATION_USE_TJ = {option};\n", encoding="utf-8")
    nltepops = pl.DataFrame({"modelgridindex": [0], "Z": [26], "ion_stage": [2], "level": [0], "n_NLTE": [10.0]})
    cell = at.plottransitions.CellConditions(
        modelgridindex=0,
        timestep=0,
        time_days=1.0,
        velocity=1000.0,
        estimators={"Te": 5000.0, "TR": 6000.0, "TJ": 8000.0},
    )
    with mock.patch.object(at.plottransitions, "read_nltepops", return_value=nltepops):
        conditions = at.plottransitions.get_model_conditions(tmp_path, cell, [(26, 2)], [])
    assert conditions.vardict["T_exc"] == pytest.approx(expected)
