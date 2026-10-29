import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(
    mockstep: mock.MagicMock,
    mockplot: mock.MagicMock,
    get_plot_xy: Callable[[t.Any], tuple[np.ndarray, np.ndarray]],
    tmp_path: Path,
) -> None:
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=tmp_path, showbinedges=True
    )

    plot_calls = {
        label.strip(): call for call in mockplot.call_args_list if isinstance((label := call.kwargs.get("label")), str)
    }
    dilute_xarr, dilute_yarr = get_plot_xy(plot_calls["Dilute blackbody model"])
    assert np.isclose(dilute_xarr.min(), 1000.0, rtol=1e-4)
    assert np.isclose(dilute_xarr.max(), 20000.0, rtol=1e-4)
    assert np.isclose(dilute_yarr.mean(), 21.27744616064978, rtol=1e-4)
    assert np.isclose(dilute_yarr.std(), 26.77850448874471, rtol=1e-4)

    fitted_xarr, fitted_yarr = get_plot_xy(plot_calls["Radiation field model"])
    assert np.isclose(fitted_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(fitted_xarr.max(), 20000.030554517798, rtol=1e-4)
    # nine bins of this cell have no fit (W < 0). Their model takes the full-spectrum fit, as ARTIS radfield() does,
    # and their band average takes J. Both were zero before
    assert np.isclose(fitted_yarr.mean(), 45.14912462715829, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(fitted_yarr, fitted_xarr)), 492291.22990819113, rtol=1e-4)

    bandavg_xarr, bandavg_yarr = get_plot_xy(mockstep.call_args_list[0])
    assert np.isclose(bandavg_xarr.min(), 2000.0030554517798, rtol=1e-4)
    assert np.isclose(bandavg_xarr.max(), 20000.030554517798, rtol=1e-4)
    assert np.isclose(bandavg_yarr.mean(), 43.854670073915386, rtol=1e-4)
    assert np.isclose(abs(np.trapezoid(bandavg_yarr, bandavg_xarr)), 476000.8181544015, rtol=1e-4)


def test_radfield_bin_with_no_fit_takes_the_full_spectrum_fit() -> None:
    """A bin with a negative W takes the full-spectrum dilute blackbody, as radfield() of ARTIS does.

    The plot drew zero for such a bin, and the band average drew zero although the bin holds its J.
    """
    from artistools.radfield.core import get_binaverage_field
    from artistools.radfield.core import get_fitted_field
    from artistools.radfield.core import j_nu_dbb

    radfielddata = pl.DataFrame({
        "timestep": [5, 5, 5],
        "modelgridindex": [0, 0, 0],
        "bin_num": [-1, 0, 1],
        "nu_lower": [0.0, 5.0e14, 6.0e14],
        "nu_upper": [0.0, 6.0e14, 7.0e14],
        "J": [1.0, 40.0, 30.0],
        "T_R": [6000.0, -1.0, 7000.0],
        "W": [0.5, -1.0, 0.1],
    })

    arr_lambda, j_lambda = get_fitted_field(radfielddata, modelgridindex=0, timestep=5)
    arr_nu = at.constants.c_ang_per_s / np.array(arr_lambda)
    j_nu_fullspec = np.array(j_nu_dbb(arr_nu, 0.5, 6000.0))
    j_nu_bins = np.array(j_nu_dbb(arr_nu, 0.1, 7000.0))
    j_nu = np.array(j_lambda) * np.array(arr_lambda) / arr_nu
    assert np.allclose(j_nu[:200], j_nu_fullspec[:200], rtol=1e-10, atol=0.0)
    assert np.allclose(j_nu[200:], j_nu_bins[200:], rtol=1e-10, atol=0.0)

    # before MULTIBIN_RADFIELD_FIRST_TIMESTEP, ARTIS takes the full-spectrum fit for every bin
    _, j_lambda_fullspec = get_fitted_field(radfielddata, modelgridindex=0, timestep=5, usebinfits=False)
    assert np.allclose(np.array(j_lambda_fullspec) * np.array(arr_lambda) / arr_nu, j_nu_fullspec, rtol=1e-10, atol=0.0)

    _, bandaverage = get_binaverage_field(radfielddata, modelgridindex=0, timestep=5)
    dlambda = at.constants.c_ang_per_s * (1 / 5.0e14 - 1 / 6.0e14)
    assert np.isclose(bandaverage[1], 40.0 / dlambda, rtol=1e-10, atol=0.0)


def test_radfield_takes_the_last_timestep_with_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With no timestep, the plot takes the last timestep that holds data, and a run with no plot gives an error.

    The command took the last timestep of the time grid. For a run that stopped early, it wrote no file and ended
    with no error.
    """
    radfielddata = at.radfield.read_radfield(modelpath, modelgridindex=[0])
    with mock.patch.object(
        at.radfield.plotradfield, "read_radfield", return_value=radfielddata.filter(pl.col("timestep") <= 59)
    ):
        at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=tmp_path)
        assert [path.name for path in tmp_path.glob("*.pdf")] == ["plotradfield_cell00000_ts059.pdf"]

        with pytest.raises(SystemExit):
            at.radfield.plotradfield.main(
                argsraw=[], modelpath=modelpath, modelgridindex=0, timestep=5, outputfile=tmp_path
            )
    assert "No radiation field data" in capsys.readouterr().err


@mock.patch.object(mplax.Axes, "scatter", side_effect=mplax.Axes.scatter, autospec=True)
def test_radfield_line_estimators_filter_cell_zero(mockscatter: mock.MagicMock) -> None:
    """The line estimator plot must filter on cell and timestep zero, which are falsy."""
    radfielddata = pl.DataFrame({
        "bin_num": [-2, -3, -2, -3],
        "modelgridindex": [0, 0, 1, 1],
        "timestep": [0, 0, 0, 0],
        "nu_upper": [1.0e15, 2.0e15, 3.0e15, 4.0e15],
        "J_nu_avg": [1.0e-20, 2.0e-20, 3.0e-20, 4.0e-20],
    })

    fig, ax = plt.subplots()
    at.radfield.plotradfield.plot_line_estimators(ax, radfielddata, modelgridindex=0, timestep=0)
    plt.close(fig)

    assert mockscatter.call_count == 1
    lambdas = np.array(mockscatter.call_args_list[0][0][1], dtype=float)
    # only the two rows of cell zero, not all four rows
    assert len(lambdas) == 2
    assert np.allclose(sorted(lambdas), sorted(at.constants.c_ang_per_s / np.array([1.0e15, 2.0e15])))


@mock.patch.object(mplax.Axes, "set_ylim", side_effect=mplax.Axes.set_ylim, autospec=True)
def test_radfield_honours_the_ymin_that_it_accepts(mocksetylim: mock.MagicMock) -> None:
    """Plotradfield adds -ymin, thus the axis must start there and not at the hard-coded zero."""
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40, modelgridindex=0, ymin=1e-14
    )

    bottoms = [callargs.kwargs["bottom"] for callargs in mocksetylim.call_args_list if "bottom" in callargs.kwargs]
    assert bottoms, "the command must set the bottom of the axis"
    assert 1e-14 in bottoms, f"the requested -ymin is missing from {bottoms}"


def test_radfield_opens_the_merged_pdf_alone(tmp_path: Path) -> None:
    """--open must open the merged pdf and not each plot that the merge takes in.

    merge_pdf_files deletes those plots, thus an application that opened one would hold nothing.
    """
    template = str(tmp_path / "rf_cell{cell:05d}_ts{timestep:03d}.pdf")
    with mock.patch("subprocess.run") as mockrun:
        at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="40-41", open=True, outputfile=template)

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"one file must open, not {len(opened)}"
    assert Path(opened[0]).is_file(), "the file that opens must be the merged pdf, which still exists"


def test_radfield_opens_the_one_plot_that_holds_data(tmp_path: Path) -> None:
    """A range that holds data for one timestep alone makes one plot, and that plot is the product.

    The run took each plot for a part of a merge, thus no plot opened. No merge came, because one
    plot cannot merge, thus --open did nothing at all.
    """
    # the test model holds no radiation field data before timestep 10
    template = str(tmp_path / "rf_cell{cell:05d}_ts{timestep:03d}.pdf")
    with mock.patch("subprocess.run") as mockrun:
        at.radfield.plotradfield.main(argsraw=[], modelpath=modelpath, timestep="9-10", open=True, outputfile=template)

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"the one plot must open, not {len(opened)} files"
    assert Path(opened[0]).is_file()


@pytest.mark.parametrize(
    "optionname", ["MULTIBIN_RADFIELD_FIRST_TIMESTEP", "FIRST_BINNED_RADFIELD_TIMESTEP", "FIRST_NLTE_RADFIELD_TIMESTEP"]
)
def test_radfield_first_timestep_takes_each_name_of_the_option(tmp_path: Path, optionname: str) -> None:
    """ARTIS renamed the option two times on 2026-10-16, thus a current model gave no first timestep of the bin fits."""
    (tmp_path / "artisoptions.h").write_text(f"constexpr int {optionname} = 6;\n", encoding="utf-8")
    assert at.radfield.core.get_multibin_radfield_first_timestep(tmp_path) == (6, optionname)


def test_plotradfield_table_follows_the_sole_frame_to_the_product(tmp_path: Path) -> None:
    """Two timesteps with data for one timestep alone gave one figure, and the table kept the name of the frame."""
    # the test model has no radiation field data before timestep 10
    at.radfield.plotradfield.main(
        argsraw=[],
        modelpath=modelpath,
        modelgridindex=[0],
        timestep="5,40",
        outputfile=tmp_path / "rf.pdf",
        write_data=True,
    )
    assert (tmp_path / "rf.pdf").is_file()
    assert (tmp_path / "rf.txt").is_file()
    assert not list(tmp_path.glob("plotradfield_cell*.txt"))


@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotradfield_writes_the_drawn_values(
    mockplot: mock.MagicMock, mockstep: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data gives the drawn points of each series in one long table, because each series has its own grid."""
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, modelgridindex=0, timestep=40, outputfile=tmp_path / "rf.pdf", write_data=True
    )
    dftable = read_data_file(tmp_path / "rf.txt")

    drawncalls = [*mockplot.call_args_list, *mockstep.call_args_list]
    drawnseries = {str(call.kwargs["label"]).strip(): call.args for call in drawncalls if "label" in call.kwargs}
    # the emergent spectrum has the scale of the specific intensity at the surface
    assert set(drawnseries) == {
        "Dilute blackbody model",
        "Band-average field",
        "Radiation field model",
        "Emergent spectrum (scaled)",
    }
    assert set(dftable["series"]) == set(drawnseries)
    # the axis shows 1000 to 20000 Å, and the table takes the points of that range
    for label, (_, drawnxvalues, drawnyvalues) in drawnseries.items():
        dfseries = dftable.filter(pl.col("series") == label)
        drawnx, drawny = np.asarray(drawnxvalues, dtype=float), np.asarray(drawnyvalues, dtype=float)
        if label == "Band-average field":
            # a step of where="pre" shows the value of a point between the point before it and the point, thus the
            # first point outside the axis has no part in the figure
            isinside = (drawnx > 1000.0) & (drawnx < 20000.0)
            assert set(drawnx[isinside].round(6)) <= set(dfseries["lambda_angstroms"].round(6).to_list())
            assert set(dfseries["lambda_angstroms"].round(6).to_list()) <= set(drawnx.round(6))
            continue
        iskept = (drawnx >= 1000.0) & (drawnx <= 20000.0)
        assert np.allclose(dfseries["lambda_angstroms"].to_numpy(), drawnx[iskept], rtol=1e-12, atol=0.0)
        assert np.allclose(
            dfseries["J_lambda_erg_per_s_per_cm2_per_angstrom"].to_numpy(), drawny[iskept], rtol=1e-12, atol=0.0
        )
