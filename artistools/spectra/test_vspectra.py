import shutil
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import pytest

import artistools as at


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_vspectraplot(mockplot: mock.MagicMock) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[at.get_path("testdata") / "vspecpolmodel", "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=at.get_path("testoutput") / "test_vspectra.pdf",
        plotvspecpol=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        timemin=11,
        timemax=12,
        distmpc=1.0,
    )

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(10))

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(10)])
    assert np.allclose(
        arr_allvspec.std(axis=1),
        [
            2.01529689e-12,
            2.05807110e-12,
            2.01551623e-12,
            2.18216916e-12,
            2.85477069e-12,
            3.34384407e-12,
            2.94892344e-12,
            2.29084411e-12,
            2.05916843e-12,
            2.00515984e-12,
        ],
        rtol=0.001,
        atol=0.0,
    )

    assert np.allclose(
        arr_allvspec.mean(axis=1),
        [
            2.9864681492951925e-12,
            3.0063451037690416e-12,
            2.9785924608537284e-12,
            3.2028094816751935e-12,
            4.097482117229833e-12,
            4.663450168092402e-12,
            4.231106733071208e-12,
            3.350080172063692e-12,
            3.0234533505898177e-12,
            2.9721539798925583e-12,
        ],
        rtol=0.001,
        atol=0.0,
    )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_vpkt_frompackets_spectrum_plot(mockplot: mock.MagicMock) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[at.get_path("testdata") / "vpktcontrib"],
        outputfile=at.get_path("testoutput") / "test_vpktscontrib_spectra.pdf",
        plotvspecpol=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        frompackets=True,
        maxpacketfiles=2,
        timemin=130,
        timemax=135,
    )

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(9))

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(9)])
    print("expecting (std): ", [float(x) for x in arr_allvspec.std(axis=1)])
    assert np.allclose(
        arr_allvspec.std(axis=1),
        [
            2.327497886235515e-15,
            1.3598207820269362e-14,
            5.008263656577672e-15,
            2.2846087128139245e-15,
            1.3201387803043819e-14,
            4.942566647559291e-15,
            2.3914786916308798e-15,
            1.3150990321980048e-14,
            4.713373632148831e-15,
        ],
        rtol=0.001,
        atol=0.0,
    )

    print("expecting (mean): ", [float(x) for x in arr_allvspec.mean(axis=1)])
    assert np.allclose(
        arr_allvspec.mean(axis=1),
        [
            1.2260651224209835e-15,
            8.443895573486584e-15,
            3.2160750274055088e-15,
            1.2006100531102612e-15,
            8.309438486245869e-15,
            3.2461295230908496e-15,
            1.2303479856067529e-15,
            8.224253260472366e-15,
            3.1773192704100478e-15,
        ],
        rtol=0.001,
        atol=0.0,
    )


def test_average_vspecpol_files(tmp_path: Path) -> None:
    """--averagevspecpolfiles averages the vspecpol_total files of every -specpath model.

    make_averaged_vspecfiles read args.modelpath, and plotspectra stores every path under
    args.specpath, thus the command stopped with AttributeError before this test existed.
    """
    modeldirs = [tmp_path / "model_a", tmp_path / "model_b"]
    for modeldir in modeldirs:
        modeldir.mkdir()
        for filename in ("vspecpol_total-0.out", "vspecpol_total-1.out"):
            shutil.copy(at.get_path("testdata") / "vspecpolmodel" / filename, modeldir / filename)

    at.spectra.plot(argsraw=[], specpath=[str(modeldir) for modeldir in modeldirs], averagevspecpolfiles=True)

    for specindex in (0, 1):
        averagedpath = modeldirs[0] / f"vspecpol_averaged-{specindex}.out"
        assert averagedpath.is_file()
        # the two models hold the same data, thus the average equals the source
        averaged = np.loadtxt(averagedpath)
        source = np.loadtxt(modeldirs[0] / f"vspecpol_total-{specindex}.out")
        assert np.allclose(averaged, source, rtol=1e-6, atol=0.0)


def test_make_virtual_spectra_summed_file(tmp_path: Path) -> None:
    """--makevspecpol sums the flux of every rank and keeps the time header and the frequency column.

    Each rank file holds one table for each observer. Two ranks with the same data give a total of
    twice the flux.
    """
    sourcedir = at.get_path("testdata") / "vspecpolmodel"
    shutil.copy(sourcedir / "vpkt.txt", tmp_path / "vpkt.txt")
    # get_nprocs reads the rank count from line 22 of input.txt
    (tmp_path / "input.txt").write_text("\n".join(["0"] * 21 + ["2"]) + "\n", encoding="utf-8")

    nobservers = 10
    ranktext = "".join((sourcedir / f"vspecpol_total-{specindex}.out").read_text() for specindex in range(nobservers))
    for mpirank in range(2):
        (tmp_path / f"vspecpol_{mpirank:04d}.out").write_text(ranktext, encoding="utf-8")

    at.spectra.make_virtual_spectra_summed_file(tmp_path)

    for specindex in range(nobservers):
        source = np.loadtxt(sourcedir / f"vspecpol_total-{specindex}.out")
        total = np.loadtxt(tmp_path / f"vspecpol_total-{specindex}.out")
        assert total.shape == source.shape
        assert np.array_equal(total[0], source[0]), "the time header must not be summed"
        assert np.array_equal(total[1:, 0], source[1:, 0]), "the frequency column must not be summed"
        assert np.allclose(total[1:, 1:], 2 * source[1:, 1:], rtol=1e-12, atol=0.0)


def copy_vpktcontrib_model(modelfolder: Path, timewindowline: str, rangesline: str) -> Path:
    """Copy the vpktcontrib model with a new time window line and a new wavelength range line in vpkt.txt."""
    sourcedir = at.get_path("testdata") / "vpktcontrib"
    modelfolder.mkdir()
    for filename in ("input.txt", "vpackets_0000.out.zst", "vpackets_0001.out.zst"):
        shutil.copy(sourcedir / filename, modelfolder / filename)
    vpktlines = (sourcedir / "vpkt.txt").read_text(encoding="utf-8").splitlines()
    # the fifth line is the time window override, and the sixth line is the custom wavelength range flag
    vpktlines[4:6] = [timewindowline, rangesline]
    (modelfolder / "vpkt.txt").write_text("\n".join(vpktlines) + "\n", encoding="utf-8")
    return modelfolder


def test_vpkt_frompackets_spectrum_keeps_the_rows_inside_the_vpkt_ranges(tmp_path: Path) -> None:
    """The spectrum of a virtual observer bins only the rows with nu_rf inside a wavelength range of vpkt.txt.

    ARTIS also writes a row when only the absorption frequency is inside a range, and it keeps that row out of
    vspecpol. The code binned each row, thus the bins outside the ranges held a biased part of the flux.
    """
    fullmodel = copy_vpktcontrib_model(tmp_path / "full", "0 10 30", "0")
    rangesmodel = copy_vpktcontrib_model(tmp_path / "ranges", "1 130 140", "1 2 3500 6000 6400 7200")
    lambda_bin_edges = np.arange(3000.0, 8000.0, 100.0)

    def get_flambda(modelpath: Path) -> npt.NDArray[np.floating]:
        dfspectrum = at.spectra.get_from_packets(
            modelpath,
            timelowdays=131.0,
            timehighdays=139.0,
            lambda_bin_edges=lambda_bin_edges,
            directionbins_are_vpkt_observers=True,
            directionbins=[0],
        )[0].collect()
        return dfspectrum["f_lambda"].to_numpy()

    flambda_full = get_flambda(fullmodel)
    flambda_ranges = get_flambda(rangesmodel)

    binlow, binhigh = lambda_bin_edges[:-1], lambda_bin_edges[1:]
    insiderange = ((binlow >= 3500.0) & (binhigh <= 6000.0)) | ((binlow >= 6400.0) & (binhigh <= 7200.0))
    outsideranges = (binhigh <= 3500.0) | ((binlow >= 6000.0) & (binhigh <= 6400.0)) | (binlow >= 7200.0)
    assert np.allclose(flambda_ranges[insiderange], flambda_full[insiderange], rtol=1e-12, atol=0.0)
    assert flambda_full[outsideranges].sum() > 0.0
    assert not flambda_ranges[outsideranges].any()

    with pytest.raises(ValueError, match="outside the virtual packets, which cover 130"):
        at.spectra.get_from_packets(
            rangesmodel,
            timelowdays=125.0,
            timehighdays=135.0,
            lambda_bin_edges=lambda_bin_edges,
            directionbins_are_vpkt_observers=True,
            directionbins=[0],
        )


def test_vpkt_frompackets_plot_refuses_a_time_range_outside_the_vpkt_window(tmp_path: Path) -> None:
    """--frompackets with -plotvspecpol stops as the vspecpol branch does when the time range leaves the window.

    ARTIS writes a virtual packet only inside the time window of vpkt.txt. The flux of the packets took the full
    width of the time range, thus a range that left the window gave a low flux and no message. The dispatcher
    reports the ValueError without a traceback.
    """
    modelpath = copy_vpktcontrib_model(tmp_path / "window", "1 130 140", "0")
    with pytest.raises(ValueError, match="outside the virtual packets"):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath],
            outputfile=tmp_path / "vpkt_window.pdf",
            plotvspecpol=[0],
            frompackets=True,
            timemin=125,
            timemax=135,
        )
