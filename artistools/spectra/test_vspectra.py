import os
import shutil
import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import pytest

import artistools as at


@pytest.mark.parametrize(
    ("plotargs", "expected_std", "expected_mean"),
    [
        pytest.param(
            {
                "specpath": [at.get_path("testdata") / "vspecpolmodel", "sn2011fe_PTF11kly_20120822_norm.txt"],
                "plotvspecpol": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "timemin": 11,
                "timemax": 12,
                "distmpc": 1.0,
            },
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
            id="vspecpol_files",
        ),
        pytest.param(
            {
                "specpath": [at.get_path("testdata") / "vpktcontrib"],
                "plotvspecpol": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "frompackets": True,
                "maxpacketfiles": 2,
                "timemin": 130,
                "timemax": 135,
            },
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
            id="vpkt_packets",
        ),
    ],
)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_virtual_spectrum_plot(
    mockplot: mock.MagicMock,
    tmp_path: Path,
    plotargs: dict[str, t.Any],
    expected_std: list[float],
    expected_mean: list[float],
) -> None:
    """Each direction of a virtual spectrum plot has the same wavelengths, and its own fluxes."""
    at.spectra.plot(argsraw=[], outputfile=tmp_path / "vspectra.pdf", **plotargs)
    ndirections = len(plotargs["plotvspecpol"])

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(
        np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(ndirections)
    )

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(ndirections)])
    print("expecting (std): ", [float(x) for x in arr_allvspec.std(axis=1)])
    assert np.allclose(arr_allvspec.std(axis=1), expected_std, rtol=0.001, atol=0.0)

    print("expecting (mean): ", [float(x) for x in arr_allvspec.mean(axis=1)])
    assert np.allclose(arr_allvspec.mean(axis=1), expected_mean, rtol=0.001, atol=0.0)


def test_average_vspecpol_files(tmp_path: Path) -> None:
    """--averagevspecpolfiles averages the vspecpol_total files of every -specpath model.

    make_averaged_vspecfiles read args.modelpath, and plotspectra stores every path under
    args.specpath, thus the command stopped with AttributeError before this test existed. The average runs in place
    of the polarisation plot, thus a ratio keeps the two model paths.
    """
    modeldirs = [tmp_path / "model_a", tmp_path / "model_b"]
    for modeldir in modeldirs:
        modeldir.mkdir()
        for filename in ("vspecpol_total-0.out", "vspecpol_total-1.out"):
            shutil.copy(at.get_path("testdata") / "vspecpolmodel" / filename, modeldir / filename)

    at.spectra.plot(
        argsraw=[], specpath=[str(modeldir) for modeldir in modeldirs], averagevspecpolfiles=True, stokesparam="Q/I"
    )

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

    # a continued run writes the rank files again, and a total that is older than a rank file then misses flux
    totalfile = tmp_path / "vspecpol_total-0.out"
    totaltime = totalfile.stat().st_mtime - 2000.0
    os.utime(totalfile, (totaltime, totaltime))
    for ranktime, expectsnewsum in ((totaltime - 1000.0, False), (totaltime + 1000.0, True)):
        for mpirank in range(2):
            os.utime(tmp_path / f"vspecpol_{mpirank:04d}.out", (ranktime, ranktime))
        at.spectra.scan_vspecpol_data.cache_clear()
        at.spectra.scan_vspecpol_data(tmp_path, 0)
        assert (totalfile.stat().st_mtime > totaltime) == expectsnewsum, ranktime


def test_vpkt_frompackets_spectrum_keeps_the_rows_inside_the_vpkt_ranges(
    tmp_path: Path, copy_vpktcontrib_model: Callable[[Path, str, str], Path]
) -> None:
    """The spectrum of a virtual observer bins only the rows with nu_rf inside a wavelength range of vpkt.txt.

    ARTIS also writes a row when only the absorption frequency is inside a range, and it keeps that row out of
    vspecpol. The code binned each row, thus the bins outside the ranges held a biased part of the flux.
    """
    fullmodel = copy_vpktcontrib_model(tmp_path / "full", "0 10 30", "0")
    rangesmodel = copy_vpktcontrib_model(tmp_path / "ranges", "1 130 140", "1 2 3500 6000 6400 7200")
    lambda_bin_edges = np.arange(3000.0, 8000.0, 100.0)

    def get_flambda(modelpath: Path) -> npt.NDArray[np.floating]:
        dfspectrum = at.spectra.scan_spectra_from_packets(
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
        at.spectra.scan_spectra_from_packets(
            rangesmodel,
            timelowdays=125.0,
            timehighdays=135.0,
            lambda_bin_edges=lambda_bin_edges,
            directionbins_are_vpkt_observers=True,
            directionbins=[0],
        )


def test_vpkt_frompackets_plot_refuses_a_time_range_outside_the_vpkt_window(
    tmp_path: Path, copy_vpktcontrib_model: Callable[[Path, str, str], Path]
) -> None:
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


def test_vspecpol_rank_files_come_from_the_newest_run_folder_with_each_rank(tmp_path: Path) -> None:
    """A restarted run reads the sums of the run before it, thus the newest complete run folder holds the sums.

    The search took the oldest run folder first, thus it summed the flux of the first run alone. A newest run folder
    with a file for only some ranks is not complete. ARTIS writes the files in the folder vspecpol of the run folder.
    """
    # get_nprocs reads the rank count from line 22 of input.txt
    (tmp_path / "input.txt").write_text("\n".join(["0"] * 21 + ["2"]) + "\n", encoding="utf-8")
    runfolders = [tmp_path / "9876543.slurm", tmp_path / "10000001.slurm" / "vspecpol", tmp_path / "10000002.slurm"]
    for runfolder, mpiranks in zip(runfolders, ((0, 1), (0, 1), (0,)), strict=True):
        runfolder.mkdir(parents=True)
        for mpirank in mpiranks:
            (runfolder / f"vspecpol_{mpirank:04d}.out").write_text("", encoding="utf-8")

    assert at.spectra.core.get_vspecpol_rank_files(tmp_path) == [
        runfolders[1] / "vspecpol_0000.out",
        runfolders[1] / "vspecpol_0001.out",
    ]


@pytest.mark.parametrize("hasgammaspecfile", [True, False])
def test_plotvspecpol_refuses_gamma_before_it_reads_a_spectrum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], hasgammaspecfile: bool
) -> None:
    """--gamma with --plotvspecpol stops before the read, because ARTIS makes the virtual packets from the r-packets.

    With gamma_spec.out, the command read vspecpol_total-0.out and drew the r-packet spectrum. With no gamma_spec.out,
    it read the packets and stopped at an assertion.
    """
    modelfolder = tmp_path / "vspecpolmodel"
    shutil.copytree(at.get_path("testdata") / "vspecpolmodel", modelfolder)
    if hasgammaspecfile:
        # the command must stop before a read of the file, thus an empty file is sufficient
        (modelfolder / "gamma_spec.out").touch()

    with (
        mock.patch.object(
            at.spectra.plotspectra, "read_dirbin_spectra", side_effect=AssertionError("read a spectrum")
        ) as mockread,
        pytest.raises(SystemExit),
    ):
        at.spectra.plot(argsraw=[str(modelfolder), "--gamma", "--plotvspecpol", "0", "-t", "11.5", "-o", str(tmp_path)])

    mockread.assert_not_called()
    stderr = capsys.readouterr().err
    assert "--gamma" in stderr
    assert "virtual packets from the r-packets" in stderr
