import argparse
import csv
import dataclasses as dc
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.viewertools.window import draw_values

if t.TYPE_CHECKING:
    from collections.abc import Callable

modelpath = at.get_path("testdata") / "testmodel"
modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")
outputpath.mkdir(exist_ok=True, parents=True)

REPOPATH = at.get_path("artistools_repository")


def funcname() -> str:
    """Get the name of the calling function."""
    thisframe = inspect.currentframe()
    try:
        if thisframe is None or thisframe.f_back is None:
            msg = "Could not get the name of the calling function."
            raise RuntimeError(msg)

        return thisframe.f_back.f_code.co_name
    finally:
        # a frame held in one of its own locals is a reference cycle, so drop it as the inspect docs advise
        del thisframe


def get_plot_xy(callargs: t.Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array(callargs[0][1], dtype=float), np.array(callargs[0][2], dtype=float)


def test_timestep_times() -> None:
    timestartarray = at.get_timestep_times(modelpath, loc="start")
    timedeltarray = at.get_timestep_times(modelpath, loc="delta")
    timemidarray = at.get_timestep_times(modelpath, loc="mid")
    assert len(timestartarray) == 100
    assert math.isclose(timemidarray[0], 250.421, abs_tol=1e-3)
    assert math.isclose(timemidarray[-1], 349.412, abs_tol=1e-3)

    assert all(
        tstart < tmid < (tstart + tdelta)
        for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray, strict=False)
    )


def test_get_inputparams() -> None:
    inputparams = at.get_inputparams(modelpath)
    dicthash = hashlib.sha256(str(sorted(inputparams.items())).encode("utf-8")).hexdigest()
    # nusyn_min and nusyn_max moved by 7.4e-10 in relative terms when the hardcoded MeV_in_Hz became
    # 1e6 / h_ev_s, which is the same conversion expressed with the Planck constant of constants.py
    assert dicthash == "477eb9a026a0d526499ab11b53f32ed256d48898479dde9d2109213b988c4456", dicthash


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@mock.patch.object(mplax.Axes, "step", side_effect=mplax.Axes.step, autospec=True)
@pytest.mark.benchmark
def test_radfield(mockstep: mock.MagicMock, mockplot: mock.MagicMock) -> None:
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=funcoutpath, showbinedges=True
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
    assert np.allclose(j_nu[:200], j_nu_fullspec[:200], rtol=1e-10)
    assert np.allclose(j_nu[200:], j_nu_bins[200:], rtol=1e-10)

    # before MULTIBIN_RADFIELD_FIRST_TIMESTEP, ARTIS takes the full-spectrum fit for every bin
    _, j_lambda_fullspec = get_fitted_field(radfielddata, modelgridindex=0, timestep=5, usebinfits=False)
    assert np.allclose(np.array(j_lambda_fullspec) * np.array(arr_lambda) / arr_nu, j_nu_fullspec, rtol=1e-10)

    _, bandaverage = get_binaverage_field(radfielddata, modelgridindex=0, timestep=5)
    dlambda = at.constants.c_ang_per_s * (1 / 5.0e14 - 1 / 6.0e14)
    assert np.isclose(bandaverage[1], 40.0 / dlambda, rtol=1e-10)


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


@pytest.mark.benchmark
def test_plotspherical(tmp_path: Path) -> None:
    at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=tmp_path)

    assert [path.name for path in tmp_path.glob("plotspherical_*.pdf")] == ["plotspherical_256.67-333.82d.pdf"]


def test_plotspherical_gaussian_filter(tmp_path: Path) -> None:
    """-gaussian_sigma must reach smooth_direction_map, which takes the width in phi bins and not in degrees."""
    from artistools.packets.plotspherical import smooth_direction_map

    with mock.patch.object(
        at.packets.plotspherical, "smooth_direction_map", side_effect=smooth_direction_map
    ) as mocksmooth:
        at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, gaussian_sigma=20, outputfile=tmp_path)

    # one map for each of the three default plot variables, on the default grid of 64 phi bins
    assert mocksmooth.call_count == 3
    for callargs in mocksmooth.call_args_list:
        _, sigma_bins, nphibins = callargs.args
        assert nphibins == 64
        assert np.isclose(sigma_bins, 20 / 360 * 64)


def test_plotspherical_one_pass_matches_one_pass_per_time_range() -> None:
    """The direction maps of several time ranges in one pass must equal the map of each range on its own."""
    from artistools.packets.plotspherical import bin_packets_by_direction

    nprocs_read, dfpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelpath=modelpath)
    tstarts = at.get_timestep_times(modelpath, loc="start")
    tends = at.get_timestep_times(modelpath, loc="end")
    # the last range repeats the first one, as every frame does when the observer never gets light from the full ejecta
    timeranges = [(tstarts[ts], tends[ts]) for ts in (60, 61, 62, 60)]
    plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc", "emvelocityoverc_sigma"]

    dfonepass, _ = bin_packets_by_direction(
        modelpath, dfpackets, nprocs_read, timeranges, 8, 4, plotvars, dfestimators=None
    )
    assert dfonepass.height == 4 * 8 * 4
    assert dfonepass["count"].sum() > 0

    for timebin, timerange in enumerate(timeranges):
        dfsingle, _ = bin_packets_by_direction(
            modelpath, dfpackets, nprocs_read, [timerange], 8, 4, plotvars, dfestimators=None
        )
        pltest.assert_frame_equal(
            dfonepass.filter(pl.col("timebin") == timebin).with_columns(timebin=pl.lit(0, dtype=pl.Int32)), dfsingle
        )


def test_plotspherical_smoothing_wraps_phi_and_crosses_each_pole() -> None:
    """The smoothing must blend no pole with the opposite pole.

    The filter wrapped the cos theta axis as it wraps phi. Thus the row at the south pole took the light of
    the row at the north pole.
    """
    from artistools.packets.plotspherical import smooth_direction_map

    ncosthetabins, nphibins = 8, 16
    data = np.zeros((ncosthetabins, nphibins))
    data[-1, 0] = 1.0  # the north polar cap, at phi bin 0

    smoothed = smooth_direction_map(data, 1.0, nphibins)

    assert np.allclose(smoothed[0], 0.0), "the south polar cap must stay dark"
    # phi wraps, thus the last phi bin is a neighbour of phi bin 0
    assert smoothed[-1, -1] > 0.0
    # the bin on the other side of the north pole is at phi + pi in the same row
    assert smoothed[-1, nphibins // 2] > smoothed[-1, nphibins // 4]


@mock.patch.object(mplax.Axes, "pcolormesh", side_effect=mplax.Axes.pcolormesh, autospec=True)
def test_plotspherical_puts_phi_zero_at_the_centre(mockpcolormesh: mock.MagicMock) -> None:
    """The direction phi = 0 (+X) must be at the longitude zero, which is the centre of the map.

    The longitude of the map started at -pi with phi bin 0, thus +X was at the edge and -X at the centre.
    """
    from artistools.packets.plotspherical import plot_spherical

    ncosthetabins, nphibins = 4, 8
    isphibinzero = [phibin == 0 for _ in range(ncosthetabins) for phibin in range(nphibins)]
    dirbins = pl.DataFrame({"count": [1] * len(isphibinzero), "luminosity": [float(x) for x in isphibinzero]})

    fig, _ = plot_spherical(dirbins, ["luminosity"], nphibins=nphibins, ncosthetabins=ncosthetabins)
    plt.close(fig)

    # the colour bar draws a mesh of its own after the map
    _, meshgrid_phi, _, data = mockpcolormesh.call_args_list[0].args
    (litcolumn,) = np.flatnonzero(np.asarray(data)[0])
    assert np.allclose(meshgrid_phi[0, litcolumn : litcolumn + 2], [0.0, 2 * np.pi / nphibins])


@pytest.mark.parametrize("nphibins", [1, 5, 8])
def test_plotspherical_map_columns_cover_each_longitude_with_the_correct_phi_bin(nphibins: int) -> None:
    """Each column of the map must show the phi bin that holds its longitude, from -pi to +pi.

    For an odd -nphibins, the map ran from -0.8 pi to 1.2 pi for five bins, thus a strip at -pi stayed blank.
    """
    from artistools.packets.plotspherical import get_map_columns

    longitude_edges, column_phibins = get_map_columns(nphibins)

    assert len(longitude_edges) == len(column_phibins) + 1
    assert np.isclose(longitude_edges[0], -np.pi)
    assert np.isclose(longitude_edges[-1], np.pi)
    assert np.all(np.diff(longitude_edges) > 0.0)
    assert np.any(np.isclose(longitude_edges, 0.0)), "phi = 0 must be an edge at the centre of the map"
    column_centre_phi = np.mod(0.5 * (longitude_edges[:-1] + longitude_edges[1:]), 2 * np.pi)
    assert np.array_equal(np.floor(column_centre_phi / (2 * np.pi) * nphibins).astype(int), column_phibins)


def test_plotspherical_smoothing_with_odd_phi_bins_crosses_the_pole_at_phi_plus_pi() -> None:
    """For an odd -nphibins, the row beyond a pole must take the two bins on each side of phi + pi."""
    from artistools.packets.plotspherical import get_rows_across_pole

    rows = np.arange(5, dtype=np.float64).reshape((1, 5))

    # the centre of bin 0 is at 0.2 pi, thus phi + pi is at 1.2 pi, which is the edge of bins 2 and 3
    assert np.allclose(get_rows_across_pole(rows, 5)[0], [2.5, 3.5, 2.0, 0.5, 1.5])


def test_plotspherical_gif(tmp_path: Path) -> None:
    """The gif holds one frame for each timestep in the time range, in the order of time.

    The frames and the gif go in a folder of this test alone. A pytest run clears the test output folder when it
    starts. Thus a second run deleted the frames of this test before the gif read them.
    """
    from PIL import Image

    # the timesteps 0, 1, and 2 end before 253 d. Three frames cover the code of the gif
    at.packets.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=tmp_path
    )

    framenames = sorted(path.name for path in tmp_path.glob("plotspherical_*.png"))
    assert framenames == [
        "plotspherical_250.00-250.84d.png",
        "plotspherical_250.84-251.69d.png",
        "plotspherical_251.69-252.54d.png",
    ]
    with Image.open(tmp_path / "sphericalplot.gif") as gif:
        assert getattr(gif, "n_frames", 1) == 3


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_logfiles(mockplot: mock.MagicMock) -> None:
    """Log file timings are parsed for every stage and rank, and plotted one page per timestep."""
    logfilepaths = at.logs.get_logfiles(Path(modelpath_classic_3d), None)
    # compressed log files must be read too, not skipped
    assert sorted(path.name for path in logfilepaths) == [
        "output_0-0.txt",
        "output_0-0.txt",
        "output_1-0.txt.zst",
        "output_1-0.txt.zst",
    ]

    timetaken = at.logs.core.read_time_taken(logfilepaths)
    assert set(timetaken) == {"update_grid", "update_packets", "communicate_estimators"}
    for stage, bytimestep in timetaken.items():
        assert len(bytimestep) == 30, f"expected 30 timesteps of {stage} timings"
        for byrank in bytimestep.values():
            assert set(byrank) == {0, 1}, f"expected both mpi ranks for {stage}"
    assert timetaken["update_grid"][0] == {0: 2, 1: 2}
    assert timetaken["update_packets"][0] == {0: 1, 1: 1}
    assert timetaken["communicate_estimators"][2] == {0: 1, 1: 1}

    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    at.logs.plotlogfiles.main(argsraw=[], modelpath=[modelpath_classic_3d], outputfile=funcoutpath / "logfiles.pdf")

    # one line per stage on each of the 30 per-timestep pages
    assert len(mockplot.call_args_list) == 3 * 30


def test_logfiles_read_the_old_and_the_current_log_format(tmp_path: Path) -> None:
    """The timings of the current ARTIS log lines and of the old log lines must both be read.

    The patterns held the Unix time and the whole seconds of the old lines. The current lines hold no Unix time,
    the grid line says "on all processes", and the seconds have decimals. Thus the command found no timing data.
    """
    (tmp_path / "output_0-0.txt").write_text(
        "2026-10-04T12:00:00Z timestep 3: time after update grid on all processes"
        " (rank 0 took 1.2s, waited 0.3s, total 1.5s)\n"
        "2026-10-04T12:00:05Z timestep 3: time after update packets for all processes"
        " (rank 0 took 3.5s, waited 0.1s, total 3.6s)\n"
        "2026-10-04T12:00:06Z timestep 3: time after estimators have been communicated (took 0.4 seconds)\n"
        "2023-11-14T15:01:59Z timestep 2: time after update grid for all processes 1699974119"
        " (rank 0 took 2s, waited 0s, total 2s)\n"
        "2023-11-14T15:02:09Z timestep 2: time after update packets for all processes 1699974129"
        " (rank 0 took 7s, waited 0s, total 7s)\n"
        "2023-11-14T15:02:10Z timestep 2: time after estimators have been communicated 1699974130 (took 1 seconds)\n",
        encoding="utf-8",
    )

    timetaken = at.logs.core.read_time_taken([tmp_path / "output_0-0.txt"])

    assert timetaken["update_grid"] == {3: {0: pytest.approx(1.2)}, 2: {0: pytest.approx(2.0)}}
    assert timetaken["update_packets"] == {3: {0: pytest.approx(3.5)}, 2: {0: pytest.approx(7.0)}}
    assert timetaken["communicate_estimators"] == {3: {0: pytest.approx(0.4)}, 2: {0: pytest.approx(1.0)}}


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


def test_plotspherical_element_arguments_give_messages(capsys: pytest.CaptureFixture[str]) -> None:
    """-elem with -atomic_number, or an -elem that is no element symbol, gives a message.

    The first stopped with a bare AssertionError, and the second plotted the packets of "Z=-1".
    """
    with pytest.raises(SystemExit):
        at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, elem="Fe", atomic_number=26)
    assert "Give one of --elem and --atomic-number" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, elem="Xx")
    assert "-elem Xx is not an element symbol" in capsys.readouterr().err


def test_plotviewingangles_writes_an_animation_of_each_direction_bin(tmp_path: Path) -> None:
    """The html file holds one frame of the animation for each of the 100 direction bins."""
    pytest.importorskip("plotly")
    outputfile = tmp_path / "viewingangles.html"
    at.plotviewingangles.main(argsraw=[str(modelpath_3d), "-o", str(outputfile)])

    framenames = set(re.findall(r'"name":"(\d\d)"', outputfile.read_text(encoding="utf-8")))
    assert framenames == {f"{dirbin:02d}" for dirbin in range(100)}


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


@pytest.mark.benchmark
def test_writecomparisondata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The files take the name of the model folder, and the spectra keep the luminosity of spec.out.

    The default model path "." gave an empty model name, e.g. spectra__artisnebular.txt.
    """
    monkeypatch.chdir(modelpath)
    # CodSpeed runs a benchmark more than one time in one folder, thus the later runs find the files of the first
    at.writecomparisondata.main(
        argsraw=[], modelpath=Path(), outputpath=tmp_path, selected_timesteps=list(range(99)), force=True
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "edep_testmodel_artisnebular.txt",
        "ionfrac_co_testmodel_artisnebular.txt",
        "ionfrac_fe_testmodel_artisnebular.txt",
        "phys_testmodel_artisnebular.txt",
        "spectra_testmodel_artisnebular.txt",
    ]

    from artistools.spectra.core import scan_spec

    # the spectra give L_lambda, and spec.out gives F_nu at 1 Mpc
    dfspecout = scan_spec(modelpath).collect()
    arr_specout_nu = dfspecout["nu"].to_numpy()
    arr_written = np.loadtxt(tmp_path / "spectra_testmodel_artisnebular.txt", comments="#")
    assert arr_written.shape == (len(arr_specout_nu), 1 + 99)
    area = 4.0 * math.pi * at.constants.megaparsec_to_cm**2
    for timestep in (10, 50):
        lum_written = np.trapezoid(arr_written[:, 1 + timestep], arr_written[:, 0])
        lum_specout = abs(np.trapezoid(dfspecout[:, 1 + timestep].to_numpy(), arr_specout_nu)) * area
        assert np.isclose(lum_written, lum_specout, rtol=1e-4)

    # the ion fractions of an element sum to one, as the format of the workshop requires
    arr_ionfrac_fe = np.loadtxt(tmp_path / "ionfrac_fe_testmodel_artisnebular.txt", comments="#")
    assert np.allclose(arr_ionfrac_fe[:, 1:].sum(axis=1), 1.0, rtol=1e-3)


def test_get_z_a_nucname() -> None:
    assert at.get_z_a_nucname("Pb208") == (82, 208)
    assert at.get_z_a_nucname("X_Pb208") == (82, 208)
    assert at.get_z_a_nucname("nniso_Pb208") == (82, 208)
    assert at.get_z_a_nucname("Fe56") == (26, 56)
    assert at.get_z_a_nucname("Ni56") == (28, 56)
    assert at.get_z_a_nucname("Co56") == (27, 56)
    assert at.get_z_a_nucname("H1") == (1, 1)


def test_get_atomic_number_and_elsymbol() -> None:
    assert at.get_atomic_number("Fe") == 26
    assert at.get_atomic_number("Ni") == 28
    assert at.get_atomic_number("Co") == 27
    assert at.get_atomic_number("H") == 1
    assert at.get_atomic_number("He") == 2
    assert at.get_atomic_number("X_Fe") == 26
    assert at.get_atomic_number("UnknownXYZ") == -1

    # the free neutron is "n1", and a title-case lookup gave nitrogen for it
    assert at.get_atomic_number("X_n1") == 0
    assert at.get_z_a_nucname("X_n1") == (0, 1)
    assert at.get_z_a_nucname("n1") == (0, 1)
    # a symbol is not case sensitive, thus every other "n" is nitrogen
    assert at.get_atomic_number("n") == 7
    assert at.get_atomic_number("N") == 7
    assert at.get_atomic_number("X_N14") == 7
    assert at.get_z_a_nucname("X_N14") == (7, 14)
    assert at.get_z_a_nucname("n14") == (7, 14)
    assert at.get_ion_tuple("nII") == (7, 2)

    # a symbol that the caller gave in the wrong case still resolves
    assert at.get_atomic_number("fe") == 26

    assert at.get_elsymbol(26) == "Fe"
    assert at.get_elsymbol(28) == "Ni"
    assert at.get_elsymbol(1) == "H"
    assert at.get_elsymbol(2) == "He"


def test_decode_roman_numeral() -> None:
    assert at.decode_roman_numeral("I") == 1
    assert at.decode_roman_numeral("II") == 2
    assert at.decode_roman_numeral("III") == 3
    assert at.decode_roman_numeral("IV") == 4
    assert at.decode_roman_numeral("V") == 5
    assert at.decode_roman_numeral("X") == 10
    assert at.decode_roman_numeral("XX") == 20
    assert at.decode_roman_numeral("i") == 1  # case-insensitive
    assert at.decode_roman_numeral("INVALID") == -1


def test_get_ionstring() -> None:
    assert at.get_ionstring(26, 2) == "Fe II"
    assert at.get_ionstring(26, 1) == "Fe I"
    assert at.get_ionstring(28, 3) == "Ni III"
    assert at.get_ionstring(26, 2, sep="") == "FeII"
    assert at.get_ionstring(26, None) == "Fe"
    assert at.get_ionstring(26, "ALL") == "Fe"
    assert at.get_ionstring(26, 2, style="charge") == "Fe+"
    assert at.get_ionstring(26, 3, style="charge") == "Fe2+"
    assert at.get_ionstring(26, 1, style="charge") == "Fe0"


def test_get_ion_tuple() -> None:
    assert at.get_ion_tuple("nnelement_I") == 53
    assert at.get_ion_tuple("nnion_I_II") == (53, 2)
    assert at.get_ion_tuple("Fe_II") == (26, 2)
    assert at.get_ion_tuple("Fe II") == (26, 2)
    assert at.get_ion_tuple("Fe I") == (26, 1)
    assert at.get_ion_tuple("Ni III") == (28, 3)
    assert at.get_ion_tuple("Co II") == (27, 2)
    assert at.get_ion_tuple("Ni") == 28
    assert at.get_ion_tuple("26") == 26


def test_get_ion_tuple_no_separator() -> None:
    """Two-letter symbols must not be split on their first letter, e.g. 'FeII' is Fe II and not F + 'eII'."""
    assert at.get_ion_tuple("FeII") == (26, 2)
    assert at.get_ion_tuple("CoII") == (27, 2)
    assert at.get_ion_tuple("NiIII") == (28, 3)
    assert at.get_ion_tuple("HeII") == (2, 2)
    # single-letter symbols still work, and are not shadowed by a longer symbol that starts the same way
    assert at.get_ion_tuple("FII") == (9, 2)
    assert at.get_ion_tuple("CIV") == (6, 4)
    assert at.get_ion_tuple("OI") == (8, 1)


def test_make_vpkt_input_default_contents() -> None:
    """The default vpkt.txt must keep the exact layout ARTIS parses, field by field."""
    expected = (
        "3\n"  # number of viewing directions
        "1 0 -1\n"  # costheta of each direction
        "0 0 0\n"  # phi of each direction
        "0 \n"  # no opacity exclusions
        "0 0.2 1.5\n"  # override_tminmax off, then the time window
        "0\n"  # no custom wavelength ranges
        "1 100\n"  # override thick cell tau, and the threshold
        "10\n"  # tau_max_vpkt
        "0\n"  # velocity grid map off
        "0.2 1.5\n"  # velocity grid map time range
        "1 3500 6000"  # one wavelength range for the velocity grid map
    )
    assert at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig()) == expected


def test_make_vpkt_input_optional_blocks() -> None:
    """The opacity exclusion and custom wavelength blocks must be prefixed by their own counts."""
    config = at.misc.modelinfo.VpktConfig(
        directions_costheta_phi=[(-1, 0), (0.5, 90)],
        opacityexclusions=[0, -1, 26],
        custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
        override_tminmax=True,
        vgrid_on=True,
        override_thickcell_tau=False,
        tau_max_vpkt=7.5,
    )
    contents = at.make_vpkt_input.format_vpkt_input(config).splitlines()

    assert contents[0] == "2"
    assert contents[1] == "-1 0.5"
    assert contents[2] == "0 90"
    assert contents[3] == "1 3 0 -1 26"
    assert contents[4] == "1 0.2 1.5"
    assert contents[5] == "1 2 3500 6000 10000 12000"
    assert contents[6] == "0 100"
    assert contents[7] == "7.5"
    assert contents[8] == "1"


def test_make_vpkt_input_roundtrip() -> None:
    """Parsing a written file must recover exactly the settings it was written from."""
    for config in (
        at.misc.modelinfo.VpktConfig(),
        at.misc.modelinfo.VpktConfig(
            directions_costheta_phi=[(-1, 0), (0.5, 90)],
            opacityexclusions=[0, -1, 26],
            custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
            override_tminmax=True,
            vgrid_on=True,
            override_thickcell_tau=False,
            tau_max_vpkt=7.5,
            vspec_tmin_in_days=1.25,
        ),
    ):
        assert at.misc.parse_vpkt_input(at.make_vpkt_input.format_vpkt_input(config)) == config


def test_make_vpkt_input_rejects_inconsistent_file() -> None:
    """A truncated or out-of-range file must be reported, not silently accepted."""
    contents = at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig())
    truncated = "\n".join(contents.splitlines()[:5])
    with pytest.raises(ValueError, match="ended while reading"):
        at.misc.parse_vpkt_input(truncated)

    # a file ARTIS would reject still loads, so it can be repaired, but reports the problem
    badcostheta = at.misc.parse_vpkt_input(contents.replace("1 0 -1", "1 0 -2", 1))
    assert "outside [-1, 1]" in str(at.make_vpkt_input.fatal_config_error(badcostheta))
    with pytest.raises(ValueError, match="outside"):
        at.make_vpkt_input.format_vpkt_input(badcostheta)


def test_make_vpkt_input_matches_artis_token_reader() -> None:
    """ARTIS reads vpkt.txt with fscanf, which ignores line breaks, so the parser must too."""
    config = at.misc.modelinfo.VpktConfig(
        directions_costheta_phi=[(0.5, 90)], opacityexclusions=[0, -1], custom_lambda_ranges=[(4000, 7000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    allonelongline = " ".join(contents.split())
    assert at.misc.parse_vpkt_input(allonelongline) == config

    onetokenperline = "\n".join(contents.split())
    assert at.misc.parse_vpkt_input(onetokenperline) == config


def test_make_vpkt_input_velocity_grid_ranges_roundtrip() -> None:
    """ARTIS reads as many velocity grid ranges as the count declares, so every one must be written."""
    config = at.misc.modelinfo.VpktConfig(
        vgrid_on=True, vgrid_lambda_ranges=[(3500, 6000), (6000, 9000), (9000, 10000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    assert contents.splitlines()[-1] == "3 3500 6000 6000 9000 9000 10000"
    assert at.misc.parse_vpkt_input(contents) == config


def test_make_vpkt_input_accepts_file_without_velocity_grid_block() -> None:
    """ARTIS only reads the velocity grid block when the map is on, so a file may legitimately omit it."""
    contents = at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig())
    withoutvgridblock = "\n".join(contents.splitlines()[:9])

    config = at.misc.parse_vpkt_input(withoutvgridblock)
    assert not config.vgrid_on
    assert config.vgrid_lambda_ranges == [(3500.0, 6000.0)]


def test_make_vpkt_input_rejects_nonzero_first_opacity_choice() -> None:
    """ARTIS asserts opacityexclusions[0] == 0, so artistools must not be able to write such a file."""
    config = at.misc.modelinfo.VpktConfig(opacityexclusions=[26, 0])
    with pytest.raises(ValueError, match="first opacity choice must be 0"):
        at.make_vpkt_input.format_vpkt_input(config)


def test_make_vpkt_input_warns_outside_compiled_limits() -> None:
    """Bounds that ARTIS asserts against compile-time constants must warn rather than fail."""
    outsidetime = at.misc.modelinfo.VpktConfig(override_tminmax=True, vspec_tmin_in_days=0.2, vspec_tmax_in_days=1.5)
    assert any("time window" in warning for warning in at.make_vpkt_input.check_config(outsidetime))

    outsidelambda = at.misc.modelinfo.VpktConfig(custom_lambda_ranges=[(1000, 2000)])
    assert any("wavelength range" in warning for warning in at.make_vpkt_input.check_config(outsidelambda))

    assert not at.make_vpkt_input.check_config(at.misc.modelinfo.VpktConfig()), "the defaults must not warn"


def test_make_vpkt_input_cli_writes_file(tmp_path: Path) -> None:
    """The subcommand must honour -directions, including a negative leading costheta, and -outputfile."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-directions=-1,0 1,0", "-o", str(outfile), "--non-interactive"])

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "2"
    assert lines[1] == "-1 1"
    assert lines[2] == "0 0"


def test_make_vpkt_input_cli_keeps_existing_settings(tmp_path: Path) -> None:
    """Rerunning on an existing file must preserve settings that were not given on the command line."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-tau-max", "7.5", "-o", str(outfile), "--non-interactive"])
    assert at.misc.parse_vpkt_input(outfile.read_text(encoding="utf-8")).tau_max_vpkt == 7.5

    at.make_vpkt_input.main(argsraw=["-vspec-tmax", "9.0", "-o", str(outfile), "--non-interactive"])
    config = at.misc.parse_vpkt_input(outfile.read_text(encoding="utf-8"))
    assert config.vspec_tmax_in_days == 9.0
    assert config.tau_max_vpkt == 7.5, "the tau-max from the first run must survive the second"


def test_make_vpkt_input_interactive_edit() -> None:
    """An empty reply keeps the current value, and an invalid reply is asked again."""
    replies = iter([
        "",  # keep the default viewing directions
        "26 -1",  # rejected: the first opacity choice must be 0, so this question repeats
        "0 26 -1",  # set opacity choices
        "maybe",  # rejected, so this question repeats
        "yes",  # override_tminmax
        "",  # keep vspec_tmin
        "3.5",  # vspec_tmax
    ])
    # pad so the reply script does not have to track how many settings there are
    replies = itertools.chain(replies, itertools.repeat(""))
    asked: list[str] = []

    def fakeprompt(text: str) -> str:
        asked.append(text)
        return next(replies)

    config = at.make_vpkt_input.edit_config_interactively(at.misc.modelinfo.VpktConfig(), promptfunc=fakeprompt)

    assert config.directions_costheta_phi == [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
    assert config.opacityexclusions == [0, 26, -1]
    assert config.override_tminmax
    assert config.vspec_tmin_in_days == 0.2
    assert config.vspec_tmax_in_days == 3.5
    # the rejected reply must have caused the same question to be asked twice
    assert sum(question.startswith("Restrict virtual packets") for question in asked) == 2
    assert "[1,0 0,0 -1,0]" in asked[0], "the prompt must show the current value"


def test_make_vpkt_input_interactive_clears_list() -> None:
    """A single '-' must clear a list-valued setting."""
    config = at.misc.modelinfo.VpktConfig(opacityexclusions=[26])
    replies = itertools.chain(iter(["", "-"]), itertools.repeat(""))
    config = at.make_vpkt_input.edit_config_interactively(config, promptfunc=lambda _: next(replies))

    assert config.opacityexclusions == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"opacityexclusions": [0, -5]}, "below -4"),
        ({"custom_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
        ({"tau_max_vpkt": 0.0}, "tau_max_vpkt 0 must be more than zero"),
        ({"cell_is_optically_thick_vpkt": -1.0}, "thick cell optical depth -1"),
        ({"override_tminmax": True, "vspec_tmin_in_days": 4.0, "vspec_tmax_in_days": 4.0}, "time window"),
        ({"vgrid_on": True, "tmin_vgrid_in_days": 2.0, "tmax_vgrid_in_days": 1.0}, "velocity grid map time range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": []}, "at least one wavelength range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
    ],
)
def test_make_vpkt_input_refuses_each_fatal_setting_of_artis(changes: dict[str, t.Any], reason: str) -> None:
    """Each setting that read_vpktparameterfile() in vpkt.cc stops on must give an error, not a file.

    The command wrote a vpkt.txt with such a setting, and ARTIS then stopped at the start of the run.
    """
    config = dc.replace(at.misc.modelinfo.VpktConfig(), **changes)
    assert reason in str(at.make_vpkt_input.fatal_config_error(config))
    with pytest.raises(ValueError, match=re.escape(reason)):
        at.make_vpkt_input.format_vpkt_input(config)

    # ARTIS reads the ranges of the velocity grid map only when the map is on
    assert (
        at.make_vpkt_input.fatal_config_error(dc.replace(at.misc.modelinfo.VpktConfig(), vgrid_lambda_ranges=[]))
        is None
    )
    assert (
        at.make_vpkt_input.fatal_config_error(
            dc.replace(at.misc.modelinfo.VpktConfig(), vgrid_lambda_ranges=[(0.0, 6000.0)])
        )
        is None
    )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_hesma_plots_take_the_times_and_the_tables_of_each_file(mockplot: mock.MagicMock, tmp_path: Path) -> None:
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
def test_hesma_spectrum_takes_the_column_names_of_the_file(mockplot: mock.MagicMock, tmp_path: Path) -> None:
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


def test_opacity_condition_labels_match_artis() -> None:
    """The codes must match trace_vpkt_direction() in vpkt.cc: -2 bound-free, -3 free-free, -4 electron scattering."""
    assert not at.misc.get_opacity_condition_label(0)
    assert at.misc.get_opacity_condition_label(-1) == "no-bb"
    assert at.misc.get_opacity_condition_label(-2) == "no-bf"
    assert at.misc.get_opacity_condition_label(-3) == "no-ff"
    assert at.misc.get_opacity_condition_label(-4) == "no-es"
    assert at.misc.get_opacity_condition_label(26) == "no-Fe"


def test_make_vpkt_input_rejects_bad_arguments() -> None:
    for baddirection in ("1", "1,0,0", "north,0", "1.5,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_directions(baddirection)

    for badrange in ("3500", "3500,6000,7000", "blue,red", "6000,3500"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_lambda_range(badrange)

    for badbool in ("maybe", "", "2"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_bool(badbool)


def test_trim_or_pad() -> None:
    result = at.misc.trim_or_pad(3, [1, 2, 3, 4], [10, 20])
    assert list(result[0]) == [1, 2, 3]
    assert list(result[1]) == [10, 20, None]

    result2 = at.misc.trim_or_pad(2, "single_string")
    assert list(result2[0]) == ["single_string", None]


def test_get_npts_model(tmp_path: Path) -> None:
    # The 3D test model has 10^3 = 1000 cells
    assert at.misc.get_npts_model(modelpath_3d) == 1000

    # Single-number format used by 1D models
    (tmp_path / "model.txt").write_text("20\n")
    assert at.misc.get_npts_model(tmp_path) == 20

    # Two-number format (Nx Ny): total cells = Nx * Ny
    two_num_dir = tmp_path / "twonum"
    two_num_dir.mkdir()
    (two_num_dir / "model.txt").write_text("10 10\n")
    assert at.misc.get_npts_model(two_num_dir) == 100


def test_get_nprocs(tmp_path: Path) -> None:
    """The 22nd line that holds a value gives nprocs. ARTIS skips comment and blank lines, and keeps them."""
    lines = ["# a comment of the user\n", "\n", *(["placeholder\n"] * 21), "4 #nprocs\n"]
    (tmp_path / "input.txt").write_text("".join(lines))
    assert at.get_nprocs(tmp_path) == 4


def test_get_cellsofmpirank(tmp_path: Path) -> None:
    def make_model(path: Path, npts: int, nprocs: int) -> None:
        lines = ["placeholder\n"] * 21 + [f"{nprocs} #nprocs\n"]
        (path / "input.txt").write_text("".join(lines))
        (path / "model.txt").write_text(f"{npts}\n")

    for npts, nprocs in ((20, 4), (21, 4), (7, 3)):
        subdir = tmp_path / f"npts{npts}_nprocs{nprocs}"
        subdir.mkdir()
        make_model(subdir, npts=npts, nprocs=nprocs)

        all_cells: list[int] = []
        cells_per_rank = []
        for rank in range(nprocs):
            cells = list(at.misc.get_cellsofmpirank(rank, subdir))
            cells_per_rank.append(cells)
            all_cells.extend(cells)

        # Every cell index appears exactly once and all cells are covered
        assert sorted(all_cells) == list(range(npts))

        # Load balancing: ranks differ by at most 1 cell
        sizes = [len(c) for c in cells_per_rank]
        assert max(sizes) - min(sizes) <= 1

        # Cells within each rank are contiguous
        for cells in cells_per_rank:
            assert cells == list(range(cells[0], cells[0] + len(cells)))

    # Verify specific assignments for evenly divisible case (npts=20, nprocs=4)
    even_dir = tmp_path / "even"
    even_dir.mkdir()
    make_model(even_dir, npts=20, nprocs=4)
    assert list(at.misc.get_cellsofmpirank(0, even_dir)) == list(range(5))
    assert list(at.misc.get_cellsofmpirank(3, even_dir)) == list(range(15, 20))

    # Verify specific assignments for uneven case (npts=21, nprocs=4):
    # rank 0 gets one extra cell (leftover), ranks 1-3 get the base count
    uneven_dir = tmp_path / "uneven"
    uneven_dir.mkdir()
    make_model(uneven_dir, npts=21, nprocs=4)
    assert list(at.misc.get_cellsofmpirank(0, uneven_dir)) == list(range(6))
    assert list(at.misc.get_cellsofmpirank(1, uneven_dir)) == list(range(6, 11))


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


def test_ejectaopacity(capsys: pytest.CaptureFixture[str]) -> None:
    """The global Planck mean opacity of the binned expansion opacities agrees with the reference value.

    The opacities need the statistical weights of the levels. A sum without them gives 13.90 cm^2/g.
    """
    at.lines.ejectaopacity.main(
        argsraw=[], modelpath=modelpath, timestep=40, lambdamin=3000.0, lambdamax=4000.0, deltalambda=10.0
    )

    (planckmeanline,) = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Global Planck mean opacity:")
    ]
    assert np.isclose(float(planckmeanline.split(":")[1].split()[0]), 13.21, rtol=1e-3)


def test_line_contributions_follow_the_level_populations_of_artis() -> None:
    """Each method of the level populations gives the same tau_sobolev and emission as a calculation by hand.

    The test model has NLTE levels and a superlevel for Fe II and Fe III, and LTE populations for Ni II.
    """
    timestep = 10
    dfcells = at.lines.linecontributions.get_line_contributions(
        modelpath, timestep, 3000.0, 30000.0, [(26, 2), (26, 3), (28, 2)], percell=True
    )
    time_s = at.get_timestep_times(modelpath, loc="mid")[timestep] * at.constants.day_to_s
    dfnltepops = at.nltepops.read_nltepops(modelpath, timestep=timestep).filter(pl.col("timestep") == timestep)
    dfestimators = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True).collect()
    kT_ev = at.constants.K_B_ev_per_K * dfestimators["Te"].item()
    adata = at.atomic.get_levels(modelpath)

    def get_population(
        level: int,
        g: npt.NDArray[np.float64],
        energy_ev: npt.NDArray[np.float64],
        dfionpops: pl.DataFrame,
        nnion: float,
    ) -> float:
        if dfionpops.is_empty():
            boltzmann = g * np.exp(-energy_ev / kT_ev)
            return float(nnion * boltzmann[level] / boltzmann.sum())
        lastnltelevel = dfionpops["level"].max()
        assert isinstance(lastnltelevel, int)
        if level <= lastnltelevel:
            return float(dfionpops.filter(pl.col("level") == level)["n_NLTE"].item())
        superlevel = slice(lastnltelevel + 1, None)
        boltzmann = g[superlevel] * np.exp(-(energy_ev[superlevel] - energy_ev[lastnltelevel + 1]) / kT_ev)
        superlevelpop = dfionpops.filter(pl.col("level") == -1)["n_NLTE"].item()
        return float(superlevelpop * boltzmann[level - lastnltelevel - 1] / boltzmann.sum())

    for ion, Z, ion_stage, source in (
        ("Fe III", 26, 3, "NLTE"),
        ("Fe II", 26, 2, "superlevel"),
        ("Ni II", 28, 2, "LTE"),
    ):
        row = (
            dfcells
            .filter((pl.col("ion") == ion) & (pl.col("n_upper_source") == source))
            .sort("emission_erg_per_s", descending=True)
            .row(0, named=True)
        )
        dflevels = adata.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["levels"].item()
        levelvalues = (
            dflevels["g"].to_numpy().astype(np.float64),
            dflevels["energy_ev"].to_numpy().astype(np.float64),
            dfnltepops.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)),
            float(dfestimators[f"nnion_{ion.replace(' ', '_')}"].item()),
        )
        g = levelvalues[0]

        lambda_cm = row["lambda_angstroms"] / at.constants.angstroms_per_cm
        n_upper = get_population(row["upper"], *levelvalues)
        tau = (
            row["A_per_s"]
            * lambda_cm**3
            * time_s
            / (8 * math.pi)
            * (g[row["upper"]] / g[row["lower"]] * get_population(row["lower"], *levelvalues) - n_upper)
        )
        emission = (
            n_upper
            * row["A_per_s"]
            * at.constants.h_erg_s
            * at.constants.C_cm_per_s
            / lambda_cm
            * -math.expm1(-tau)
            / tau
            * dfestimators["volume"].item()
        )
        assert np.isclose(row["tau_sobolev"], tau, rtol=1e-6), ion
        assert np.isclose(row["emission_erg_per_s"], emission, rtol=1e-6), ion


def test_line_populations_take_the_boltzmann_factor_of_the_ground_level_with_no_superlevel() -> None:
    """For an ion with NLTE rows and no superlevel, a level above the NLTE levels has n_0 g / g_0 exp(-E / k T_exc)."""
    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 10000.0]})
    levels = pl.DataFrame({"level": [0, 1, 2], "g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    dfnltepops = pl.DataFrame({"modelgridindex": [0, 1, 0, 1], "level": [0, 0, 1, 1], "n_NLTE": [10.0, 20.0, 3.0, 4.0]})
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [1, 2], dfnltepops, "nnion_X_II")
    dflevel2 = dfpops.filter(pl.col("level") == 2).sort("modelgridindex")
    assert dflevel2["population_source"].unique().to_list() == ["Boltzmann from the ground level"]
    expected = np.array([10.0, 20.0]) * 3.0 * np.exp(-2.0 / at.constants.K_B_ev_per_K / np.array([5000.0, 10000.0]))
    assert np.allclose(dflevel2["population"].to_numpy(), expected, rtol=1e-12, atol=0.0)
    assert dfpops.filter(pl.col("level") == 1).sort("modelgridindex")["population"].to_list() == [3.0, 4.0]


def test_line_summary_gives_the_velocity_of_the_largest_tau_with_a_cell_that_has_no_superlevel_row() -> None:
    """A cell with no superlevel row gave a NaN population, and the summary took the velocity of that NaN tau."""
    from artistools.lines.linecontributions import get_line_summary

    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 5000.0]})
    levels = pl.DataFrame({"level": [0, 1, 2], "g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    dfnltepops = pl.DataFrame({
        "modelgridindex": [0, 1, 0, 1, 0],
        "level": [0, 0, 1, 1, -1],
        "n_NLTE": [10.0, 20.0, 3.0, 4.0, 1.0],
    })
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [2], dfnltepops, "nnion_X_II")
    assert dfpops.filter(pl.col("modelgridindex") == 1)["population"].to_list() == [None]

    dfsummary = get_line_summary(
        pl.DataFrame({
            "modelgridindex": [0, 1],
            "velocity_kmps": [1000.0, 2000.0],
            "Te": [5000.0, 5000.0],
            "ion": ["X II", "X II"],
            "lambda_angstroms": [5000.0, 5000.0],
            "lower": [1, 1],
            "upper": [2, 2],
            "A_per_s": [0.1, 0.1],
            "n_lower_source": ["NLTE", "NLTE"],
            "n_upper_source": ["superlevel", "superlevel"],
            "tau_sobolev": [5.0, None],
            "emission_erg_per_s": [1.0, None],
        })
    )
    assert dfsummary["tau_sobolev_max"].item() == 5.0
    assert dfsummary["velocity_kmps_at_tau_sobolev_max"].item() == 1000.0
    assert dfsummary["cellcount_tau_sobolev_above_1"].item() == 1


def test_line_populations_of_a_cell_with_no_nlte_row_take_the_lte_populations() -> None:
    """ARTIS writes no NLTE row of an element that fell back to LTE in a cell, and that cell got no populations."""
    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 8000.0], "nnion_X_II": [100.0, 200.0]})
    levels = pl.DataFrame({"level": [0, 1], "g": [2.0, 4.0], "energy_ev": [0.0, 1.0]})
    dfnltepops = pl.DataFrame({"modelgridindex": [0, 0], "level": [0, 1], "n_NLTE": [90.0, 10.0]})
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [0, 1], dfnltepops, "nnion_X_II")
    dfcell1 = dfpops.filter(pl.col("modelgridindex") == 1).sort("level")
    assert dfcell1["population_source"].to_list() == ["LTE", "LTE"]
    boltzmann = np.array([2.0, 4.0 * np.exp(-1.0 / at.constants.K_B_ev_per_K / 8000.0)])
    assert np.allclose(dfcell1["population"].to_numpy(), 200.0 * boltzmann / boltzmann.sum(), rtol=1e-12, atol=0.0)
    assert dfpops.filter(pl.col("modelgridindex") == 0).sort("level")["population"].to_list() == [90.0, 10.0]


def test_line_summary_joins_the_cells_of_a_line_with_different_population_methods() -> None:
    """A line with NLTE populations in one cell and LTE populations in another gave two rows of the summary."""
    from artistools.lines.linecontributions import get_line_summary
    from artistools.lines.linecontributions import get_line_summary_of_batches

    def get_cell_table(modelgridindex: int, source: str, emission: float) -> pl.DataFrame:
        return pl.DataFrame({
            "modelgridindex": [modelgridindex],
            "velocity_kmps": [1000.0 * (modelgridindex + 1)],
            "Te": [5000.0],
            "ion": ["Fe II"],
            "lambda_angstroms": [7155.0],
            "lower": [0],
            "upper": [10],
            "A_per_s": [0.1],
            "n_lower_source": [source],
            "n_upper_source": [source],
            "tau_sobolev": [0.5],
            "emission_erg_per_s": [emission],
        })

    dfsummary = get_line_summary_of_batches([
        get_line_summary(get_cell_table(0, "NLTE", 1.0)),
        get_line_summary(get_cell_table(1, "LTE", 3.0)),
    ])
    assert dfsummary.height == 1
    assert dfsummary["n_lower_source"].item() == "LTE/NLTE"
    assert dfsummary["emission_erg_per_s"].item() == 4.0
    assert np.isclose(dfsummary["velocity_kmps_emission_mean"].item(), 1750.0, rtol=1e-12)


def test_plotlinecontributions_bins_the_cells_of_each_line_by_velocity(tmp_path: Path) -> None:
    """The emission and the mean tau_sobolev of each bin agree with a sum by hand over the selected cells of the line.

    -costhetamin selects the cells by the direction of their centre, thus the ranking and the emission of each line
    also take only these cells.
    """
    commonargs: dict[str, t.Any] = {"modelpath": modelpath_classic_3d, "lambdamin": 6000.0, "lambdamax": 6400.0}
    nbins = 6
    costhetamin = 0.3
    at.lines.plotlinecontributions.main(
        argsraw=[],
        timestep=25,
        nvelocitybins=nbins,
        maxlines=3,
        rankby="tau",
        costhetamin=costhetamin,
        outputfile=tmp_path / "lines.pdf",
        write_data=True,
        **commonargs,
    )
    dfdrawn = pl.read_csv(tmp_path / "lines.txt", separator=" ", null_values="nan", comment_prefix="#")
    # the rows of the black curve of all the lines come first, and they have no line
    dfalllines = dfdrawn.filter(pl.col("ion").is_null())
    dfdrawn = dfdrawn.filter(pl.col("ion").is_not_null())

    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    dfdirections = (
        at
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .select("modelgridindex", costheta=pl.col("vel_z_mid") / pl.col("vel_r_mid"))
        .collect()
    )
    dfcells = (
        at.lines.linecontributions
        .get_line_contributions(timestep=25, percell=True, **commonargs)
        .join(dfdirections, on="modelgridindex", how="left")
        .filter(pl.col("costheta") >= costhetamin)
    )
    linekeys = list(at.lines.linecontributions.LINEKEYS)
    dfranked = (
        dfcells
        .group_by(linekeys)
        .agg(pl.col("tau_sobolev").max(), pl.col("emission_erg_per_s").sum())
        .sort("tau_sobolev", descending=True)
        .head(3)
    )
    assert dfdrawn.select(linekeys).unique().sort(linekeys).equals(dfranked.select(linekeys).sort(linekeys))

    # the default bins end at the corner of the 3D grid
    binwidth_kmps = modelmeta["vmax_cmps"] / at.constants.km_to_cm * math.sqrt(3) / nbins
    # the black curve sums the emission of all the lines, and the tables did not give it
    dfallbins = (
        dfcells
        .group_by(velocitybin=(pl.col("velocity_kmps") / binwidth_kmps).floor().cast(pl.Int64))
        .agg(pl.col("emission_erg_per_s").sum())
        .sort("velocitybin")
    )
    assert np.allclose(
        dfalllines.sort("x_lower_kmps")["emission_erg_per_s"].to_numpy(),
        dfallbins.filter(pl.col("velocitybin") < nbins)["emission_erg_per_s"].to_numpy(),
        rtol=1e-6,
        atol=0.0,
    )
    for line in dfranked.iter_rows(named=True):
        dflinecells = dfcells.filter(
            (pl.col("ion") == line["ion"]) & (pl.col("lower") == line["lower"]) & (pl.col("upper") == line["upper"])
        ).with_columns(velocitybin=(pl.col("velocity_kmps") / binwidth_kmps).floor().cast(pl.Int64))
        dflinebins = dfdrawn.filter(
            (pl.col("ion") == line["ion"]) & (pl.col("lower") == line["lower"]) & (pl.col("upper") == line["upper"])
        )
        assert dflinebins.height > 1
        for row in dflinebins.iter_rows(named=True):
            dfbincells = dflinecells.filter(pl.col("velocitybin") == round(row["x_lower_kmps"] / binwidth_kmps))
            emission = float(dfbincells["emission_erg_per_s"].sum())
            tau = pl.col("tau_sobolev")
            tau_mean = dfbincells.select((tau * pl.col("volume")).sum() / pl.col("volume").sum()).item()
            assert row["cellcount"] == dfbincells.height
            assert np.isclose(row["emission_erg_per_s"], emission, rtol=1e-6, atol=0.0)
            assert np.isclose(row["tau_sobolev_volume_mean"], tau_mean, rtol=1e-6, atol=0.0)
            assert np.isclose(
                row["emission_fraction_per_kmps"],
                emission / binwidth_kmps / line["emission_erg_per_s"],
                rtol=1e-6,
                atol=0.0,
            )
            # the median is the smallest tau_sobolev whose cumulative volume reaches half of the volume of the bin
            dfsorted = dfbincells.sort("tau_sobolev")
            # equal volumes put half of the volume exactly at the end of a cell, and that tie takes the lower value
            cumulativevolume = dfsorted["volume"].cum_sum().to_numpy()
            halfvolume = 0.5 * cumulativevolume[-1] * (1.0 - 1e-12)
            tau_median = dfsorted["tau_sobolev"][int(np.searchsorted(cumulativevolume, halfvolume))]
            assert np.isclose(row["tau_sobolev_volume_median"], tau_median, rtol=1e-6, atol=0.0)
            volumefraction = float(dfbincells.filter(tau > 1.0)["volume"].sum()) / float(dfbincells["volume"].sum())
            assert np.isclose(row["volume_fraction_tau_sobolev_above_1"], volumefraction, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    ("photospherevelocity", "errortext"),
    [("350000km/s", "above the speed of light"), ("0.05", "has no unit"), ("10km/s", "a third of the width of a cell")],
)
def test_plotlinecontributions_stops_for_a_photosphere_velocity_with_no_unit_or_no_resolution(
    photospherevelocity: str, errortext: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop before the estimators for a velocity above c, a number with no unit, or a disk smaller than a cell.

    A bare number took the unit of -velocityunit, whose default depends on the model, thus 0.05 meant for 0.05 c gave
    0.05 km/s.
    """
    with pytest.raises(SystemExit):
        at.lines.plotlinecontributions.main(
            argsraw=[
                *("-modelpath", str(modelpath_classic_3d), "-timestep", "25"),
                *("-lambdamin", "6000", "-lambdamax", "6400", "-velocityunit", "kmps"),
                *("-troughdirbin", "90", "-photospherevelocity", photospherevelocity),
                *("-o", str(tmp_path / "lines.pdf")),
            ]
        )
    captured = capsys.readouterr()
    assert errortext in captured.out + captured.err


def test_plotlinecontributions_reports_an_arrival_time_outside_the_timesteps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An arrival time before the first timestep or after the last one is outside each timestep.

    The log clipped such a time to the first or the last timestep, but the packet commands keep no packet there.
    """
    from artistools.lines.linecontributions import LineVelocityProfiles

    tstarts = at.get_timestep_times(modelpath_classic_3d, loc="start")
    tends = at.get_timestep_times(modelpath_classic_3d, loc="end")
    dfrankedcells = pl.DataFrame({"velocity_kmps": [0.5 * at.constants.C_cm_per_s / at.constants.km_to_cm]})
    profiles = LineVelocityProfiles(
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        dfrankedcells.with_columns(emission_erg_per_s=pl.lit(1.0)),
    )
    at.lines.plotlinecontributions.print_arrival_times(modelpath_classic_3d, profiles, tstarts[0] * 1.01, "kmps")
    assert "the first time in no timestep, because it is before the start" in capsys.readouterr().out
    at.lines.plotlinecontributions.print_arrival_times(modelpath_classic_3d, profiles, tends[-1] * 0.99, "kmps")
    assert "the last time in no timestep, because it is after the end" in capsys.readouterr().out


def test_sobolev_trough_of_a_shell_agrees_with_the_analytic_depth() -> None:
    """A shell of constant tau_sobolev from 0.9 v_ph to 0.9 vmax gives the depth that the geometry gives.

    For a line-of-sight velocity u above v_ph, each ray in front of the disk crosses the shell, thus the depth is
    1 - exp(-tau). For u below v_ph, only the ring of the disk with an impact parameter above sqrt(v_ph^2 - u^2) is in
    front, thus the depth is (1 - exp(-tau)) u^2 / v_ph^2, as a mean over the bin. A shell gives the same depth for
    each direction. On a coarse grid with bins narrower than a cell, a cell covers each bin that it crosses. The
    earlier code put the whole volume of a cell in the bin of its centre, and the depths then went from 0.24 to 1.11.

    Bins that end below the photosphere, e.g. with -velocitymax, gave no plateau. One bin gave a blend above zero at
    only one wavelength, because the blend went to zero at the centre of the outer bins. The wavelength grid of the
    blend went to the fastest part of a cell, although the bins above the shell had no depth.
    """
    from artistools.lines.linecontributions import CellLineInputs
    from artistools.lines.linecontributions import get_sobolev_trough
    from artistools.lines.linecontributions import LineVelocityProfiles

    vmax_kmps, vph_kmps, tau, time_days = 1.0e4, 4000.0, 2.0, 1.0
    line = {"ion": "X I", "lambda_angstroms": 5000.0, "lower": 0, "upper": 1, "A_per_s": 1.0}
    absorbed = -math.expm1(-tau)

    def get_shell_trough(
        ncells: int, edges: list[float], dirbin: int, *, relativisticdoppler: bool
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        cellwidth_kmps = 2 * vmax_kmps / ncells
        centres = -vmax_kmps + cellwidth_kmps * (np.arange(ncells) + 0.5)
        vx, vy, vz = (axis.ravel() for axis in np.meshgrid(centres, centres, centres, indexing="ij"))
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        cellvolume = (cellwidth_kmps * at.constants.km_to_cm * time_days * at.constants.day_to_s) ** 3
        modelgridindex = np.arange(ncells**3)
        dfrankedcells = pl.DataFrame({
            **{key: [value] * ncells**3 for key, value in line.items()},
            "modelgridindex": modelgridindex,
            "volume": np.full(ncells**3, cellvolume),
            # the shell starts inside the photosphere, thus each part of a cell outside the photosphere has tau
            "tau_sobolev": np.where((speed > 0.9 * vph_kmps) & (speed < 0.9 * vmax_kmps), tau, 0.0),
        })
        dfcells = pl.DataFrame({
            "modelgridindex": modelgridindex,
            **{
                f"vel_{axis}_mid": values * at.constants.km_to_cm
                for axis, values in zip("xyz", (vx, vy, vz), strict=True)
            },
        })
        profiles = LineVelocityProfiles(
            pl.DataFrame(), pl.DataFrame([line]), pl.DataFrame(), pl.DataFrame(), dfrankedcells
        )
        inputs = CellLineInputs(dfcells, pl.DataFrame(), {}, pl.DataFrame(), time_days)
        return get_sobolev_trough(
            profiles,
            inputs,
            dirbin,
            vph_kmps,
            edges,
            (cellwidth_kmps, cellwidth_kmps, cellwidth_kmps),
            relativisticdoppler=relativisticdoppler,
            averageoverphi=True,
        )

    edges = [1000.0 * index for index in range(9)]
    expected = [
        absorbed * (upper**3 - lower**3) / (3 * (upper - lower) * vph_kmps**2) if upper <= vph_kmps else absorbed
        for lower, upper in itertools.pairwise(edges)
    ]
    for dirbin, relativisticdoppler in ((90, True), (50, False)):
        dfdepths, dfblend = get_shell_trough(64, edges, dirbin, relativisticdoppler=relativisticdoppler)
        assert np.allclose(dfdepths["trough_depth"].to_numpy()[: len(expected)], expected, rtol=0.0, atol=0.01), dirbin
        deepest = dfblend.top_k(1, by="trough_depth").row(0, named=True)
        assert np.isclose(deepest["trough_depth"], absorbed, rtol=0.01, atol=0.0)
        # the plateau from 4000 to 8000 km/s gives the wavelengths of the Doppler shift of 5000 Å
        assert 4866.0 < deepest["lambda_angstroms"] < 4934.0

    # a cell of the 10^3 grid is 2000 km/s wide, and each bin is 400 km/s wide. Near the z axis the shell of these
    # cells ends at the cell face at z = 8000 km/s, thus the test takes the bins from 4400 to 7200 km/s
    dfdepths, _ = get_shell_trough(10, [400.0 * index for index in range(21)], 90, relativisticdoppler=False)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4400.0, 6800.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), absorbed, rtol=0.0, atol=0.02)

    # the bins continue past the last edge with the same width, thus the plateau above the photosphere stays
    dfshort, dfshortblend = get_shell_trough(64, edges[:4], 90, relativisticdoppler=False)
    assert np.allclose(dfshort["trough_depth"].to_numpy()[: len(expected)], expected, rtol=0.0, atol=0.01)
    assert np.isclose(dfshortblend["trough_depth"].to_numpy().max(), absorbed, rtol=0.01, atol=0.0)
    # the shell ends at 9000 km/s, and the blend ends one bin after the last depth. The grid went to the fastest part
    # of a cell at 17 000 km/s, and the 400 points then covered a range twice as wide
    c_kmps = at.constants.C_cm_per_s / at.constants.km_to_cm
    shortestlambda_angstroms = float(dfshortblend["lambda_angstroms"].to_numpy().min())
    assert 5000.0 * (1 - 11000.0 / c_kmps) < shortestlambda_angstroms < 5000.0 * (1 - 9000.0 / c_kmps)

    # the blend samples the rays at each wavelength, thus one wide bin still gives the plateau of the shell
    dfonebin, dfonebinblend = get_shell_trough(10, [0.0, 2.0 * vmax_kmps], 90, relativisticdoppler=False)
    assert dfonebin.height == 1
    assert np.isclose(dfonebinblend["trough_depth"].to_numpy().max(), absorbed, rtol=0.0, atol=0.07)


def get_trough_of_box_grid(
    gridshape: tuple[int, int, int],
    tauofcell: "Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]",
    nlines: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the trough of nlines lines at 5000 Å with the same tau_sobolev, on a grid from -10000 to 10000 km/s.

    tauofcell gives the tau_sobolev of each cell from the velocities of the cell centres [km/s].
    """
    from artistools.lines.linecontributions import CellLineInputs
    from artistools.lines.linecontributions import get_sobolev_trough
    from artistools.lines.linecontributions import LineVelocityProfiles

    vmax_kmps = 1.0e4
    widths = (2 * vmax_kmps / gridshape[0], 2 * vmax_kmps / gridshape[1], 2 * vmax_kmps / gridshape[2])
    centres = [-vmax_kmps + width * (np.arange(ncells) + 0.5) for ncells, width in zip(gridshape, widths, strict=True)]
    vx, vy, vz = (axis.ravel() for axis in np.meshgrid(*centres, indexing="ij"))
    modelgridindex = np.arange(len(vx))
    lines = [
        {"ion": "X I", "lambda_angstroms": 5000.0, "lower": 0, "upper": upper, "A_per_s": 1.0}
        for upper in range(1, nlines + 1)
    ]
    dfrankedcells = pl.concat([
        pl.DataFrame({
            **{key: [value] * len(vx) for key, value in line.items()},
            "modelgridindex": modelgridindex,
            "volume": np.ones(len(vx)),
            "tau_sobolev": tauofcell(vx, vy, vz),
        })
        for line in lines
    ])
    dfcells = pl.DataFrame({
        "modelgridindex": modelgridindex,
        **{f"vel_{axis}_mid": values * at.constants.km_to_cm for axis, values in zip("xyz", (vx, vy, vz), strict=True)},
    })
    profiles = LineVelocityProfiles(pl.DataFrame(), pl.DataFrame(lines), pl.DataFrame(), pl.DataFrame(), dfrankedcells)
    inputs = CellLineInputs(dfcells, pl.DataFrame(), {}, pl.DataFrame(), 1.0)
    edges = [1000.0 * index for index in range(9)]
    return get_sobolev_trough(
        profiles, inputs, 90, 4000.0, edges, widths, relativisticdoppler=False, averageoverphi=True
    )


def test_sobolev_trough_blends_lines_on_each_ray_and_takes_each_cell_width() -> None:
    """Two lines that absorb the same half of the disk remove half of its flux, and a cell has a width on each axis.

    The blend was 1 - prod(1 - depth) of the depths over the disk, which gave 0.75 for two saturated lines on the same
    half of the disk. The rays also took the x width of a cell for each axis.
    """
    tau = 50.0

    def get_halfshell_tau(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        return np.where((speed > 3600.0) & (speed < 9000.0) & (vx > 0.0), tau, 0.0)

    dfdepths, dfblend = get_trough_of_box_grid((48, 48, 48), get_halfshell_tau, nlines=2)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4000.0, 6000.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), 0.5, rtol=0.0, atol=0.03)
    assert np.isclose(float(dfblend["trough_depth"].to_numpy().max()), 0.5, rtol=0.0, atol=0.03)

    def get_shell_tau(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        return np.where((speed > 3600.0) & (speed < 9000.0), 2.0, 0.0)

    # the cells are twice as wide along z as along x and y
    dfdepths, _ = get_trough_of_box_grid((64, 64, 32), get_shell_tau, nlines=1)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4000.0, 6000.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), -math.expm1(-2.0), rtol=0.0, atol=0.02)


def test_linecontributions_taumin_keeps_each_line_that_reaches_it() -> None:
    """-taumin leaves out only the lines whose tau_sobolev stays below it in each cell.

    The upper bound of tau_sobolev is never below the tau_sobolev of a cell, thus -taumin keeps each line that reaches
    it.
    """
    commonargs: dict[str, t.Any] = {"modelpath": modelpath_classic_3d, "lambdamin": 3000.0, "lambdamax": 5000.0}
    linekeys = list(at.lines.linecontributions.LINEKEYS)
    dfall = at.lines.linecontributions.get_line_contributions(timestep=25, **commonargs)
    dfkept = at.lines.linecontributions.get_line_contributions(timestep=25, taumin=1.0, **commonargs)
    reachingkeys = set(dfall.filter(pl.col("tau_sobolev_max") >= 1.0).select(linekeys).iter_rows())
    keptkeys = set(dfkept.select(linekeys).iter_rows())
    assert reachingkeys
    assert reachingkeys <= keptkeys
    assert len(keptkeys) < dfall.height


def test_taumin_keeps_a_line_whose_lower_level_is_above_the_nlte_levels() -> None:
    """The -taumin bound of an ion with NLTE rows takes n_l <= n_ion max(1, g_l / g_0).

    A level above the NLTE levels of an ion with no superlevel takes n_0 (g_l / g_0) exp(-(E_l - E_0) / k T_exc), which
    can be above n_ion. The earlier bound n_l <= n_ion left out such a line, although its tau_sobolev reached -taumin.
    """
    from artistools.lines.linecontributions import get_cell_line_table
    from artistools.lines.linecontributions import get_lines_that_can_reach_tau
    from artistools.lines.linecontributions import NLTEPOPS_SCHEMA

    dfcells = pl.DataFrame({
        "modelgridindex": [0],
        "velocity_kmps": [5000.0],
        "volume": [1e45],
        "Te": [1e4],
        "nne": [1e8],
        "T_exc": [1e4],
        "nnion_Fe_II": [1.0],
    })
    levelsofion = {
        (26, 2): pl.DataFrame({
            "level": [0, 1, 2],
            "g": [1.0, 9.0, 5.0],
            "energy_ev": [0.0, 0.01, 2.0],
            "levelname": ["a", "b", "c"],
        })
    }
    dflines = pl.DataFrame({
        "Z": [26],
        "ion_stage": [2],
        "ion": ["Fe II"],
        "lambda_angstroms": [6230.4],
        "lower": [1],
        "upper": [2],
        "A_per_s": [1e4],
        "lower_g": [9.0],
        "upper_g": [5.0],
    })
    # an NLTE row for the ground level and no superlevel row
    dfnltepops = pl.DataFrame(
        {"modelgridindex": [0], "Z": [26], "ion_stage": [2], "level": [0], "n_NLTE": [0.9]}, schema=NLTEPOPS_SCHEMA
    )
    dfcelllines = get_cell_line_table(dfcells, dflines, levelsofion, dfnltepops, time_days=1.0)
    assert dfcelllines["n_lower_per_cm3"].item() > dfcells["nnion_Fe_II"].item()
    tau = dfcelllines["tau_sobolev"].item()
    dfkept = get_lines_that_can_reach_tau(dflines, dfcells, levelsofion, {(26, 2)}, tau, time_days=1.0)
    assert dfkept.height == 1

    # a second cell with a NaN estimator gave a bound of NaN, and the bound then left out the line
    dfwithnan = pl.concat([
        dfcells,
        dfcells.with_columns(modelgridindex=pl.col("modelgridindex") + 1, nnion_Fe_II=pl.lit(math.nan)),
    ])
    assert get_lines_that_can_reach_tau(dflines, dfwithnan, levelsofion, {(26, 2)}, tau, time_days=1.0).height == 1


def test_linecontributions_takes_the_nlte_levels_of_each_cell() -> None:
    """A level that only a different cell resolves takes the superlevel population of this cell.

    The highest NLTE level was the maximum over all the cells. A cell that resolves fewer levels then got a null
    population for a level of its superlevel, and the line had no tau_sobolev in that cell.
    """
    from artistools.lines.linecontributions import get_cell_line_table
    from artistools.lines.linecontributions import NLTEPOPS_SCHEMA

    dfcells = pl.DataFrame({
        "modelgridindex": [0, 1],
        "velocity_kmps": [5000.0, 6000.0],
        "volume": [1e45, 1e45],
        "Te": [1e4, 1e4],
        "nne": [1e8, 1e8],
        "T_exc": [1e4, 1e4],
        "nnion_Fe_II": [1.0, 1.0],
    })
    levelsofion = {
        (26, 2): pl.DataFrame({
            "level": [0, 1, 2, 3],
            "g": [1.0, 3.0, 5.0, 7.0],
            "energy_ev": [0.0, 0.5, 1.0, 3.0],
            "levelname": ["a", "b", "c", "d"],
        })
    }
    dflines = pl.DataFrame({
        "Z": [26],
        "ion_stage": [2],
        "ion": ["Fe II"],
        "lambda_angstroms": [6200.0],
        "lower": [2],
        "upper": [3],
        "A_per_s": [1e4],
        "lower_g": [5.0],
        "upper_g": [7.0],
    })
    # cell 0 resolves levels 0 and 1, and cell 1 resolves levels 0 to 2. Each cell has a superlevel row (level -1)
    dfnltepops = pl.DataFrame(
        {
            "modelgridindex": [0, 0, 0, 1, 1, 1, 1],
            "Z": [26] * 7,
            "ion_stage": [2] * 7,
            "level": [0, 1, -1, 0, 1, 2, -1],
            "n_NLTE": [0.5, 0.2, 0.3, 0.5, 0.2, 0.2, 0.1],
        },
        schema=NLTEPOPS_SCHEMA,
    )
    dfcelllines = get_cell_line_table(dfcells, dflines, levelsofion, dfnltepops, time_days=1.0).sort("modelgridindex")
    assert dfcelllines["n_lower_source"].to_list() == ["superlevel", "NLTE"]
    assert dfcelllines["n_upper_source"].to_list() == ["superlevel", "superlevel"]
    assert dfcelllines["n_lower_per_cm3"].null_count() == 0
    assert np.isclose(dfcelllines["n_lower_per_cm3"][1], 0.2, rtol=1e-12, atol=0.0)
    # in cell 0, the superlevel holds levels 2 and 3, thus level 2 takes its Boltzmann share of 0.3
    assert 0.0 < dfcelllines["n_lower_per_cm3"][0] < 0.3


def test_linecontributions_writes_a_file_only_with_the_output_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command wrote linecontributions.txt into the working folder with no -o, because -o had a default name."""
    monkeypatch.chdir(tmp_path)
    commonargs: dict[str, t.Any] = {"modelpath": modelpath, "timestep": 10, "lambdamin": 4600.0, "lambdamax": 4700.0}
    at.lines.linecontributions.main(argsraw=[], **commonargs)
    assert not list(tmp_path.iterdir())
    assert '"Fe III" 4659.' in capsys.readouterr().out
    # the skill gives the model path as the last positional argument, and the parser rejected it
    at.lines.linecontributions.main(
        argsraw=["-timestep", "10", "-lambdamin", "4600", "-lambdamax", "4700", str(modelpath)]
    )
    assert '"Fe III" 4659.' in capsys.readouterr().out

    outputfile = tmp_path / "lines.txt"
    at.lines.linecontributions.main(argsraw=[], outputfile=outputfile, **commonargs)
    filetext = outputfile.read_text(encoding="utf-8")
    assert "# column tau_sobolev_max: the maximum Sobolev optical depth of the line in the cells" in filetext


@mock.patch.object(mplax.Axes, "axhline", side_effect=mplax.Axes.axhline, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_ratios_and_the_planck_mean(
    mockplot: mock.MagicMock, mockaxhline: mock.MagicMock, tmp_path: Path
) -> None:
    """The plot has the bins and a moving average of each opacity, a dashed capped opacity, ratios, and the Planck mean.

    The Planck mean takes the bins of the x range, with the Planck function at the temperature of the cell. Only
    --showplanckmean draws it.
    """
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "--showplanckmean",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = [call.args[0] for call in mockplot.call_args_list]
    assert len(set(axes)) == 2, "the opacities and the ratios need two frames"
    mainplots = [call for call in mockplot.call_args_list if call.args[0] is axes[0]]
    assert len(mainplots) == 6, "each opacity needs its bins and its moving average"
    assert [call.kwargs["linestyle"] for call in mainplots if call.kwargs.get("label")] == ["-", "--", "-"]

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    dfbins = at.lines.core.get_expansion_opacities(lines, dfcell, edges, time_days)
    lambda_cm = dfbins["lambda_angstroms_bin_mid"].to_numpy() * 1e-8
    temperature = dfcell["T_exc"].item()
    planck = lambda_cm**-5 / np.expm1(
        at.constants.h_erg_s * at.constants.C_cm_per_s / lambda_cm / temperature / at.constants.K_B_erg_per_K
    )
    expected = float(np.sum(planck * dfbins["exopac"].to_numpy()) / np.sum(planck))
    assert mockaxhline.call_count == 1
    assert np.isclose(mockaxhline.call_args.args[1], expected, rtol=1e-9, atol=0.0)

    # the line and its calculation need --showplanckmean
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-o",
            str(tmp_path / "noplanck.pdf"),
        ]
    )
    assert mockaxhline.call_count == 1


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_each_cap_and_the_line_count(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """-taucaps draws an opacity for each cap, and --showlinecount adds a panel with the number of lines in each bin.

    The panel counts each line that the opacities sum. A cap that is given two times gives one opacity. The :g format
    gave 1.0000001 the column name of 1, thus the kernel stopped with an error.
    """
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-taucaps",
            "0.1",
            "1",
            "10",
            "1",
            "1.0000001",
            "--showlinecount",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    assert len(axes) == 3, "the opacities, the ratios, and the number of lines need three panels"
    labels = [call.kwargs["label"] for call in mockplot.call_args_list if call.kwargs.get("label")]
    assert labels == [
        "Expansion opacity",
        *(rf"Line-binned, $\tau_\mathrm{{l,max}}$ = {taucap}" for taucap in ("0.1", "1", "10", "1.0000001")),
        "Line-binned",
    ]
    ratioplots = [call for call in mockplot.call_args_list if call.args[0] is axes[1]]
    assert len(ratioplots) == 5, "each line-binned opacity needs a ratio"

    (linecountplot,) = [call for call in mockplot.call_args_list if call.args[0] is axes[2]]
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    # each bin draws its count at its lower edge and at its upper edge
    assert np.nansum(linecountplot.args[2]) == 2 * lines.dflines.height


def test_plotopacity_average_cell_takes_the_mean_composition(capsys: pytest.CaptureFixture[str]) -> None:
    """--averagecell takes one cell with the mass-weighted mean of n_ion / rho, of rho, and of T_exc, and logs T_exc.

    A model of one cell gives the same cell. A cell with no temperature does not count in the mean temperature.
    """
    dfone = at.lines.core.get_cell_estimators(modelpath, 40, None, "Te")
    pltest.assert_frame_equal(
        at.lines.plotopacity.get_average_cell(dfone),
        dfone.select(at.lines.plotopacity.get_average_cell(dfone).columns),
        check_dtypes=False,
        rel_tol=1e-12,
    )

    dfcells = pl.DataFrame({
        "modelgridindex": [0, 1, 2],
        "timestep": [5, 5, 5],
        "T_exc": [1000.0, 3000.0, None],
        "rho": [1.0, 2.0, 4.0],
        "mass_g": [1.0, 3.0, 4.0],
        "nnion_Fe_II": [2.0, 4.0, 8.0],
    })
    capsys.readouterr()
    dfmean = at.lines.plotopacity.get_average_cell(dfcells)
    assert np.isclose(dfmean["T_exc"].item(), (1000.0 + 3 * 3000.0) / 4, rtol=1e-12, atol=0.0)
    meanrho = (1.0 + 3 * 2.0 + 4 * 4.0) / 8
    assert np.isclose(dfmean["rho"].item(), meanrho, rtol=1e-12, atol=0.0)
    assert np.isclose(dfmean["nnion_Fe_II"].item(), meanrho * (2.0 + 3 * 2.0 + 4 * 2.0) / 8, rtol=1e-12, atol=0.0)
    assert "T_exc = 2500 K" in capsys.readouterr().out
    assert (
        at.lines.plotopacity.get_cells_text(None, None, None, "TJ = 2500 K")
        == "mean composition of all cells at TJ = 2500 K"
    )


def test_excitation_temperature_is_the_temperature_that_artis_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-exctemperature auto takes TJ or Te from LTEPOP_EXCITATION_USE_TJ of the model, and the log names it.

    The kernel took Te before. A classic model sets LTEPOP_EXCITATION_USE_TJ = true, and after the LTE timesteps its Te
    differs from TJ, thus the level populations did not match ARTIS. A commented line does not count.
    """
    get_column = at.lines.core.get_excitation_temperature_column
    capsys.readouterr()
    assert get_column(tmp_path, "auto") == "Te"
    assert "gives no LTEPOP_EXCITATION_USE_TJ" in capsys.readouterr().err

    (tmp_path / "artis").mkdir()
    optionspath = tmp_path / "artis" / "artisoptions.h"
    for value, expectedcolumn in (("true", "TJ"), ("false", "Te")):
        optionspath.write_text(
            f"// constexpr bool LTEPOP_EXCITATION_USE_TJ = {'false' if value == 'true' else 'true'};\n"
            f"constexpr bool LTEPOP_EXCITATION_USE_TJ = {value};\n",
            encoding="utf-8",
        )
        assert get_column(tmp_path, "auto") == expectedcolumn
        assert f"T_exc = {expectedcolumn}," in capsys.readouterr().out
    assert get_column(tmp_path, "TJ") == "TJ"
    assert "-exctemperature TJ" in capsys.readouterr().out


def test_expansion_opacities_keep_the_values_of_the_join_query() -> None:
    """The Rust kernel gives the values of the earlier polars query.

    The earlier query joined each cell with each line, and the reference sums come from it. Only the order
    of the additions is different, thus the values agree within the rounding error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    assert dfopacities.height == 100
    for column, expectedsum in {
        "exopac": 1397.6901658607103,
        "linebinned": 11675.7786539805,
        "linebinned_cap1": 1652.4668052744682,
    }.items():
        assert math.isclose(dfopacities[column].sum(), expectedsum, rel_tol=1e-12), column


def test_expansion_opacities_give_each_cap_its_own_column() -> None:
    """The kernel gives the sum of each cap to the column of that cap. The order of the caps has no effect.

    A cap of 1 gives the value of the regression test above. A larger cap gives a larger sum, and a cap above each
    tau_sobolev gives the sum with no cap.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (10.0, 1e300, 0.1, 1.0)
    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    sums = {taucap: dfopacities[at.lines.core.get_capped_column(taucap)].sum() for taucap in sorted(taucaps)}
    assert at.lines.core.get_capped_column(1234567.0) != at.lines.core.get_capped_column(1234568.0)
    assert math.isclose(sums[1.0], 1652.4668052744682, rel_tol=1e-12)
    assert math.isclose(sums[1e300], dfopacities["linebinned"].sum(), rel_tol=1e-12)
    assert list(sums.values()) == sorted(sums.values())
    assert len(set(sums.values())) == len(sums)


def test_expansion_opacities_of_a_null_population_are_zero() -> None:
    """A null population of an ion gives no opacity from that ion, and a null temperature gives no opacity.

    The kernel reads contiguous columns with no nulls, thus get_expansion_opacities() must replace each null.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )
    ionstr = opacitylines.ionstrs[0]

    def get_opacities(dfcells: pl.DataFrame) -> pl.DataFrame:
        return at.lines.core.get_expansion_opacities(opacitylines, dfcells, lambda_bin_edges, time_days).select(
            at.lines.core.get_opacity_columns(at.lines.core.DEFAULT_TAUCAPS)
        )

    pltest.assert_frame_equal(
        get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
        get_opacities(dfcell.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
    )
    dfnotemperature = get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias("T_exc")))
    assert np.allclose(dfnotemperature.select(pl.all().abs().max()).row(0), 0.0, rtol=0.0, atol=0.0)


def test_expansion_opacities_keep_a_nan_in_each_sum() -> None:
    """A NaN temperature gives NaN level populations, and each of the sums must then be NaN.

    f64::min(NaN, 1) is 1, and NaN.abs() >= 1e-18 is false, thus the kernel gave a finite capped opacity
    and a finite exopac for such a cell.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te").with_columns(
        pl.lit(float("nan"), dtype=pl.Float32).alias("T_exc")
    )
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (0.1, 1.0)
    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    opacitycolumns = at.lines.core.get_opacity_columns(taucaps)
    isnan = dfopacities.select(pl.col(opacitycolumns).is_nan())
    assert isnan["linebinned"].any()
    for column in opacitycolumns:
        assert isnan[column].equals(isnan["linebinned"]), column


def test_expansion_opacities_skip_a_line_with_a_null_constant() -> None:
    """A line with a null A value adds nothing, as a line that the atomic data does not hold.

    The kernel reads columns with no nulls, thus a null constant stopped the sum with an error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    isfirstline = pl.int_range(pl.len()) == 0

    def get_opacities(transitions: list[pl.LazyFrame]) -> pl.DataFrame:
        adatachanged = adata.with_columns(pl.Series("transitions", transitions, dtype=pl.Object))
        opacitylines = at.lines.core.get_opacity_lines(adatachanged, dfcell.columns, lambda_bin_edges, time_days)
        return at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    dfnullconstant = get_opacities([
        dftransitions.lazy().with_columns(A=pl.when(~isfirstline).then(pl.col("A")))
        for dftransitions in adata["transitions"]
    ])
    dfnoline = get_opacities([dftransitions.lazy().filter(~isfirstline) for dftransitions in adata["transitions"]])

    pltest.assert_frame_equal(dfnullconstant, dfnoline)


def test_lambda_bin_edges_reject_a_range_with_no_bin() -> None:
    """A range with no bin stops with a message, not with an IndexError in get_expansion_opacities()."""
    with pytest.raises(ValueError, match="holds no bin"):
        at.lines.core.get_lambda_bin_edges(5000.0, 4000.0, 10.0)


def test_lambda_bin_edges_cover_the_full_range() -> None:
    """The bins cover the full wavelength range, also when the division of the range rounds down.

    (4000 - 3000) / 0.1 is 9999.999999999998, and int() of it gave 9999 bins. Thus the last bin was missing.
    A range that does not hold a whole number of bins lost the part after the last whole bin.
    """
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 0.1)
    assert len(edges) == 10001
    assert math.isclose(edges[-1], 4000.0, rel_tol=1e-12)

    edges = at.lines.core.get_lambda_bin_edges(1000.0, 25010.0, 20.0)
    assert len(edges) == 1202
    assert math.isclose(edges[-1], 25020.0, rel_tol=1e-12)


def test_plotopacity_weights_the_cells_by_mass() -> None:
    """The mean over the cells weights each cell by its mass, and it sums the cells of every batch.

    Cell k holds k times the ion populations and k times the mass of the test cell. The line-binned
    opacity is linear in the populations. Thus, for n cells, the mean is sum(k^2) / sum(k) = (2n + 1) / 3
    times the opacity of the test cell. The first CELLSPERBATCH cells fill one batch, and the last 8 cells
    go into a second batch.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    assert dfcell.height == 1

    cellcount = at.lines.core.CELLSPERBATCH + 8
    dfcells = pl.concat([
        dfcell.with_columns(
            pl.lit(k - 1, dtype=dfcell.schema["modelgridindex"]).alias("modelgridindex"),
            pl.col("mass_g") * k,
            # a population is Float32, and k times it rounds in Float32
            pl.col("^nnion_.*$").cast(pl.Float64) * k,
        )
        for k in range(1, cellcount + 1)
    ])

    def get_linebinned(dfestimators: pl.DataFrame) -> npt.NDArray[np.float64]:
        dfopacities, _ = at.lines.plotopacity.get_massweighted_opacities(
            adata, time_days, dfestimators, at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
        )
        return dfopacities["linebinned"].to_numpy()

    linebinned_onecell = get_linebinned(dfcell)
    assert linebinned_onecell.max() > 0.0
    meanfactor = (2 * cellcount + 1) / 3
    assert np.allclose(get_linebinned(dfcells), meanfactor * linebinned_onecell, rtol=1e-10, atol=0.0)


def test_plotopacity_calculates_only_the_bins_of_the_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The bins of the plot range give the same plotted data as a calculation of the full grid of rpkt.h.

    The range also holds one bin more than half the window of the moving average at each end. The width of a bin
    comes from rpkt.h, and the command prints the range and the number of the bins.
    """
    (tmp_path / "artis").mkdir()
    (tmp_path / "artis" / "rpkt.h").write_text(
        "constexpr double expopac_lambdamin = 3000.;\n"
        "constexpr double expopac_lambdamax = 4000.;\n"
        "constexpr double expopac_deltalambda = 10.;\n",
        encoding="utf-8",
    )
    assert at.lines.core.get_expopac_grid(tmp_path) == (3000.0, 4000.0, 10.0)
    assert at.lines.core.get_expopac_grid(modelpath) is None

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    dffull, _ = at.lines.plotopacity.get_massweighted_opacities(
        adata, time_days, dfcell, at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    )
    xmin, xmax, width = 3213.0, 3517.0, 50.0
    capsys.readouterr()
    edges, deltalambda = at.lines.plotopacity.get_computed_bin_edges(tmp_path, xmin, xmax, None, width)
    assert deltalambda == pytest.approx(10.0, rel=1e-12, abs=0.0)
    assert len(edges) - 1 < 50 < dffull.height
    assert (
        f"{len(edges) - 1} wavelength bins of 10 Angstroms from {edges[0]:g} to {edges[-1]:g} Angstroms of the 100 bins"
        " of rpkt.h from 3000 to 4000 Angstroms"
    ) in capsys.readouterr().out
    dfpart, _ = at.lines.plotopacity.get_massweighted_opacities(adata, time_days, dfcell, edges)

    windowbins = at.lines.plotopacity.get_window_bins(width, deltalambda)
    plotted = [
        (
            df.filter(pl.col("lambda_angstroms_upper") > xmin, pl.col("lambda_angstroms_lower") < xmax),
            at.misc.df_filter_minmax_bracketed(
                at.lines.plotopacity.get_moving_averages(df, windowbins, ["exopac", "linebinned", "linebinned_cap1"]),
                "lambda_angstroms_bin_mid",
                xmin,
                xmax,
            ).collect(),
        )
        for df in (dffull, dfpart)
    ]
    pltest.assert_frame_equal(plotted[0][0], plotted[1][0], rel_tol=1e-9)
    pltest.assert_frame_equal(plotted[0][1], plotted[1][1], rel_tol=1e-9)


def test_plotopacity_window_holds_the_nearest_odd_number_of_bins() -> None:
    """The window of the moving average holds the odd number of bins that is nearest to its width.

    2 * round(n / 2) + 1 gave the odd number nearest to n + 1, e.g. 5 bins for a width of 3 bins.
    """
    windowbins = {
        width: at.lines.plotopacity.get_window_bins(width, 20.0) for width in (20.0, 60.0, 62.0, 140.0, 220.0)
    }
    assert windowbins == {20.0: 1, 60.0: 3, 62.0: 3, 140.0: 7, 220.0: 11}
    # a width of an even number of bins is one bin from two odd numbers, and it takes the larger one
    assert at.lines.plotopacity.get_window_bins(200.0, 20.0) == 11
    assert at.lines.plotopacity.get_window_bins(1.2, 0.2) == 7


def test_expansion_opacity_keeps_a_weak_line() -> None:
    """A line of a very small optical depth adds to the expansion opacity of its bin.

    1 - exp(-tau) is exactly zero below tau = 5.6e-17, thus a bin of weak lines had no expansion opacity.
    """
    from artistools.rustext import sum_binned_line_opacities

    taus = [1e-18, 1e-6, 0.5]
    dflevels = pl.DataFrame({"ionindex": [0], "g": [1.0], "energy_ev": [0.0]}, schema_overrides={"ionindex": pl.UInt32})
    dflines = pl.DataFrame(
        {
            "lambda_angstroms_binindex": [0, 1, 2],
            "lower": [0, 0, 0],
            "upper": [0, 0, 0],
            "sobolev_lower": taus,
            "sobolev_upper": [0.0, 0.0, 0.0],
            "lambda_angstroms": [1.0, 1.0, 1.0],
        },
        schema_overrides={"lambda_angstroms_binindex": pl.UInt32, "lower": pl.UInt32, "upper": pl.UInt32},
    )
    dfcells = pl.DataFrame({"T_exc": [5000.0], "nnion_0": [1.0]})
    exopac = sum_binned_line_opacities(dflevels, dflines, dfcells, ["nnion_0"], [], 3, at.constants.K_B_ev_per_K)[
        "exopac"
    ]
    assert np.allclose(exopac.to_numpy(), -np.expm1(-np.array(taus)), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("edges", [[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 2.0, 3.0]])
def test_rust_bin_sums_match_numpy_histogram(edges: list[float]) -> None:
    """The Rust kernel gives the bins of np.histogram, for uniform and for other edges.

    A bin is [lower, upper), the last bin also holds its upper edge, and a value outside the edges or NaN is in no
    bin. The uniform edges take a guess that the exact edges correct, thus a value on an inner edge tests it.
    """
    from artistools.rustext import sum_weights_in_bins

    values = [-0.1, 0.0, 0.5, 0.9999999, 1.0, 2.0, 2.5, 3.0, 3.0000001, math.nan]
    df = pl.DataFrame({"x": values, "e": [float(2**index) for index in range(len(values))]})
    # 3.0000001 is 3.0 in 32 bits, thus each type of value has its own reference
    for dtype in (pl.Float64, pl.Float32):
        dftyped = df.with_columns(pl.col("x").cast(dtype))
        typedvalues = dftyped["x"].to_numpy()
        isnumber = ~np.isnan(typedvalues)
        refcounts, _ = np.histogram(typedvalues[isnumber], bins=edges)
        refsums, _ = np.histogram(typedvalues[isnumber], bins=edges, weights=df["e"].to_numpy()[isnumber])
        refsquaresums, _ = np.histogram(typedvalues[isnumber], bins=edges, weights=df["e"].to_numpy()[isnumber] ** 2)
        sums = sum_weights_in_bins(dftyped, "x", "e", edges, sumsquares=True)
        assert np.allclose(sums["sum"].to_numpy(), refsums, rtol=1e-12, atol=0.0), dtype
        assert np.allclose(sums["sumsquares"].to_numpy(), refsquaresums, rtol=1e-12, atol=0.0), dtype
        assert sums["count"].to_list() == refcounts.tolist(), dtype

    # each group has its own bins, in the order [group][bin]
    rng = np.random.default_rng(seed=1)
    dfgroups = pl.DataFrame({
        "x": rng.uniform(-0.5, 3.5, 100_000),
        "e": rng.uniform(0.0, 1.0, 100_000),
        "group": rng.integers(0, 3, 100_000, dtype=np.int32),
    })
    grouped = sum_weights_in_bins(dfgroups, "x", "e", edges, "group", 3)
    for group in range(3):
        dfgroup = dfgroups.filter(pl.col("group") == group)
        expectedcounts, _ = np.histogram(dfgroup["x"].to_numpy(), bins=edges)
        expectedsums, _ = np.histogram(dfgroup["x"].to_numpy(), bins=edges, weights=dfgroup["e"].to_numpy())
        rows = slice(group * (len(edges) - 1), (group + 1) * (len(edges) - 1))
        assert np.allclose(grouped["sum"].to_numpy()[rows], expectedsums, rtol=1e-12, atol=0.0)
        assert grouped["count"].to_numpy()[rows].tolist() == expectedcounts.tolist()
    # pyo3-polars raises its own ComputeError, which is not the class of the polars package
    with pytest.raises(Exception, match="a group is outside"):
        sum_weights_in_bins(dfgroups, "x", "e", edges, "group", 2)


def test_rust_bin_indices_match_the_bins_of_the_sums() -> None:
    """get_bin_indices gives the bin of each value with the rules of sum_weights_in_bins, and -1 outside the edges."""
    from artistools.rustext import get_bin_indices
    from artistools.rustext import sum_weights_in_bins

    rng = np.random.default_rng(seed=2)
    edges = [0.0, 0.5, 2.0, 3.0]
    values = np.concatenate([rng.uniform(-0.5, 3.5, 10_000), edges, [math.nan]])
    df = pl.DataFrame({"x": values, "e": np.ones(len(values))})
    bins = get_bin_indices(df, "x", edges)["binindex"].to_numpy()
    counts = sum_weights_in_bins(df, "x", "e", edges)["count"].to_numpy()
    assert np.array_equal(np.bincount(bins[bins >= 0], minlength=len(edges) - 1), counts)
    # each edge starts its bin, the last edge is in the last bin, and NaN is in no bin
    assert bins[-5:].tolist() == [0, 1, 2, 2, -1]


def test_opacity_cell_batches_hold_fewer_cells_for_more_bins() -> None:
    """A batch has one row for each cell and bin, thus a batch of more bins must hold fewer cells.

    Each batch held 4096 cells, and the 4998 bins of the ejectaopacity defaults then took 3.6 GB for one batch.
    """
    dfcells = pl.DataFrame({"modelgridindex": range(10000)})
    for numbins in (100, 1200, 4998, 49980):
        batches = at.lines.core.get_cell_batches(dfcells, numbins)
        assert sum(batch.height for batch in batches) == dfcells.height
        assert max(batch.height for batch in batches) * numbins <= at.lines.core.ROWSPERBATCH, numbins


def test_plotopacity_median_of_the_cells_agrees_with_the_median_of_each_bin() -> None:
    """-cellaverage median takes the mass-weighted median of the cells from a histogram of log10 of the opacity.

    The median of each wavelength bin agrees with the exact mass-weighted median within half a step of the histogram.
    A NaN opacity stopped the median with an error of the strict cast of its bin. It now gives NaN, as in the mean.
    """
    from artistools.lines.core import get_expansion_opacities
    from artistools.lines.core import get_opacity_lines

    timestep = 25
    time_days = at.get_timestep_times(modelpath_classic_3d, loc="mid")[timestep]
    dfestimators = at.lines.core.get_cell_estimators(modelpath_classic_3d, timestep, None, "TJ").head(60)
    adata = at.lines.core.get_opacity_atomic_data(modelpath_classic_3d)
    edges = [4000.0 + 100.0 * index for index in range(11)]
    dfmedian, _ = at.lines.plotopacity.get_massweighted_opacities(
        adata, time_days, dfestimators, edges, taucaps=[], cellaverage="median"
    )
    dfcells = get_expansion_opacities(
        get_opacity_lines(adata, dfestimators.columns, edges, time_days), dfestimators, edges, time_days, []
    )
    dfexact = (
        dfcells
        .group_by("lambda_angstroms_binindex")
        .agg(exact=at.misc.get_weighted_quantile_expr("exopac", "mass_g", 0.5))
        .sort("lambda_angstroms_binindex")
    )
    ratio = dfmedian["exopac"].to_numpy() / dfexact["exact"].to_numpy()
    assert np.all(np.abs(np.log10(ratio)) <= 0.5 * at.lines.plotopacity.OPACITYMEDIAN_LOGSTEP + 1e-9)

    dfnancell = dfestimators.with_columns(
        T_exc=pl
        .when(pl.col("modelgridindex") == dfestimators["modelgridindex"][0])
        .then(math.nan)
        .otherwise(pl.col("T_exc"))
    )
    for cellaverage in ("mean", "median"):
        dfnan, _ = at.lines.plotopacity.get_massweighted_opacities(
            adata, time_days, dfnancell, edges, taucaps=[], cellaverage=cellaverage
        )
        assert dfnan["exopac"].is_nan().all(), cellaverage


def test_plotopacity_velocity_range_takes_the_cells_of_the_range(capsys: pytest.CaptureFixture[str]) -> None:
    """-vmin and -vmax of plotopacity select the cells with a mid-point velocity in the range, and no other cell.

    Each velocity needs a unit. The count of the cells in the range and the title give each velocity in the unit
    of the user.
    """
    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    dfvelocities = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid_on_c", costheta=pl.col("vel_z_mid_on_c") / pl.col("vel_r_mid_on_c"))
        .collect()
    )
    expectedcells = set(dfvelocities.filter(pl.col("vel_r_mid_on_c").is_between(0.1, 0.2))["modelgridindex"])
    assert 0 < len(expectedcells) < dfvelocities.height

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "0.2c"])
    dfestimators = at.lines.core.get_cell_estimators(modelpath_classic_3d, 5, None, "Te")
    capsys.readouterr()
    dfcells = at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax)
    assert set(dfcells["modelgridindex"]) == expectedcells
    assert (
        f"{len(expectedcells)} of {dfestimators.height} cells with estimators are in the range"
        " with vmin = 0.1c and vmax = 0.2c" in (capsys.readouterr().out)
    )

    # -costhetamin selects the cells near the pole at +z, and -dirbin 90 selects the same cells
    expectedpolar = set(
        dfvelocities.filter((pl.col("vel_r_mid_on_c") >= 0.1) & (pl.col("costheta") >= 0.8))["modelgridindex"]
    )
    assert 0 < len(expectedpolar) < len(set(dfestimators["modelgridindex"]))
    for selectionargs in (["-costhetamin", "0.8"], ["-dirbin", "90"]):
        args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", *selectionargs])
        costhetarange = at.misc.get_costheta_range(args)
        dfpolarcells = at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax, costhetarange)
        assert set(dfpolarcells["modelgridindex"]) == expectedpolar & set(dfestimators["modelgridindex"])

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "60000km/s"])
    assert at.lines.plotopacity.get_cells_text(None, args.vmin, args.vmax) == (
        "mass-weighted mean of the cells with vmin = 0.1c and vmax = 60000 km/s"
    )
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1"])

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "150000km/s"])
    with pytest.raises(ValueError, match=re.escape("No cell with estimators is in the range with vmin = 150000 km/s")):
        at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax)


def test_cell_estimators_of_an_empty_cell_give_an_error() -> None:
    """A cell with no matter has no estimators, and the error names the cell and the next cell with estimators.

    The command stopped with "cannot concat empty list" before.
    """
    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    emptycell = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") == 0.0)
        .select(pl.col("modelgridindex").min())
        .collect()
        .item()
    )
    estimatorcells = at.scan_estimators(modelpath_classic_3d, timestep=5).select("modelgridindex").collect()
    nextcell = estimatorcells.filter(pl.col("modelgridindex") > emptycell)["modelgridindex"].min()
    with pytest.raises(ValueError, match=rf"hold no values for cell {emptycell} .* is cell {nextcell}$"):
        at.lines.core.get_cell_estimators(modelpath_classic_3d, 5, emptycell, "Te")


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


def test_linefluxes_emfeaturesearch_parsing() -> None:
    """Emission features given on the command line must arrive as tuples of ints, not as raw strings."""
    parser = argparse.ArgumentParser()
    at.plotlinefluxes.addargs(parser)

    args = parser.parse_args(["-emfeaturesearch", "(26, 2, 7155, 7150, 7160)", "(28, 2, 7378, 7373, 7383)"])
    assert args.emfeaturesearch == [(26, 2, 7155, 7150, 7160), (28, 2, 7378, 7373, 7383)]

    # the default must already be usable for the two-feature flux ratio plot
    assert len(parser.parse_args([]).emfeaturesearch) >= 2

    # time bins are floats, not appended lists of strings
    args = parser.parse_args(["-timebins_tstart", "200", "250", "-timebins_tend", "250", "300"])
    assert args.timebins_tstart == [200.0, 250.0]
    assert args.timebins_tend == [250.0, 300.0]

    # unset time bins stay None, so that each model falls back to its own timestep grid
    args = parser.parse_args([])
    assert args.timebins_tstart is None
    assert args.timebins_tend is None

    # the wavelengths may be fractional, but the atomic number, ion stage, and level indices must be integers
    assert parser.parse_args(["-emfeaturesearch", "(26, 2, 12570.5, 12470.5, 12670.5)"]).emfeaturesearch == [
        (26, 2, 12570.5, 12470.5, 12670.5)
    ]

    for badfeature in ("not a tuple", "(26.0, 2, 7155)", "(26, True, 7155)", "(26, 2, 7155, 7150, 7160, 1.5, 2)"):
        with pytest.raises(SystemExit):
            parser.parse_args(["-emfeaturesearch", badfeature])


def test_linefluxes_default_timebins_use_each_models_timesteps() -> None:
    """With no explicit time bins, the packet binning must fall back to the model's own timestep grid."""
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_packets

    emfeatures = [get_closelines(modelpath_classic_3d, 26, 2, 7155, 7100, 7200)]

    dflcdata_default = get_line_luminosities_from_packets("trueemissiontype", emfeatures, modelpath_classic_3d)
    dflcdata_explicit = get_line_luminosities_from_packets(
        "trueemissiontype",
        emfeatures,
        modelpath_classic_3d,
        arr_tstart=at.get_timestep_times(modelpath_classic_3d, loc="start"),
        arr_tend=at.get_timestep_times(modelpath_classic_3d, loc="end"),
    )

    pltest.assert_frame_equal(dflcdata_default, dflcdata_explicit)


def test_linefluxes_timebins_keep_a_rounding_gap_and_drop_a_real_gap() -> None:
    """A packet between two bins that meet within rounding takes a bin, and a packet in a real gap takes none.

    Each bin was [tstart, tstart + twidth), and timesteps.out gives six significant figures. A packet in the
    rounding gap before the next start then had no bin, and the sums did not include it.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [0.9, 1.0, 1.999995, 2.0, 2.9999, 3.0, 3.5, 4.0, 5.0, 5.1]})
    timebins = dftimes.select(get_timebin_expr(pl.col("t"), [1.0, 2.0, 4.0], [1.99999, 3.0, 5.0]))
    assert timebins.to_series().to_list() == [None, 0, 0, 1, 1, None, None, 2, 2, None]


def test_linefluxes_from_pops_takes_the_cell_volumes_of_a_3d_model(tmp_path: Path) -> None:
    """A 3D model has no shell velocities, thus the luminosity takes the volume of each cell.

    The function selected vel_r_min_kmps, which only a 1D model has, and a 3D model stopped with ColumnNotFoundError.
    """
    import itertools

    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    # a 3D grid of 2^3 cells with the files of the 1D test model. Only cell 0 has NLTE populations
    model3dpath = tmp_path / "model3d"
    model3dpath.mkdir()
    for filepath in modelpath.iterdir():
        if filepath.is_file() and filepath.name != "model.txt" and not filepath.name.startswith("."):
            (model3dpath / filepath.name).symlink_to(filepath)
    t_model_days = 0.00115740740741
    vmax = 8.0e8
    cellwidth = vmax * t_model_days * 86400.0
    cellrows = "".join(
        f"{cellid} {x} {y} {z} {10**-0.18}\n1.0 0.9 0.0 0.0 0.0\n"
        for cellid, (z, y, x) in enumerate(itertools.product((-cellwidth, 0.0), repeat=3), start=1)
    )
    (model3dpath / "model.txt").write_text(f"8\n{t_model_days}\n{vmax}\n{cellrows}", encoding="utf-8")

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    arr_tstart = [at.get_timestep_times(modelpath, loc="start")[10]]
    arr_tend = [at.get_timestep_times(modelpath, loc="end")[10]]
    lum1d = get_line_luminosities_from_pops(emfeatures, modelpath, arr_tstart, arr_tend)["Fe II 1-0"].item()
    lum3d = get_line_luminosities_from_pops(emfeatures, model3dpath, arr_tstart, arr_tend)["Fe II 1-0"].item()

    # the cell of the 3D model is a cube of side vmax, and the shell of the 1D model is a sphere of radius vmax
    assert np.isclose(lum3d / lum1d, 3.0 / (4.0 * math.pi), rtol=1e-5)


def test_linefluxes_from_pops_names_a_level_or_a_time_with_no_populations() -> None:
    """A line with no NLTE population at a time gives a message, not a bare AssertionError.

    The test model writes NLTE populations only at even timesteps from 10.
    """
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_closelines
    from artistools.plotlinefluxes import get_line_luminosities_from_pops

    emfeatures = [FeatureTuple("Fe II 1-0", "Fe II", 0.0, [0], 0.0, 0.0, 26, 2, [1], [0])]
    with pytest.raises(ValueError, match="hold no value of the upper level 1 at timestep 11"):
        get_line_luminosities_from_pops(
            emfeatures,
            modelpath,
            arr_tstart=[at.get_timestep_times(modelpath, loc="start")[11]],
            arr_tend=[at.get_timestep_times(modelpath, loc="end")[11]],
        )

    with pytest.raises(ValueError, match="holds no Fe II line for the feature 7155"):
        get_closelines(modelpath_classic_3d, 26, 2, 7155, 7155.0, 7155.01)


def test_linefluxes_feature_includes_a_line_at_a_wavelength_bound() -> None:
    """A line at lower_wavelength or upper_wavelength of a feature is part of the feature."""
    from artistools.plotlinefluxes import get_closelines

    linelambda = (
        at.atomic
        .scan_linelist(modelpath_classic_3d)
        .filter(pl.col("atomic_number") == 26, pl.col("ion_stage") == 2)
        .select(pl.col("lambda_angstroms").first())
        .collect()
        .item()
    )
    feature = get_closelines(modelpath_classic_3d, 26, 2, "bound", linelambda, linelambda)
    assert feature.lowestlambda == feature.highestlambda == linelambda


def test_linefluxes_pops_luminosity_matches_a_loop_over_the_cells() -> None:
    """The vectorised sum over the lines and the cells must give what the loop over each cell gave.

    A cell without population data gives its volume to the next cell outward that has data, and the
    outermost empty cells give their volume to no cell. The loop here is the former algorithm.
    """
    from artistools.plotlinefluxes import sum_line_luminosities

    rng = np.random.default_rng(seed=3)
    ncells = 6
    shell_volumes_at_1s = rng.uniform(1e40, 2e40, size=ncells)
    dftimes = pl.DataFrame({"timeindex": [0, 1, 2], "timestep": [4, 7, 7], "t_sec_cubed": [1e21, 8e21, 8e21]})
    dflines = pl.DataFrame({"lineindex": [0, 1], "level": [12, 9], "A": [0.3, 0.02], "delta_ergs": [2e-12, 1.5e-12]})

    # cell 0 and cell 3 hold no data in timestep 4, and the last two cells hold no data in timestep 7
    poprows = [
        (timestep, level, modelgridindex, float(rng.uniform(1e5, 1e6)))
        for timestep, emptycells in ((4, {0, 3}), (7, {4, 5}))
        for level in (12, 9)
        for modelgridindex in range(ncells)
        if modelgridindex not in emptycells
    ]
    dfnltepops = pl.DataFrame(
        poprows,
        schema={"timestep": pl.Int32, "level": pl.Int32, "modelgridindex": pl.Int32, "n_NLTE": pl.Float64},
        orient="row",
    )
    levelpop_of_key = {(timestep, level, mgi): n_nlte for timestep, level, mgi, n_nlte in poprows}

    expected = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        shell_volumes = shell_volumes_at_1s * t_sec_cubed
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            unaccounted_shellvol = 0.0
            for modelgridindex in range(ncells):
                levelpop = levelpop_of_key.get((timestep, level, modelgridindex))
                if levelpop is None:
                    unaccounted_shellvol += shell_volumes[modelgridindex]
                    continue
                expected[timeindex] += (
                    delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                )
                unaccounted_shellvol = 0.0

    lumdata = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=True)

    assert lumdata.shape == expected.shape
    assert np.allclose(lumdata, expected, rtol=1e-13)

    # the cells of a 2D or 3D model have no radial order, thus an empty cell gives its volume to no other cell
    expected_cells = np.zeros(dftimes.height)
    for timeindex, timestep, t_sec_cubed in dftimes.iter_rows():
        for level, A_val, delta_ergs in dflines.select("level", "A", "delta_ergs").iter_rows():
            for modelgridindex in range(ncells):
                if (levelpop := levelpop_of_key.get((timestep, level, modelgridindex))) is not None:
                    expected_cells[timeindex] += (
                        delta_ergs * A_val * levelpop * shell_volumes_at_1s[modelgridindex] * t_sec_cubed
                    )
    lumdata_cells = sum_line_luminosities(dfnltepops, dflines, dftimes, shell_volumes_at_1s, fillshellgaps=False)
    assert np.allclose(lumdata_cells, expected_cells, rtol=1e-13)
    assert not np.allclose(lumdata_cells, expected, rtol=1e-6)


def test_linefluxes_rejects_lone_timebin_argument() -> None:
    """Giving only one of the two time bin edge lists must be rejected before any data is read."""
    with pytest.raises(ValueError, match="must be given together"):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], timebins_tstart=[200.0, 250.0])


def test_linefluxes_gives_a_colour_to_each_model_after_the_default_palette() -> None:
    """Each model gets a colour, also after the ten colours of the default palette.

    The command gave None to each model after the tenth, thus --plotemittingregions stopped with an error.
    """
    with mock.patch.object(at.plotlinefluxes, "make_emitting_regions_plot") as mockplot:
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d] * 11, plotemittingregions=True)
    colors = mockplot.call_args.args[0].color
    assert len(colors) == 11
    assert all(color is not None for color in colors)


def test_linefluxes_lineflux_ratio_plot() -> None:
    """The line flux ratio plot must run with no arguments beyond the model path.

    The plot of Fe II 7155 Å and 12570 Å also draws the fit of Flörs et al. (2020), and --write_data gave no table of
    it.
    """
    funcoutpath = outputpath / funcname()
    funcoutpath.mkdir(exist_ok=True, parents=True)
    # the fit is near 0.02 at the 3 to 7 days of the test model, which the fixed bottom of 0.05 hides
    at.plotlinefluxes.main(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
        ymin=0.001,
        outputfile=funcoutpath / "linefluxes.pdf",
        write_data=True,
    )
    assert (funcoutpath / "linefluxes.pdf").is_file()
    dfcomparisons = pl.read_csv(
        funcoutpath / "linefluxes_comparisons.txt", separator=" ", null_values="nan", comment_prefix="#"
    )
    assert dfcomparisons["series"].unique().to_list() == ["Flörs+2020 fit"]
    assert dfcomparisons.height == 3


def test_linefluxes_table_of_three_models_with_one_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Three models of one name gave the same column, and the join stopped the plot with a repeated column.

    The axis shows the times from -xmin to -xmax, and the table also gave the points outside that range. The test model
    has no packets of these lines, thus the ratio of each time is NaN, and the test gives each line a luminosity.
    """

    def get_line_luminosities(
        _emtypecolumn: str, emfeatures: Sequence[t.Any], _modelpath: Path, **_kwargs: t.Any
    ) -> pl.DataFrame:
        times = np.linspace(3.0, 8.0, 11)
        return pl.DataFrame({"time": times, **{feature.colname: np.full(len(times), 1e38) for feature in emfeatures}})

    with mock.patch.object(at.plotlinefluxes, "get_line_luminosities_from_packets", get_line_luminosities):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d] * 3,
            emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
            xmin=4.5,
            xmax=6.5,
            ymin=0.001,
            outputfile=tmp_path / "linefluxes.pdf",
            write_data=True,
        )
    assert "The data table gives the column of" in capsys.readouterr().err
    dftable = read_data_file(tmp_path / "linefluxes.txt")
    assert len([column for column in dftable.columns if column.startswith("fratio_")]) == 3
    assert dftable["time"].is_between(4.5, 6.5).all()
    assert dftable.height > 0
    assert read_data_file(tmp_path / "linefluxes_comparisons.txt")["time"].is_between(4.5, 6.5).all()


def test_linefluxes_refuse_two_features_that_share_a_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A packet of a line that both features hold counted in both sums, thus the ratio noise was not correct.

    Two identical features gave a ratio of exactly 1 and a noise above 0.
    """
    with pytest.raises(SystemExit):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 7155, 7100, 7200)],
            outputfile=tmp_path / "linefluxes.pdf",
        )
    assert "of the ratio share" in capsys.readouterr().err
    assert not (tmp_path / "linefluxes.pdf").exists()


def test_linefluxes_floers_ratio_file_with_no_w7_model(tmp_path: Path) -> None:
    """A file with no W7 model gave a frame with no columns, and the concatenation of the comparisons then failed."""
    ratiofile = tmp_path / "floersratios.csv"
    ratiofile.write_text("file,epoch,NIR_VIS_ratio\nfig-nne_Te_allcells-subch-300d.txt,300,1.5\n", encoding="utf-8")
    at.plotlinefluxes.main(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        emfeaturesearch=[(26, 2, 7155, 7100, 7200), (26, 2, 12570, 12400, 12700)],
        floersmodelratiofile=str(ratiofile),
        outputfile=tmp_path / "linefluxes.pdf",
    )
    assert (tmp_path / "linefluxes.pdf").is_file()


def test_get_ion_tuple_rejects_missing_element() -> None:
    """A string with a separator but no element symbol must raise rather than return atomic number -1."""
    for badionstr in (" II", "_II", "notanion", "Fe "):
        with pytest.raises(ValueError, match="Could not parse ionstr"):
            at.get_ion_tuple(badionstr)


def test_default_plotitem_keeps_estimator_columns_named_like_elements() -> None:
    """Te and W are estimator names as well as element symbols, so a real column must win over the element reading."""
    from artistools.estimators.plotestimators import get_default_plotitem_skip_reason

    estimatorcolumns = ["timestep", "modelgridindex", "Te", "TR", "W", "nne", "rho", "nnelement_Fe"]

    for plotitem in (["Te"], ["W"], ["TR"], ["rho"], ["nne"], [["averageionisation", ["Fe"]]]):
        assert get_default_plotitem_skip_reason(plotitem, estimatorcolumns) is None, plotitem

    # an element that the model does not contain is still dropped
    assert get_default_plotitem_skip_reason([["averageionisation", ["Sr"]]], estimatorcolumns) == (
        "the estimators have no Sr"
    )
    assert get_default_plotitem_skip_reason([["populations", ["Sr I", "Sr II"]]], estimatorcolumns) is not None

    # initabundances/initmasses come from the input model file, so they must not be gated on estimator columns
    assert get_default_plotitem_skip_reason([["initabundances", ["Sr", "Ni_stable"]]], estimatorcolumns) is None
    assert get_default_plotitem_skip_reason([["initmasses", ["Sr", "Ni_56"]]], estimatorcolumns) is None


def test_write_lbol_edep(tmp_path: Path) -> None:
    """The bolometric luminosity / deposition writer must produce one line per selected timestep.

    test_writecomparisondata() uses a model with no deposition.out, so its FileNotFoundError guard skipped this
    function entirely and hid the fact that it read column names the light curve frame does not have.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    selected_timesteps = [0, 1, 2, 5]
    write_lbol_edep(modelpath_classic_3d, selected_timesteps, outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"#NTIMES: {len(selected_timesteps)}"
    datalines = [line for line in lines if not line.startswith("#")]
    assert len(datalines) == len(selected_timesteps)
    for line in datalines:
        timedays, lbol, edep = (float(x) for x in line.split())
        assert timedays > 0.0
        assert lbol > 0.0
        assert edep > 0.0


def test_write_lbol_edep_ntimes_matches_rows(tmp_path: Path) -> None:
    """The NTIMES header must count the rows actually written, not the timesteps asked for.

    A selected timestep with no light curve or deposition data is dropped by the join, so a header quoting
    len(selected_timesteps) would promise more times than the file contains and misalign every reader.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    write_lbol_edep(modelpath_classic_3d, [0, 1, 2, 5, 9999], outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    datalines = [line for line in lines if not line.startswith("#")]
    assert lines[0] == f"#NTIMES: {len(datalines)}"
    # timestep 9999 does not exist, so it must not be counted
    assert len(datalines) == 4


def test_path_is_artis_model_accepts_a_compressed_output_file() -> None:
    """A compressed ARTIS output file is a model, and not a reference data file."""
    assert all(at.misc.path_is_artis_model(f"light_curve.out{ext}") for ext in ("", ".zst", ".gz", ".xz"))
    assert not at.misc.path_is_artis_model("AT2017gfo_smarttetal2017.txt")


@mock.patch.object(mplax.Axes, "set_ylim", side_effect=mplax.Axes.set_ylim, autospec=True)
def test_radfield_honours_the_ymin_that_it_accepts(mocksetylim: mock.MagicMock) -> None:
    """Plotradfield adds -ymin, thus the axis must start there and not at the hard-coded zero."""
    at.radfield.plotradfield.main(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40, modelgridindex=0, ymin=1e-14
    )

    bottoms = [callargs.kwargs["bottom"] for callargs in mocksetylim.call_args_list if "bottom" in callargs.kwargs]
    assert bottoms, "the command must set the bottom of the axis"
    assert 1e-14 in bottoms, f"the requested -ymin is missing from {bottoms}"


def test_firstexisting_gives_the_purpose_of_a_missing_file(tmp_path: Path) -> None:
    """A list of file names alone does not tell a user what the file holds or which command reads it."""
    with pytest.raises(FileNotFoundError, match=r"linestat\.out gives the wavelength"):
        at.misc.firstexisting("linestat.out", folder=tmp_path, purpose="linestat.out gives the wavelength")

    # a caller that gives no purpose keeps the plain message
    with pytest.raises(FileNotFoundError) as noreason:
        at.misc.firstexisting("linestat.out", folder=tmp_path)

    assert "None of these files exist" in str(noreason.value)
    assert "gives the wavelength" not in str(noreason.value)


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


def test_plotspectra_x_still_gives_the_unit() -> None:
    """-x named the unit on plotspectra before -xunit, thus a script that holds it still runs.

    -x names the axis variable on plotestimators, but each parser reads its own arguments.
    """
    import artistools.spectra.plotspectra

    parser = argparse.ArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    # each spelling gives the canonical name of the unit, thus the plot code reads one name
    assert parser.parse_args([".", "-x", "nm"]).xunit == "nm"
    assert parser.parse_args([".", "-xunits", "micron"]).xunit == "micron"
    assert parser.parse_args([".", "-xunit", "Hz"]).xunit == "hz"


def test_timesteps_command_lists_the_days_of_each_timestep(capsys: pytest.CaptureFixture[str]) -> None:
    """The timesteps command gives the table that a user needs to select a -timestep value.

    Before this command, the mapping from a timestep to its days appeared only inside the error message
    for a wrong value.
    """
    at.listtimesteps.main(argsraw=["-modelpath", str(modelpath)])
    table = capsys.readouterr().out

    lines = table.splitlines()
    assert lines[0] == "TEST MODEL (folder testmodel): 100 timesteps from 250.000 to 350.000 days"
    assert lines[1].split() == ["timestep", "start_days", "mid_days", "end_days", "width_days"]
    assert len(lines) == 103, "a header, a column line, 100 rows, and a closing hint"

    firstrow = lines[2].split()
    assert firstrow[0] == "0"
    assert np.isclose(float(firstrow[1]), 250.0)

    # the hint names the ways to select a time, and the keyword that names the final timestep
    assert "-timestep" in lines[-1]
    assert "-timedays" in lines[-1]
    assert "`last` names timestep 99" in lines[-1]


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


def test_timesteps_command_answers_a_reverse_lookup(capsys: pytest.CaptureFixture[str]) -> None:
    """-timedays names the timestep that covers a time, and -timestep gives the days of one timestep."""
    at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-t", "300"])
    assert capsys.readouterr().out.strip() == "300 days falls in timestep 54, which covers 299.812 to 300.823 days"

    at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-ts", "last"])
    assert capsys.readouterr().out.strip() == "Timestep 99 covers 348.824 to 350.000 days"


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


def test_plotspherical_makes_the_output_folder(tmp_path: Path) -> None:
    """A -o path that has no file extension names a folder, which the command makes.

    The command wrote a file that had the name of the folder and no extension, and a run with
    --makegif stopped, because it built the path of each frame below a folder that did not exist.
    """
    outfolder = tmp_path / "frames"
    at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=str(outfolder), timemin=250, timemax=300)

    assert outfolder.is_dir(), "the command must make the folder that -o names"
    assert list(outfolder.glob("plotspherical_*.pdf")), f"no plot in {list(outfolder.iterdir())}"


def test_timesteps_command_refuses_a_timestep_outside_the_model() -> None:
    """A timestep that the model does not hold must name the range that it holds.

    The command indexed the list of times with the given value, thus 999 gave an IndexError and -1
    read the last row of the list and gave it the wrong label.
    """
    for timestep in ("999", "-1"):
        with pytest.raises(ValueError, match=r"is not in this model\. It has 100 timesteps, 0 to 99"):
            at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])

    # the timesteps at each end of the model are in it
    for timestep in ("0", "99", "last"):
        at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-timestep", timestep])


def test_plotspherical_gif_keeps_the_name_that_o_gives(tmp_path: Path) -> None:
    """A -o path that has a file extension names the gif, and its folder then holds the frames.

    A path such as movie.gif became a folder, thus the file that the user asked for was never written.
    """
    gifpath = tmp_path / "movie.gif"
    at.packets.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=str(gifpath)
    )

    assert gifpath.is_file(), f"the gif must keep its name, but {list(tmp_path.iterdir())}"
    assert list(gifpath.parent.glob("plotspherical_*.png")), "the frames go in the folder of the gif"

    # a path with no file extension still names a folder that holds the gif and the frames
    outfolder = tmp_path / "movie"
    at.packets.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=str(outfolder)
    )
    assert (outfolder / "sphericalplot.gif").is_file(), f"no gif in {list(outfolder.iterdir())}"


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


def test_writecomparisondata_rejects_an_empty_timestep_list(tmp_path: Path) -> None:
    """An empty timestep list wrote files that hold a header and no data."""
    with pytest.raises(ValueError, match="selected-timesteps"):
        at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[])


def test_writecomparisondata_of_a_model_with_no_estimators_gives_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model with no estimator files stopped with a raw ColumnNotFoundError for Te."""
    modelfolder = tmp_path / "noestimators"
    modelfolder.mkdir()
    for sourcefile in modelpath.iterdir():
        if sourcefile.is_file() and not sourcefile.name.startswith("estimators"):
            (modelfolder / sourcefile.name).symlink_to(sourcefile.resolve())

    with pytest.raises(SystemExit) as excinfo:
        at.writecomparisondata.main(
            argsraw=[], modelpath=modelfolder, outputpath=tmp_path / "out", selected_timesteps=[50]
        )

    assert excinfo.value.code == 1
    assert "holds no estimator files" in capsys.readouterr().err


def test_ionfrac_header_counts_the_stages_from_neutral(tmp_path: Path) -> None:
    """The format labels the neutral stage 0, thus the stages below the lowest ion of an element hold zero.

    The header gave each column its ARTIS ion stage, which is 1 for the neutral stage, thus Co II took the label co2.
    """
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=list(range(10))
    )

    ionfracfiles = sorted(tmp_path.glob("ionfrac_*_artisnebular.txt"))
    assert ionfracfiles, "the run wrote no ion fraction file"

    elementlist = at.atomic.get_composition_data(modelpath)
    lowermost_of_elsymbol = {
        at.get_elsymbol(row["Z"]).lower(): row["lowermost_ion_stage"] for row in elementlist.iter_rows(named=True)
    }
    assert any(lowermost > 1 for lowermost in lowermost_of_elsymbol.values())

    for ionfracfile in ionfracfiles:
        elsymbol = ionfracfile.name.split("_")[1]
        lines = ionfracfile.read_text(encoding="utf-8").splitlines()
        nstages = int(next(line for line in lines if line.startswith("#NSTAGES:")).split()[1])
        headers = [line for line in lines if line.startswith("#vel_mid")]
        assert headers
        for header in headers:
            assert header.split()[1:] == [f"{elsymbol}{stage}" for stage in range(nstages)]

        datarows = [line.split()[1:] for line in lines if not line.startswith("#")]
        assert datarows
        for row in datarows:
            assert len(row) == nstages
            assert all(float(value) == 0.0 for value in row[: lowermost_of_elsymbol[elsymbol] - 1])


def test_writecomparisondata_keeps_a_tiny_ion_fraction(tmp_path: Path) -> None:
    """An ion fraction below 1e-38 must keep its digits.

    The estimator cache stores a population as Float32. A Float32 ratio below 1e-38 is subnormal, thus it lost
    digits or became zero. The old reader divided Python floats, and the files of the two readers differed.
    """
    from artistools.writecomparisondata import write_ionfracts

    dfestimators = pl.DataFrame(
        {"timestep": [0], "vel_r_mid": [1e9], "nnelement_Fe": [1e8], "nnion_Fe_II": [1e-37]},
        schema={"timestep": pl.Int32, "vel_r_mid": pl.Float64, "nnelement_Fe": pl.Float32, "nnion_Fe_II": pl.Float32},
    )
    write_ionfracts(modelpath, "tiny", [0], dfestimators, tmp_path)

    datalines = [
        line
        for line in (tmp_path / "ionfrac_fe_tiny_artisnebular.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    expected = float(np.float32(1e-37)) / float(np.float32(1e8))
    # the columns are the velocity, then the stages from the neutral stage, thus Fe II is the third value
    assert float(datalines[0].split()[2]) == pytest.approx(expected, rel=1e-4, abs=0.0)


def test_installskill_replaces_a_link_and_keeps_the_skill_of_the_package(tmp_path: Path) -> None:
    """A copy must replace the link itself, because rmtree through the link deletes the skill of the package."""
    from artistools.installskill import get_skill_folder
    from artistools.installskill import install_skill

    skillsdir = tmp_path / "skills"
    packageskill = get_skill_folder() / "SKILL.md"
    assert install_skill(skillsdir, copy=False, force=False)
    assert not install_skill(skillsdir, copy=False, force=False)

    with pytest.raises(SystemExit):
        install_skill(skillsdir, copy=True, force=False)

    assert install_skill(skillsdir, copy=True, force=True)
    destination = skillsdir / "artis-data"
    assert not destination.is_symlink()
    assert packageskill.is_file()
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == packageskill.read_text(encoding="utf-8")


def test_installskill_names_each_file_that_rmtree_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The warning named only SKILL.md, but rmtree then also deleted the other files of a real skill folder."""
    from artistools.installskill import install_skill

    skillsdir = tmp_path / "skills"
    (skillsdir / "artis-data" / "references").mkdir(parents=True)
    (skillsdir / "artis-data" / "SKILL.md").write_text("an older copy", encoding="utf-8")
    usernotes = skillsdir / "artis-data" / "references" / "notes.md"
    usernotes.write_text("notes of the user", encoding="utf-8")

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        install_skill(skillsdir, copy=True, force=False)
    assert usernotes.is_file()
    assert "notes.md" in capsys.readouterr().err

    assert install_skill(skillsdir, copy=True, force=True)
    assert "notes.md" in capsys.readouterr().err


def test_installskill_refuses_a_destination_in_the_skill_of_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comparison of the path text let rmtree delete the skill of the package, or let the command write into it."""
    from artistools.installskill import install_skill

    packageskill = tmp_path / "package" / "skills" / "artis-data"
    packageskill.mkdir(parents=True)
    (packageskill / "SKILL.md").write_text("the skill of the package", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: packageskill)

    # on a case-insensitive file system, e.g. the macOS default, SKILLS is a different spelling of the same folder
    for skillsdir in (packageskill, packageskill.parent, packageskill.parent.with_name("SKILLS")):
        if skillsdir.exists():
            with pytest.raises(SystemExit):
                install_skill(skillsdir, copy=True, force=True)
    assert [path.name for path in packageskill.iterdir()] == ["SKILL.md"]

    # the destination tmp/artis-data holds the package, and rmtree of the destination deleted the skill of the package
    nestedskill = tmp_path / "artis-data" / "artistools" / "data" / "skills" / "artis-data"
    nestedskill.mkdir(parents=True)
    (nestedskill / "SKILL.md").write_text("the skill of the package", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: nestedskill)
    with pytest.raises(SystemExit):
        install_skill(tmp_path, copy=True, force=True)
    assert (nestedskill / "SKILL.md").is_file()


def test_installskill_makes_no_link_into_the_uv_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With uvx, artistools runs from the uv cache.

    When uv deletes the cache, the link breaks and no message tells the user.
    """
    from artistools.installskill import install_skill

    cachedskill = tmp_path / "uv" / "archive-v0" / "abc123" / "artistools" / "data" / "skills" / "artis-data"
    cachedskill.mkdir(parents=True)
    (cachedskill / "SKILL.md").write_text("the skill in the cache", encoding="utf-8")
    monkeypatch.setattr("artistools.installskill.get_skill_folder", lambda: cachedskill)

    with pytest.raises(SystemExit):
        install_skill(tmp_path / "skills", copy=False, force=False)
    assert install_skill(tmp_path / "skills", copy=True, force=False)
    assert not (tmp_path / "skills" / "artis-data").is_symlink()


def test_artis_data_skill_names_each_function_that_exists() -> None:
    """A rename needs no shim in artistools, thus a stale name in the skill gives an agent an error and no warning."""
    from artistools.installskill import get_skill_folder

    skilltext = "\n".join(path.read_text(encoding="utf-8") for path in sorted(get_skill_folder().glob("*.md")))
    names = sorted(set(re.findall(r"\bat\.([a-z_]+(?:\.[a-z_]+)*)\(", skilltext)))
    assert len(names) > 10

    stalenames = []
    for name in names:
        try:
            function = functools.reduce(getattr, name.split("."), at)
        except AttributeError:
            stalenames.append(name)
        else:
            if not callable(function):
                stalenames.append(name)
    assert not stalenames, f"SKILL.md names functions that artistools does not have: {stalenames}"


def test_artis_data_skill_summary_examples_match_the_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The skill shows an example of each summary line, and a change of the format made the example incorrect.

    A number of an example matches any number, "mymodel" matches any model name, and "..." matches any text.
    """
    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    section = skilltext.split("one summary line for each drawn series.", 1)[1].split("A line that starts with", 1)[0]
    examples = [" ".join(span.split()) for span in re.findall(r"`([^`]+)`", section) if not span.startswith("-")]
    assert len(examples) == 5, examples

    numberpattern = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"

    def get_example_pattern(example: str) -> str:
        wordpatterns = []
        for word in example.split():
            if word.startswith("..."):
                wordpatterns.append(".*" + re.escape(word.removeprefix("...")))
            elif word.startswith("mymodel"):
                wordpatterns.append(".+" + re.escape(word.removeprefix("mymodel")))
            else:
                parts = re.split(f"({numberpattern})", word)
                wordpatterns.append(
                    "".join(numberpattern if index % 2 else re.escape(part) for index, part in enumerate(parts))
                )
        return r"\s+".join(wordpatterns)

    at.estimators.plot(argsraw=[], modelpath=modelpath_classic_3d, outputfile=tmp_path, timedays=5, plotlist=[["Te"]])
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], filter=["B"], timemin=250, timemax=300, outputfile=tmp_path)
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], frompackets=True, outputfile=tmp_path)
    at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", outputfile=tmp_path / "spec.pdf")
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        timedays="5",
        showemission=True,
        showabsorption=True,
        outputfile=tmp_path / "em.pdf",
    )
    output = capsys.readouterr().out
    unmatched = [example for example in examples if re.search(get_example_pattern(example), output) is None]
    assert not unmatched, f"SKILL.md gives summary lines that the commands do not print: {unmatched}"


def test_artis_data_skill_python_ignores_a_module_in_the_model_folder(tmp_path: Path) -> None:
    """The command of the skill runs Python in a model folder, and "python -" imported an artistools.py of that folder.

    A model from a different person can hold such a file, thus the agent ran it with the permission to read data.
    """
    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    command = next(block for block in re.findall(r"```sh\n(.*?)```", skilltext, re.DOTALL) if "ATPY=" in block)
    command = command.replace("...", "print(at.__file__)")
    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    marker = tmp_path / "ran.txt"
    (modelfolder / "artistools.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")

    # the command of pip install --user is in ~/.local/bin, and that folder holds no python. Thus the command of this
    # test is in a folder of its own. The editable installation of a checkout can need PYTHONPATH, and -P keeps it
    commandfolder = tmp_path / "bin"
    commandfolder.mkdir()
    # a console script does not put the working folder on sys.path, and -m does without -P
    (commandfolder / "artistools").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -P -m artistools "$@"\n', encoding="utf-8"
    )
    (commandfolder / "artistools").chmod(0o755)
    env = os.environ | {
        "PATH": os.pathsep.join([str(commandfolder), os.environ.get("PATH", "")]),
        "PYTHONPATH": str(Path(at.__file__).resolve().parent.parent),
    }
    bashpath = shutil.which("bash")
    assert bashpath is not None
    # the command is the fixed text of the skill, which the agent runs in a shell
    completed = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [bashpath, "-c", command], cwd=modelfolder, env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert Path(completed.stdout.strip()).resolve() == Path(at.__file__).resolve()


def test_artis_data_skill_examples_run_on_the_test_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each example of the skill must run, because a change of a reader can stop it with no other failed test.

    The examples use 10, 12, and 50 days, but the test model ends at 7.9 days. Thus the test uses 5 days and 6 days.
    """
    import runpy

    from artistools.installskill import get_skill_folder

    examples = [
        (path.name, skilltext, example)
        for path in sorted(get_skill_folder().glob("*.md"))
        for skilltext in [path.read_text(encoding="utf-8")]
        for example in re.finditer(r"```python\n(.*?)```", skilltext, re.DOTALL)
    ]
    assert len(examples) > 3

    monkeypatch.chdir(modelpath_classic_3d)
    for index, (filename, skilltext, example) in enumerate(examples):
        code = (
            example
            .group(1)
            .replace("timedays(modelpath, 10)", "timedays(modelpath, 5)")
            .replace("timedays(modelpath, 50)", "timedays(modelpath, 5)")
            .replace("timemin=10, timemax=12", "timemin=5, timemax=6")
        )
        scriptpath = tmp_path / f"example{index}.py"
        scriptpath.write_text(code, encoding="utf-8")
        try:
            runpy.run_path(str(scriptpath), run_name="__main__")
        except Exception as exc:
            exc.add_note(f"in the example at line {skilltext.count(chr(10), 0, example.start()) + 1} of {filename}")
            raise


def test_artis_data_skill_compares_ionisation_processes_on_a_model_with_both_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 3D test model has no non-thermal rates, thus the test of all the examples runs only one branch."""
    import runpy

    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "SKILL.md").read_text(encoding="utf-8")
    example = next(block for block in re.findall(r"```python\n(.*?)```", skilltext, re.DOTALL) if "gamma_NT" in block)
    scriptpath = tmp_path / "ionisationprocesses.py"
    scriptpath.write_text(example.replace("timedays(modelpath, 50)", "timedays(modelpath, 300)"), encoding="utf-8")

    monkeypatch.chdir(modelpath)
    runpy.run_path(str(scriptpath), run_name="__main__")
    output = capsys.readouterr().out
    # non-thermal ionisation gives most ionisations of Fe II at 300 days, and photoionisation most of Fe I
    rows = {line.split()[1]: line.split() for line in output.splitlines() if line.startswith("│ Fe_")}
    assert float(rows["Fe_II"][-2]) > 0.9
    assert float(rows["Fe_I"][-2]) < 0.1


def test_artis_data_skill_balances_ionisation_and_recombination_on_a_model_with_both_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Alpha_R*nne_<ion> is the recombination to the ion below, thus an incorrect pair of ions gives no balance.

    The 3D test model has no recombination rates, thus the test of all the examples runs only the branch that prints
    this.
    """
    import runpy

    from artistools.installskill import get_skill_folder

    skilltext = (get_skill_folder() / "recipes.md").read_text(encoding="utf-8")
    example = next(block for block in re.findall(r"```python\n(.*?)```", skilltext, re.DOTALL) if "Alpha_R" in block)
    scriptpath = tmp_path / "ionisationbalance.py"
    scriptpath.write_text(example.replace("timedays(modelpath, 50)", "timedays(modelpath, 300)"), encoding="utf-8")

    monkeypatch.chdir(modelpath)
    runpy.run_path(str(scriptpath), run_name="__main__")
    output = capsys.readouterr().out
    ratios = {line.split()[1]: float(line.split()[-2]) for line in output.splitlines() if line.startswith("│ Fe_")}
    assert set(ratios) == {"Fe_I", "Fe_II", "Fe_III", "Fe_IV"}
    assert all(ratio == pytest.approx(1.0, rel=0.05) for ratio in ratios.values())


def test_writecomparisondata_edep_takes_the_next_timestep(tmp_path: Path) -> None:
    """ARTIS writes the deposition of timestep n into the estimators of timestep n + 1.

    The edep file took the row of timestep n, thus it gave the deposition of the timestep before, and it
    disagreed with the lbol_edep file of the same export by about 5 % at 200 days.
    """
    at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[50])

    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    nextrow = at.scan_estimators(modelpath=modelpath, timestep=(51,)).select("total_dep").collect()
    assert float(datalines[0].split()[1]) == pytest.approx(nextrow.item(), rel=1e-4, abs=0.0)

    # ARTIS wrote no timestep after the last one, thus the file gives zero there and the export goes on
    lasttimestep = len(at.get_timestep_times(modelpath)) - 1
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[lasttimestep], force=True
    )
    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    assert float(datalines[0].split()[1]) == 0.0


def test_linefluxes_refuse_overlapping_time_bins() -> None:
    """Overlapping bins gave each shared interval to the last bin, and each bin still divided by its full width.

    Thus the bins [100, 200) and [150, 250) gave the first bin the packets of 100 to 150 days alone.
    """
    with pytest.raises(ValueError, match="time bins overlap"):
        at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 150.0], [200.0, 250.0])

    # bins that meet at an edge are no overlap, in either order
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [100.0, 200.0], [200.0, 300.0])
    at.plotlinefluxes.get_timebin_expr(pl.col("t_arrive_d"), [200.0, 100.0], [300.0, 200.0])


def test_linefluxes_timebins_in_reverse_order_take_the_order_in_time() -> None:
    """The bins in reverse order must give each time the same bin as the bins in the order of time.

    The upper edge went to the last bin of the list, and a rounding gap closed only between neighbours of the list.
    """
    from artistools.plotlinefluxes import get_timebin_expr

    dftimes = pl.DataFrame({"t": [100.0, 150.0, 200.0, 250.0, 300.0]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [200.0, 100.0], [300.0, 200.0])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        0,
    ]

    dftimes = pl.DataFrame({"t": [1.0, 1.999995, 2.0, 3.0, 3.1]})
    assert dftimes.select(get_timebin_expr(pl.col("t"), [2.0, 1.0], [3.0, 1.99999])).to_series().to_list() == [
        1,
        1,
        0,
        0,
        None,
    ]


def test_linefluxes_emitting_regions_give_one_file_for_each_time_bin(tmp_path: Path) -> None:
    """Each time bin of the emitting regions has its own figure, thus two bins can overlap.

    The command wrote each figure to the one name that -o gives, thus only the last one stayed.
    """
    # the test data holds no floers_te_nne.json, thus one reference point stands in for it
    refdata = (["5"], np.array([5.0]), [{"ne": [5.0], "temp": [5000.0]}])
    with mock.patch.object(at.plotlinefluxes, "read_te_nne_refdata", return_value=refdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            use_lastemissiontype=True,
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
        )

    assert sorted(path.name for path in tmp_path.glob("*.pdf")) == ["emreg_5.0d.pdf", "emreg_6.0d.pdf"]
    # the model has no packets in the features, and the table left out the reference curve, thus it gave no file
    dfpoints = pl.read_csv(tmp_path / "emreg_5.0d.txt", separator=" ", null_values="nan", comment_prefix="#")
    assert dfpoints.select("modelpath", "label", "feature", "log10nne", "Te").rows() == [
        (None, "Flörs+2020 +5d", "reference", 5.0, 5000.0)
    ]


def test_linefluxes_emitting_regions_write_the_drawn_points(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--write_data wrote no file for the emitting regions. The 3D test model has no packets in these features.

    --print-data --output-format json printed one JSON document for each time bin, which a parser of JSON refused.
    """
    import collections

    emdata = collections.defaultdict(lambda: {"em_log10nne": np.array([5.0, 6.0]), "em_Te": np.array([4e3, 5e3])})
    with mock.patch.object(at.plotlinefluxes, "get_emitting_regions_data", return_value=emdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            norefdata=True,
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
            print_data=True,
            output_format="json",
        )

    # a direct call of main prints the progress messages to the standard output, thus the JSON starts at "{"
    stdout = capsys.readouterr().out
    document = json.loads(stdout[stdout.index("\n{") :])
    assert {row["figure_time_days"] for row in document["data"]} == {5.0, 6.0}

    # --output-format json also gives each data file the JSON format
    for tmid in (5.0, 6.0):
        dfpoints = pl.DataFrame(json.loads((tmp_path / f"emreg_{tmid:.1f}d.json").read_text(encoding="utf-8"))["rows"])
        assert dfpoints.columns == ["modelpath", "label", "feature", "time_days", "log10nne", "Te"]
        assert dfpoints["time_days"].unique().to_list() == [tmid]
        assert dfpoints["log10nne"].to_list()[:2] == [5.0, 6.0]


def test_linefluxes_emitting_regions_keep_two_models_with_one_label_apart(tmp_path: Path) -> None:
    """The table named each model by its name alone, thus two models with one label gave rows with one key."""
    import collections

    # a second path to the same model folder gives a second model that has the line list
    secondpath = tmp_path / "secondmodel"
    secondpath.symlink_to(modelpath_classic_3d, target_is_directory=True)
    emdata = collections.defaultdict(lambda: {"em_log10nne": np.array([5.0]), "em_Te": np.array([4e3])})
    with mock.patch.object(at.plotlinefluxes, "get_emitting_regions_data", return_value=emdata):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d, secondpath],
            label=["same", "same"],
            plotemittingregions=True,
            norefdata=True,
            timebins_tstart=[4.0],
            timebins_tend=[6.0],
            outputfile=tmp_path / "emreg.pdf",
            write_data=True,
        )

    dfpoints = pl.read_csv(tmp_path / "emreg.txt", separator=" ", null_values="nan", comment_prefix="#")
    assert set(dfpoints["label"].to_list()) == {"same"}
    assert set(dfpoints["modelpath"].to_list()) == {str(modelpath_classic_3d), str(secondpath)}


def test_linefluxes_emitting_regions_take_the_timestep_of_the_thermal_emission() -> None:
    """The cell and the timestep of the emitting regions must come from the same event.

    The cell came from the last thermal emission and the timestep from the last interaction, thus a
    packet that scattered in a later timestep read the estimators of the wrong time.
    """
    dfpackets = pl.LazyFrame({
        "em_timestep": [52, 52, 3],
        "emtrue_timestep": [50, -1, 3],
        "emtrue_modelgridindex": [40, 40, 2],
    })

    assert at.plotlinefluxes.get_emission_columns(dfpackets, "trueemissiontype") == (
        "emtrue_timestep",
        "emtrue_modelgridindex",
    )

    # the last interaction keeps its own timestep and cell
    assert at.plotlinefluxes.get_emission_columns(dfpackets, "emissiontype") == ("em_timestep", "em_modelgridindex")

    # an old packets file has no trueem_time, thus the cell and the timestep both come from the last interaction
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_timestep"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )

    # a 3D packets file with trueem_time and no trueem_posx gives no cell of the thermal emission. The join on
    # emtrue_modelgridindex then stopped with ColumnNotFoundError
    assert at.plotlinefluxes.get_emission_columns(dfpackets.drop("emtrue_modelgridindex"), "trueemissiontype") == (
        "em_timestep",
        "em_modelgridindex",
    )


def test_linefluxes_emitting_regions_without_the_reference_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The package holds no floers_te_nne.json, thus the message names --norefdata, which plots the model alone.

    Each figure of a time bin has its own legend, and a label that holds LaTeX braces stays as it is.
    str.format read the braces of such a label as a field and raised KeyError, and only the last figure had a legend.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.plotlinefluxes.main(argsraw=[], modelpath=[modelpath_classic_3d], plotemittingregions=True)
    assert "--norefdata" in capsys.readouterr().err

    # the packets give no position of the thermal emission, thus the cell comes from the last interaction
    with (
        mock.patch.object(at.plotlinefluxes, "set_legend") as mocklegend,
        mock.patch.object(
            at.plotlinefluxes, "plot_nne_te_points", side_effect=at.plotlinefluxes.plot_nne_te_points
        ) as mockpoints,
    ):
        at.plotlinefluxes.main(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotemittingregions=True,
            norefdata=True,
            label=["$M_{\\rm ej}$ = 1 at {timeavg:.0f}d"],
            timebins_tstart=[4.0, 5.0],
            timebins_tend=[6.0, 7.0],
            outputfile=tmp_path / "emreg.png",
        )

    assert sorted(path.name for path in tmp_path.glob("*.png")) == ["emreg_5.0d.png", "emreg_6.0d.png"]
    assert mocklegend.call_count == 2, "each figure needs a legend"
    assert [callargs.args[1] for callargs in mockpoints.call_args_list] == [
        "$M_{\\rm ej}$ = 1 at 5d",
        "$M_{\\rm ej}$ = 1 at 6d",
    ]


def test_linefluxes_label_fields_keep_latex_braces() -> None:
    """Only the fields timeavg and modeltag change, thus a LaTeX brace of a label stays."""
    from artistools.plotlinefluxes import format_label_fields

    assert (
        format_label_fields("$M_{\\rm ej}$ {modeltag} {timeavg:.0f}d", timeavg=250.4, modeltag="w7")
        == "$M_{\\rm ej}$ w7 250d"
    )


def make_viewer(kind: str, tokens: "Sequence[str]") -> t.Any:
    """Return the viewer of a command, with a canvas and no window, after its first plot."""
    module = importlib.import_module(f"artistools.{kind}.interactive")
    viewerclass = next(
        value for name, value in vars(module).items() if name.endswith("Viewer") and isinstance(value, type)
    )
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = viewerclass(tokens, fig)
    assert draw_values(viewer) is None
    return viewer


def test_describe_counts_the_cells_and_the_rank_files_of_each_run_folder(capsys: pytest.CaptureFixture[str]) -> None:
    """The summary must count the cells with matter and the rank files of all the run folders, e.g. job0 and job1."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["describe", str(modelpath_classic_3d), "--quiet"])
    output = capsys.readouterr().out
    assert "1000 cells, 695 cells with matter" in output
    assert "timesteps: 30, from 3.000 to 8.000 days" in output
    assert "run folders with estimators: job0, job1" in output
    assert "estimators: 4 rank files" in output
    assert "nlte: 0 rank files" in output
    assert "Fe I to Fe V" in output
    assert "direction bins: 100 tables in light_curve_res.out = 10 cos θ bins times 10 φ bins" in output
    # the model folder of the classic-mode 3D test holds artisoptions.h and no folder artis
    assert "temperature limits: MINTEMP = 3500.; MAXTEMP = 140000." in output
    # an old ARTIS gave a different name to an option, e.g. NT_ON and not NT_SCHEME
    assert "non-thermal: NT_ON = false" in output
    assert "ARTIS of run folder job0: v2023.10-106-gfcb7e2c (branch develop)" in output


def test_describe_gives_the_effect_of_the_physics_options() -> None:
    """The ion balance of a kilonova model with FORCE_SAHA_ION_BALANCE is Saha at Te, and not the rate balance.

    The effect of the options must follow the code of ARTIS, thus each case here comes from ltepop.cc and
    update_packets.cc.
    """
    from artistools.describemodel import get_option_effects

    kilonova = {
        "ION_NLEVELS_EXCITED_NLTE": "(int element_z, int ionstage) { return 0; }",
        "FORCE_SAHA_ION_BALANCE": "(int element_z) { return true; }",
        "NT_SCHEME": "NonThermalScheme::NT_OFF",
        "PARTICLE_THERMALISATION_SCHEME": "ParticleThermalisationScheme::BARNES",
    }
    assert get_option_effects("NLTE", kilonova) == [
        "the ion balance of each element is the Saha equation at Te",
        "no level is NLTE, thus the levels of an ion follow the Boltzmann distribution",
    ]
    assert get_option_effects("non-thermal", kilonova) == [
        "ARTIS gives no non-thermal ionisation and no non-thermal excitation"
    ]
    thermalisation = get_option_effects("thermalisation", kilonova)
    assert "belong to the time step before their row" in thermalisation[0]
    assert "Barnes et al. (2016)" in thermalisation[1]

    # a condition in the function of an option gives no claim about its effect
    nebular = {
        "ION_NLEVELS_EXCITED_NLTE": "(int element_z, int ionstage) { if (element_z == 26) { return 80; } return 0; }",
        "FORCE_SAHA_ION_BALANCE": "(int element_z) { if (element_z == 1) { return true; } return false; }",
        "INSTANT_PARTICLE_DEPOSITION": "true",
    }
    nebulareffects = get_option_effects("NLTE", nebular)
    assert "the Saha equation at Te to the ion balance of some elements" in nebulareffects[0]
    assert "rate equations of the NLTE levels" in nebulareffects[1]
    assert "analytic" in get_option_effects("thermalisation", nebular)[0]


def test_describe_names_the_packets_files_of_an_earlier_run_with_more_ranks(tmp_path: Path) -> None:
    """The count took the ranks of input.txt alone, thus it said nothing of the packets of an earlier run with more."""
    from artistools.describemodel import get_output_lines

    shutil.copy(modelpath_classic_3d / "input.txt", tmp_path / "input.txt")
    nprocs = at.misc.get_nprocs(tmp_path)
    (tmp_path / "100.slurm").mkdir()
    for rank in [*range(nprocs), nprocs + 1]:
        (tmp_path / "100.slurm" / f"packets00_{rank:04d}.out").write_text("", encoding="utf-8")

    lines = get_output_lines(tmp_path)
    assert any(line.startswith(f"packets: {nprocs} rank files") for line in lines)
    assert "  1 more packets files have a rank above the ranks of input.txt" in "\n".join(lines)

    # the search stopped at the model folder when it held every rank, thus the count missed the run folder
    for rank in range(nprocs):
        (tmp_path / "100.slurm" / f"packets00_{rank:04d}.out").rename(tmp_path / f"packets00_{rank:04d}.out")
    lines = get_output_lines(tmp_path)
    assert "  1 more packets files have a rank above the ranks of input.txt" in "\n".join(lines)

    # a second earlier run with the same rank gave one rank number, and the count gave one file for the two
    (tmp_path / "200.slurm").mkdir()
    (tmp_path / "200.slurm" / f"packets00_{nprocs + 1:04d}.out").write_text("", encoding="utf-8")
    assert "  2 more packets files have a rank above the ranks of input.txt" in "\n".join(get_output_lines(tmp_path))


def test_run_versions_follow_the_runs_and_leave_out_a_copied_log(tmp_path: Path) -> None:
    """A copy of the log of an earlier run in the model folder came last, thus the catalog gave its old version."""
    from artistools.catalog import get_version_columns
    from artistools.describemodel import get_run_versions

    timestepline = "timestep {}: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n"
    for foldername, version, firsttimestep in [("200.slurm", "newbuild", 10), ("100.slurm", "oldbuild", 0)]:
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "output_0-0.txt").write_text(
            f"git version: {version}\n{timestepline.format(firsttimestep)}", encoding="utf-8"
        )
    shutil.copy(tmp_path / "100.slurm" / "output_0-0.txt", tmp_path / "output_0-0.txt")

    assert list(get_run_versions(tmp_path)) == ["100.slurm", "200.slurm"]
    assert get_version_columns(tmp_path) == {"artis_version": "newbuild"}

    # the log of the last run gives no version, and the catalog gave the version of the run before it
    (tmp_path / "300.slurm").mkdir()
    (tmp_path / "300.slurm" / "output_0-0.txt").write_text(timestepline.format(20), encoding="utf-8")
    assert list(get_run_versions(tmp_path)) == ["100.slurm", "200.slurm", "300.slurm"]
    assert get_version_columns(tmp_path) == {"artis_version": None}


def test_describe_counts_the_direction_bins_of_the_file(tmp_path: Path) -> None:
    """ARTIS sets MABINS when it compiles, and the summary gave the 100 bins of artistools for a file of 2 bins."""
    from artistools.describemodel import get_selection_lines

    (tmp_path / "light_curve_res.out").write_text("1.0 1.0 1.0\n2.0 1.0 1.0\n" * 2, encoding="utf-8")
    (selectionline,) = [line for line in get_selection_lines(tmp_path) if "direction bins" in line]
    assert "direction bins: 2 tables in light_curve_res.out, and artistools gives the angles of 100 bins" in (
        selectionline
    )
    assert "selects bin N of 0 to 1," in selectionline


def test_describe_finds_the_light_curve_in_a_run_folder(tmp_path: Path) -> None:
    """A run on a cluster writes light_curve.out into its run folder, and the summary searched only the model folder.

    The light curve reader searches the run folders, and scan_deposition reads only the model folder.
    """
    from artistools.describemodel import get_output_lines

    modelfolder = tmp_path / "model"
    (modelfolder / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "12411848.slurm" / "light_curve.out").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    (modelfolder / "12411848.slurm" / "deposition.out").write_text("1.0 2.0 3.0\n", encoding="utf-8")

    lines = get_output_lines(modelfolder)
    assert "light_curve.out: 12411848.slurm/light_curve.out (read with: artistools plotlightcurves)" in lines
    assert "deposition.out: none" in lines

    # a run can join the estimator files of its ranks into one file, and the summary counted the rank files alone
    allranksmodel = tmp_path / "allranksmodel"
    (allranksmodel / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", allranksmodel / "input.txt")
    (allranksmodel / "12411848.slurm" / "estimators_allranks.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 3000 W 1 TJ 2000 nne 1.0e5\n\n", encoding="utf-8"
    )
    lines = get_output_lines(allranksmodel)
    assert any(
        line.startswith("estimators: 0 rank files, 1 files of all ranks (estimators_allranks.out) (read with:")
        for line in lines
    )


@pytest.mark.parametrize(
    "optionname", ["MULTIBIN_RADFIELD_FIRST_TIMESTEP", "FIRST_BINNED_RADFIELD_TIMESTEP", "FIRST_NLTE_RADFIELD_TIMESTEP"]
)
def test_radfield_first_timestep_takes_each_name_of_the_option(tmp_path: Path, optionname: str) -> None:
    """ARTIS renamed the option two times on 2026-10-16, thus a current model gave no first timestep of the bin fits."""
    (tmp_path / "artisoptions.h").write_text(f"constexpr int {optionname} = 6;\n", encoding="utf-8")
    assert at.radfield.core.get_multibin_radfield_first_timestep(tmp_path) == (6, optionname)


def test_corehours_counts_the_link_to_a_log_one_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The model folder holds a link to the rank-0 log of the last run, and the sum counts that run one time.

    The second run stopped early and has no last line with its core hours, thus its value is an estimate from the
    timestamps: 1 hour * 4 processes * 1 thread.
    """
    import json

    import artistools.__main__

    modelfolder = tmp_path / "model"
    nprocsline = "Simulation propagates 100 packets on this rank (total 400 with nprocs 4)"
    logtext_of_runfolder = {
        "job_from_ts0000": (
            f"2026-10-15T09:00:00Z sn3d launched\n2026-10-15T09:00:00Z {nprocsline}\n"
            "2026-10-15T11:00:00Z sn3d finished (job: pktprop ts 0 to ts 9 grid-preprop, 2.000 wallclock hours"
            " * 4 processes * 1 threads = 8.000 core hours)\n"
        ),
        "job_from_ts0010": (
            f"2026-10-15T12:00:00Z sn3d launched\n2026-10-15T12:00:00Z {nprocsline}\n"
            "2026-10-15T12:30:00Z timestep 10: start update_packets\n"
            "2026-10-15T13:00:00Z timestep 10: time after update grid on all processes\n"
        ),
    }
    for runfolder, logtext in logtext_of_runfolder.items():
        (modelfolder / runfolder).mkdir(parents=True)
        (modelfolder / runfolder / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    (modelfolder / "output_0-0.txt").symlink_to(Path("job_from_ts0010", "output_0-0.txt"))

    artistools.__main__.main(argsraw=["corehours", str(modelfolder), "--json"])
    corehours = json.loads(capsys.readouterr().out)
    assert [job["duplicate"] for job in corehours["jobs"]] == [False, False, True]
    assert [job["estimate"] for job in corehours["jobs"]] == [False, True, False]
    # the JSON gives the inputs of the estimate
    assert np.isclose(corehours["jobs"][1]["wallclock_hours"], 1.0)
    assert corehours["jobs"][1]["tasks"] == 4
    assert np.isclose(corehours["total_core_hours"], 12.0)
    assert np.isclose(corehours["wallclock_hours"], 3.0)
    assert corehours["tasks"] == 4
    assert corehours["timestep_ranges"] == [[0, 9], [10, 10]]

    artistools.__main__.main(argsraw=["corehours", str(modelfolder)])
    table = capsys.readouterr().out
    assert "ignored duplicate log" in table
    assert "warning: sn3d did not finish. Estimated: (job: pktprop ts 10 to ts 10 grid-preprop" in table
    assert "CPU time:             12.000  core-h" in table

    # a damaged log gives no core hours and no task count, thus the sum is not complete and the tasks are not known.
    # The run folder comes before the target of the link, thus the search for the copy also reads the damaged log
    (modelfolder / "job_from_ts0005").mkdir()
    (modelfolder / "job_from_ts0005" / "output_0-0.txt.gz").write_bytes(b"not gzip data")
    # a run folder with no rank-0 log and a last line with no number also give a run with no core hours
    (modelfolder / "job_from_ts0030").mkdir()
    (modelfolder / "job_from_ts0030" / "output_1-0.txt").write_text(
        logtext_of_runfolder["job_from_ts0000"], encoding="utf-8"
    )
    (modelfolder / "job_from_ts0040").mkdir()
    (modelfolder / "job_from_ts0040" / "output_0-0.txt").write_text("sn3d finished = core hours\n", encoding="utf-8")
    (modelfolder / "light_curve.out").write_text("", encoding="utf-8")
    artistools.__main__.main(argsraw=["corehours", str(modelfolder / "light_curve.out"), "--json"])
    captured = capsys.readouterr()
    corehours = json.loads(captured.out)
    assert "output_0-0.txt.gz is unreadable" in captured.err
    assert "job_from_ts0030 has the logs of other ranks and no rank-0 log" in captured.err
    assert "job_from_ts0040/output_0-0.txt has a line with values that the command cannot read" in captured.err
    assert [job["core_hours"] is None for job in corehours["jobs"]] == [False, True, False, True, True, True]
    assert corehours["total_core_hours"] is None
    assert corehours["tasks"] is None

    artistools.__main__.main(argsraw=["corehours", str(modelfolder)])
    table = capsys.readouterr().out
    assert "warning: The log is missing, empty, or unreadable." in table
    assert "warning: Some logs give no core hours. The CPU time omits their runs." in table
    assert "Tasks:" not in table


def test_logsummary_reads_a_run_that_stopped_before_its_estimators(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that stops early writes a log and no estimator file, and its log holds the error that stopped it."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "12411848.slurm" / "output_0-0.txt").write_text(
        "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n",
        encoding="utf-8",
    )

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan", comment_prefix="#")
    assert dfsummary["level"].to_list() == ["error"]
    assert dfsummary["first_timestep"].item() == 0


def test_logsummary_all_ranks_reads_the_ranks_of_each_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A restart can use fewer ranks, and -ranks all read only the ranks of input.txt, which left out an error."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    for runfolder, rank, text in [
        ("100.slurm", 7, "[error] cell 3 has no matter\n"),
        ("200.slurm", 0, "[warning] cell 4 is cold\n"),
    ]:
        (modelfolder / runfolder).mkdir()
        (modelfolder / runfolder / f"output_{rank}-0.txt").write_text(
            f"timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n{text}", encoding="utf-8"
        )

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-ranks", "all", "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan", comment_prefix="#")
    assert sorted(dfsummary["level"].to_list()) == ["error", "warning"]


def test_logsummary_cell_reads_the_rank_of_the_cell_in_each_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A restart can use more ranks, and the rank of a cell in the current input.txt is a different cell in a run.

    With 1000 cells, the blocks of ARTIS give cell 999 to rank 1 of 2 ranks, and to rank 3 of 4 ranks.
    """
    import artistools.__main__

    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "model.txt").write_text("1000\n", encoding="utf-8")
    header = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n"
    for runfolder, rankcount, cellrank in [("100.slurm", 2, 1), ("200.slurm", 4, 3)]:
        (modelfolder / runfolder).mkdir()
        for rank in range(rankcount):
            # ARTIS writes the MPI layout near the start of each log
            mpiline = f"  rank {rank} of [0..{rankcount - 1}] in MPI_COMM_WORLD\n"
            message = "[error] cell 999 has no matter\n" if rank == cellrank else "[warning] cell 5 is cold\n"
            (modelfolder / runfolder / f"output_{rank}-0.txt").write_text(mpiline + header + message, encoding="utf-8")

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["level"].to_list() == ["error"]
    assert dfsummary["count"].item() == 2

    # the names of the logs gave 2 ranks when the logs of ranks 2 and 3 were missing, and then rank 1 took cell 999
    for rank in (2, 3):
        (modelfolder / "200.slurm" / f"output_{rank}-0.txt").unlink()
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    captured = capsys.readouterr()
    assert pl.read_csv(io.StringIO(captured.out), separator=" ", null_values="nan")["count"].item() == 1
    # the summary left out the run of 200.slurm with no message
    assert "200.slurm has no log of the ranks 3" in captured.err

    # a folder that gives no number of ranks gives a warning, and the summary leaves out its logs
    (modelfolder / "300.slurm").mkdir()
    (modelfolder / "300.slurm" / "output_0-0.txt").write_text(header + "[error] cell 999 is odd\n", encoding="utf-8")
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "-modelgridindex", "999", "--quiet"])
    captured = capsys.readouterr()
    assert "300.slurm has no modelgridrankassignments.out" in captured.err
    assert pl.read_csv(io.StringIO(captured.out), separator=" ", null_values="nan")["count"].item() == 1


def test_logsummary_log_that_cannot_be_read_is_not_a_copy(tmp_path: Path) -> None:
    """A log with no read permission stopped the search for a copy, and the reader of that log reports the problem."""
    from artistools.logs.core import are_copies

    modellog = tmp_path / "output_0-0.txt"
    modellog.write_text("sn3d launched\n", encoding="utf-8")
    (tmp_path / "100.slurm").mkdir()
    runlog = tmp_path / "100.slurm" / "output_0-0.txt"
    runlog.write_text("sn3d launched\n", encoding="utf-8")
    runlog.chmod(0)
    try:
        if os.access(runlog, os.R_OK):
            pytest.skip("the user can read a file with no read permission, e.g. root")
        assert not are_copies(modellog, runlog, comparetext=True)
    finally:
        runlog.chmod(0o644)


def test_logsummary_reads_a_compressed_copy_of_a_log_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A gzip copy of the log of a run in the model folder had a different name, and its messages counted twice."""
    import gzip

    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "100.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    logtext = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    (modelfolder / "100.slurm" / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    with gzip.open(modelfolder / "output_0-0.txt.gz", "wt", encoding="utf-8") as fcopy:
        fcopy.write(logtext)

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [1]


def test_logsummary_combines_the_summary_of_each_log() -> None:
    """The messages of all the logs did not fit in the memory, thus each log gives its own summary first.

    The combination must give the summary of all the messages: the sums, the bounds, the cells, and the first example.
    """
    from artistools.logs.core import get_log_summary
    from artistools.logs.core import get_logfiles
    from artistools.logs.core import read_log_messages
    from artistools.logs.core import summarise_log_messages

    logfiles = get_logfiles(modelpath_classic_3d, None)
    assert len(logfiles) > 2
    dfall = summarise_log_messages(pl.concat([read_log_messages(logfile) for logfile in logfiles]))
    pltest.assert_frame_equal(get_log_summary(logfiles), dfall)


def test_logsummary_reads_two_gzip_copies_of_a_log_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two gzip files of one text can have different bytes, e.g. another time in the header of gzip."""
    import gzip

    import artistools.__main__

    modelfolder = tmp_path / "model"
    (modelfolder / "100.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    logtext = b"timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    for folder, mtime in ((modelfolder / "100.slurm", 1), (modelfolder, 2)):
        with gzip.GzipFile(folder / "output_0-0.txt.gz", "wb", mtime=mtime) as fcopy:
            fcopy.write(logtext)
    assert (modelfolder / "output_0-0.txt.gz").read_bytes() != (
        modelfolder / "100.slurm" / "output_0-0.txt.gz"
    ).read_bytes()

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [1]


def test_log_reader_finds_a_message_on_the_line_of_a_timestep(tmp_path: Path) -> None:
    """The reader took the timestep from the line and went on to the next line, thus it left out the message."""
    from artistools.logs.core import read_log_messages

    logfile = tmp_path / "output_0-0.txt"
    logfile.write_text(
        "timestep 3: time before update grid 0.1 (tstart + tdelta/2) 2.0 days ...[error] cell 7 has no matter\n",
        encoding="utf-8",
    )
    dfmessages = read_log_messages(logfile)
    assert dfmessages.select("level", "timestep", "modelgridindex").rows() == [("error", 3, 7)]

    # the progress text names the cell, and the text after the tag does not
    logfile.write_text("Finding T_e in cell 4...[warning] the solver did not converge\n", encoding="utf-8")
    assert read_log_messages(logfile).select("level", "modelgridindex").rows() == [("warning", 4)]


def test_logsummary_keeps_identical_logs_of_two_run_folders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two retries can stop with the same error and write the same text, and the second retry counted as a copy."""
    import artistools.__main__

    modelfolder = tmp_path / "model"
    logtext = "timestep 0: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n[error] cell 3 has no matter\n"
    for foldername in ("100.slurm", "200.slurm"):
        (modelfolder / foldername).mkdir(parents=True)
        (modelfolder / foldername / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")

    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [2]

    # a local run writes its estimators and its log into the model folder, and its log is a run and not a copy
    (modelfolder / "output_0-0.txt").write_text(logtext, encoding="utf-8")
    (modelfolder / "estimators_0000.out").write_text("", encoding="utf-8")
    artistools.__main__.main(argsraw=["logsummary", str(modelfolder), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan")
    assert dfsummary["count"].to_list() == [3]


def test_logsummary_counts_each_kind_of_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """A log holds thousands of warnings of a few kinds, thus the table names each kind, its timesteps, and its cells.

    In the classic-mode 3D test, heating is bigger than cooling in many cells, thus ARTIS sets Te to MAXTEMP. The first
    timesteps are in LTE, where ARTIS solves no thermal balance.
    """
    import artistools.__main__

    artistools.__main__.main(argsraw=["logsummary", str(modelpath_classic_3d), "--quiet"])
    dfsummary = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ", null_values="nan", comment_prefix="#")
    dfmaxtemp = dfsummary.filter(pl.col("kind").str.contains("T_e forced to be MAXTEMP"))
    assert dfmaxtemp.height == 1
    assert dfmaxtemp["count"].item() > 1000
    assert dfmaxtemp["first_timestep"].item() == 5
    assert dfmaxtemp["cells"].item() > 100
    assert dfsummary["count"].is_sorted(descending=True)


def read_data_file(path: Path) -> pl.DataFrame:
    """Read a table that --write_data wrote."""
    return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_logfiles_write_data_holds_the_plotted_times(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The table of --write_data must give each line of each page: the mpi ranks and the seconds of each stage."""
    at.logs.plotlogfiles.main(
        argsraw=[], modelpath=[modelpath_classic_3d], outputfile=tmp_path / "logfiles.pdf", write_data=True
    )

    dftable = read_data_file(tmp_path / "logfiles.txt")
    assert dftable.columns == ["timestep", "mpirank", "update_grid_s", "update_packets_s", "communicate_estimators_s"]
    assert len(mockplot.call_args_list) == 3 * 30
    for call in mockplot.call_args_list:
        axis, mpirank, timetaken = call.args
        timestep = int(axis.get_title().split()[-1])
        dfpage = dftable.filter(pl.col("timestep") == timestep).sort("mpirank")
        assert dfpage["mpirank"].to_list() == list(mpirank)
        assert np.allclose(dfpage[f"{call.kwargs['label']}_s"].to_numpy(), timetaken, rtol=1e-12, atol=0)


def test_logfiles_with_no_timing_data_replace_the_table_of_an_earlier_run(tmp_path: Path) -> None:
    """Logs with no timing line returned before the data output, thus the table of an earlier run stayed."""
    tablefile = tmp_path / "logfiles.txt"
    tablefile.write_text("timestep mpirank update_grid_s\n5 0 1.0\n", encoding="utf-8")
    args = argparse.Namespace(print_data=False, write_data=True)
    at.logs.plotlogfiles.make_plot({}, tmp_path / "logfiles.pdf", args=args)
    assert "the table has no rows" in tablefile.read_text(encoding="utf-8")


@mock.patch.object(mplax.Axes, "pcolormesh", side_effect=mplax.Axes.pcolormesh, autospec=True)
def test_plotspherical_write_data_holds_the_drawn_maps(mockpcolormesh: mock.MagicMock, tmp_path: Path) -> None:
    """The table of --write_data must give each pixel of each map after the smoothing, at the drawn coordinates.

    The longitude of the map does not follow the ascending phi bins, and --phireverse changes its sign. Thus the
    table must give the edges that the axes received, and not the edges of the phi bins.
    """
    ncosthetabins, nphibins = 4, 5
    plotvars = ["luminosity", "emvelocityoverc"]
    at.packets.plotspherical.main(
        argsraw=[],
        modelpath=modelpath,
        outputfile=tmp_path / "map.pdf",
        nphibins=nphibins,
        ncosthetabins=ncosthetabins,
        plotvars=plotvars,
        gaussian_sigma=40,
        phireverse=True,
        write_data=True,
    )

    dftable = read_data_file(tmp_path / "map.txt")
    assert dftable.columns[-2:] == ["luminosity_erg_per_s", "emvelocityoverc"]
    # the colour bar draws a mesh of its own after each map
    mapcalls = [call for call in mockpcolormesh.call_args_list if call.args[0].name == "mollweide"]
    assert len(mapcalls) == len(plotvars)
    for call, colname in zip(mapcalls, ["luminosity_erg_per_s", "emvelocityoverc"], strict=True):
        _, meshgrid_phi, meshgrid_theta, data = call.args
        drawn = np.asarray(data, dtype=np.float64)
        assert drawn.shape == (ncosthetabins, nphibins + 1)
        assert np.allclose(dftable[colname].to_numpy().reshape(drawn.shape), drawn, rtol=1e-6, atol=0, equal_nan=True)

    longitude_edges = np.sort(meshgrid_phi[0])
    assert np.allclose(dftable["longitude_lower_rad"].to_numpy()[: nphibins + 1], longitude_edges[:-1][::-1])
    assert np.allclose(dftable["longitude_upper_rad"].to_numpy()[: nphibins + 1], longitude_edges[1:][::-1])
    assert np.allclose(
        dftable["costheta_lower"].to_numpy()[:: nphibins + 1], np.sin(meshgrid_theta[:-1, 0]), rtol=1e-12, atol=1e-15
    )


def test_plotviewingangles_write_data_holds_the_volume_and_the_lines(tmp_path: Path) -> None:
    """The tables of --write_data must give the density of each cell of the volume and the points of each line."""
    pytest.importorskip("plotly")
    import plotly.express as px
    import plotly.graph_objects as go

    outputfile = tmp_path / "viewingangles.html"
    with (
        mock.patch.object(go, "Volume", side_effect=go.Volume) as mockvolume,
        mock.patch.object(px, "line_3d", side_effect=px.line_3d) as mockline3d,
    ):
        at.plotviewingangles.main(argsraw=[str(modelpath_3d), "-o", str(outputfile), "--write_data"])

    dfvolume = read_data_file(tmp_path / "viewingangles.txt")
    volumekwargs = mockvolume.call_args.kwargs
    for colname, kwarg in (
        ("pos_x_mid_cm", "x"),
        ("pos_y_mid_cm", "y"),
        ("pos_z_mid_cm", "z"),
        ("rho_g_per_cm3", "value"),
    ):
        assert np.allclose(dfvolume[colname].to_numpy(), volumekwargs[kwarg], rtol=1e-12, atol=0)

    dflines = read_data_file(tmp_path / "viewingangles_directions.txt")
    dfdrawnlines = mockline3d.call_args.args[0]
    assert dflines["dirbin"].to_list() == [int(dirbin) for dirbin in dfdrawnlines["Angle-bin"]]
    for axis in ("x", "y", "z"):
        assert np.allclose(
            dflines[f"{axis}_cm"].to_numpy(), dfdrawnlines[f"{axis}_coord"].to_numpy(), rtol=1e-12, atol=0
        )


@mock.patch.object(mplax.Axes, "axhline", side_effect=mplax.Axes.axhline, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_writes_the_drawn_values(
    mockplot: mock.MagicMock, mockaxhline: mock.MagicMock, tmp_path: Path
) -> None:
    """--write_data gives the drawn bins, the moving averages, the ratios, the line counts, and the Planck mean.

    The moving average has one more point at each end of the range, thus it goes in a second table. With no moving
    average, the ratios go in the table of the bins, with one value for each bin.
    """
    commonargs = ["-modelpath", str(modelpath), "-timestep", "40", "-xmin", "3000", "-xmax", "4000", "--write_data"]
    at.lines.plotopacity.main(
        argsraw=[*commonargs, "--showplanckmean", "--showlinecount", "-o", str(tmp_path / "opac.pdf")]
    )
    dfbins = read_data_file(tmp_path / "opac.txt")
    dfaverages = read_data_file(tmp_path / "opac_movingaverage.txt")

    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    mainplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[0]]
    ratioplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[1]]
    (linecountplot,) = [call.args for call in mockplot.call_args_list if call.args[0] is axes[2]]
    columns = ["exopac", "linebinned_cap1", "linebinned"]
    # the moving average draws one more point at each end of -xmin and -xmax, and the table takes the points inside
    averagexvalues = np.asarray(mainplots[1][1], dtype=np.float64)
    isinrange = (averagexvalues >= 3000.0) & (averagexvalues <= 4000.0)
    # each opacity draws its bins, then its moving average. The line of each bin has a point at each edge and a NaN
    for column, (_, binx, biny), (_, averagexall, averageyall) in zip(
        columns, mainplots[0::2], mainplots[1::2], strict=True
    ):
        assert np.allclose(dfbins["lambda_angstroms_lower"].to_numpy(), binx[0::3], rtol=1e-12, atol=0.0)
        assert np.allclose(dfbins["lambda_angstroms_upper"].to_numpy(), binx[1::3], rtol=1e-12, atol=0.0)
        assert np.allclose(dfbins[f"{column}_cm2_per_g"].to_numpy(), biny[0::3], rtol=1e-12, atol=0.0)
        averagex, averagey = np.asarray(averagexall)[isinrange], np.asarray(averageyall)[isinrange]
        assert np.allclose(dfaverages["lambda_angstroms_bin_mid"].to_numpy(), averagex, rtol=1e-12, atol=0.0)
        assert np.allclose(dfaverages[f"{column}_movingaverage_cm2_per_g"].to_numpy(), averagey, rtol=1e-12, atol=0.0)
    for column, (_, ratioxall, ratioyall) in zip(columns[1:], ratioplots, strict=True):
        ratiox, ratioy = np.asarray(ratioxall)[isinrange], np.asarray(ratioyall)[isinrange]
        assert np.allclose(dfaverages["lambda_angstroms_bin_mid"].to_numpy(), ratiox, rtol=1e-12, atol=0.0)
        assert np.allclose(
            dfaverages[f"ratio_{column}_to_exopac"].to_numpy(), ratioy, rtol=1e-12, atol=0.0, equal_nan=True
        )
    assert np.array_equal(dfbins["linecount"].to_numpy(), linecountplot[2][0::3])
    assert np.allclose(dfbins["planckmean_exopac_cm2_per_g"], mockaxhline.call_args.args[1], rtol=1e-12, atol=0.0)

    mockplot.reset_mock()
    at.lines.plotopacity.main(argsraw=[*commonargs, "-movingaveragewidth", "0", "-o", str(tmp_path / "noaverage.pdf")])
    assert not (tmp_path / "noaverage_movingaverage.txt").exists()
    dfbins = read_data_file(tmp_path / "noaverage.txt")
    assert "planckmean_exopac_cm2_per_g" not in dfbins.columns
    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    ratioplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[1]]
    for column, (_, ratiox, ratioy) in zip(columns[1:], ratioplots, strict=True):
        assert np.allclose(dfbins["lambda_angstroms_lower"].to_numpy(), ratiox[0::3], rtol=1e-12, atol=0.0)
        assert np.allclose(
            dfbins[f"ratio_{column}_to_exopac"].to_numpy(), ratioy[0::3], rtol=1e-12, atol=0.0, equal_nan=True
        )


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
    mockplot: mock.MagicMock, mockstep: mock.MagicMock, tmp_path: Path
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


def test_catalog_gives_one_row_for_each_model_and_warns_about_a_broken_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A folder of models gives one row for each model, and a broken model gives null values and one warning."""
    from artistools.catalog import main

    (tmp_path / "group").mkdir()
    (tmp_path / "group" / "classic3d").symlink_to(modelpath_classic_3d, target_is_directory=True)
    (tmp_path / "testmodel").symlink_to(modelpath, target_is_directory=True)
    # a link back to the top folder must not make the search loop
    (tmp_path / "group" / "loop").symlink_to(tmp_path, target_is_directory=True)
    brokenfolder = tmp_path / "broken"
    brokenfolder.mkdir()
    (brokenfolder / "input.txt").write_text("not an input file\n", encoding="utf-8")
    (brokenfolder / "model.txt").write_text("not a model\n", encoding="utf-8")

    outputfile = tmp_path / "out" / "catalog.txt"
    main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    captured = capsys.readouterr()

    dfcatalog = read_data_file(outputfile)
    assert dfcatalog["modelfolder"].to_list() == ["broken", "group/classic3d", "testmodel"]
    assert "#   peak_lum_erg_per_s, peak_time_days: the largest point of light_curve.out" in outputfile.read_text(
        encoding="utf-8"
    )

    rowbroken, rowclassic, rowtestmodel = dfcatalog.iter_rows(named=True)
    assert rowbroken["cells"] is None
    assert rowbroken["timesteps"] is None
    assert captured.err.count("warning: ") == 1
    assert "Model broken in broken: the table gives nan for the cells of model.txt" in captured.err

    assert rowclassic["dimensions"] == 3
    assert rowclassic["cells"] == 1000
    assert rowclassic["timesteps"] == 30
    assert rowclassic["nlte"] is False
    assert rowclassic["nonthermal"] is False
    assert rowclassic["thermalisation"] == "INSTANT_PARTICLE_DEPOSITION=true"
    # MPKTS = 15000 packets for each of the 2 ranks of input.txt
    assert rowclassic["num_packets"] == 30000
    assert rowclassic["artis_version"] == "v2023.10-106-gfcb7e2c"

    # the test model has no artisoptions.h and no log
    assert rowtestmodel["modelname"] == "TEST MODEL"
    assert rowtestmodel["nlte"] is None
    assert rowtestmodel["artis_version"] is None

    for row, folder in ((rowclassic, modelpath_classic_3d), (rowtestmodel, modelpath)):
        # light_curve.out holds the time and the luminosity in Lsun, and an older file repeats the times in a second
        # table, thus the first table ends where the time decreases
        lightcurve = np.loadtxt(folder / "light_curve.out")
        firsttablelength = next(
            (index + 1 for index in range(len(lightcurve) - 1) if lightcurve[index + 1, 0] < lightcurve[index, 0]),
            len(lightcurve),
        )
        times, lums_lsun = lightcurve[:firsttablelength, 0], lightcurve[:firsttablelength, 1]
        assert np.isclose(row["peak_lum_erg_per_s"], lums_lsun.max() * at.constants.Lsun_to_erg_per_s, rtol=1e-6)
        assert np.isclose(row["peak_time_days"], times[lums_lsun.argmax()], rtol=1e-6)


def test_catalog_prints_the_methods_and_the_table_and_asks_before_it_overwrites_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standard output gives the method of each column and the table, and -o asks before it replaces a file."""
    from artistools.catalog import main

    (tmp_path / "testmodel").symlink_to(modelpath, target_is_directory=True)
    main(argsraw=[str(tmp_path)])
    printedlines = capsys.readouterr().out.splitlines()
    # the progress messages come before the product, which is the lines of the methods and then the table
    productstart = printedlines.index("# the method of each column:")
    tablestart = next(index for index, line in enumerate(printedlines) if line.startswith("modelname "))
    assert all(line.startswith("# ") for line in printedlines[productstart:tablestart])
    dfcatalog = pl.read_csv(io.StringIO("\n".join(printedlines[tablestart:])), separator=" ", null_values="nan")
    assert dfcatalog["modelname"].to_list() == ["TEST MODEL"]
    assert dfcatalog["modelfolder"].to_list() == ["testmodel"]

    outputfile = tmp_path / "out" / "catalog.txt"
    main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    assert "# command: " in outputfile.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    main(argsraw=[str(tmp_path), "-o", str(outputfile), "--force"])


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


def test_check_counts_the_cells_at_the_temperature_limits() -> None:
    """The worst time step and the fraction over all time steps agree with a count of the estimators."""
    from artistools.check import get_check_results

    dfresults = get_check_results(modelpath_classic_3d)
    assert dfresults.columns == ["check", "status", "detail"]
    status, detail = dfresults.filter(pl.col("check") == "temperature limits").select("status", "detail").row(0)

    dfte = at.scan_estimators(modelpath_classic_3d).select("timestep", "Te").collect()
    atlimit = dfte["Te"].is_in([3500.0, 140000.0])
    assert dfte.filter(pl.col("timestep") == 0)["Te"].eq(3500.0).sum() == 562
    overallpercent = 100 * float(atlimit.to_numpy().mean())
    assert np.isclose(overallpercent, 54.6, atol=0.05)
    assert status == "warning"
    assert "at the worst time step 0 (3.05 d), of the 695 cells with Te, 562 (80.9%) are at MINTEMP = 3500 K" in detail
    assert f"over all 30 time steps, {overallpercent:.1f}% of the values of Te are at a limit" in detail


def test_check_gives_an_error_for_a_te_that_is_not_finite() -> None:
    """A NaN Te counted as a cell with Te at no limit, thus it lowered the fraction at the limits."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({"timestep": [0, 0, 0, 1], "Te": [3500.0, 5000.0, float("nan"), float("inf")]})
    with mock.patch("artistools.check.scan_estimators", return_value=dfte):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "error"
    assert "of the 2 cells with Te, 1 (50.0%) are at MINTEMP = 3500 K" in detail
    assert "2 values of Te are NaN or infinite" in detail


def test_check_temperature_limits_warns_about_one_bad_timestep() -> None:
    """The mean over 20 time steps was 5% for one time step with each cell at MINTEMP, thus the status was ok."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({
        "timestep": [timestep for timestep in range(20) for _ in range(10)],
        "Te": [3500.0 if timestep == 0 else 5000.0 for timestep in range(20) for _ in range(10)],
    })
    with (
        mock.patch("artistools.check.scan_estimators", return_value=dfte),
        mock.patch("artistools.check.get_run_versions", return_value={"job0": {"git version": "v2023.10"}}),
    ):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "warning"
    assert "over all 20 time steps, 5.0% of the values of Te are at a limit" in detail


def test_check_names_runs_of_other_builds_for_the_temperature_limits() -> None:
    """artisoptions.h gives the limits of the current build, and a run of an earlier build can have other limits."""
    from artistools.check import check_temperature_limits

    dfte = pl.LazyFrame({"timestep": [0, 0], "Te": [3500.0, 5000.0]})
    versions = {"job0": {"git version": "v2023.10"}, "job1": {"git version": "v2026.5"}}
    with (
        mock.patch("artistools.check.scan_estimators", return_value=dfte),
        mock.patch("artistools.check.get_run_versions", return_value=versions),
    ):
        status, detail = check_temperature_limits(modelpath_classic_3d)
    assert status == "warning"
    assert "the runs used 2 ARTIS versions (v2023.10, v2026.5)" in detail


def test_check_complete_timesteps_finds_a_missing_row() -> None:
    """The last row gave the last complete time step, thus a row that a bad restart left out gave the status ok."""
    from artistools.check import check_complete_timesteps

    dfdeposition = pl.LazyFrame({"timestep": [0, 2]})
    with mock.patch("artistools.check.scan_deposition", return_value=dfdeposition):
        status, detail = check_complete_timesteps(modelpath_classic_3d)
    assert status == "warning"
    assert "deposition.out has no row for the time steps 1" in detail

    # each row through one time step past the model gave the status ok, and more complete time steps than the model
    ntimesteps = len(at.get_timestep_times(modelpath_classic_3d))
    with mock.patch(
        "artistools.check.scan_deposition", return_value=pl.LazyFrame({"timestep": list(range(ntimesteps + 1))})
    ):
        status, detail = check_complete_timesteps(modelpath_classic_3d)
    assert status == "warning"
    assert f"deposition.out has a row for time step {ntimesteps}" in detail


def test_check_restarts_takes_the_number_of_ranks_from_the_rank_0_log(tmp_path: Path) -> None:
    """A run of 4 ranks that lost one log and a run of 3 ranks each held 3 logs, and the check gave the status ok."""
    from artistools.check import check_restarts

    runfolders = []
    for foldername, rankcount in [("100.slurm", 4), ("200.slurm", 3)]:
        folder = tmp_path / foldername
        folder.mkdir()
        runfolders.append(folder)
        for rank in range(3):
            mpiline = f"  rank {rank} of [0..{rankcount - 1}] in MPI_COMM_WORLD\n"
            (folder / f"output_{rank}-0.txt").write_text(mpiline, encoding="utf-8")
    with (
        mock.patch("artistools.check.get_runfolders", return_value=runfolders),
        mock.patch("artistools.check.get_run_versions", return_value={}),
    ):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "the number of ranks changes from 4 in 100.slurm to 3 in 200.slurm" in detail
    assert "100.slurm has no log of the ranks 3 of its 4 ranks" in detail

    # logs of higher ranks of an earlier run made the count of the logs equal to the count of the ranks
    (runfolders[0] / "output_2-0.txt").unlink()
    for rank in (4, 5):
        (runfolders[0] / f"output_{rank}-0.txt").write_text("", encoding="utf-8")
    with (
        mock.patch("artistools.check.get_runfolders", return_value=runfolders[:1] * 2),
        mock.patch("artistools.check.get_run_versions", return_value={}),
    ):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "100.slurm has no log of the ranks 2-3 of its 4 ranks" in detail
    assert "100.slurm holds logs of the ranks 4-5, above its 4 ranks" in detail


def test_check_restarts_counts_a_run_that_stopped_before_its_estimators(tmp_path: Path) -> None:
    """A run that stops before it writes an estimator has a log, and the check said that the model had no restart."""
    from artistools.check import check_restarts

    for foldername, version in [("100.slurm", "v1"), ("200.slurm", "v2")]:
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "output_0-0.txt").write_text(
            f"git version: {version}\n  rank 0 of [0..0] in MPI_COMM_WORLD\n", encoding="utf-8"
        )
    # only the first run wrote estimators
    with mock.patch("artistools.check.get_runfolders", return_value=[tmp_path / "100.slurm"]):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "2 run folders with estimators or logs (100.slurm, 200.slurm)" in detail
    assert "the ARTIS version changes from v1" in detail


def test_check_restarts_gives_no_rank_count_from_the_names_of_the_logs(tmp_path: Path) -> None:
    """With no line of MPI_COMM_WORLD, the number of the logs stood for the number of ranks, also with a log missing."""
    from artistools.check import check_restarts

    runfolders = []
    for foldername in ("100.slurm", "200.slurm"):
        folder = tmp_path / foldername
        folder.mkdir()
        runfolders.append(folder)
        for rank in range(3):
            (folder / f"output_{rank}-0.txt").write_text("git version: v1\n", encoding="utf-8")
    with mock.patch("artistools.check.get_runfolders", return_value=runfolders):
        status, detail = check_restarts(tmp_path)
    assert status == "warning"
    assert "gives no line of MPI_COMM_WORLD, thus the number of its ranks is not known" in detail


def test_check_restarts_warns_about_a_run_with_no_version() -> None:
    """A run with no version left the comparison, and the agreement of the other runs then gave the status ok."""
    from artistools.check import check_restarts

    versions = {"job0": {"git version": "v2023.10-106-gfcb7e2c"}}
    with mock.patch("artistools.check.get_run_versions", return_value=versions):
        status, detail = check_restarts(modelpath_classic_3d)
    assert status == "warning"
    assert "job1 gives no ARTIS version, thus the check cannot compare the versions" in detail

    # a branch alone does not name the build, and two runs of one branch gave the status ok
    branches = {"job0": {"git branch": "main"}, "job1": {"git branch": "main"}}
    with mock.patch("artistools.check.get_run_versions", return_value=branches):
        status, detail = check_restarts(modelpath_classic_3d)
    assert status == "warning"
    assert "job0, job1 give no ARTIS version" in detail


def test_check_reads_each_log_once_and_finds_the_restarts() -> None:
    """The model folder holds a copy of the log of job1, thus the log check reads the logs of job0 and job1 alone.

    Almost each run logs warnings, thus a log with no error gives the status ok.
    """
    from artistools.check import get_check_results

    dfresults = get_check_results(modelpath_classic_3d)
    statusofcheck = dict(zip(dfresults["check"], dfresults["status"], strict=True))
    detailofcheck = dict(zip(dfresults["check"], dfresults["detail"], strict=True))
    assert statusofcheck["log messages"] == "ok"
    assert (
        "the rank-0 logs of 2 run folders (job0, job1) hold 0 errors and 7419 warnings" in detailofcheck["log messages"]
    )
    assert statusofcheck["restarts"] == "ok"
    assert "2 run folders with estimators or logs (job0, job1)" in detailofcheck["restarts"]
    assert "v2023.10-106-gfcb7e2c" in detailofcheck["restarts"]
    assert statusofcheck["complete time steps"] == "ok"
    assert "30 of 30 time steps as complete" in detailofcheck["complete time steps"]
    assert statusofcheck["packet noise"] == "not checked"
    assert detailofcheck["packet noise"] == "give --frompackets to read the packets"


def test_check_katz_ratio_agrees_with_a_direct_integration(tmp_path: Path) -> None:
    """The energy check reads only light_curve.out and deposition.out, thus a model with no estimators also has it.

    The time range of the test model starts at 3 days with hot ejecta, thus the ratio is far from 1 with no fault.
    """
    from artistools.check import get_check_results

    for filename in (
        "input.txt",
        "model.txt",
        "abundances.txt",
        "compositiondata.txt",
        "light_curve.out",
        "spec.out",
        "deposition.out",
    ):
        shutil.copy(modelpath_classic_3d / filename, tmp_path / filename)

    # light_curve.out holds a second table of 30 rows, which repeats the times
    lightcurve = np.loadtxt(modelpath_classic_3d / "light_curve.out")[:30]
    deposition = np.loadtxt(modelpath_classic_3d / "deposition.out")
    lumtime_s = lightcurve[:, 0] * at.constants.day_to_s
    deptime_s = deposition[:, 1] * at.constants.day_to_s
    lumintegral = np.trapezoid(lightcurve[:, 1] * at.constants.Lsun_to_erg_per_s * lumtime_s, x=lumtime_s)
    depintegral = np.trapezoid(deposition[:, 3] * at.constants.Lsun_to_erg_per_s * deptime_s, x=deptime_s)

    for model in (modelpath_classic_3d, tmp_path):
        dfresults = get_check_results(model)
        status, detail = dfresults.filter(pl.col("check") == "energy").select("status", "detail").row(0)
        assert status == "warning"
        assert f"∫ t L dt = {lumintegral:.3e} erg s" in detail
        assert f"∫ t Q dt = {depintegral:.3e} erg s" in detail
        assert f"L/Q = {lumintegral / depintegral:.3f}" in detail
        assert "gives a ratio far from 1 with no fault in the model" in detail

    dfresults = get_check_results(tmp_path)
    statusofcheck = dict(zip(dfresults["check"], dfresults["status"], strict=True))
    assert statusofcheck["restarts"] == "not checked"
    assert statusofcheck["temperature limits"] == "not checked"
    assert statusofcheck["log messages"] == "not checked"

    # a light curve that starts late has no NaN row for its first times, thus the check said that L covers each time
    np.savetxt(tmp_path / "light_curve.out", lightcurve[3:])
    detail = get_check_results(tmp_path).filter(pl.col("check") == "energy")["detail"].item()
    omitteddays = lightcurve[3, 0] - deposition[:, 1].min()
    assert f"leave out {omitteddays:.3g} d of L and 0 d of Q with no value or no row" in detail

    # a light curve with no row at one inner time has no NaN either, and the trapezoid rule spanned the gap
    np.savetxt(tmp_path / "light_curve.out", np.delete(lightcurve, 10, axis=0))
    status, detail = get_check_results(tmp_path).filter(pl.col("check") == "energy").select("status", "detail").row(0)
    assert status == "warning"
    assert f"light_curve.out has no row at 1 mid times of deposition.out, e.g. at {lightcurve[10, 0]:.2f} d" in detail


def test_check_that_fails_gives_an_error_row_and_the_other_checks_run() -> None:
    """An exception in one check gives a row of status error, and the other checks still run."""
    from artistools.check import get_check_results

    with mock.patch("artistools.check.check_temperature_limits", side_effect=RuntimeError("no\nestimators")):
        dfresults = get_check_results(modelpath_classic_3d)
    assert dfresults.height == 9
    assert dfresults.filter(pl.col("check") == "temperature limits").select("status", "detail").row(0) == (
        "error",
        "RuntimeError: no estimators",
    )
    assert dfresults.filter(pl.col("check") == "complete time steps")["status"].item() == "ok"


def test_check_from_packets_gives_the_largest_noise(capsys: pytest.CaptureFixture[str]) -> None:
    """--frompackets gives the largest relative noise of the light curve and its time, with a warning."""
    from artistools.check import main

    dflightcurve = (
        at.lightcurve.scan_lightcurve_from_packets(modelpath)[-1].collect().filter(pl.col("relnoise").is_not_null())
    )
    worst = dflightcurve.sort("relnoise", descending=True).row(0, named=True)
    # a check with a warning gives the exit status 1, as ruff check does for a finding
    with pytest.raises(SystemExit) as exitinfo:
        main(argsraw=[str(modelpath), "--frompackets"])
    assert exitinfo.value.code == 1
    captured = capsys.readouterr()
    assert "\ncheck status detail\n" in captured.out
    noiseline = next(line for line in captured.out.splitlines() if line.startswith('"packet noise"'))
    assert noiseline.startswith('"packet noise" warning ')
    assert f"the largest relative noise is {100 * worst['relnoise']:.1f}% at {worst['time_days']:.2f} d" in noiseline
    assert re.search(r"warning: packet noise \(warning\): the largest relative noise", captured.err)
    # a check that is not checked gives no warning line
    assert "complete time steps (" not in captured.err

    # --exit-zero gives the table and the warnings, and the status 0
    main(argsraw=[str(modelpath), "--frompackets", "--exit-zero"])
    assert "warning: packet noise (warning)" in capsys.readouterr().err


def test_check_packet_noise_names_a_timestep_with_no_escaped_packet() -> None:
    """A time step with no escaped packet has a null noise, and the check left it out and gave the status ok."""
    from artistools.check import check_packet_noise

    dflightcurve = pl.LazyFrame({
        "timestep": [0, 1, 2],
        "time_days": [1.0, 2.0, 3.0],
        "packetcount": [1000, 0, 1000],
        "luminosity_Lsun": [1e8, 0.0, 1e8],
        "relnoise": [0.01, None, 0.01],
    })
    with mock.patch("artistools.check.scan_lightcurve_from_packets", return_value={-1: dflightcurve}):
        status, detail = check_packet_noise(modelpath_classic_3d, frompackets=True)
    assert status == "warning"
    assert "the time steps 1 have no escaped r-packet" in detail


def test_values_of_one_cell_agree_with_the_estimators() -> None:
    """The values command must give the raw estimators of the cell, and not a mean over cells or bins."""
    from artistools.estimators.values import get_values_table

    dfvalues = get_values_table(modelpath_classic_3d, ["Te", "TR", "nne", "rho"], timesteps=[29], modelgridindex=[23])
    assert dfvalues.columns == [
        "timestep",
        "time_days",
        "modelgridindex",
        "vel_r_mid_kmps",
        "Te[K]",
        "TR[K]",
        "nne[e^-/cm^3]",
        "rho[g/cm^3]",
    ]
    assert dfvalues.height == 1

    expected = (
        at
        .scan_estimators(modelpath_classic_3d, modelgridindex=23, timestep=29, join_modeldata=True)
        .select("tmid_days", "vel_r_mid", "Te", "TR", "nne", "rho")
        .collect()
        .row(0, named=True)
    )
    row = dfvalues.row(0, named=True)
    assert row["timestep"] == 29
    assert row["modelgridindex"] == 23
    assert np.isclose(row["time_days"], at.get_timestep_times(modelpath_classic_3d, loc="mid")[29], rtol=1e-6)
    assert np.isclose(row["vel_r_mid_kmps"], expected["vel_r_mid"] / at.constants.km_to_cm, rtol=1e-6)
    for variable, column in (("Te", "Te[K]"), ("TR", "TR[K]"), ("nne", "nne[e^-/cm^3]"), ("rho", "rho[g/cm^3]")):
        assert np.isclose(row[column], expected[variable], rtol=1e-6, atol=0.0), variable


def test_values_of_a_range_of_cells_give_one_row_for_each_cell_and_timestep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A range of cells and timesteps gives one row for each pair, and a cell with no matter gives a warning.

    The cells 12, 13, and 16 of the 3D model hold no matter, thus ARTIS wrote no estimators for them.
    """
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(argsraw=["Te", "nne", "-ts", "28-29", "-cell", "12-16", "-o", str(outputfile), str(modelpath_classic_3d)])

    assert "modelgridindex 12-13,16 has no estimators" in capsys.readouterr().err

    dfwritten = pl.read_csv(outputfile, separator=" ", comment_prefix="#")
    assert dfwritten.select("timestep", "modelgridindex").rows() == [(28, 14), (28, 15), (29, 14), (29, 15)]

    dfexpected = (
        at
        .scan_estimators(modelpath_classic_3d, modelgridindex=[14, 15], timestep=[28, 29])
        .select("timestep", "modelgridindex", "Te", "nne")
        .sort("timestep", "modelgridindex")
        .collect()
    )
    assert np.allclose(dfwritten["Te[K]"].to_numpy(), dfexpected["Te"].to_numpy(), rtol=1e-6, atol=0.0)
    assert np.allclose(dfwritten["nne[e^-/cm^3]"].to_numpy(), dfexpected["nne"].to_numpy(), rtol=1e-6, atol=0.0)

    # the header of the file tells how the command made the table
    header = [line for line in outputfile.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert any("with no average" in line for line in header)
    assert any(line.startswith("# command:") for line in header)


def test_values_give_the_time_of_volume_prevtimestep(tmp_path: Path) -> None:
    """The method lines gave the mid time of the row for volume_prevtimestep, which takes the timestep before."""
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(
        argsraw=[
            "volume",
            "volume_prevtimestep",
            "-ts",
            "0-1",
            "-cell",
            "14",
            "-o",
            str(outputfile),
            str(modelpath_classic_3d),
        ]
    )

    text = outputfile.read_text(encoding="utf-8")
    assert (
        "volume_prevtimestep = init_volume (t_prev / t_model)^3, where t_prev is the mid time of the timestep before"
        in text
    )
    dfwritten = pl.read_csv(outputfile, separator=" ", null_values="nan", comment_prefix="#").sort("timestep")
    volumecolumn = next(column for column in dfwritten.columns if column.startswith("volume["))
    prevcolumn = next(column for column in dfwritten.columns if column.startswith("volume_prevtimestep"))
    assert dfwritten[prevcolumn].to_list()[0] is None
    assert np.isclose(dfwritten[prevcolumn][1], dfwritten[volumecolumn][0], rtol=1e-6)


def test_values_ask_about_the_overwrite_before_they_read_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command found the model folder before the question, e.g. through ssh for a remote model."""
    from artistools.estimators import values

    outputfile = tmp_path / "values.txt"
    outputfile.write_text("old\n", encoding="utf-8")
    with (
        mock.patch.object(values, "get_model_folder", side_effect=AssertionError("the model was read")),
        mock.patch("sys.stdin.isatty", return_value=False),
        pytest.raises(SystemExit),
    ):
        values.main(argsraw=["Te", "-ts", "28", "-o", str(outputfile), str(modelpath_classic_3d)])
    assert "did not overwrite the files" in capsys.readouterr().err


def test_values_take_a_file_of_the_model_as_the_model_path(tmp_path: Path) -> None:
    """A model path can name a file of the model, and the command stopped because the file holds no input.txt."""
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(argsraw=["Te", "-ts", "28", "-cell", "14", "-o", str(outputfile), str(modelpath_classic_3d / "input.txt")])
    dfwritten = pl.read_csv(outputfile, separator=" ", comment_prefix="#")
    assert dfwritten.select("timestep", "modelgridindex").rows() == [(28, 14)]


def test_values_warn_about_a_cell_with_no_estimators_at_one_timestep(capsys: pytest.CaptureFixture[str]) -> None:
    """A cell that has estimators at one selected timestep and none at a second timestep gives a warning.

    The check tested the cells and the timesteps apart, thus each cell and each timestep had a row and no warning came.
    """
    from artistools.estimators.values import warn_missing_rows

    dfvalues = pl.DataFrame({"timestep": [0, 1, 0], "modelgridindex": [0, 0, 1], "Te[K]": [1e4, 1e4, 1e4]})
    warn_missing_rows(dfvalues, [0, 1], modelgridindex=None)

    assert "modelgridindex 1 has no estimators at timestep 1" in capsys.readouterr().err


def test_values_warn_about_a_requested_cell_in_the_direction_range(capsys: pytest.CaptureFixture[str]) -> None:
    """-dirbin with -cell warns about a requested cell in the range that has no estimators, e.g. a cell with no matter.

    The warning took only the cells of the table, thus such a cell gave no warning. Cell 932 is near the pole and holds
    no matter. Cell 402 is near the equator, thus it is outside the range, and it gives no warning.
    """
    from artistools.estimators.values import main

    main(argsraw=["Te", "-ts", "25", "-dirbin", "90", "-cell", "402,644,932", str(modelpath_classic_3d)])

    stderr = capsys.readouterr().err
    assert "modelgridindex 932 has no estimators" in stderr
    assert "402" not in stderr


def test_values_with_quiet_print_only_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    """--quiet must leave only the table on the standard output, thus a script can read it directly."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["values", "Te", "-t", "7.8", "-cell", "14", "--quiet", str(modelpath_classic_3d)])

    dfprinted = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ")
    assert dfprinted.columns == ["timestep", "time_days", "modelgridindex", "vel_r_mid_kmps", "Te[K]"]
    assert dfprinted.select("timestep", "modelgridindex").rows() == [(29, 14)]


def test_values_of_an_unknown_variable_name_the_listing(capsys: pytest.CaptureFixture[str]) -> None:
    """A variable that the model does not hold stops the command, and the help names a near name and the listing."""
    from artistools.estimators.values import main

    with pytest.raises(SystemExit) as excinfo:
        main(argsraw=["Tee", "-t", "7.8", str(modelpath_classic_3d)])
    assert excinfo.value.code == 1

    stderr = capsys.readouterr().err
    assert "'Tee' is not a variable" in stderr
    assert "Did you mean Te?" in stderr
    assert "--listvariables" in stderr


def test_cache_delete_keeps_the_lock_files_and_other_parquet_files(tmp_path: Path) -> None:
    """Delete the caches and the leftover files of a stopped conversion, and keep each file that is not a cache.

    A deleted lock file can give the lock to two writers. A parquet file of a different name belongs to the user, also
    when it ends in .out.parquet.
    """
    from artistools.cache import main as cachemain

    deleted = [
        "model.txt.parquet",
        "linelist.out.parquet.tmp",
        "packets/packetsbatch00_0000_0003.out.parquet",
        "12411848.slurm/estimbatch00_0000_0003.out.parquet",
        "packets/.packetsbatch01_0004_0007.out.parquet.partialab12.partial",
        "12411848.slurm/.estimators_allranks.out.parquet.parts9x/ts00001_part00000.parquet",
    ]
    kept = [
        "packets/.packetsbatch00_0000_0003.out.parquet.replace-lock",
        "spectrum_data.parquet",
        "science.out.parquet",
        "analysis/.analysis.parts.backup/notes.txt",
        "model.txt",
    ]
    for name in [*deleted, *kept]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")

    cachemain(argsraw=["delete", "-f", str(tmp_path)])

    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == sorted(kept)


def test_cache_make_gives_no_data_for_virtual_packets_without_their_files(tmp_path: Path) -> None:
    """Give no data, and not an error, for a model that has vpkt.txt but no virtual packet files.

    sn3d writes the virtual packet files only with VPKT_WRITE_CONTRIBS, thus vspecpolmodel has vpkt.txt and none of
    these files. vpktcontrib has the files, thus it gets its cache.
    """
    from artistools.cache import make_caches

    vspecpolmodel = tmp_path / "vspecpolmodel"
    shutil.copytree(at.get_path("testdata") / "vspecpolmodel", vspecpolmodel)
    assert (vspecpolmodel / "vpkt.txt").is_file()
    dfresults = make_caches(vspecpolmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets").row(0) == (
        "virtual packets",
        "no data",
        "the model holds no virtual packet files",
    )
    assert "error" not in dfresults["status"].to_list()

    # the copy leaves out the caches of earlier runs of the tests
    vpktmodel = tmp_path / "vpktcontrib"
    shutil.copytree(at.get_path("testdata") / "vpktcontrib", vpktmodel, ignore=shutil.ignore_patterns("vpackets"))
    dfresults = make_caches(vpktmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets").row(0) == (
        "virtual packets",
        "ok",
        "1 caches of the packets of 2 ranks",
    )


def test_cache_make_gives_an_error_for_incomplete_or_remote_packet_files(tmp_path: Path) -> None:
    """Give an error, and not "no data", when the file of the first rank is missing or the model is remote.

    The other files of the model show that it has data. A remote model needs a run of the command on its host.
    """
    from artistools.cache import make_caches
    from artistools.cache import make_packet_caches

    vpktmodel = tmp_path / "vpktcontrib"
    shutil.copytree(at.get_path("testdata") / "vpktcontrib", vpktmodel, ignore=shutil.ignore_patterns("vpackets"))
    (vpktmodel / "vpackets_0000.out.zst").unlink()
    dfresults = make_caches(vpktmodel)
    assert dfresults.filter(pl.col("data") == "virtual packets")["status"].item() == "error"

    with pytest.raises(ValueError, match="on a different host"):
        make_packet_caches(Path("modelhost:/models/kilonova"), virtual=True)


def test_check_names_the_missing_input_files_and_the_leftover_files(tmp_path: Path) -> None:
    """Give a warning for each missing input file and for the files that a stopped conversion left."""
    from artistools.check import check_caches
    from artistools.check import check_input_files

    (tmp_path / "input.txt").write_text("")
    (tmp_path / "model.txt.zst").write_text("")
    (tmp_path / "packets").mkdir()
    (tmp_path / "packets" / ".packetsbatch00_0000_0003.out.parquet.partialab12.partial").write_text("x")

    status, detail = check_input_files(tmp_path)
    assert status == "warning"
    assert "model.txt" not in detail
    assert "no abundances.txt" in detail
    assert "no adata.txt" in detail
    status, detail = check_caches(tmp_path)
    assert status == "warning"
    assert "1 leftover files of conversions that stopped" in detail


def test_linefluxes_ratio_noise_adds_the_noise_of_the_two_features_in_quadrature() -> None:
    """Give each feature the noise sqrt(sum(e^2)) / sum(e) of its packets, and the ratio the root of the sum of squares.

    A time bin with no packet of a feature has no noise value, thus its ratio has none either.
    """
    from artistools.plotlinefluxes import add_ratio_noise
    from artistools.plotlinefluxes import FeatureTuple
    from artistools.plotlinefluxes import get_line_luminosities_from_packets

    features = [
        FeatureTuple(f"feature{index}", "label", 1000.0, [index], 990.0, 1010.0, 26, 2, [1], [0]) for index in (1, 2)
    ]
    dfpackets = pl.LazyFrame({
        "trueemissiontype": [1, 1, 1, 2, 2],
        "e_rf": [1.0, 2.0, 3.0, 4.0, 4.0],
        "t_arrive_d": [1.5, 1.5, 1.5, 1.5, 2.5],
    })
    with mock.patch.object(at.plotlinefluxes, "scan_packets", return_value=(1, dfpackets)):
        dflcdata = get_line_luminosities_from_packets(
            "trueemissiontype", features, modelpath_classic_3d, arr_tstart=[1.0, 2.0], arr_tend=[2.0, 3.0]
        )
    assert np.isclose(dflcdata["feature1_relnoise"][0], math.sqrt(1 + 4 + 9) / 6, rtol=1e-12)
    assert dflcdata["feature1_relnoise"][1] is None
    assert np.allclose(dflcdata["feature2_relnoise"].to_numpy(), [1.0, 1.0], rtol=1e-12, atol=0)

    dfratio = add_ratio_noise(dflcdata.with_columns(fratio=pl.lit(1.0)), ["feature1", "feature2"])
    assert np.isclose(dfratio["fratio_relnoise"][0], math.sqrt((math.sqrt(14) / 6) ** 2 + 1), rtol=1e-12)
    assert dfratio["fratio_relnoise"][1] is None
