import argparse
import dataclasses as dc
import gzip
import lzma
import shlex
import shutil
import typing as t
import warnings
from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.markers as mplmarkers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.container import ErrorbarContainer
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Mbol_sun
from artistools.lightcurve import interactive
from artistools.lightcurve import viewingangleanalysis
from artistools.lightcurve.core import bracket_spectrum_to_band

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot(mockplot: mock.MagicMock, benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    benchmark(lambda: at.lightcurve.plot(argsraw=[], modelpath=[modelpath], outputfile=tmp_path, frompackets=False))

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.334, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.2348791, rtol=1e-4)

    integral = np.trapezoid(arr_lum, arr_time_d)
    assert np.isclose(integral, 2.4189054554e42, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 3.231155e40, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 7.2115e39, rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_frompackets(mockplot: mock.MagicMock, benchmark: BenchmarkFixture) -> None:
    benchmark(
        lambda: at.lightcurve.plot(
            argsraw=[],
            modelpath=modelpath,
            frompackets=True,
            outputfile=Path(outputpath, "lightcurve_from_packets.pdf"),
        )
    )

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.33389, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.23483, rtol=1e-4)

    integral = np.trapezoid(arr_lum, arr_time_d)

    assert np.isclose(integral, 9.0323767e40, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 1.2039713396033405e39, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 3.614004402353378e38, rtol=1e-4)


def test_lightcurve_of_a_virtual_observer_holds_the_energy_of_its_packets() -> None:
    """The light curve of a virtual observer bins the arrival time and the energy of that observer, times 4 pi.

    Each observer and each opacity choice has its own columns. The columns of a different observer, or no factor of
    4 pi, give a different energy.
    """
    modelpath = at.get_path("testdata") / "vpktcontrib"
    nprocs_read, dfvpackets = at.packets.get_virtual_packets(modelpath)
    # observer 1 with opacity choice 1
    dirbin = at.misc.get_vpkt_config(modelpath)["nspectraperobs"] + 1

    dflightcurve = at.lightcurve.get_from_packets(
        modelpath, directionbins=[dirbin], directionbins_are_vpkt_observers=True
    )[dirbin].collect()

    dftimesteps = at.misc.get_timesteps(modelpath).collect()
    assert dflightcurve["timestep"].to_list() == dftimesteps["timestep"].to_list()
    dfinrange = dfvpackets.filter(
        pl.col("dir1_t_arrive_d").is_between(dftimesteps["tstart_days"].min(), dftimesteps["tend_days"].max())
    ).collect()
    assert dflightcurve["packetcount"].sum() == dfinrange.height > 0
    energy = (
        np.sum(dflightcurve["luminosity_Lsun"].to_numpy() * dftimesteps["twidth_days"].to_numpy())
        * at.constants.day_to_s
        * Lsun_to_erg_per_s
        * nprocs_read
        / (4 * np.pi)
    )
    assert np.isclose(energy, dfinrange["dir1_e_rf_1"].to_numpy().sum(dtype=np.float64), rtol=1e-10, atol=0.0)


def test_lightcurve_of_a_virtual_observer_keeps_the_vpkt_window_and_ranges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The light curve of a virtual observer holds only the timesteps inside the time window of vpkt.txt.

    The energy comes only from the rows with nu_rf inside a wavelength range of vpkt.txt, as for the spectrum. A
    timestep outside the window gave zero luminosity, and a timestep across an edge gave a low luminosity.
    """
    sourcedir = at.get_path("testdata") / "vpktcontrib"
    for filename in ("input.txt", "vpackets_0000.out.zst", "vpackets_0001.out.zst"):
        shutil.copy(sourcedir / filename, tmp_path / filename)
    vpktlines = (sourcedir / "vpkt.txt").read_text(encoding="utf-8").splitlines()
    # the fifth line is the time window override, and the sixth line is the custom wavelength range flag
    vpktlines[4:6] = ["1 130 140", "1 2 3500 6000 6400 7200"]
    (tmp_path / "vpkt.txt").write_text("\n".join(vpktlines) + "\n", encoding="utf-8")
    nprocs_read, dfvpackets = at.packets.get_virtual_packets(tmp_path)
    dirbin = 4

    dflightcurve = at.lightcurve.get_from_packets(
        tmp_path, directionbins=[dirbin], directionbins_are_vpkt_observers=True
    )[dirbin].collect()

    assert "leaves out" in capsys.readouterr().err
    dftimesteps = (
        at.misc.get_timesteps(tmp_path).filter(pl.col("timestep").is_in(dflightcurve["timestep"].implode())).collect()
    )
    windowstart, windowend = (
        float(dftimesteps["tstart_days"].to_numpy().min()),
        float(dftimesteps["tend_days"].to_numpy().max()),
    )
    assert windowstart >= 130.0
    assert windowend <= 140.0
    assert 0 < dftimesteps.height < at.misc.get_timesteps(tmp_path).collect().height

    c = at.constants.c_ang_per_s
    lambda_rf = c / pl.col("dir1_nu_rf")
    dfinrange = dfvpackets.filter(
        pl.col("dir1_t_arrive_d").is_between(windowstart, windowend)
        & (lambda_rf.is_between(3500.0, 6000.0, closed="none") | lambda_rf.is_between(6400.0, 7200.0, closed="none"))
    ).collect()
    assert dflightcurve["packetcount"].sum() == dfinrange.height > 0
    energy = (
        np.sum(dflightcurve["luminosity_Lsun"].to_numpy() * dftimesteps["twidth_days"].to_numpy())
        * at.constants.day_to_s
        * Lsun_to_erg_per_s
        * nprocs_read
        / (4 * np.pi)
    )
    assert np.isclose(energy, dfinrange["dir1_e_rf_1"].to_numpy().sum(dtype=np.float64), rtol=1e-10, atol=0.0)

    vpktlines[4] = "1 100 110"
    (tmp_path / "vpkt.txt").write_text("\n".join(vpktlines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No selected timestep lies fully inside the time window"):
        at.lightcurve.get_from_packets(tmp_path, directionbins=[dirbin], directionbins_are_vpkt_observers=True)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_lightcurve_plot_reflightcurves_keep_their_errorbars(mockerrorbar: mock.MagicMock) -> None:
    """Reference light curves keep their error bars by either route, the model path list or -reflightcurves."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_waxmanetal2018.txt"],
        reflightcurves=["AT2017gfo_smarttetal2017.txt"],
        outputfile=Path(outputpath, "lightcurve_reflightcurves.pdf"),
    )

    # one call for the reference light curve given as a model path, one for the -reflightcurves file
    assert mockerrorbar.call_count == 2

    labels = [callitem[1]["label"] for callitem in mockerrorbar.call_args_list]
    assert labels == ["AT2017gfo (Waxman+2018)", "AT2017gfo (Smartt+2017)"]

    # the -reflightcurves file continues the grey sequence rather than starting it again
    assert [callitem[1]["color"] for callitem in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    for callitem, expected_time_d_min in zip(mockerrorbar.call_args_list, (0.5, 0.638), strict=True):
        arr_time_d = np.array(callitem[0][1])
        arr_lum = np.array(callitem[0][2])
        arr_errminus, arr_errplus = (np.array(err) for err in callitem[1]["yerr"])

        assert np.isclose(arr_time_d.min(), expected_time_d_min, rtol=1e-4)
        assert arr_errminus.shape == arr_lum.shape
        assert arr_errplus.shape == arr_lum.shape
        assert (arr_errminus > 0.0).all()
        assert (arr_errplus > 0.0).all()


def test_filter_data_is_sorted_by_wavelength() -> None:
    """Filter curves must come back in ascending wavelength order with transmissions still paired.

    The files under data/filters/NOT are stored out of order. Callers interpolate on this grid and
    integrate over it with np.trapezoid, which silently cancels flux across negative-width segments,
    so the sort must happen here rather than at each use.
    """
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")

    rawlines = (filterdir / "NOT" / "B.txt").read_text(encoding="utf-8").splitlines()[4:]
    rawpairs = {float(row.split()[0]): float(row.split()[1]) for row in rawlines if row.split()}
    assert sorted(rawpairs) != list(rawpairs), "this fixture is only meaningful while the file is unsorted"

    _, _, wavefilter, transmission, wavefilter_min, wavefilter_max = at.lightcurve.get_filter_data(filterdir, "NOT/B")

    assert np.all(np.diff(wavefilter) > 0), "wavelengths must be strictly ascending after sorting"
    assert wavefilter_min == wavefilter[0]
    assert wavefilter_max == wavefilter[-1]

    # the sort must move transmissions with their wavelengths, not just reorder one of the two
    assert len(wavefilter) == len(rawpairs)
    for wavelength, transmit in zip(wavefilter, transmission, strict=True):
        assert transmit == rawpairs[wavelength]


def test_spectrum_filter_range_includes_bracketing_points() -> None:
    """Band integration must retain the spectrum point on each side of the filter range."""
    spectrum = pl.DataFrame({"lambda_angstroms": [1000.0, 2000.0, 3000.0, 4000.0], "f_lambda": [1.0, 2.0, 4.0, 8.0]})

    wavelength, flux = bracket_spectrum_to_band(spectrum, wavefilter_min=2200.0, wavefilter_max=2800.0)

    assert np.allclose(wavelength, np.array([2000.0, 3000.0]))
    assert np.allclose(flux, np.array([2.0, 4.0]))


def test_band_magnitude_calculations() -> None:
    band_magnitude_data = at.lightcurve.generate_band_lightcurve_data(
        modelpath,
        plotvspecpol=False,
        plotviewingangle=False,
        filter=["bol", "U", "B", "V", "I"],
        timemin=290.0,
        timemax=300.0,
        average_over_phi_angle=False,
        average_over_theta_angle=False,
    )

    expected_summary = {
        "bol": ((290.381, -12.522955565351443), (299.309, -12.290504747994545), -12.486325221679541),
        "U": ((290.381, -11.72755172004823), (299.309, -10.940395871435907), -11.552651445171437),
        "B": ((290.381, -12.803113515779813), (299.309, -12.468614058886018), -12.72946419132831),
        "V": ((290.381, -13.134621676461588), (299.309, -12.922533596999637), -13.018642833106346),
        "I": ((290.381, -12.353786573875853), (299.309, -12.099768401014884), -12.443185093790348),
    }
    expected_brightest = {
        "bol": (291.359, -12.572391488690043),
        "U": (298.303, -11.927722052704134),
        "B": (298.303, -12.8852578600296),
        "V": (293.327, -13.16654427664928),
        "I": (295.307, -12.701314848818333),
    }

    assert band_magnitude_data.keys() == expected_summary.keys()
    for band_name, (expected_first, expected_last, expected_mean) in expected_summary.items():
        magnitudes = band_magnitude_data[band_name]
        assert len(magnitudes) == 10
        assert magnitudes[0] == pytest.approx(expected_first)
        assert magnitudes[-1] == pytest.approx(expected_last)
        assert np.mean([magnitude for _, magnitude in magnitudes]) == pytest.approx(expected_mean)
        assert min(magnitudes, key=itemgetter(1)) == pytest.approx(expected_brightest[band_name])


def test_band_magnitude_selection_and_colour() -> None:
    band_magnitude_data = at.lightcurve.generate_band_lightcurve_data(
        modelpath,
        plotvspecpol=False,
        plotviewingangle=False,
        filter=["B", "V"],
        timemin=290.0,
        timemax=300.0,
        average_over_phi_angle=False,
        average_over_theta_angle=False,
    )

    times, b_magnitudes = at.lightcurve.get_band_lightcurve(band_magnitude_data, "B", timemin=293.0, timemax=296.0)

    assert times == pytest.approx([293.327, 294.315, 295.307])
    assert b_magnitudes == pytest.approx([-12.708769275316618, -12.656976514620492, -12.794120144812808])

    colour_times, b_minus_v = at.lightcurve.get_colour_delta_mag(band_magnitude_data, ["B", "V"])
    assert colour_times == pytest.approx([time for time, _ in band_magnitude_data["B"]])
    assert b_minus_v == pytest.approx([
        0.3315081606817749,
        0.2742048700430537,
        0.2331501773840028,
        0.4577750013326618,
        0.34488607340471944,
        0.09842377150842907,
        0.2854726927324762,
        0.38998827189773166,
        0.022457860681898367,
        0.4539195381136185,
    ])


def test_band_lightcurve_peakmag_risetime_plot(tmp_path: Path) -> None:
    """The export writes one file for each band, with the fit values of the angle average and delta m40."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath,
        filter=["bol", "B"],
        include_delta_m40=True,
        plotviewingangle=-1,
        timemin=250,
        timemax=300,
        save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
        outputfile=tmp_path,
    )

    for band_name in ("bol", "B"):
        dfdata = at.misc.read_wsv(tmp_path / f"{band_name}band_TEST MODEL_viewing_angle_data.txt")
        assert dfdata.columns == [
            "dirbin",
            "peak_mag_polyfit",
            "risetime_polyfit",
            "deltam15_polyfit",
            "deltam40_polyfit",
        ]
        assert dfdata["dirbin"].to_list() == [-1]
        assert 250.0 <= dfdata["risetime_polyfit"].item() <= 300.0
        assert -20.0 < dfdata["peak_mag_polyfit"].item() < -5.0


def test_viewing_angle_peakmag_risetime_scatter_plot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The viewing angle scatter plot must not read -xmin/-xmax, which this parser does not define.

    Its time range is named -timemin/-timemax, so args has no xmin and reading it raised AttributeError
    before any scatter plot could be saved. The plot reads the data file that the first call writes.
    """
    monkeypatch.chdir(tmp_path)
    commonargs: dict[str, t.Any] = {
        "modelpath": [modelpath],
        "filter": ["B"],
        "timemin": 250,
        "timemax": 300,
        "outputfile": tmp_path,
    }
    at.lightcurve.plot(argsraw=[], save_viewing_angle_peakmag_risetime_delta_m15_to_file=True, **commonargs)
    at.lightcurve.plot(argsraw=[], make_viewing_angle_peakmag_risetime_scatter_plot=True, **commonargs)

    assert list(tmp_path.glob("*risetime_peakmag.pdf"))


@mock.patch.object(mplax.Axes, "scatter", side_effect=mplax.Axes.scatter, autospec=True)
def test_viewing_angle_scatter_plot_colours_each_direction_bin_of_the_data_file(
    mockscatter: mock.MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The colour bar mode gives one colour to each direction bin of the data file, and not to all the direction bins.

    The scatter plot raised ValueError, because 100 colours went to the rows of the selected direction bins.
    """
    monkeypatch.chdir(tmp_path)
    commonargs: dict[str, t.Any] = {
        "modelpath": [modelpath_classic_3d],
        "timemin": 3.2,
        "timemax": 7.5,
        "outputfile": tmp_path,
    }
    at.lightcurve.plot(
        argsraw=[], plotviewingangle=[0, 10], save_viewing_angle_peakmag_risetime_delta_m15_to_file=True, **commonargs
    )
    scatterplotargs: dict[str, t.Any] = {
        "make_viewing_angle_peakmag_risetime_scatter_plot": True,
        "colorbarcostheta": True,
    }
    at.lightcurve.plot(argsraw=[], plotviewingangle=[0, 10], **scatterplotargs, **commonargs)

    assert list(tmp_path.glob("*risetime_peakmag.pdf"))
    dirbincolors = mockscatter.call_args_list[0].kwargs["color"]
    assert len(dirbincolors) == 2
    assert not np.allclose(dirbincolors[0], dirbincolors[1]), "the two cos(theta) bins share a colour"

    # the data file names the direction bin of each row, thus a different selection gives the same colours
    mockscatter.reset_mock()
    at.lightcurve.plot(argsraw=[], plotviewingangle=[0, 10, 20], **scatterplotargs, **commonargs)
    assert np.allclose(mockscatter.call_args_list[0].kwargs["color"], dirbincolors)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_lightcurve_subplots(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Two bands give two panels, and each panel takes the light curve of its band."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["bol", "B"], outputfile=tmp_path)

    assert (tmp_path / "plotlightcurves.pdf").is_file()
    assert len(mockplot.call_args_list) == 2
    assert mockplot.call_args_list[0].args[0] is not mockplot.call_args_list[1].args[0]


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_colour_evolution_plot_ylabel(mockylabel: mock.MagicMock, tmp_path: Path) -> None:
    """A colour evolution plot must be labelled in delta magnitudes, not as a band magnitude.

    colour_evolution_plot assigns args.filter before asking for the labels, so reading the plot kind back off
    args labelled these axes "None Magnitude".
    """
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=tmp_path)

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert r"$\Delta$m" in ylabels, ylabels


@pytest.mark.parametrize("flag", ["--legendframe", "--legendframeon"])
@mock.patch.object(mplax.Axes, "legend", side_effect=mplax.Axes.legend, autospec=True)
def test_legend_frame_takes_both_spellings(mocklegend: mock.MagicMock, flag: str, tmp_path: Path) -> None:
    """The command plotlightcurves had its own --legendframeon, and each command now takes --legendframe.

    A script can hold the old spelling, thus it gives the same white frame.
    """
    at.lightcurve.plot(argsraw=[str(modelpath), flag, "-o", str(tmp_path / "lightcurve.pdf")])

    legendkwargs = mocklegend.call_args.kwargs
    assert legendkwargs["frameon"]
    assert legendkwargs["facecolor"] == "white"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_falls_back_to_the_model_name(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A series with no -label is named after its model, not after the None that pads the -label list."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], outputfile=tmp_path)

    assert [callargs.kwargs["label"] for callargs in mockplot.call_args_list] == ["TEST MODEL"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_uses_the_label_arg(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A -label value names its series, in place of the model name."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], label=["My model"], outputfile=tmp_path)

    assert [callargs.kwargs["label"] for callargs in mockplot.call_args_list] == ["My model"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_direction_bin_keeps_the_label_arg(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A direction bin is named after the -label value of its model, with the bin appended."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        filter=["B"],
        plotviewingangle=[0],
        label=["My model"],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    assert len(labels) == 1
    assert labels[0].startswith("My model "), labels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_color_arg(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A -color value must reach the plotted line."""
    at.lightcurve.plot(
        argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], color=["magenta"], outputfile=tmp_path
    )

    assert [callargs.kwargs["color"] for callargs in mockplot.call_args_list] == ["magenta"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_viewingangle_colours(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Direction bins get one colour each from the tab20 colour map, whatever -color says.

    The -color list has one entry per model, so it cannot colour the direction bins. This used to warn about
    a -color argument that the user never gave, and then leave the colour unset. More direction bins than the
    colour map has colours must wrap around rather than run off the end of the list.
    """
    palette = ["#111111", "#222222"]
    ndirbins = len(palette) + 1  # one more than the palette holds, so the last bin wraps to the first colour

    # a short palette exercises the wrap with three direction bins rather than the twenty-one the tab20 map
    # would need, and the palette itself is covered by test_colour_evolution_plot_dirbin_colours_*
    with mock.patch.object(at.lightcurve.plotlightcurve, "get_dirbin_palette", return_value=palette):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=modelpath_classic_3d,
            colour_evolution=["B-V"],
            plotviewingangle=list(range(ndirbins)),
            timemin=5,
            timemax=8,
            color=["magenta"],
            outputfile=tmp_path,
        )

    colors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list]
    assert colors == [*palette, palette[0]]


def test_dirbin_palette_avoids_the_assigned_colours() -> None:
    """The direction bin palette holds tab20 colours, minus any that a whole series was given."""
    tab20colors = [mplcolors.to_hex(color) for color in plt.get_cmap("tab20")(np.linspace(0, 1.0, 20))]

    palette = [mplcolors.to_hex(color) for color in at.lightcurve.plotlightcurve.get_dirbin_palette([tab20colors[0]])]
    assert palette == tab20colors[1:]

    # every colour of the map assigned leaves none free, and a bin still has to be drawn in something
    assert len(at.lightcurve.plotlightcurve.get_dirbin_palette(tab20colors)) == len(tab20colors)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_dirbin_colour_is_stable_across_subplots(
    mockplot: mock.MagicMock, tmp_path: Path
) -> None:
    """A direction bin keeps one colour across the subplots of the filter pairs.

    The legend is drawn on one subplot only, so a bin drawn in a different colour in each subplot
    contradicts the legend everywhere else.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=["U-B", "B-V"],
        plotviewingangle=[0, 1],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    # one call per (direction bin, filter pair), the filter pairs of a bin together
    colors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list]
    assert len(colors) == 4
    assert np.allclose(colors[0], colors[1]), "bin 0 changed colour between the subplots"
    assert np.allclose(colors[2], colors[3]), "bin 1 changed colour between the subplots"
    assert not np.allclose(colors[0], colors[2]), "the two direction bins share a colour"


@mock.patch.object(mplax.Axes, "legend", side_effect=mplax.Axes.legend, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_dirbin_labels_take_the_line_colours(
    mockplot: mock.MagicMock, mocklegend: mock.MagicMock, tmp_path: Path
) -> None:
    """The legend label of a direction bin has the colour of its line."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=["B-V"],
        plotviewingangle=[0, 1],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    linecolours = [mplcolors.to_hex(callargs.kwargs["color"]) for callargs in mockplot.call_args_list]
    assert len(set(linecolours)) == 2

    mocklegend.assert_called_once()
    legend = mocklegend.call_args.args[0].get_legend()
    assert [mplcolors.to_hex(text.get_color()) for text in legend.get_texts()] == linecolours


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_single_dirbin_colour(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Use the -color value when a model contributes a single line, even if that line is one direction bin."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=["B-V"],
        plotviewingangle=[5],
        color=["magenta"],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    assert [callargs.kwargs["color"] for callargs in mockplot.call_args_list] == ["magenta"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_subplots(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """Two colours give two panels, and each panel takes the colour of its pair of bands."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["U-B", "B-V"], outputfile=tmp_path)

    assert (tmp_path / "plotcolorevolutionU-B_B-V.pdf").is_file()
    assert len(mockplot.call_args_list) == 2
    assert mockplot.call_args_list[0].args[0] is not mockplot.call_args_list[1].args[0]
    bandmags = at.lightcurve.generate_band_lightcurve_data(modelpath, filter=["U", "B", "V"])
    for callargs, bands in zip(mockplot.call_args_list, (["U", "B"], ["B", "V"]), strict=True):
        _, expectedcolours = at.lightcurve.get_colour_delta_mag(bandmags, bands)
        assert np.allclose(callargs.args[2], expectedcolours)


def test_get_colour_delta_mag_unequal_sampling() -> None:
    """A band with no flux at some time has no point there, so the two bands can be sampled differently."""
    band_lightcurve_data: dict[str, list[tuple[float, float]]] = {
        "B": [(10.0, -18.0), (20.0, -17.0), (30.0, -16.0)],
        "V": [(10.0, -18.5), (30.0, -16.5), (40.0, -15.0)],
    }

    times, colours = at.lightcurve.get_colour_delta_mag(band_lightcurve_data, ["B", "V"])

    assert times == [10.0, 30.0]
    assert colours == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("z", "dist_mpc"),
    [
        (0.0, 0.0),
        (0.005791, 24.912483443375777),  # SN 1991T
        (0.0133, 57.54493109140769),  # iPTF13ebh
        (0.01433, 62.04993050233093),  # SN 1999dq
        (0.1, 460.2999363904721),
        (0.5, 2832.9380939001253),
        (1.0, 6607.6576117749355),
        (3.0, 25422.741745189862),
    ],
)
def test_luminosity_distance(z: float, dist_mpc: float) -> None:
    """Reference values are astropy's FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(z), which this replaced.

    astropy evaluates the same Baes et al. (2017) function through scipy's hyp2f1 where this integrates it,
    so the two agree to ~4e-14 and the tolerance is set well inside the 1e-10 that any use here needs.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z) == pytest.approx(dist_mpc, rel=1e-11, abs=1e-12)


def test_luminosity_distance_planck18_parameters() -> None:
    """The cosmology parameters must be used, not baked in (reference values from astropy)."""
    for z, dist_mpc in ((0.01433, 64.43316428422708), (0.5, 2927.080479237606), (2.0, 15936.22617736705)):
        assert at.lightcurve.luminosity_distance(H0=67.4, Om0=0.315, z=z) == pytest.approx(dist_mpc, rel=1e-11)


def test_luminosity_distance_negative_dark_energy() -> None:
    """Om0 > 1 leaves a flat universe with a negative dark energy density, which is still integrable.

    astropy's s = ((1 - Om0) / Om0)^(1/3) is a cube root of a negative number here, which it survives only
    by carrying a complex s through and discarding the imaginary part. The u form never forms s at all, so
    these come from adaptive quadrature rather than from astropy, which is itself ~7e-15 off at the first.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.5, z=0.1) == pytest.approx(425.1405279377727, rel=1e-12)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=1.0) == pytest.approx(4066.82076478827, rel=1e-12)


def test_luminosity_distance_dense_matter() -> None:
    """A negative dark energy density leaves an integrable pole in 1 / E at the turnaround.

    The larger Om0 is, the closer that turnaround crowds up to z = 0 and the further its pole reaches, so
    integrating over u alone ran 1.1% low for Om0 = 1e6 at z = 1 and ~1% over the last 0.01 in z above the
    turnaround of any Om0 > 1. The first two values agree with adaptive quadrature in ln(1 + z), the third
    with 32 to 1024 node rules in q, which converge where the adaptive routine warns and stops short.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e6, z=1.0) == pytest.approx(8.56941262664432, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=100.0, z=1.0) == pytest.approx(803.7609074618974, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e6, z=1e-10) == pytest.approx(
        4.28242824230702e-07, rel=1e-11, abs=0
    )

    # just above the turnaround of Om0 = 2, where 1 / E is unbounded and u alone was 1.3% out
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=-0.2062994740159002) == pytest.approx(
        -1523.69803, rel=1e-6
    )

    # the q width has to come from the difference of cubes: taking qhi - qlo directly was 1.8e-7 out here
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (1e-10, 1e-8):
        expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * 5.0 * z)
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=5.0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)


def test_luminosity_distance_past_turnaround_rejected() -> None:
    """A negative dark energy density halts the expansion, and nothing beyond that turnaround has a distance.

    E(z)^2 is negative there, which no domain check on z alone would catch. Below the turnaround the
    quadrature returns a NaN that would reach the magnitudes as one, and immediately below it something
    worse: the interior nodes still straddle positive radicands, so it returns a plausible finite number.
    """
    for Om0, z in ((2.0, -0.5), (2.0, -0.999), (1.5, -0.4), (1.5, -0.306638726649365)):
        with pytest.raises(ValueError, match="stops expanding"):
            at.lightcurve.luminosity_distance(H0=70.0, Om0=Om0, z=z)

    # just above its turnaround of z = -0.2063 the distance exists and must still be returned
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=-0.1) == pytest.approx(-462.8923430069639, rel=1e-11)


def test_luminosity_distance_matter_only() -> None:
    """For Om0 = 1 the integral is analytic: D_L = 2 (c / H0) (1 + z) (1 - 1 / sqrt(1 + z)).

    Its z integrand diverges as z' -> -1, so the blueshifts here are the ones quadrature cannot follow.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (-0.999999, -0.99, -0.9, -0.01, 0.01, 0.5, 2.0, 10.0, 1e8):
        expected = 2 * hubble_dist_mpc * (1.0 + z) * (1.0 - 1.0 / np.sqrt(1.0 + z))
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.0, z=z) == pytest.approx(expected, rel=1e-10, abs=0)


def test_luminosity_distance_matter_only_tiny_redshift() -> None:
    """Expanding the Om0 = 1 closed form gives D_L = (c / H0) z (1 + z) (1 - 3z/4 + O(z^2)) as z -> 0.

    The tolerance is tight enough to fail if that closed form is evaluated as 1 - 1 / sqrt(1 + z), which
    loses all but a few digits to cancellation here (8e-8 relative at z = 1e-10).
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (1e-10, 1e-8, 1e-6):
        expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * z)
        # abs=0 because these distances are ~1e-7 Mpc, so the default absolute tolerance of approx would
        # swallow the cancellation error entirely
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)


def test_luminosity_distance_blueshift() -> None:
    """A blueshift is a valid redshift, and the closed forms hold for negative z as well.

    A blueshift lies below the crossover for any Om0 < 0.5, so these integrate over z rather than over u,
    which would stretch the range towards infinity as z -> -1.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (-1e-6, -0.001, -0.01, -0.1, -0.5, -0.9, -0.999999):
        # Om0 = 0 has E(z) = 1, so the comoving distance is exactly (c / H0) z
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.0, z=z) == pytest.approx(
            hubble_dist_mpc * z * (1.0 + z), rel=1e-13, abs=0
        )
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z) < 0.0

    # the mildly negative redshifts of nearby blueshifted galaxies are the only ones of practical interest
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=-0.001) == pytest.approx(-4.279429096775459)

    # at the edge of the domain a cosmology with no closed form has to come from the z quadrature: this
    # value agrees with both a 2048-node rule and adaptive quadrature, where the u substitution used for
    # redshifts would give a magnitude 41% too small
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=-0.999999) == pytest.approx(
        -0.0048851852579134521, rel=1e-12, abs=0
    )


def test_luminosity_distance_below_minus_one_rejected() -> None:
    """1 + z is a ratio of scale factors, so z <= -1 is not a redshift and must not silently give a NaN."""
    for z in (-1.0, -1.5, -10.0):
        with pytest.raises(ValueError, match="must be greater than -1"):
            at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z)


def test_luminosity_distance_quadrature_extremes() -> None:
    """Pin the ends of the range that the quadrature, rather than a closed form, has to cover.

    A tiny z falls on whichever side of the crossover z = 0 sits: below it for Om0 = 0.3, whose width is
    z itself, and above it for Om0 = 0.9, whose width has to come from the difference of the roots.
    """
    hubble_dist_mpc = 299792.458 / 70.0

    # expanding the integrand gives D_L = (c / H0) z (1 + z) (1 - 3 Om0 z / 4 + O(z^2)) as z -> 0
    for Om0 in (0.3, 0.9):
        for z in (1e-10, 1e-8):
            expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * Om0 * z)
            assert at.lightcurve.luminosity_distance(H0=70.0, Om0=Om0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)

    # adaptive quadrature in ln(1 + z), which resolves every regime of the integrand, gives this
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=1e8) == pytest.approx(1415324782383.9055, rel=1e-11)


def test_luminosity_distance_across_the_crossover() -> None:
    """1 / E has a plateau below Om0 (1 + z')^3 = 1 - Om0 and a (1 + z')^(-3/2) tail above it.

    Neither variable covers both once they differ by decades, which is why the range is split there. These
    are the two ways that goes wrong with a single variable: integrating a strongly blueshifted, nearly
    matter-only cosmology over z alone was 64% low, and taking a nearly matter-free one to high z over u
    alone was ~1e-4 out. Reference values are from adaptive quadrature in ln(1 + z).
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1 - 1e-12, z=-0.999999) == pytest.approx(
        -1.1881950472843847, rel=1e-11
    )
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.999, z=-0.999999) == pytest.approx(
        -0.02942354607438868, rel=1e-11
    )
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e-8, z=1e8) == pytest.approx(556188062894733.94, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e-4, z=1e5) == pytest.approx(25177127557.013027, rel=1e-11)


def test_luminosity_distance_matter_free() -> None:
    """A matter-free universe expands with E(z) = 1, so D_L = (c / H0) z (1 + z) at every redshift.

    The 2 / u^3 integrand of this limit defeats any fixed-node quadrature at high z, so it is a closed form.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (0.0, 0.01, 1.0, 1e3, 1e8):
        expected = hubble_dist_mpc * z * (1.0 + z)
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.0, z=z) == pytest.approx(expected, rel=1e-13)


def test_read_hesma_lightcurve_file_header(tmp_path: Path) -> None:
    """Column names must come from splitting the comment header into words, not into characters."""
    hesmafile = tmp_path / "hesma_model.dat"
    hesmafile.write_text("# time bol B V\n1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n", encoding="utf-8")

    dfhesma = at.lightcurve.read_hesma_lightcurve_file(hesmafile)

    assert list(dfhesma.columns) == ["time", "bol", "B", "V"]
    assert dfhesma["time"].to_list() == [1.0, 5.0]
    assert dfhesma["V"].to_list() == [4.0, 8.0]


def test_read_hesma_lightcurve_file_no_header(tmp_path: Path) -> None:
    """A file with no comment header uses its first line as the header."""
    hesmafile = tmp_path / "hesma_model_noheader.dat"
    hesmafile.write_text("time bol\n1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    dfhesma = at.lightcurve.read_hesma_lightcurve_file(hesmafile)

    assert list(dfhesma.columns) == ["time", "bol"]
    assert dfhesma["bol"].to_list() == [2.0, 4.0]


def test_hesma_lightcurve_on_a_grid_with_more_panels_than_bands(tmp_path: Path) -> None:
    """Four bands take a grid of six panels. The strict zip of the panels and the bands then stopped the plot."""
    from artistools.lightcurve.plotlightcurve import plot_hesma_lightcurve

    hesmafile = tmp_path / "hesma_model.dat"
    hesmafile.write_text("# t U B V R\n1.0 1.0 2.0 3.0 4.0\n5.0 5.0 6.0 7.0 8.0\n", encoding="utf-8")
    fig, axesgrid = plt.subplots(2, 3)
    axes = list(axesgrid.flatten())
    plot_hesma_lightcurve(axes, ["U", "B", "V", "R"], argparse.Namespace(plot_hesma_model=str(hesmafile)), None)

    assert [len(axis.get_lines()) for axis in axes] == [1, 1, 1, 1, 0, 0]
    assert np.allclose(np.asarray(axes[3].get_lines()[0].get_ydata(), dtype=float), [4.0, 8.0])
    plt.close(fig)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_reference_colors(
    mockplot: mock.MagicMock, mockerrorbar: mock.MagicMock, tmp_path: Path
) -> None:
    """The reference light curves get black and then grey, and the model keeps the first colour of the cycle."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_smarttetal2017.txt", modelpath, "AT2017gfo_waxmanetal2018.txt"],
        outputfile=tmp_path,
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    modelcolors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list if "color" in callargs.kwargs]
    assert modelcolors == ["C0"]


@pytest.mark.parametrize("plotkwargs", [{}, {"filter": ["B"]}, {"colour_evolution": ["B-V"]}])
@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_linewidth_arg(
    mockplot: mock.MagicMock, mockerrorbar: mock.MagicMock, plotkwargs: dict[str, t.Any], tmp_path: Path
) -> None:
    """The -linewidth list sets the line width of each series on the bolometric, band, and colour plots."""
    bolometric = not plotkwargs
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, modelpath, *(["AT2017gfo_waxmanetal2018.txt"] if bolometric else [])],
        linewidth=[0.5, 7.0, 2.5],
        outputfile=tmp_path,
        **plotkwargs,
    )

    linewidths = [
        callargs.kwargs.get("linewidth") for callargs in mockplot.call_args_list if "label" in callargs.kwargs
    ]
    assert linewidths == [0.5, 7.0]

    if bolometric:
        assert mockerrorbar.call_args.kwargs["elinewidth"] == 2.5
        assert mockerrorbar.call_args.kwargs["capthick"] == 2.5


@mock.patch.object(mplax.Axes, "legend", side_effect=mplax.Axes.legend, autospec=True)
def test_lightcurve_plot_keeps_the_series_order(mocklegend: mock.MagicMock, tmp_path: Path) -> None:
    """The legend and the draw order follow the model path list, whatever the position of a reference light curve."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_smarttetal2017.txt", modelpath, "AT2017gfo_waxmanetal2018.txt"],
        label=["first", "second", "third"],
        outputfile=tmp_path,
    )

    assert mocklegend.call_args.kwargs["labels"] == ["first", "second", "third"]

    # matplotlib draws artists of equal zorder in the order that the axes received them
    seriesartists = [
        artist
        for handle in mocklegend.call_args.kwargs["handles"]
        for artist in (handle.get_children() if isinstance(handle, ErrorbarContainer) else [handle])
    ]
    assert len({artist.get_zorder() for artist in seriesartists}) == 1


@mock.patch.object(mplax.Axes, "legend", side_effect=mplax.Axes.legend, autospec=True)
def test_lightcurve_plot_legend_labels_take_the_series_colours(mocklegend: mock.MagicMock, tmp_path: Path) -> None:
    """Each legend label has the colour of its series, for a model line and for a series with error bars."""
    seriescolours = ["tab:green", "tab:red", "0.4"]
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_smarttetal2017.txt", modelpath, "AT2017gfo_waxmanetal2018.txt"],
        color=seriescolours,
        outputfile=tmp_path,
    )

    mocklegend.assert_called_once()
    legend = mocklegend.call_args.args[0].get_legend()
    labelcolours = [mplcolors.to_hex(text.get_color(), keep_alpha=True) for text in legend.get_texts()]
    assert labelcolours == [mplcolors.to_hex(color, keep_alpha=True) for color in seriescolours]


@mock.patch.object(mplax.Axes, "scatter", side_effect=mplax.Axes.scatter, autospec=True)
@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_lightcurve_plot_reflightcurves_continue_the_greys(
    mockerrorbar: mock.MagicMock, mockscatter: mock.MagicMock, tmp_path: Path
) -> None:
    """A -reflightcurves file follows the reference files of the model path list, thus no two series are black."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, "AT2017gfo_smarttetal2017.txt"],
        reflightcurves=["AT2017gfo_waxmanetal2018.txt"],
        outputfile=tmp_path,
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    # both files have error columns, so neither route falls back to a plain scatter
    assert mockscatter.call_args_list == []


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_colors_survive_a_skipped_model(
    mockplot: mock.MagicMock, mockerrorbar: mock.MagicMock, tmp_path: Path
) -> None:
    """A model path that plots nothing must not shift the colour of every later series."""
    at.lightcurve.plot(
        argsraw=[], modelpath=["nonexistentmodelfolder", "AT2017gfo_smarttetal2017.txt", modelpath], outputfile=tmp_path
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0"]

    modelcolors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list if "color" in callargs.kwargs]
    assert modelcolors == ["C1"]


def test_find_bol_reflightcurve_file_reads_a_compressed_file(tmp_path: Path) -> None:
    """A reference light curve that is compressed must be found under the name of the plain file."""
    import lzma

    with lzma.open(tmp_path / "myref.txt.xz", "wt", encoding="utf-8") as compressedfile:
        compressedfile.write("#time_days lum\n1.0 2.0\n")

    assert at.lightcurve.find_bol_reflightcurve_file(tmp_path / "myref.txt") == tmp_path / "myref.txt.xz"
    assert at.lightcurve.path_is_reference_lightcurve(tmp_path / "myref.txt")
    assert at.lightcurve.find_bol_reflightcurve_file(tmp_path / "notafile.txt") is None


REFLIGHTCURVE = "AT2017gfo_smarttetal2017.txt"


def get_reflightcurve_errorbar_call(
    mockerrorbar: mock.MagicMock,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times and luminosities that the reference light curve was drawn with."""
    assert mockerrorbar.call_count == 1
    callargs = mockerrorbar.call_args_list[0]

    return np.array(callargs[0][1]), np.array(callargs[0][2])


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_erg_per_s(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve is plotted in erg/s by default."""
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], outputfile=outputpath / "lightcurve_reflightcurve_ergpers.pdf"
    )

    arr_time_d, arr_lum = get_reflightcurve_errorbar_call(mockerrorbar)

    assert np.isclose(arr_time_d[0], 0.638, rtol=1e-4)
    assert np.isclose(arr_lum[0], 1.1246049739669314e42, rtol=1e-4)

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.isclose(yerr[0][0], 2.7737755982632654e41, rtol=1e-4)
    assert np.isclose(yerr[1][0], 3.681894356120629e41, rtol=1e-4)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_lsun(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve must be converted to Lsun when the axis is in Lsun.

    Before the conversion was applied, the reference data stayed in erg/s and was drawn a factor of
    Lsun_to_erg_per_s above the ARTIS light curves on the same axis.
    """
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], Lsun=True, outputfile=outputpath / "lc_reflc_Lsun.pdf"
    )

    _arr_time_d, arr_lum = get_reflightcurve_errorbar_call(mockerrorbar)

    assert np.isclose(arr_lum[0], 1.1246049739669314e42 / Lsun_to_erg_per_s, rtol=1e-4)
    assert np.isclose(arr_lum[0], 2.9393752586e8, rtol=1e-4)

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.isclose(yerr[0][0], 2.7737755982632654e41 / Lsun_to_erg_per_s, rtol=1e-4)
    assert np.isclose(yerr[1][0], 3.681894356120629e41 / Lsun_to_erg_per_s, rtol=1e-4)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_magnitude(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve must be converted to magnitudes when the axis is in magnitudes."""
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], magnitude=True, outputfile=outputpath / "lc_reflc_mag.pdf"
    )

    _arr_time_d, arr_mag = get_reflightcurve_errorbar_call(mockerrorbar)

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(REFLIGHTCURVE)
    lum_lsun = dflightcurve["luminosity_erg/s"].to_numpy() / Lsun_to_erg_per_s
    assert np.allclose(arr_mag, Mbol_sun - 2.5 * np.log10(lum_lsun), rtol=1e-10)
    assert np.isclose(arr_mag[0], -16.4306375858, rtol=1e-9)

    # the file gives a symmetric error in log10(luminosity), so both magnitude error bars are 2.5 times it
    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    expected_magerr = 2.5 * dflightcurve["log_lbol_err"].to_numpy()
    assert np.allclose(yerr[0], expected_magerr, rtol=1e-3)
    assert np.allclose(yerr[1], expected_magerr, rtol=1e-3)


def test_convert_lum_lsun_to_plotunits() -> None:
    """Deposition rates and reference data must use the same unit conversion as the ARTIS light curves."""
    lum_lsun = np.array([1.0, 1e8])

    assert np.allclose(at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "Lsun"), lum_lsun)

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "erg/s"), lum_lsun * Lsun_to_erg_per_s
    )

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "mag"), [Mbol_sun, Mbol_sun - 20.0]
    )


def test_convert_lum_ergs_to_plotunits() -> None:
    """The erg/s axis must pass the reference data through untouched, with no round trip through Lsun."""
    lum_erg_per_s = np.array([1.1246049739669314e42, 3.0e41])

    assert (at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, "erg/s") == lum_erg_per_s).all()

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, "Lsun"),
        lum_erg_per_s / Lsun_to_erg_per_s,
    )


def test_convert_lum_to_plotunits_nonpositive() -> None:
    """A zero or negative luminosity has no magnitude, and must not raise a numpy warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(np.array([0.0, -1.0, 1.0]), "mag")

    assert np.isinf(result[0])
    assert np.isnan(result[1])
    assert np.isclose(result[2], Mbol_sun)


def test_get_reflightcurve_yerr_magnitude() -> None:
    """Magnitude error bars must reach the magnitudes of the brightest and faintest luminosity bounds.

    matplotlib draws the bar from y - yerr[0] to y + yerr[1], and a brighter (larger) luminosity is a smaller
    magnitude, so the two rows are asymmetric and must not be swapped.
    """
    lum = np.array([1e42, 1e42, 1e42])
    errplus = np.array([1e42, 0.0, 1e42])
    errminus = np.array([0.9e42, 0.0, 2e42])  # the last error bar reaches past zero luminosity

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "mag")
    mag = at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum, "mag")

    assert np.isclose(yerr[0][0], 2.5 * np.log10(2.0))  # brighter by a factor of two
    assert np.isclose(yerr[1][0], 2.5)  # fainter by a factor of ten
    assert yerr[1][0] > yerr[0][0], "the fainter (upper) row must not be swapped with the brighter (lower) row"

    assert np.isclose(mag[0] - yerr[0][0], Mbol_sun - 2.5 * np.log10(2e42 / Lsun_to_erg_per_s))
    assert np.isclose(mag[0] + yerr[1][0], Mbol_sun - 2.5 * np.log10(0.1e42 / Lsun_to_erg_per_s))

    assert np.isclose(yerr[0][1], 0.0)  # a zero error stays zero on both sides
    assert np.isclose(yerr[1][1], 0.0)

    # a bar reaching zero luminosity has no faintest magnitude. A NaN there makes matplotlib drop the whole
    # bar, so the faint side has no length and the point is flagged for an arrow instead
    assert np.allclose(unbounded, [False, False, True])
    assert np.isfinite(yerr[1][2])
    assert np.isclose(yerr[1][2], 0.0)
    assert np.isclose(yerr[0][2], 2.5 * np.log10(2.0))


def test_get_reflightcurve_yerr_nonpositive_luminosity_draws_no_bar() -> None:
    """A luminosity at or below zero has no magnitude, so it must not reach matplotlib as an infinite bar.

    matplotlib adds the error to the value, and inf + -inf warns and yields NaN.
    """
    lum = np.array([0.0, -1e42, 1e42])
    errminus = np.array([1e41, 1e41, 1e41])
    errplus = np.array([1e41, 1e41, 1e41])

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "mag")

    assert np.all(np.isfinite(yerr[0])), yerr[0]
    assert np.all(np.isfinite(yerr[1])), yerr[1]
    assert np.allclose(yerr[0][:2], 0.0)
    assert np.allclose(yerr[1][:2], 0.0)
    # a point with no magnitude is not an open faint side, it is not drawn at all
    assert not unbounded[:2].any()
    assert yerr[0][2] > 0.0


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_unbounded_bar_keeps_the_bright_half_and_gets_an_arrow(
    mockerrorbar: mock.MagicMock, tmp_path: Path
) -> None:
    """An error bar reaching zero luminosity keeps its bright half, plus an arrow pointing at the faint side.

    Putting NaN in the faint row instead makes matplotlib collapse the segment to a single vertex, so the
    finite bright half disappears with it. The arrow direction is fixed at the errorbar call, and a
    magnitude axis is inverted only after every series is drawn, so the default direction is the wrong one.
    """
    reffile = tmp_path / "unbounded_reflightcurve.txt"
    reffile.write_text(
        "#time_days luminosity_erg/s luminosity_errminus_erg/s luminosity_errplus_erg/s\n"
        "2.0 1e42 5e41 5e41\n"
        "3.0 1e42 2e42 5e41\n",
        encoding="utf-8",
    )

    _fig, axis = plt.subplots()
    at.lightcurve.plotlightcurve.plot_bol_reflightcurve(axis, reffile, "mag", color="0.0")
    at.plottools.invert_magnitude_yaxis(axis)

    barcall, arrowcall = mockerrorbar.call_args_list
    yerr = barcall[1]["yerr"]
    assert np.all(np.isfinite(yerr[0]))
    assert np.all(np.isfinite(yerr[1]))
    assert yerr[0][1] > 0.0, "the bright half of the unbounded bar must still be drawn"
    assert np.isclose(yerr[1][1], 0.0), "the faint half has no length, so matplotlib keeps the bright one"

    assert arrowcall[1]["lolims"] is True
    assert np.allclose(arrowcall[0][1], [3.0]), "only the unbounded point gets an arrow"

    ymin, ymax = axis.get_ylim()
    assert ymin > ymax, "this test only means anything on an inverted magnitude axis"
    arrowcontainer = axis.containers[-1]
    assert isinstance(arrowcontainer, ErrorbarContainer)
    carets = [line.get_marker() for line in arrowcontainer.lines[1]]
    assert carets == [mplmarkers.CARETDOWNBASE], carets
    # a downward caret is drawn towards the lower screen positions, which on this axis are the fainter
    # (larger) magnitudes, so the arrow marks the side of the bar that has no end
    screen_y_of_fainter = axis.transData.transform((0.0, max(ymin, ymax)))[1]
    screen_y_of_brighter = axis.transData.transform((0.0, min(ymin, ymax)))[1]
    assert screen_y_of_fainter < screen_y_of_brighter


def test_bol_reflightcurve_empty_label_stays_out_of_the_legend() -> None:
    """An explicit -label "" keeps a positional reference curve out of the legend.

    Only a missing label takes the file metadata: matplotlib gives no legend entry to a series
    labelled "", which is how one is suppressed.
    """
    _fig, axis = plt.subplots()
    plotlabel = at.lightcurve.plotlightcurve.plot_bol_reflightcurve(
        axis, "AT2017gfo_smarttetal2017.txt", "erg/s", color="0.0", label=""
    )
    assert not plotlabel

    plotlabel = at.lightcurve.plotlightcurve.plot_bol_reflightcurve(
        axis, "AT2017gfo_smarttetal2017.txt", "erg/s", color="0.0", label=None
    )
    assert plotlabel
    assert plotlabel != "None"


def test_get_reflightcurve_yerr_scaling() -> None:
    """Without magnitudes the error bar sizes are scaled the same way as the luminosities."""
    lum = np.array([1e42, 2e42])
    errminus = np.array([1e41, 3e41])
    errplus = np.array([2e41, 4e41])

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "erg/s")
    assert np.allclose(yerr[0], errminus)
    assert np.allclose(yerr[1], errplus)
    assert not unbounded.any(), "a luminosity axis reaches zero, so no bar is open-ended"

    yerr, _unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "Lsun")
    assert np.allclose(yerr[0], errminus / Lsun_to_erg_per_s)
    assert np.allclose(yerr[1], errplus / Lsun_to_erg_per_s)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_magnitude_asymmetric(mockerrorbar: mock.MagicMock) -> None:
    """A reference curve with asymmetric luminosity errors gets asymmetric magnitude error bars.

    AT2017gfo_smarttetal2017.txt has errors that are symmetric in log10(luminosity), so its two magnitude
    error bar rows are equal and cannot catch a swapped or symmetric-only implementation.
    """
    reflightcurve = "AT2017gfo_waxmanetal2018.txt"
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[reflightcurve, modelpath],
        magnitude=True,
        outputfile=outputpath / "lc_reflc_mag_asym.pdf",
    )

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(reflightcurve)
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()
    errminus = dflightcurve["luminosity_errminus_erg/s"].to_numpy()
    errplus = dflightcurve["luminosity_errplus_erg/s"].to_numpy()

    # computed from the luminosity ratios rather than from the function under test, so that swapping the two
    # rows of get_reflightcurve_yerr fails here instead of swapping the expectation with it
    expected = [
        2.5 * np.log10((lum_erg_per_s + errplus) / lum_erg_per_s),
        2.5 * np.log10(lum_erg_per_s / (lum_erg_per_s - errminus)),
    ]

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])
    assert not np.allclose(yerr[0], yerr[1], rtol=1e-2), "this file must exercise the asymmetric branch"


@pytest.mark.parametrize("lumunit", ["erg/s", "Lsun", "magnitude"])
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition(mockplot: mock.MagicMock, lumunit: str) -> None:
    """Deposition curves are drawn in the y axis units, in every unit mode.

    plot_energy_rates() appends a suffix to the caller's label and selects its own linestyle and
    colour, so those keys must not also arrive in the **plotkwargs splat, which used to raise
    "got multiple values for keyword argument".
    """
    plotkwargs: dict[str, t.Any] = {} if lumunit == "erg/s" else {lumunit: True}

    with warnings.catch_warnings():
        # a zero deposition rate has no magnitude, but it must not warn
        warnings.simplefilter("error", RuntimeWarning)

        # a free-threading build warns when it takes the GIL back to import fontTools, which matplotlib
        # needs for a pdf. That belongs to the library, thus it must not fail this test
        warnings.filterwarnings("ignore", message=".*global interpreter lock.*", category=RuntimeWarning)

        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotdeposition=True,
            outputfile=outputpath / f"lc_deposition_{lumunit.replace('/', '')}.pdf",
            **plotkwargs,
        )

    gammadep_lsun = at.get_deposition(modelpath_classic_3d).collect()["gammadep_Lsun"].to_numpy()
    with np.errstate(divide="ignore"):
        expected = {
            "erg/s": gammadep_lsun * Lsun_to_erg_per_s,
            "Lsun": gammadep_lsun,
            "magnitude": Mbol_sun - 2.5 * np.log10(gammadep_lsun),
        }[lumunit]

    gammacurves = [
        callargs
        for callargs in mockplot.call_args_list
        if str(callargs.kwargs.get("label", "")).endswith(r"$\dot{E}_{dep,\gamma}$")
    ]
    assert len(gammacurves) == 1
    assert np.allclose(np.asarray(gammacurves[0][0][2], dtype=float), expected, equal_nan=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotthermalisation(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """--plotthermalisation draws the deposition rate over the emission rate of each particle, and the Barnes curves.

    The ratios and the Barnes curves go in the panel below the light curves, and each ratio takes the -linewidth value.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotthermalisation=True,
        linewidth=[2.5],
        outputfile=tmp_path / "lc_thermalisation.pdf",
    )

    depdata = at.get_deposition(modelpath_classic_3d).collect()
    ratiocalls = {
        callargs.kwargs["label"]: callargs
        for callargs in mockplot.call_args_list
        if r"\middle/" in str(callargs.kwargs.get("label", ""))
    }
    assert len(ratiocalls) == 3
    gammacall = next(callargs for label, callargs in ratiocalls.items() if r"dep,\gamma" in label)
    assert np.allclose(
        np.asarray(gammacall.args[2], dtype=float),
        (depdata["gammadep_Lsun"] / depdata["eps_gamma_Lsun"]).to_numpy(),
        equal_nan=True,
    )
    assert {callargs.kwargs["linewidth"] for callargs in ratiocalls.values()} == {2.5}
    barnescalls = [callargs for callargs in mockplot.call_args_list if "Barnes" in str(callargs.kwargs.get("label"))]
    assert len(barnescalls) == 3
    thermaxes = {id(callargs.args[0]) for callargs in [*ratiocalls.values(), *barnescalls]}
    assert len(thermaxes) == 1
    lightcurveaxis = mockplot.call_args_list[0].args[0]
    assert id(lightcurveaxis) not in thermaxes


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_reflightcurves_arg_draws_error_bars(mockerrorbar: mock.MagicMock) -> None:
    """-reflightcurves must draw the same curve as a positional reference file, error bars included.

    This branch used to scatter the points with no uncertainties while the positional branch drew error bars
    from the same file, and it kept its own copy of the unit conversion.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        reflightcurves=[REFLIGHTCURVE],
        magnitude=True,
        outputfile=outputpath / "lc_reflightcurves_arg.pdf",
    )

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(REFLIGHTCURVE)
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()

    _arr_time_d, arr_mag = get_reflightcurve_errorbar_call(mockerrorbar)
    assert np.allclose(arr_mag, Mbol_sun - 2.5 * np.log10(lum_erg_per_s / Lsun_to_erg_per_s))

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    expected, _unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(
        lum_erg_per_s,
        dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
        dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
        "mag",
    )
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])


def test_get_plot_lum_unit_and_column() -> None:
    """The unit choice and the light curve column to plot come from one place."""
    for magnitude, lsun, expected_unit, expected_col in (
        (True, False, "mag", "mag"),
        (True, True, "mag", "mag"),  # magnitude wins over Lsun
        (False, True, "Lsun", "luminosity_Lsun"),
        (False, False, "erg/s", "luminosity_erg/s"),
    ):
        args = argparse.Namespace(magnitude=magnitude, Lsun=lsun)
        lumunit = at.lightcurve.plotlightcurve.get_plot_lum_unit(args)
        assert lumunit == expected_unit
        assert at.lightcurve.plotlightcurve.get_plot_lum_column(lumunit) == expected_col


@pytest.mark.parametrize(
    ("dirbins", "colour_evolution"),
    [([0], ["B-V"]), ([0, 1], ["B-V"]), ([0], ["U-B", "B-V"]), ([0, 1], ["U-B", "B-V"])],
)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_yaxis_is_inverted(
    mockplot: mock.MagicMock, dirbins: list[int], colour_evolution: list[str]
) -> None:
    """The colour axis points downwards however many direction bins and filter pairs are drawn.

    invert_yaxis() toggles rather than sets, and the subplots share one y axis, so inverting once per plotted
    line left the axis the right way up whenever an even number of lines was drawn.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=colour_evolution,
        plotviewingangle=dirbins,
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_colourevolution_inverted.pdf",
    )

    axes = {id(callargs[0][0]): callargs[0][0] for callargs in mockplot.call_args_list}
    assert axes
    for axis in axes.values():
        ymin, ymax = axis.get_ylim()
        assert ymin > ymax, f"the colour axis is not inverted for {len(dirbins)} bins and {colour_evolution}"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_dirbin_colours_avoid_the_model_colours(mockplot: mock.MagicMock) -> None:
    """A direction bin must not be given the colour that a whole model was assigned.

    The direction bin palette and the per-model colours are two independent sources drawn on one set of axes,
    and both used to start at the first colour of the matplotlib cycle.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, modelpath_classic_3d],
        colour_evolution=["B-V"],
        plotviewingangle=[0, 1],
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_colourevolution_colours.pdf",
    )

    colors = [mplcolors.to_hex(callargs.kwargs["color"]) for callargs in mockplot.call_args_list]
    assert len(colors) == 3
    assert len(set(colors)) == len(colors), colors


@pytest.mark.parametrize("plotalpha", [False, True])
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotalphadeposition_draws_the_alpha_curves(mockplot: mock.MagicMock, plotalpha: bool) -> None:
    """The alpha decay curves are drawn only when -plotalphadeposition asks for them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotdeposition=not plotalpha,
        plotalphadeposition=plotalpha,
        outputfile=outputpath / "lc_alphadep.pdf",
    )

    labels = [str(callargs.kwargs.get("label")) for callargs in mockplot.call_args_list]
    assert any(r"\alpha" in label for label in labels) == plotalpha, labels
    # the gamma and beta curves are drawn either way, so the alpha rates can be compared against them
    assert any(r"\gamma" in label for label in labels), labels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_deposition_curves_do_not_reuse_a_model_colour(mockplot: mock.MagicMock) -> None:
    """The deposition curves take their colours from the axis cycle, which the model colours were removed from.

    A hardcoded colour would collide with whichever model get_series_colors happened to give that colour to.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d, modelpath_classic_3d],
        plotalphadeposition=True,
        timemin=3,
        timemax=8,
        outputfile=outputpath / "lc_deposition_colours.pdf",
    )

    lines = [
        (str(callargs.kwargs.get("label")), color)
        for callargs in mockplot.call_args_list
        if (color := callargs.kwargs.get("color")) is not None
    ]
    modelcolors = {mplcolors.to_hex(color) for label, color in lines if "dot" not in label}
    depositioncolors = {mplcolors.to_hex(color) for label, color in lines if "dot" in label}

    assert modelcolors
    assert depositioncolors
    assert not modelcolors & depositioncolors, f"{modelcolors} and {depositioncolors} share a colour"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition_does_not_inherit_the_light_curve_style(mockplot: mock.MagicMock) -> None:
    """The deposition curves take their style from the command line, not from the last direction bin drawn.

    plot_artis_lightcurve mutates its plot kwargs per direction bin, and those used to ride into the
    deposition curves, drawing them semi-transparent and above the light curves for no stated reason.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotdeposition=True,
        plotviewingangle=[0, 1],
        colorbarphi=True,
        outputfile=outputpath / "lc_deposition_style.pdf",
    )

    depositionkwargs = [
        callargs.kwargs for callargs in mockplot.call_args_list if r"\dot{E}" in str(callargs.kwargs.get("label"))
    ]
    assert depositionkwargs
    for kwargs in depositionkwargs:
        assert "alpha" not in kwargs
        assert "zorder" not in kwargs


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition_is_drawn_once_per_model(mockplot: mock.MagicMock) -> None:
    """A model draws its deposition rates once, however many light curve series it contributes.

    plot_artis_lightcurve runs once per escape type and once per top nuclide, and the deposition rates were
    drawn on every run: the gamma-ray series repeated all three curves in new colours, labelled with the
    gamma-ray series name even though the deposition rates have nothing to do with the escape type.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        gamma=True,
        rpkt=True,
        plotdeposition=True,
        outputfile=outputpath / "lc_deposition_gamma_and_rpkt.pdf",
    )

    depositionlabels = [
        str(callargs.kwargs.get("label"))
        for callargs in mockplot.call_args_list
        if r"\dot{E}" in str(callargs.kwargs.get("label"))
    ]
    assert depositionlabels
    assert len(depositionlabels) == len(set(depositionlabels)), depositionlabels
    # the deposition rates are a model property, so they keep the model name rather than the gamma series name
    assert not any(r"$\gamma$ $\dot{E}" in label for label in depositionlabels), depositionlabels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_one_sided_time_limit_keeps_the_data_in_view(mockplot: mock.MagicMock) -> None:
    """A -timemin with no -timemax leaves the right hand side fitted to the light curve.

    Setting a one-sided limit before anything is drawn turns autoscaling off, freezing the other side at the
    default 0-1 view instead of the range of the data.
    """
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, timemin=250.0, outputfile=outputpath / "lc_timemin_only.pdf")

    axis = mockplot.call_args_list[0][0][0]
    xmin, xmax = axis.get_xlim()
    assert np.isclose(xmin, 250.0)
    assert xmax > 300.0, xmax


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_gamma_lightcurve_magnitude_ylabel(mockylabel: mock.MagicMock) -> None:
    """A gamma-ray light curve in magnitudes is not a bolometric magnitude, and must not be labelled as one."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        gamma=True,
        rpkt=False,
        magnitude=True,
        outputfile=outputpath / "lc_gamma_mag.pdf",
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert ylabels == [r"Absolute $\gamma$-ray Magnitude"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_averaged_direction_bin_magnitude_is_rebuilt(mockplot: mock.MagicMock) -> None:
    """Averaging direction bins averages the luminosity, so the magnitude must be rebuilt from the result.

    average_direction_bins takes the mean of every column, and a magnitude is logarithmic: the mean of the
    magnitudes is not the magnitude of the mean luminosity, and one dark bin sends the mean to inf.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        magnitude=True,
        plotviewingangle=[0],
        average_over_phi_angle=True,
        outputfile=outputpath / "lc_averagedphi_mag.pdf",
    )

    lcpath = at.firstexisting("light_curve_res.out", folder=modelpath_classic_3d, tryzipped=True)
    averaged = at.misc.average_direction_bins(
        at.lightcurve.scan_lightcurve(lcpath, directionresolved=True), overangle="phi"
    )[0].collect()
    lum_lsun_by_time = dict(zip(averaged["time_days"], averaged["luminosity_Lsun"], strict=True))
    meanofmags_by_time = dict(zip(averaged["time_days"], averaged["mag"], strict=True))

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    arr_mag = np.array(mockplot.call_args_list[0][0][2])
    assert arr_time_d.size > 0

    with np.errstate(divide="ignore"):
        expected = Mbol_sun - 2.5 * np.log10(np.array([lum_lsun_by_time[t_d] for t_d in arr_time_d]))
    assert np.allclose(arr_mag, expected, equal_nan=True)

    meanofmags = np.array([meanofmags_by_time[t_d] for t_d in arr_time_d])
    assert not np.allclose(arr_mag, meanofmags, equal_nan=True), "the stale mean of the magnitudes was plotted"
    assert np.isfinite(arr_mag).sum() > np.isfinite(meanofmags).sum(), "a dark bin still poisons the average"


def test_colour_evolution_plot_leaves_the_filter_arg_alone(tmp_path: Path) -> None:
    """The band selection must not be written back onto args.filter, which main() dispatches on.

    A caller that reuses one Namespace would otherwise take the band light curve branch on the second call
    and silently draw a different plot.
    """
    parser = argparse.ArgumentParser()
    at.lightcurve.addargs(parser)
    args = parser.parse_args([])
    args.modelpath = [modelpath]
    args.colour_evolution = ["U-B", "B-V"]
    args.outputfile = tmp_path

    at.lightcurve.plot(args=args)

    assert not args.filter


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_lightcurve_ylabel_names_deposition_only_when_it_is_drawn(mockylabel: mock.MagicMock) -> None:
    """Asking for deposition rates is not drawing them, so a run with no ARTIS model must not claim them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[REFLIGHTCURVE],
        plotdeposition=True,
        outputfile=outputpath / "lc_reflc_only_deposition.pdf",
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert all(r"\dot{E}" not in ylabel for ylabel in ylabels), ylabels


def run_set_scatterplot_plot_params(
    magnitudes: Sequence[float], ymin: float | None, ymax: float | None
) -> tuple[float, float, float]:
    """Plot the magnitudes, apply the shared scatter plot setup, and return xmax, ymin, ymax."""
    _fig, axis = plt.subplots()
    axis.plot([1.0, 10.0], list(magnitudes))
    args = argparse.Namespace(colouratpeak=False, ymin=ymin, ymax=ymax, colorbarcostheta=False, colorbarphi=False)

    viewingangleanalysis.set_scatterplot_plot_params(axis, args)

    ymin, ymax = axis.get_ylim()
    return axis.get_xlim()[1], ymin, ymax


def test_set_scatterplot_plot_params_without_xmin() -> None:
    """The light curve parser names its range -timemin/-timemax, so the scatter setup must not read args.xmin.

    Reading args.xmin raised AttributeError and made every --make_viewing_angle_peakmag_* run fail.
    """
    xmax, ymin, ymax = run_set_scatterplot_plot_params([15.0, 19.0], ymin=None, ymax=None)

    assert ymin > ymax, "the magnitude axis must point downwards"
    # neither side falls back to the default 0-1 view when no limit was given
    assert xmax >= 10.0
    assert ymin >= 19.0


def test_readfile_rebuilds_the_magnitude_after_averaging() -> None:
    """The loader owns the averaging, so the magnitude cannot be left as the mean of the magnitudes.

    Averaging is linear and a magnitude is logarithmic, so a single dark bin sends the arithmetic mean of
    the magnitudes to inf while the magnitude of the averaged luminosity stays finite.
    """
    lcpath = at.firstexisting("light_curve_res.out", folder=modelpath_classic_3d, tryzipped=True)

    averaged = at.lightcurve.scan_lightcurve(lcpath, directionresolved=True, average_over_phi=True)[0].collect()
    stalemean = at.misc.average_direction_bins(
        at.lightcurve.scan_lightcurve(lcpath, directionresolved=True), overangle="phi"
    )[0].collect()

    with np.errstate(divide="ignore"):
        magofmeanlum = Mbol_sun - 2.5 * np.log10(averaged["luminosity_Lsun"].to_numpy())

    assert np.allclose(averaged["mag"].to_numpy(), magofmeanlum, equal_nan=True)
    assert not np.allclose(averaged["mag"].to_numpy(), stalemean["mag"].to_numpy(), equal_nan=True)


def test_color_arg_rejects_a_colour_matplotlib_cannot_parse() -> None:
    """A mistyped colour must be named by the argument that took it, not by a colour cycle helper."""
    parser = argparse.ArgumentParser()
    at.lightcurve.addargs(parser)

    assert parser.parse_args(["-color", "tab:blue", "#1F77B4"]).color == ["tab:blue", "#1F77B4"]

    for argname in ("-color", "-refspeccolors"):
        with pytest.raises(SystemExit):
            parser.parse_args([argname, "notacolour"])


def test_transparent_series_colour_leaves_the_cycle_alone() -> None:
    """A series drawn in "none" holds no colour, so it must not remove black from the palette."""
    assert not at.plottools.get_assigned_colors(["none"])
    assert mplcolors.to_hex("#000000") in {
        mplcolors.to_hex(color) for color in at.plottools.get_unused_colors(["#000000", "#ffffff"], ["none"])
    }


@pytest.mark.parametrize("plottype", ["bolometric", "band", "colour_evolution"])
@mock.patch.object(mplax.Axes, "set_xlim", side_effect=mplax.Axes.set_xlim, autospec=True)
def test_time_limits_reach_every_figure(mockxlim: mock.MagicMock, plottype: str, tmp_path: Path) -> None:
    """-timemin/-timemax are this command's x limits, so every figure it draws must honour them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        timemin=260,
        timemax=300,
        outputfile=tmp_path,
        filter=["B"] if plottype == "band" else [],
        colour_evolution=["B-V"] if plottype == "colour_evolution" else [],
    )

    limits = [callargs[0][1:3] for callargs in mockxlim.call_args_list if len(callargs[0]) > 2]
    assert (260, 300) in limits, limits


def test_nonpositive_limit_on_a_log_axis_is_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    """A log axis cannot show a limit at or below zero, so it is dropped with a message naming the argument."""
    _fig, axis = plt.subplots()
    axis.plot([1.0, 10.0], [1e40, 1e44])
    args = argparse.Namespace(logscaley=True, ymin=0.0, ymax=1e45, xmin=None, xmax=None)

    at.plottools.set_axis_properties(axis, args)

    # a warning goes to the standard error, thus --quiet keeps it
    assert "-ymin" in capsys.readouterr().err
    ymin, ymax = axis.get_ylim()
    assert ymin > 0.0, "the data must stay in view rather than the axis freezing at a rejected limit"
    assert np.isclose(ymax, 1e45)


def test_viewing_angle_scatter_needs_the_angle_averaged_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Writing the per-direction-bin data and plotting it are two runs, so asking for both must say so."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="angle-averaged"):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B"],
            timemin=250,
            timemax=300,
            outputfile=tmp_path,
            save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
            make_viewing_angle_peakmag_risetime_scatter_plot=True,
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_reflightcurve_is_drawn_once_per_panel(mockplot: mock.MagicMock) -> None:
    """A band reference file is read and drawn once for the whole figure, not once per band.

    plot_lightcurve_from_refdata already draws every band of the file onto its own panel, so calling it
    from inside the band loop drew each reference curve once per band. It also asserted a single axes,
    which a figure with more than one band never has, so -filter B V -reflightcurves raised AssertionError.
    """
    refdata = pl.DataFrame({
        "band": ["B", "B", "V", "V"],
        "time": [260.0, 280.0, 260.0, 280.0],
        "magnitude": [-13.0, -12.5, -13.2, -12.7],
    })

    with mock.patch.object(
        at.lightcurve.plotlightcurve, "read_reflightcurve_band_data", return_value=(refdata, {"label": "refband"})
    ) as mockread:
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B", "V"],
            reflightcurves=["fakeref.dat"],
            outputfile=outputpath / "lc_band_ref_once.pdf",
        )

    assert mockread.call_count == 1, "the reference file must be read once, not once per band"

    reflines = [callargs for callargs in mockplot.call_args_list if callargs[1].get("label") == "refband"]
    assert len(reflines) == 2, "one reference curve per band panel"
    assert {tuple(np.asarray(callargs[0][1])) for callargs in reflines} == {(260.0, 280.0)}


# no autospec: on Python 3.15 the imported name is a lazy proxy until its first use, and a spec of it is not callable
@mock.patch.object(at.lightcurve.plotlightcurve, "get_next_color", wraps=at.plottools.get_next_color)
def test_alpha_deposition_colour_is_taken_only_when_it_is_drawn(mockcolor: mock.MagicMock) -> None:
    """A colour taken but not drawn steps every later series along the cycle for nothing."""
    for plotalphadeposition, expected_extra in ((False, 0), (True, 1)):
        _fig, axis = plt.subplots()
        args = argparse.Namespace(
            plotdeposition=True, plotalphadeposition=plotalphadeposition, plotthermalisation=False,
            magnitude=False, Lsun=False,
        )  # fmt: skip
        at.lightcurve.plotlightcurve.resolve_energy_rate_args(args)

        mockcolor.reset_mock()
        at.lightcurve.plotlightcurve.plot_energy_rates(axis, None, modelpath_classic_3d, "modelname", args)

        # one colour skipped plus gamma and beta, and the alpha colour only when its curves are drawn
        assert mockcolor.call_count == 3 + expected_extra


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_plot_colorbar_keeps_the_labels_it_does_not_name(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The colour bar names a direction bin, thus it takes the legend entry of a direction bin alone.

    The band plot gave no label to any series of a model with a colour bar. Thus the angle-averaged curve
    and a -label that the user gave both left the legend.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        filter=["B"],
        plotviewingangle=[-1, 0],
        colorbarcostheta=True,
        label=["My model"],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    assert len(labels) == 2
    assert labels[0] == "My model", "the angle-averaged curve is not a direction bin of the colour bar"
    assert labels[1] is None, labels
    assert labels.count("My model") == 1, "the legend gives a custom -label one time only"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotcmf_leaves_the_next_direction_bin_alone(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The comoving frame curve is thin and dashed, and the next direction bin keeps the rest-frame style.

    The two styles went into the plot kwargs that every direction bin shares. Thus every curve after
    the first comoving frame curve was dashed. Such a curve also lost the width that -linewidth gave.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotcmf=True,
        plotviewingangle=[0, 1],
        linewidth=[3.0],
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    styles = [
        (callargs.kwargs.get("linewidth"), callargs.kwargs.get("linestyle")) for callargs in mockplot.call_args_list
    ]
    assert len(styles) == 4
    # one rest-frame curve and one comoving frame curve for each of the two direction bins
    assert styles[0] == styles[2] == (3.0, None)
    assert styles[1] == styles[3] == (1, "dashed")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotcmf_draws_the_unlabelled_series_of_a_colorbar(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A colour bar leaves a series with no label, thus the comoving frame curve has no label to mark.

    The command asserted that the label was there, thus --plotcmf with a colour bar stopped with an
    AssertionError before it drew anything.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotcmf=True,
        plotviewingangle=[0, 1],
        colorbarphi=True,
        timemin=5,
        timemax=8,
        outputfile=tmp_path,
    )

    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    assert labels == [None] * 4


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_plot_first_dirbin_takes_the_color_arg(mockplot: mock.MagicMock) -> None:
    """A -color value is removed from the cycle for the model, so one of its lines has to use it.

    The band figure gave every line of a multi-bin model a cycle colour, so the requested colour was
    reserved and then drawn nowhere. The bolometric figure has always given it to the first bin.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        filter=["B"],
        plotviewingangle=[0, 1],
        color=["magenta"],
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_band_dirbin_color.pdf",
    )

    colors = [
        mplcolors.to_hex(callargs[1]["color"]) for callargs in mockplot.call_args_list if callargs[1].get("color")
    ]
    assert mplcolors.to_hex("magenta") in colors, colors


def test_scatterplot_magnitude_axis_stays_inverted_with_limits() -> None:
    """set_ylim re-sorts the pair it is given, so an inversion applied before it is lost."""
    _xmax, ymin, ymax = run_set_scatterplot_plot_params([-19.0, -15.0], ymin=-20.0, ymax=-14.0)

    assert ymin > ymax, "the magnitude axis must point downwards even when -ymin/-ymax are given"
    assert np.isclose(ymin, -14.0)
    assert np.isclose(ymax, -20.0)


def test_lightcurve_print_data_survives_quiet(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """--print_data prints the plotted data, thus --quiet must keep it.

    addarg_quiet names the arguments whose product is the standard output, and run_command reads them.
    Only the dispatcher applies the redirection, thus the test gives a command line and not a keyword.
    """
    import artistools.__main__

    common = ["plotlightcurves", str(modelpath), "-o", str(tmp_path)]
    artistools.__main__.main(argsraw=[*common, "--print_data", "--quiet"])
    withdata = capsys.readouterr().out

    artistools.__main__.main(argsraw=[*common, "--quiet"])
    withoutdata = capsys.readouterr().out

    assert not withoutdata, "--quiet alone must hide the progress messages"
    assert withdata, "print_product writes the product, thus --quiet must keep it"

    # a script reads the product, thus no progress message may stand around it
    assert "====>" not in withdata
    assert "Reading " not in withdata
    assert withdata.lstrip().startswith("shape:")


def test_lightcurve_reference_data_takes_a_day_range(tmp_path: Path) -> None:
    """A range in days needs no timestep data, thus reference data alone still takes -timedays.

    apply_time_range_args gave the first path to get_time_range, which searches for the files of an
    ARTIS model. A reference light curve holds none, thus the command stopped before it drew anything.
    """
    at.lightcurve.plot(argsraw=["AT2017gfo_smarttetal2017.txt", "-timedays", "2-5", "-o", str(tmp_path / "ref.pdf")])

    assert (tmp_path / "ref.pdf").is_file()


def test_lightcurve_day_range_of_a_model_clamps_to_its_timesteps() -> None:
    """A model in the list gives the timestep times, thus the range still clamps to a timestep edge."""
    import argparse

    from artistools.misc import apply_time_range_args

    withmodel = argparse.Namespace(timestep=None, timedays="260-300", timemin=None, timemax=None)
    apply_time_range_args(withmodel, [modelpath])

    reference = argparse.Namespace(timestep=None, timedays="260-300", timemin=None, timemax=None)
    apply_time_range_args(reference, ["AT2017gfo_smarttetal2017.txt"])

    # the model clamps to the start and the end of a timestep, and the reference data take the range
    assert withmodel.timemin > 260.0
    assert withmodel.timemax < 300.0
    assert np.isclose(reference.timemin, 260.0)
    assert np.isclose(reference.timemax, 300.0)


CLASSIC1DPATH = at.get_path("testdata") / "test-classicmode_1d"


def test_lightcurve_timestep_must_mean_the_same_days_for_every_model() -> None:
    """A timestep names different days on a different timestep grid.

    The plot holds one time axis, thus the days of the first model filtered every model, and the second
    curve showed another timestep without a word. Two grids that disagree now stop the command, and
    -timedays serves both because a day means the same everywhere.
    """
    import argparse

    from artistools.misc import apply_time_range_args

    def build(timestep: str | None, timedays: str | None) -> argparse.Namespace:
        return argparse.Namespace(timestep=timestep, timedays=timedays, timemin=None, timemax=None)

    # one grid, or the same grid twice, resolves the timestep as before
    sameargs = build("40", None)
    apply_time_range_args(sameargs, [modelpath, modelpath])
    assert sameargs.timemin is not None

    with pytest.raises(SystemExit) as excinfo:
        apply_time_range_args(build("40", None), [modelpath, CLASSIC1DPATH])
    assert excinfo.value.code == 1

    # a range in days means the same for every model, thus it needs no agreement between the grids
    daysargs = build(None, "260-300")
    apply_time_range_args(daysargs, [modelpath, CLASSIC1DPATH])
    assert daysargs.timemin is not None


def test_lightcurve_timestep_takes_a_path_that_names_a_file(tmp_path: Path) -> None:
    """A path can name the light curve file of a run, and -timestep must read the folder of that run.

    get_time_range reads timesteps.out or input.txt of a run. The file went to it as it came, thus
    "light_curve.out -timestep 20" asked for light_curve.out/input.txt and stopped.
    """
    outputfile = tmp_path / "lc.pdf"
    at.lightcurve.plot(argsraw=[str(modelpath / "light_curve.out"), "-timestep", "20", "-o", str(outputfile)])

    assert outputfile.is_file(), "the command must draw the light curve of the file"


def test_angle_averaged_peakmag_without_filter_reads_the_averaged_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--save_angle_averaged_peakmag_risetime_delta_m15_to_file without -filter fits dirbin -1.

    The branch forced dirbins to [-1] after it read the direction-resolved file, whose keys are
    0 to 99. Thus the command stopped with KeyError before this test existed.
    """
    monkeypatch.chdir(tmp_path)
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotviewingangle=-2,
        save_angle_averaged_peakmag_risetime_delta_m15_to_file=True,
        timemin=3.2,
        timemax=7.5,
        outputfile=tmp_path,
    )

    assert list(tmp_path.glob("*angle_averaged_all_models_data.txt"))


def test_viewing_angle_fit_plot_keeps_magnitude_limits_inverted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-ymin and -ymax in the order of a magnitude axis must stay inverted. invert_yaxis() toggled them back."""
    import argparse

    import matplotlib.figure as mplfig

    parser = argparse.ArgumentParser()
    at.lightcurve.plotlightcurve.addargs(parser)
    args = parser.parse_args(["-ymin", "-14", "-ymax", "-20"])
    args.timemin, args.timemax = 1.0, 20.0

    savedfigures: list[mplfig.Figure] = []

    def save_figure_spy(fig: mplfig.Figure, *_args: t.Any, **_kwargs: t.Any) -> None:
        savedfigures.append(fig)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(at.lightcurve.viewingangleanalysis, "save_figure", save_figure_spy)
    at.lightcurve.viewingangleanalysis.make_plot_test_viewing_angle_fit(
        time=[1.0, 5.0, 20.0],
        magnitude=np.array([-16.0, -18.0, -15.0]),
        xfit=[1.0, 5.0, 20.0],
        fxfit=[-16.0, -18.0, -15.0],
        key="B",
        mag_after15days_polyfit=-15.5,
        tmax_polyfit=5.0,
        time_after15days_polyfit=20.0,
        modelname="model",
        angle=0,
        args=args,
    )

    ymin, ymax = savedfigures[0].axes[0].get_ylim()
    assert ymin > ymax, "the plot must draw a brighter magnitude higher"


def test_viewing_angle_peakmag_export_without_filter_fits_each_direction_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--save_viewing_angle_peakmag_risetime_delta_m15_to_file without -filter fits each selected bin.

    Only the angle-averaged modes force dirbin -1. This export must keep the parsed direction bins
    and read the direction-resolved light curve file. The first column names the bin of each row, thus
    a reader needs no direction bin selection of its own.
    """
    monkeypatch.chdir(tmp_path)

    def export_dirbins(dirbins: list[int], folder: Path) -> npt.NDArray[np.float64]:
        folder.mkdir()
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotviewingangle=dirbins,
            save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
            timemin=3.2,
            timemax=7.5,
            outputfile=folder,
        )
        (datafile,) = folder.glob("*_viewing_angle_data.txt")
        return np.loadtxt(datafile, skiprows=1, ndmin=2)

    twobins = export_dirbins([0, 1], tmp_path / "twobins")
    assert twobins.shape == (2, 4), "the export holds one row per selected direction bin"
    assert twobins[:, 0].tolist() == [0.0, 1.0], "the first column names the direction bin of each row"

    # the values of a direction bin must not depend on the other bins of the selection
    onebin = export_dirbins([1], tmp_path / "onebin")
    assert onebin.shape == (1, 4)
    assert np.allclose(onebin[0], twobins[1])


def test_viewing_angle_peakmag_without_filter_refuses_virtual_observers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without -filter the fit reads light_curve_res.out, which holds no virtual packet observer.

    The command read the direction bins of the real packets that had the numbers of the observers, and wrote that data
    as observer data.
    """
    modelcopy = tmp_path / "model"
    modelcopy.mkdir()
    for sourcefile in modelpath_classic_3d.iterdir():
        (modelcopy / sourcefile.name).symlink_to(sourcefile)
    (modelcopy / "vpkt.txt").symlink_to(at.get_path("testdata") / "vspecpolmodel" / "vpkt.txt")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelcopy],
            plotvspecpol=[0],
            save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
            timemin=3.2,
            timemax=7.5,
            outputfile=tmp_path,
        )

    # SystemExit holds the status alone, thus the message of the command is the text that it printed
    assert "light_curve_res.out does not hold" in capsys.readouterr().err
    assert not list(tmp_path.glob("*_viewing_angle_data.txt"))


def test_band_peakmag_export_writes_one_file_for_each_band(tmp_path: Path) -> None:
    """Each band has its own data file with one row for each direction bin, and the decline rate is positive."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath,
        filter=["bol", "B"],
        plotviewingangle=-1,
        timemin=250,
        timemax=300,
        save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
        outputfile=tmp_path,
    )

    for band in ("bol", "B"):
        (datafile,) = tmp_path.glob(f"{band}band_*_viewing_angle_data.txt")
        data = np.loadtxt(datafile, skiprows=1, ndmin=2)
        assert data.shape == (1, 4)
        assert data[0, 0] == -1, "the first column names the angle-averaged direction bin"
        assert data[0, 3] > 0.0


def test_angle_averaged_export_with_two_bands_gives_one_row_for_each_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The angle-averaged writers index their lists by model, thus two bands must not give two rows for one model."""
    monkeypatch.chdir(tmp_path)
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        filter=["bol", "B"],
        timemin=250,
        timemax=300,
        save_angle_averaged_peakmag_risetime_delta_m15_to_file=True,
        outputfile=tmp_path,
    )

    (datafile,) = tmp_path.glob("*angle_averaged_all_models_data.txt")
    # a header line and one line for the one model
    assert len(datafile.read_text(encoding="utf-8").splitlines()) == 2


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_escape_type_selects_the_packet_type(mockylabel: mock.MagicMock) -> None:
    """-escape_type is the old spelling of --rpkt and --gamma, and it must select the packet type.

    The argument reached no reader, thus -escape_type TYPE_GAMMA gave an R-packet light curve. A
    script that holds the old spelling must still run, thus the argument stays as a hidden alias.
    """
    # parse_cli_args takes argsraw only when it gets no keyword argument, thus every option goes here
    at.lightcurve.plot(
        argsraw=[
            "-modelpath",
            str(modelpath_classic_3d),
            "-escape_type",
            "TYPE_GAMMA",
            "--magnitude",
            "-o",
            str(outputpath / "lc_escapetype_gamma.pdf"),
        ]
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert ylabels == [r"Absolute $\gamma$-ray Magnitude"]

    mockylabel.reset_mock()

    at.lightcurve.plot(
        argsraw=[
            "-modelpath",
            str(modelpath_classic_3d),
            "-escape_type",
            "TYPE_RPKT",
            "--magnitude",
            "-o",
            str(outputpath / "lc_escapetype_rpkt.pdf"),
        ]
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert ylabels == ["Absolute Bolometric Magnitude"]


def test_find_lightcurve_file_refuses_a_direction_resolved_gamma_request() -> None:
    """ARTIS writes no direction-resolved gamma-ray light curve, thus the request must not read the UVOIR file."""
    with pytest.raises(FileNotFoundError, match="direction-resolved gamma"):
        at.lightcurve.find_lightcurve_file(modelpath_classic_3d, directionresolved=True, gamma=True)

    # each request on its own still names the file that holds it
    assert at.lightcurve.find_lightcurve_file(modelpath_classic_3d).name.startswith("light_curve.out")
    assert at.lightcurve.find_lightcurve_file(modelpath_classic_3d, directionresolved=True).name.startswith(
        "light_curve_res.out"
    )
    assert at.lightcurve.find_lightcurve_file(modelpath_classic_3d, gamma=True).name.startswith("gamma_light_curve.out")


@pytest.mark.parametrize("refispositional", [False, True])
def test_bolometric_residual_panel_gives_the_rms_residual(tmp_path: Path, refispositional: bool) -> None:
    """A reference luminosity of 1.2 times the model gives a relative RMS residual of 0.2 / 1.2."""
    dfmodel = (
        at.lightcurve
        .scan_lightcurve(at.lightcurve.find_lightcurve_file(modelpath))[-1]
        .filter(pl.col("time_days").is_between(260.0, 330.0))
        .gather_every(5)
        .collect()
    )
    obsfile = tmp_path / "fakebolobs.txt"
    lum = pl.col("luminosity_erg/s")
    obstext = "#time_days luminosity_erg/s luminosity_errminus_erg/s luminosity_errplus_erg/s\n" + dfmodel.select(
        "time_days", lum * 1.2, (lum * 0.2).alias("errminus"), (lum * 0.4).alias("errplus")
    ).write_csv(separator=" ", include_header=False)
    obsfile.write_text(obstext, encoding="utf-8")
    obsfile.with_name(f"{obsfile.name}.meta.yml").write_text("dist_mpc: 1\nlabel: fake bolometric\n", encoding="utf-8")

    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, obsfile] if refispositional else [modelpath],
        reflightcurves=[] if refispositional else [str(obsfile)],
        residuals=1,
        write_data=True,
        outputfile=tmp_path / "bolresiduals.pdf",
    )
    dfstats = pl.read_csv(tmp_path / "bolresiduals_residuals.csv")
    assert dfstats["reference"].item() == "fake bolometric"
    assert dfstats["npoints"].item() == dfmodel.height
    assert np.isclose(dfstats["rms_relative"].item(), 0.2 / 1.2, rtol=0.05)
    assert np.isclose(
        dfstats["rms_relative"].item(), dfstats["rms"].item() / (1.2 * dfmodel["luminosity_erg/s"].to_numpy().mean())
    )


@pytest.mark.parametrize("filtername", [None, "B"])
def test_lightcurve_residual_panel_compares_two_models(tmp_path: Path, filtername: str | None) -> None:
    """The default baseline is the first model in a bolometric plot or a band plot."""
    at.lightcurve.plot(
        argsraw=["-residuals"],
        modelpath=[modelpath, modelpath],
        filter=[filtername] if filtername is not None else None,
        label=["baseline", "comparison"],
        write_data=True,
        outputfile=tmp_path / "models.pdf",
    )
    dfstats = pl.read_csv(tmp_path / "models_residuals.csv")
    assert dfstats["model"].item() == "comparison"
    assert dfstats["reference"].item() == "baseline"
    assert np.isclose(dfstats["rms"].item(), 0.0)


@pytest.mark.parametrize(
    ("rpkt", "baselineindex"), [(False, 0), (False, 1), (True, 0), (True, 1), (True, 2), (True, 3)]
)
def test_residual_baseline_counts_gamma_lightcurves(tmp_path: Path, rpkt: bool, baselineindex: int) -> None:
    """Gamma light curves count in plot order in gamma-only plots and mixed plots."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d, modelpath_classic_3d],
        label=["model1", "model2"],
        gamma=True,
        rpkt=rpkt,
        residuals=baselineindex,
        write_data=True,
        outputfile=tmp_path / "gamma.pdf",
    )
    labels = [r"model1 $\gamma$", r"model2 $\gamma$"]
    if rpkt:
        labels = ["model1", labels[0], "model2", labels[1]]
    dfstats = pl.read_csv(tmp_path / "gamma_residuals.csv")
    assert dfstats["reference"].to_list() == [labels[baselineindex]] * (len(labels) - 1)
    assert dfstats["model"].to_list() == [label for index, label in enumerate(labels) if index != baselineindex]
    assert (dfstats["npoints"] > 0).all()


@pytest.mark.parametrize(("modelcount", "baselineindex"), [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)])
def test_residual_baseline_counts_comoving_frame_curves(tmp_path: Path, modelcount: int, baselineindex: int) -> None:
    """Comoving frame curves count in plot order, with their plotted points and colours."""
    with mock.patch.object(
        at.lightcurve.plotlightcurve, "draw_residual_panel", wraps=at.lightcurve.plotlightcurve.draw_residual_panel
    ) as mockdraw:
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d] * modelcount,
            label=["first", "second"][:modelcount],
            plotcmf=True,
            residuals=baselineindex,
            write_data=True,
            outputfile=tmp_path / "cmf.pdf",
        )
    series = mockdraw.call_args.args[2]
    mainaxis = mockdraw.call_args.args[1]
    assert len(series) == len(mainaxis.lines) == 2 * modelcount
    for residualseries, line in zip(series, mainaxis.lines, strict=True):
        assert np.allclose(residualseries.x, line.get_xdata())
        assert np.allclose(residualseries.y, line.get_ydata())
        assert residualseries.color == line.get_color()
    dfstats = pl.read_csv(tmp_path / "cmf_residuals.csv")
    assert dfstats.height == len(series) - 1
    assert dfstats["reference"].to_list() == [series[baselineindex].label] * dfstats.height
    baseline = series[baselineindex]
    expectedrms = []
    xmin, xmax = mainaxis.get_xlim()
    for index, comparison in enumerate(series):
        if index != baselineindex:
            inrange = (baseline.x >= max(xmin, comparison.x.min())) & (baseline.x <= min(xmax, comparison.x.max()))
            residual = np.interp(baseline.x[inrange], comparison.x, comparison.y) - baseline.y[inrange]
            expectedrms.append(np.sqrt(np.mean(residual**2)))
    assert np.allclose(dfstats["rms"].to_numpy(), expectedrms)


@pytest.mark.parametrize("rateflag", ["deposition", "emission", "analyticemission"])
@pytest.mark.parametrize("baselineindex", [0, 1, 2])
@pytest.mark.parametrize("lumunit", ["erg/s", "Lsun", "mag"])
def test_residual_baseline_counts_energy_rate_curves(
    tmp_path: Path, rateflag: str, baselineindex: int, lumunit: str
) -> None:
    """Energy-rate curves count before the reference curve and keep the units of the main axis."""
    times = np.array([5.0, 6.0, 7.0])
    rate_lsun = np.array([1e6, 2e6, 3e6])
    depdata = pl.DataFrame({
        "tmid_days": times,
        "elecdep_Lsun": rate_lsun,
        "eps_elec_Lsun": rate_lsun,
        "eps_elec_ana_Lsun": rate_lsun,
    })
    reffile = tmp_path / "reference.txt"
    reffile.write_text("#time_days luminosity_erg/s\n5 1e40\n6 2e40\n7 3e40\n", encoding="utf-8")
    with (
        mock.patch.object(at.lightcurve.plotlightcurve, "get_deposition", return_value=depdata.lazy()),
        mock.patch.object(
            at.lightcurve.plotlightcurve, "draw_residual_panel", wraps=at.lightcurve.plotlightcurve.draw_residual_panel
        ) as mockdraw,
    ):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d, reffile],
            label=["model", "reference"],
            residuals=baselineindex,
            write_data=True,
            outputfile=tmp_path / "rates.pdf",
            deposition=["betaminus"] if rateflag == "deposition" else [],
            emission=["betaminus"] if rateflag == "emission" else [],
            analyticemission=["betaminus"] if rateflag == "analyticemission" else [],
            Lsun=lumunit == "Lsun",
            magnitude=lumunit == "mag",
        )
    series = mockdraw.call_args.args[2]
    assert len(series) == 3
    assert series[0].label == "model"
    assert series[2].label == "reference"
    assert np.allclose(series[1].x, times)
    expectedrate = rate_lsun
    if lumunit == "erg/s":
        expectedrate = rate_lsun * Lsun_to_erg_per_s
    elif lumunit == "mag":
        expectedrate = Mbol_sun - 2.5 * np.log10(rate_lsun)
    assert np.allclose(series[1].y, expectedrate)
    rateline = mockdraw.call_args.args[1].lines[1]
    assert series[1].label == rateline.get_label()
    assert series[1].color == rateline.get_color()
    dfstats = pl.read_csv(tmp_path / "rates_residuals.csv")
    assert dfstats["reference"].to_list() == [series[baselineindex].label] * 2
    assert dfstats["model"].to_list() == [item.label for index, item in enumerate(series) if index != baselineindex]
    assert (dfstats["npoints"] > 0).all()


@pytest.mark.parametrize("baselineindex", [0, 1, 2])
def test_residual_baseline_counts_hesma_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baselineindex: int
) -> None:
    """Check each baseline index with a HESMA curve after the model and the reference curve."""
    monkeypatch.chdir(tmp_path)
    hesmafile = tmp_path / "hesma.dat"
    hesmafile.write_text("# time B\n265 -13.5\n280 -13\n300 -12.5\n", encoding="utf-8")
    refdata = pl.DataFrame({"band": ["B"] * 3, "time": [265.0, 280.0, 300.0], "magnitude": [-13.0, -12.5, -12.0]})
    with (
        mock.patch.object(
            at.lightcurve.plotlightcurve, "read_reflightcurve_band_data", return_value=(refdata, {"label": "reference"})
        ),
        mock.patch.object(
            at.lightcurve.plotlightcurve, "draw_residual_panel", wraps=at.lightcurve.plotlightcurve.draw_residual_panel
        ) as mockdraw,
    ):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B"],
            reflightcurves=["reference.dat"],
            plot_hesma_model=hesmafile,
            residuals=baselineindex,
            write_data=True,
            outputfile=tmp_path,
        )
    series = mockdraw.call_args.args[2]
    axis = mockdraw.call_args.args[1]
    assert len(series) == 3
    assert series[2].label == "hesma"
    assert np.allclose(series[2].x, [265.0, 280.0, 300.0])
    assert np.allclose(series[2].y, [-13.5, -13.0, -12.5])
    assert series[2].color == axis.lines[-1].get_color()
    stats = pl.read_csv(tmp_path / "plotBlightcurves_residuals.csv")
    assert stats["reference"].to_list() == [series[baselineindex].label] * 2
    assert stats["model"].to_list() == [item.label for index, item in enumerate(series) if index != baselineindex]
    assert (stats["npoints"] > 0).all()
    assert stats["rms_relative"].null_count() == 2


def test_band_residual_panel_takes_one_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A band plot with one filter gives the RMS residual in magnitudes, and more than one filter stops the command."""
    # --write_data also writes the band data to the working folder
    monkeypatch.chdir(tmp_path)
    refdata = pl.DataFrame({"band": ["B", "B", "B"], "time": [265.0, 280.0, 300.0], "magnitude": [-13.0, -12.5, -12.0]})

    with (
        mock.patch.object(
            at.lightcurve.plotlightcurve, "read_reflightcurve_band_data", return_value=(refdata, {"label": "refband"})
        ),
        mock.patch.object(
            at.lightcurve.plotlightcurve, "save_figure", wraps=at.lightcurve.plotlightcurve.save_figure
        ) as mocksave,
    ):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B"],
            reflightcurves=["fakeref.dat"],
            residuals=1,
            write_data=True,
            outputfile=tmp_path,
        )
        # a fainter model lies below the reference in the main frame, thus it must also lie below zero in the panel
        assert all(axis.yaxis_inverted() for axis in mocksave.call_args.args[0].axes)
        dfstats = pl.read_csv(tmp_path / "plotBlightcurves_residuals.csv")
        assert dfstats["npoints"].item() == 3
        assert dfstats["rms"].item() > 0.0
        # a ratio to a mean magnitude has no meaning
        assert dfstats["rms_relative"].null_count() == 1

        with pytest.raises(SystemExit):
            at.lightcurve.plot(
                argsraw=[],
                modelpath=[modelpath],
                filter=["B", "V"],
                reflightcurves=["fakeref.dat"],
                residuals=1,
                outputfile=tmp_path,
            )
        # SystemExit holds the status alone, thus the message of the command is the text that it printed
        assert "-residuals applies to a plot of one frame" in capsys.readouterr().err


def test_reference_band_data_uses_the_given_distance_modulus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A distance modulus in the metadata is a measured value, thus it beats a distance from the redshift.

    iPTF13ebh gives dist_modulus 33.63 and z 0.0133. The reader took 57.54 Mpc from z (a modulus of 33.80),
    thus the points were 0.17 mag too bright.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "iPTF13ebh.dat").write_text("time,magnitude,band\n56610.0,15.0,B\n", encoding="utf-8")

    dfband, _ = at.lightcurve.core.read_reflightcurve_band_data("iPTF13ebh.dat")

    assert dfband["magnitude"].item() == pytest.approx(15.0 - 33.63, rel=1e-9)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_of_the_angle_average_and_a_bin(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The angle average (-1) and a direction bin in one plot must both come from the ARTIS output files.

    light_curve_res.out holds the bins 0 to 99 alone, thus -plotviewingangle -1 0 stopped with KeyError: -1.
    """
    outputfile = tmp_path / "lc.pdf"
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath_classic_3d], plotviewingangle=[-1, 0], outputfile=outputfile)

    assert outputfile.is_file()
    assert mockplot.call_count >= 2


def test_scan_lightcurve_reads_many_leading_zeros(tmp_path: Path) -> None:
    """ARTIS writes 0.0 as 0, thus a gamma-ray light curve can start with more than 100 rows of 0.

    polars inferred an integer column from the first 100 rows, and the first real value stopped the read.
    """
    rows = [f"{0.1 * (index + 1):.2f} 0 0" for index in range(120)] + ["12.10 1.5e+07 1.4e+07"]
    lcfile = tmp_path / "gamma_light_curve.out"
    lcfile.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dflc = at.lightcurve.core.scan_lightcurve(lcfile)[-1].collect()

    assert dflc.height == 121
    assert dflc["luminosity_Lsun"][-1] == pytest.approx(1.5e7, rel=1e-9)


def test_band_lightcurve_of_the_angle_average_with_virtual_packets() -> None:
    """With -plotvspecpol, bin -1 is the angle average of spec.out, and a vpkt run can have no specpol.out.

    The reader took the times of bin -1 from specpol.out, thus it stopped with FileNotFoundError.
    """
    from artistools.lightcurve.core import generate_band_lightcurve_data

    bandmags = generate_band_lightcurve_data(
        at.get_path("testdata") / "vpktcontrib", dirbin=-1, plotvspecpol=[0], filter=["B"]
    )
    assert bandmags["B"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_interactive_command_reproduces_plot(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The command of the viewer must draw the same curves as the viewer, and the old flags give the particle lists.

    --plotalphadeposition gives the deposition rates, the emission rates, and the analytical alpha rate. The command
    gives these as lists of particles, and a list takes each word after it. Thus a path after a list would be a
    particle, and the command must give the paths first.
    """
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer([str(modelpath_classic_3d), "--plotalphadeposition", "--interactive"], fig)
    assert (viewer.values.deposition, viewer.values.emission, viewer.values.analyticemission) == (
        ("gamma", "betaminus", "alpha"),
        ("betaminus", "alpha"),
        ("alpha",),
    )
    assert viewer.draw() is None
    # the deposition.out of this run is older than the fission columns, and ARTIS writes no emission rate of positrons
    assert viewer.get_energy_rate_reason("deposition", "fission") is not None
    assert viewer.get_energy_rate_reason("emission", "betaplus") is not None
    assert viewer.get_energy_rate_reason("thermalisation", "betaplus") is None

    newvalues = dc.replace(
        viewer.values,
        lumunit="Lsun",
        timemin="3.5",
        timemax="7",
        deposition=("gamma", "total"),
        thermalisation=("gamma", "betaplus"),
    )
    assert viewer.change(newvalues) is None
    command = viewer.get_command()
    assert command.endswith(" -timemin 3.5 -timemax 7 --Lsun -deposition gamma total -emission betaminus alpha"
                            " -analyticemission alpha -thermalisation gamma betaplus")  # fmt: skip
    # the thermalisation ratios go in a panel below the light curves, and not in a second figure
    frames = viewer.get_frames()
    assert len(frames) == 2
    viewercurves = {
        str(line.get_label()): np.asarray(line.get_ydata(), dtype=float)
        for frame in frames
        for line in frame.get_lines()
    }
    assert any(r"\beta^+" in label for label in viewercurves), viewercurves

    mockplot.reset_mock()
    at.lightcurve.plot(argsraw=[*shlex.split(command)[2:], "-o", str(tmp_path / "lc.pdf")])
    commandcurves = {
        str(callargs.kwargs.get("label")): np.asarray(callargs.args[2], dtype=float)
        for callargs in mockplot.call_args_list
        if callargs.kwargs.get("label")
    }
    assert commandcurves.keys() == viewercurves.keys()
    for label, ydata in viewercurves.items():
        assert np.allclose(commandcurves[label], ydata, rtol=1e-12, atol=0.0, equal_nan=True), label
    assert sorted(path.name for path in tmp_path.iterdir()) == ["lc.pdf"]


@pytest.mark.parametrize("plotoptions", [[], ["-filter", "B"], ["-colour_evolution", "B-V"]])
def test_plotlightcurves_writes_png_data_to_a_png_file(tmp_path: Path, plotoptions: list[str]) -> None:
    """A light curve with -o lc.png must hold PNG data. The saves once gave format="pdf" for each file name."""
    outputfile = tmp_path / "lc.png"
    at.lightcurve.plot(argsraw=[str(modelpath), *plotoptions, "-o", str(outputfile)])
    assert outputfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_lightcurve_alpha_stays_on_its_model(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The -linealpha of one model must not go to the next model of a band plot, which reuses its plot arguments."""
    secondmodel = tmp_path / "secondmodel"
    secondmodel.symlink_to(modelpath, target_is_directory=True)
    at.lightcurve.plot(
        argsraw=[str(modelpath), str(secondmodel), "-filter", "B", "-linealpha", "0.3", "-o", str(tmp_path / "lc.pdf")]
    )
    alphas = [callargs.kwargs.get("alpha") for callargs in mockplot.call_args_list if callargs.kwargs.get("label")]
    assert alphas[:2] == [0.3, None]


def test_thermalisation_panel_follows_the_time_range_of_the_light_curves() -> None:
    """The panel of the thermalisation ratios must not widen the shared time axis, and -ymax must not fix its legend.

    The panel had the default x margin of 5 %, and the -ymax of the luminosity axis stopped the legend room of the
    panel.
    """
    plotlightcurve = at.lightcurve.plotlightcurve

    def draw(extraoptions: list[str]) -> tuple[tuple[float, float], tuple[float, float] | None]:
        args = at.misc.parse_cli_args(
            plotlightcurve.addargs, None, None, [str(modelpath_classic_3d), *extraoptions], {}
        )
        plotlightcurve.resolve_plot_args(args)
        fig, axis, thermaxis, residualaxis = plotlightcurve.make_plot_figure(args)
        FigureCanvasAgg(fig)
        plotlightcurve.draw_plot(args, axis, thermaxis, residualaxis)
        fig.canvas.draw()
        return axis.get_xlim(), None if thermaxis is None else thermaxis.get_ylim()

    depositionxlim, _ = draw(["-deposition", "gamma"])
    thermalisationxlim, thermalisationylim = draw(["-thermalisation", "gamma"])
    _, ylimwithymax = draw(["-thermalisation", "gamma", "-ymax", "2e43"])
    assert np.allclose(thermalisationxlim, depositionxlim, rtol=1e-9, atol=0.0)
    assert thermalisationylim is not None
    assert ylimwithymax is not None
    assert np.allclose(ylimwithymax, thermalisationylim, rtol=1e-9, atol=0.0)


def test_viewer_gives_a_magnitude_plot_no_log_scale() -> None:
    """A magnitude plot of the window must not keep a log scale of the command.

    The window has no scale control for a magnitude, and a log axis of magnitudes showed no curve.
    """
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer([str(modelpath), "--magnitude", "-yscale", "log", "--interactive"], fig)
    assert viewer.values.yscale == viewer.defaultyscale
    assert "-yscale" not in viewer.get_command()


def test_plotcmf_refuses_a_magnitude_and_skips_the_virtual_packet_observers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A magnitude has no comoving frame luminosity, and the virtual packets hold no comoving frame energy.

    --plotcmf --magnitude reached an assert after the first light curve. --plotcmf -plotvspecpol stopped with
    ColumnNotFoundError, because the light curve of an observer has no luminosity_cmf_Lsun column.
    """
    with pytest.raises(SystemExit):
        at.lightcurve.plot(argsraw=[], modelpath=[modelpath], plotcmf=True, magnitude=True, outputfile=tmp_path)
    assert "has no magnitude" in capsys.readouterr().err

    outputfile = tmp_path / "observer.pdf"
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[at.get_path("testdata") / "vpktcontrib"],
        frompackets=True,
        plotvspecpol=[0],
        plotcmf=True,
        outputfile=outputfile,
    )
    assert "draws no curve of an observer" in capsys.readouterr().err
    assert outputfile.is_file()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_average_over_phi_without_a_direction_bin_plots_the_angle_average(
    mockplot: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--average_over_phi_angle with no -plotviewingangle, or with bin -1 alone, reads light_curve.out.

    The command averaged the one bin of light_curve.out and stopped with ValueError. The angle average of bin -1 is
    the mean over every direction already, thus the flag changes nothing.
    """
    at.lightcurve.plot(
        argsraw=[], modelpath=[modelpath_classic_3d], average_over_phi_angle=True, outputfile=tmp_path / "lc.pdf"
    )
    assert "gives none" in capsys.readouterr().err
    curves = len(mockplot.call_args_list)
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotviewingangle=[-1],
        average_over_phi_angle=True,
        outputfile=tmp_path / "lc2.pdf",
    )
    assert len(mockplot.call_args_list) == 2 * curves


def test_scan_lightcurve_takes_the_layout_from_the_caller(tmp_path: Path) -> None:
    """The caller says whether a file holds one table for each direction bin. The name of the file decides nothing.

    scan_lightcurve read a file as direction resolved when its name held "_res". A copy with a different name then
    gave one table. An average over the one bin of light_curve.out gave a ValueError about 100 missing bins.
    """
    lcpath = at.firstexisting("light_curve_res.out", folder=modelpath_classic_3d, tryzipped=True)
    copypath = tmp_path / "lightcurve_copy.out"
    copypath.write_bytes(lcpath.read_bytes())
    lcdataframes = at.lightcurve.scan_lightcurve(copypath, directionresolved=True)
    assert sorted(lcdataframes) == list(range(100))
    # the angle average of bin -1 is the mean over every direction already, thus it takes no average
    pltest.assert_frame_equal(
        at.lightcurve.scan_lightcurve(modelpath_classic_3d / "light_curve.out", average_over_phi=True)[-1].collect(),
        at.lightcurve.scan_lightcurve(modelpath_classic_3d / "light_curve.out")[-1].collect(),
    )
    with pytest.raises(ValueError, match="holds 100 tables"):
        at.lightcurve.scan_lightcurve(copypath)


def test_scan_lightcurve_of_a_build_with_a_different_count_of_direction_bins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ARTIS sets MABINS when it compiles. A file of a different count of direction bins gives a warning.

    scan_lightcurve stopped for each count other than 100, thus the light curves of such a run were not readable.
    """
    lcpath = tmp_path / "light_curve_res.out"
    table = "1.0 2.0 3.0\n2.0 4.0 6.0\n"
    lcpath.write_text(table * 4, encoding="utf-8")
    lcdataframes = at.lightcurve.scan_lightcurve(lcpath, directionresolved=True)

    assert sorted(lcdataframes) == [0, 1, 2, 3]
    assert np.allclose(lcdataframes[3].collect()["luminosity_Lsun"].to_numpy(), [2.0, 4.0])
    assert "holds 4 tables" in capsys.readouterr().err

    # a build with one or two direction bins writes as many tables as an angle-averaged light curve
    for nbins in (1, 2):
        lcpath.write_text(table * nbins, encoding="utf-8")
        assert sorted(at.lightcurve.scan_lightcurve(lcpath, directionresolved=True)) == list(range(nbins))
        assert f"holds {nbins} tables" in capsys.readouterr().err


def test_average_of_a_light_curve_with_more_direction_bins_stops(tmp_path: Path) -> None:
    """The average takes the geometry of 100 direction bins. It ignored each table above bin 99 with no message."""
    lcpath = tmp_path / "light_curve_res.out"
    lcpath.write_text("1.0 2.0 3.0\n2.0 4.0 6.0\n" * 101, encoding="utf-8")

    with pytest.raises(ValueError, match="1 more bins"):
        at.lightcurve.scan_lightcurve(lcpath, directionresolved=True, average_over_phi=True)


def test_named_light_curve_file_must_agree_with_the_direction_bins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A user can name light_curve_res.out with no -plotviewingangle. The command then stops with a message.

    The command read the 100 tables of the file as one table of the angle average, and plotted half of them.
    """
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[], modelpath=[modelpath_classic_3d / "light_curve_res.out"], outputfile=tmp_path / "lc.pdf"
        )
    assert "-plotviewingangle" in capsys.readouterr().err


def test_colour_at_peak_stops_before_the_fits_without_the_phillips_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The package holds no CfA3_Phillips.dat, thus --colouratpeak must stop with its name before the slow fits.

    The command fitted the light curve of each direction bin first, then stopped with FileNotFoundError.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.lightcurve.plot(argsraw=[], modelpath=[modelpath], filter=["B", "V"], colouratpeak=True)
    assert "CfA3_Phillips.dat" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_topnucs_with_virtual_packet_observers_stops_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The virtual packets hold no pellet. Thus -topnucs or --use_pellet_decay_time with -plotvspecpol gives a message.

    get_from_packets stopped with a bare AssertionError, which the viewer showed as its reason.
    """
    vpktmodelpath = at.get_path("testdata") / "vpktcontrib"
    with pytest.raises(SystemExit):
        at.lightcurve.plot(argsraw=[], modelpath=[vpktmodelpath], plotvspecpol=[0], topnucs=2, outputfile=tmp_path)
    assert "hold no pellet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[vpktmodelpath],
            frompackets=True,
            plotvspecpol=[0],
            use_pellet_decay_time=True,
            outputfile=tmp_path,
        )
    assert "hold no pellet" in capsys.readouterr().err
    with pytest.raises(ValueError, match="hold no pellet"):
        at.lightcurve.get_from_packets(
            vpktmodelpath, directionbins=[0], directionbins_are_vpkt_observers=True, pellet_nucname="Ni56"
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_nuclide_light_curve_label_keeps_the_model_name(mockplot: mock.MagicMock) -> None:
    """The label of the light curve of one nuclide holds the model name and the gamma marker.

    The label held the nuclide alone, thus a second model or --gamma gave the legend the same label again.
    """
    from artistools.lightcurve import plotlightcurve

    args = at.misc.parse_cli_args(plotlightcurve.addargs, None, None, [str(modelpath), "--frompackets", "--gamma"])
    plotlightcurve.resolve_plot_args(args)
    fig = mplfig.Figure()
    axis = fig.add_subplot()
    lcdataframes = at.lightcurve.scan_lightcurve(modelpath / "light_curve.out")
    with mock.patch.object(plotlightcurve, "get_from_packets", return_value=lcdataframes):
        plotlightcurve.plot_artis_lightcurve(
            modelpath, axis, escape_type="TYPE_GAMMA", frompackets=True, args=args, pellet_nucname="Ni56"
        )
    assert mockplot.call_args.kwargs["label"] == rf"{at.misc.get_model_name(modelpath)} $\gamma$ Ni56"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_hesma_model_is_drawn_on_the_panel_of_each_band(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """-plot_hesma_model draws each band of the HESMA file on the panel of that band, with the file name as its label.

    The command asserted one axes, which a figure with two bands never has, and the package holds no data/hesma folder.
    """
    hesmafile = tmp_path / "hesma_model.dat"
    hesmafile.write_text("# t B V\n10.0 -18.0 -18.5\n20.0 -17.0 -17.8\n", encoding="utf-8")
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        filter=["B", "V"],
        plot_hesma_model=hesmafile,
        outputfile=tmp_path / "bands.pdf",
    )
    hesmacalls = [callargs for callargs in mockplot.call_args_list if callargs.kwargs.get("color") == "black"]
    assert len(hesmacalls) == 2
    assert {callargs.kwargs["label"] for callargs in hesmacalls} == {"hesma_model"}
    assert len({id(callargs.args[0]) for callargs in hesmacalls}) == 2
    assert [list(callargs.args[2]) for callargs in hesmacalls] == [[-18.0, -17.0], [-18.5, -17.8]]
    assert (tmp_path / "bands.pdf").is_file()


def test_filter_data_gives_the_reference_wavelength() -> None:
    """The reference wavelength of a band comes from the same cached read as its transmission curve."""
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")
    assert np.isclose(
        at.lightcurve.get_filter_data(filterdir, "B")[1],
        float((filterdir / "B.txt").read_text(encoding="utf-8").splitlines()[2]),
    )


def test_viewer_drops_the_options_that_the_command_refuses_together() -> None:
    """The window starts from a command with --magnitude --plotcmf, or with -topnucs and an observer, and draws a plot.

    The command stops for these pairs, and the viewer stopped with it at the start. The controls of the window drop
    the same options, thus the start does the same.
    """
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer([str(modelpath), "--magnitude", "--plotcmf", "--interactive"], fig)
    assert (viewer.values.lumunit, viewer.values.plotcmf) == ("mag", False)
    assert viewer.draw() is None

    vpktmodelpath = at.get_path("testdata") / "vpktcontrib"
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer(
        [str(vpktmodelpath), "-plotvspecpol", "0", "-topnucs", "2", "--use_pellet_decay_time", "--interactive"], fig
    )
    assert viewer.values.directionkind == "vpkt"
    assert (viewer.values.frompackets, viewer.values.topnucs, viewer.values.usepelletdecaytime) == (True, 0, False)
    assert viewer.draw() is None
    assert "--frompackets -plotvspecpol 0" in viewer.get_command()


def test_viewer_controls_read_the_refused_options_of_the_command() -> None:
    """A change of a control drops the options that plotlightcurves refuses, and the control shows the reason.

    The viewer held its own copy of each rule of the command, thus a new rule of the command did not reach the window.
    """
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    vpktmodelpath = at.get_path("testdata") / "vpktcontrib"
    viewer = interactive.LightCurveViewer([str(vpktmodelpath), "--frompackets", "-topnucs", "2", "--interactive"], fig)
    observervalues = dc.replace(
        viewer.values, directionkind="vpkt", directionbins=(0,), topnucs=2, usepelletdecaytime=True, plotcmf=True
    )
    assert {option.flag for option in viewer.get_refused_options(observervalues)} == {
        "--plotcmf",
        "-topnucs",
        "--use_pellet_decay_time",
    }
    droppedvalues = viewer.drop_refused_options(observervalues)
    assert (droppedvalues.topnucs, droppedvalues.usepelletdecaytime, droppedvalues.plotcmf) == (0, False, False)
    assert not viewer.get_refused_options(droppedvalues)

    magnitudevalues = dc.replace(viewer.values, lumunit="mag")
    reasons = interactive.get_refused_reasons(viewer, magnitudevalues)
    assert set(reasons) == {"--plotcmf"}
    assert "has no magnitude" in reasons["--plotcmf"]


def test_ab_filter_zero_point_is_the_flux_of_an_ab_flat_spectrum() -> None:
    """The first line of an AB filter file is the energy flux of a flat 3631 Jy spectrum through the filter.

    PS1/ws.txt held the zero point of gs.txt, thus each PS1/ws magnitude was 1.62 mag too bright. The zero points of
    the Swift UVOT files uvw1_ab and uvw2_ab are 35 % below this integral. The repository cannot confirm their source.
    """
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")
    unconfirmed = {"uvw1_ab", "uvw2_ab"}
    ratios: dict[str, float] = {}
    for filterpath in sorted(filterdir.rglob("*.txt")):
        filtername = filterpath.relative_to(filterdir).with_suffix("").as_posix()
        if filtername in unconfirmed or filterpath.read_text(encoding="utf-8").splitlines()[3].strip() != "ab":
            continue
        zeropoint, _, wavelengths, transmission, _, _ = at.lightcurve.get_filter_data(filterdir, filtername)
        f_lambda_ab = 3631e-23 * at.constants.C_cm_per_s * 1e8 / wavelengths**2
        ratios[filtername] = float(np.trapezoid(f_lambda_ab * transmission, wavelengths)) / zeropoint

    assert len(ratios) > 30
    assert all(0.85 < ratio < 1.25 for ratio in ratios.values()), ratios
    assert np.isclose(ratios["PS1/ws"], 1.0, rtol=1e-3)


def test_pellet_decay_time_without_packets_stops_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the packets give the decay time of a pellet. The command and the viewer read one rule.

    The command reached a bare assert, and the viewer kept its own copy of the rule in a tooltip.
    """
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[], modelpath=[modelpath_classic_3d], use_pellet_decay_time=True, outputfile=tmp_path / "lc.pdf"
        )
    assert "decay time of a pellet" in capsys.readouterr().err

    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer([str(modelpath), "--use_pellet_decay_time", "--interactive"], fig)
    assert not viewer.values.usepelletdecaytime
    assert viewer.draw() is None
    reasons = interactive.get_refused_reasons(viewer, viewer.values)
    assert "decay time of a pellet" in reasons["--use_pellet_decay_time"]
    assert "--use_pellet_decay_time" not in interactive.get_refused_reasons(
        viewer, dc.replace(viewer.values, frompackets=True)
    )


def test_gamma_light_curve_of_a_virtual_packet_observer_stops_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ARTIS makes the virtual packets from the r-packets, thus an observer has no gamma-ray light curve.

    --gamma with -plotvspecpol drew the UVOIR light curve of the observer with a gamma-ray label.
    """
    vpktmodelpath = at.get_path("testdata") / "vpktcontrib"
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[vpktmodelpath],
            frompackets=True,
            plotvspecpol=[1],
            gamma=True,
            outputfile=tmp_path / "lc.pdf",
        )
    assert "no gamma-ray light curve" in capsys.readouterr().err

    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer([str(vpktmodelpath), "-plotvspecpol", "1", "--gamma", "--interactive"], fig)
    assert not viewer.values.gamma
    assert "no gamma-ray light curve" in interactive.get_refused_reasons(viewer, viewer.values)["--gamma"]
    assert not viewer.drop_refused_options(dc.replace(viewer.values, gamma=True)).gamma


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_direction_bins_of_a_model_with_no_direction_file_give_a_warning(
    mockplot: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model with no direction-resolved file shows the angle average for -plotviewingangle, and the command says so.

    The command gave the angle average with no message.
    """
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], plotviewingangle=[0], outputfile=tmp_path / "lc.pdf")
    assert "holds no direction-resolved file" in capsys.readouterr().err
    assert mockplot.call_args.kwargs["label"] == "TEST MODEL"


def test_band_data_files_of_several_models_hold_the_model_name(tmp_path: Path) -> None:
    """--write_data writes one band file for each model. The second model overwrote the file of the first model."""
    modelcopy = tmp_path / "secondmodel"
    modelcopy.mkdir()
    for filepath in modelpath.iterdir():
        if filepath.name != "plotlabel.txt":
            (modelcopy / filepath.name).symlink_to(filepath)
    outputfolder = tmp_path / "out"
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, modelcopy],
        filter=["B"],
        timemin=290,
        timemax=300,
        write_data=True,
        outputfile=outputfolder,
    )
    assert sorted(path.name for path in outputfolder.glob("band_*.txt")) == [
        "band_B_TEST MODEL.txt",
        "band_B_secondmodel.txt",
    ]


def test_band_name_with_an_instrument_folder_gives_valid_file_names(tmp_path: Path) -> None:
    """A filter of an instrument has a name with a folder, e.g. NOT/B. Its slash must not reach a file name.

    The default name of the plot named the folder plotNOT, and the save stopped with FileNotFoundError.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        filter=["NOT/B"],
        timemin=290,
        timemax=300,
        write_data=True,
        outputfile=tmp_path,
    )
    assert (tmp_path / "plotNOT_Blightcurves.pdf").is_file()
    assert (tmp_path / "band_NOT_B.txt").is_file()

    args = at.misc.parse_cli_args(
        at.lightcurve.plotlightcurve.addargs, None, None, [str(modelpath), "-colour_evolution", "NOT/B-NOT/V"]
    )
    at.lightcurve.plotlightcurve.resolve_plot_args(args)
    assert args.outputfile.name == "plotcolorevolutionNOT_B-NOT_V.pdf"

    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        filter=["NOT/B"],
        timemin=250,
        timemax=300,
        save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
        outputfile=tmp_path,
    )
    assert (tmp_path / "NOT_Bband_TEST MODEL_viewing_angle_data.txt").is_file()


def test_average_over_an_angle_needs_the_first_bin_of_each_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An average over phi gives one curve for each cos(theta) bin, which its first direction bin names.

    A direction bin that starts no group reached a bare assert of get_dirbin_labels.
    """
    for averageflag, dirbin in (("average_over_phi_angle", 5), ("average_over_theta_angle", 15)):
        plotkwargs: dict[str, t.Any] = {averageflag: True}
        with pytest.raises(SystemExit):
            at.lightcurve.plot(
                argsraw=[],
                modelpath=[modelpath_classic_3d],
                plotviewingangle=[dirbin],
                outputfile=tmp_path / "lc.pdf",
                **plotkwargs,
            )
        assert f"bins {dirbin} start no group" in capsys.readouterr().err

    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.LightCurveViewer(
        [str(modelpath_classic_3d), "-plotviewingangle", "5", "--average_over_phi_angle", "--interactive"], fig
    )
    assert viewer.values.directionkind == "bin"
    assert viewer.draw() is None


def test_brightness_at_time_takes_the_output_folder_and_one_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--brightnessattime saves in the folder of -o, and a range of -timedays gives a message.

    The plot went to the working folder, and a range stopped with a ValueError of float.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    at.lightcurve.plot(
        argsraw=[], modelpath=[modelpath_classic_3d], brightnessattime=True, timedays="5", outputfile="sub/x.pdf"
    )
    assert [path.name for path in (tmp_path / "sub").iterdir()] == ["plotviewinganglebrightnessat5.0days.pdf"]
    assert not list(tmp_path.glob("*.pdf"))

    with pytest.raises(SystemExit):
        at.lightcurve.plot(argsraw=[], modelpath=[modelpath_classic_3d], brightnessattime=True, timedays="4-5")
    assert "takes one time" in capsys.readouterr().err


@pytest.mark.parametrize(
    "datatext",
    [
        "#time magnitude band\n55800.0 12.0 B\n55801.0 12.5 B\n",
        "#time,magnitude,band\n55800.0,12.0,B\n55801.0,12.5,B\n",
    ],
)
def test_reference_band_data_takes_a_header_line_that_starts_with_a_hash(tmp_path: Path, datatext: str) -> None:
    """A header line can start with "#", as in a bolometric reference light curve.

    The reader cut each line at the "#", thus it lost the header and stopped with ColumnNotFoundError.
    """
    datafile = tmp_path / "refband.dat"
    datafile.write_text(datatext, encoding="utf-8")
    (tmp_path / "refband.dat.meta.yml").write_text(
        "label: ref\ntimecorrection: 55790.0\ndist_mpc: 10.0\n", encoding="utf-8"
    )

    dfband, metadata = at.lightcurve.core.read_reflightcurve_band_data(datafile)

    assert metadata["label"] == "ref"
    assert dfband["band"].to_list() == ["B", "B"]
    assert np.allclose(dfband["time"].to_numpy(), [10.0, 11.0])
    assert np.allclose(dfband["magnitude"].to_numpy(), [12.0 - 5 * np.log10(10e6) + 5, 12.5 - 5 * np.log10(10e6) + 5])


def test_band_outside_a_virtual_packet_spectrum_gives_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """The spectrum of a virtual packet observer covers the range of vpkt.txt. A band outside it has a too faint value.

    The bol band and a filter that reaches outside the spectrum gave a magnitude with no message.
    """
    from artistools.lightcurve.core import generate_band_lightcurve_data

    vspecpolmodel = at.get_path("testdata") / "vspecpolmodel"
    bandmags = generate_band_lightcurve_data(vspecpolmodel, dirbin=0, plotvspecpol=[0], filter=["bol", "uvw1"])
    assert bandmags["bol"]
    stderr = capsys.readouterr().err
    assert "the bol band of a virtual packet observer integrates only the range of its spectrum" in stderr
    assert "the uvw1 filter" in stderr


def test_band_light_curve_of_a_run_with_only_specpol_out() -> None:
    """A POL_ON run can write specpol.out and no spec.out. Its Stokes I spectrum gives the angle average.

    The band light curve of bin -1 stopped with FileNotFoundError, because get_spectra read spec.out only.
    """
    from artistools.lightcurve.core import generate_band_lightcurve_data

    bandmags = generate_band_lightcurve_data(at.get_path("testdata") / "vspecpolmodel", filter=["B"])
    assert bandmags["B"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_packets_of_a_band_plot_give_a_warning(
    mockplot: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A band light curve comes from the spectra. --frompackets has no effect there, and the command says so.

    --frompackets marked the direction bins as present, thus a model with no spec_res.out stopped with KeyError.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        filter=["B"],
        plotviewingangle=[0],
        frompackets=True,
        timemin=290,
        timemax=300,
        outputfile=tmp_path / "lc.pdf",
    )
    assert "-filter reads no packets files, thus --frompackets has no effect" in capsys.readouterr().err
    assert mockplot.call_args.kwargs["label"] == "TEST MODEL"


def test_plot_with_no_light_curve_stops_with_a_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command skips each model that has no light curve file, and a plot with no curve then stops with a message.

    The command stopped with a bare AssertionError, and python -O saved an empty figure.
    """
    with pytest.raises(SystemExit):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            gamma=True,
            plotviewingangle=[0],
            outputfile=tmp_path / "x.pdf",
        )
    assert "the plot holds no light curve" in capsys.readouterr().err
    assert not (tmp_path / "x.pdf").exists()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_packets_light_curve_of_a_model_with_no_model_file(
    mockplot: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rest frame light curve of the packets needs no model.txt. Only the comoving frame needs it.

    A copy of a run with no model.txt stopped with FileNotFoundError, also with no --plotcmf.
    """
    modelcopy = tmp_path / "nomodel"
    modelcopy.mkdir()
    for filepath in modelpath.iterdir():
        if filepath.name != "model.txt":
            (modelcopy / filepath.name).symlink_to(filepath)

    at.lightcurve.plot(
        argsraw=[], modelpath=[modelcopy], frompackets=True, plotcmf=True, outputfile=tmp_path / "lc.pdf"
    )
    stderr = capsys.readouterr().err
    assert "holds no model.txt" in stderr
    assert "--plotcmf draws no curve" in stderr
    assert len(mockplot.call_args_list) == 1
    assert (tmp_path / "lc.pdf").is_file()


def test_options_that_change_nothing_give_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """--test_viewing_angle_fit needs a fit, and --plotcmf has no time axis of the decay time. Each one gives a warning.

    --test_viewing_angle_fit with no save flag or scatter flag drew no fit, and --plotcmf with --use_pellet_decay_time
    drew the comoving frame curve on the axis of the arrival time.
    """
    args = at.misc.parse_cli_args(
        at.lightcurve.plotlightcurve.addargs,
        None,
        None,
        [str(modelpath), "--test_viewing_angle_fit", "--frompackets", "--use_pellet_decay_time", "--plotcmf"],
    )
    at.lightcurve.plotlightcurve.resolve_plot_args(args)
    stderr = capsys.readouterr().err
    assert "only a save flag or a scatter flag makes these fits" in stderr
    assert "different time axes" in stderr
    assert (args.test_viewing_angle_fit, args.plotcmf, args.use_pellet_decay_time) == (False, False, True)


def test_colour_arguments_must_name_two_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """-colour_evolution and --colouratpeak each need two bands, and a different count gives a message.

    One band stopped with IndexError, and U-B-V used the first two bands with no message.
    """
    for colour in ("B", "U-B-V"):
        with pytest.raises(SystemExit):
            at.lightcurve.plot(argsraw=[], modelpath=[modelpath], colour_evolution=[colour], outputfile=tmp_path)
        assert "-colour_evolution takes two bands for each colour" in capsys.readouterr().err

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        at.lightcurve.plot(argsraw=[], modelpath=[modelpath], filter=["B"], colouratpeak=True)
    assert "-filter must name two bands" in capsys.readouterr().err


def test_scan_lightcurve_of_a_cut_or_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """sn3d writes light_curve.out again at each timestep. A run that stops during that write leaves a bad file.

    An empty file gave a polars NoDataError with no file name. A cut last line gave its cut number as the value, and
    tables of different lengths reached a bare assert.
    """
    lcpath = tmp_path / "light_curve.out"
    lcpath.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no light curve"):
        at.lightcurve.scan_lightcurve(lcpath)

    lcpath.write_text("1.0 2.0 3.0\n2.0 4.0 6.0\n3.0 2.37598e+09 1.59", encoding="utf-8")
    dflc = at.lightcurve.scan_lightcurve(lcpath)[-1].collect()
    assert dflc["time_days"].to_list() == [1.0, 2.0]
    assert "ends with a cut line" in capsys.readouterr().err

    lcpath.write_text("1.0 2.0 3.0\n2.0 4.0 6.0\n1.0 2.5 3.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tables have different lengths"):
        at.lightcurve.scan_lightcurve(lcpath)


def test_scan_lightcurve_of_a_cut_compressed_file(tmp_path: Path) -> None:
    """A cut .gz file gave an EOFError, and a cut .xz file gave a polars panic. Neither error named the file."""
    fulltext = (modelpath / "light_curve.out").read_bytes()
    for compressedname, compress in (("light_curve.out.gz", gzip.compress), ("light_curve.out.xz", lzma.compress)):
        compressedpath = tmp_path / compressedname
        compressedpath.write_bytes(compress(fulltext)[:1000])
        with pytest.raises((OSError, pl.exceptions.PanicException)) as excinfo:
            at.lightcurve.scan_lightcurve(tmp_path / "light_curve.out")
        assert any("light_curve.out" in note for note in [str(excinfo.value), *getattr(excinfo.value, "__notes__", [])])
        compressedpath.unlink()


def test_writebollightcurvedata_writes_the_light_curve_of_each_model(tmp_path: Path) -> None:
    """The command writes the time and the luminosity of light_curve.out, or of the spectra of each direction bin."""
    from artistools.lightcurve import writebollightcurvedata

    writebollightcurvedata.main(argsraw=[str(modelpath), "-o", str(tmp_path)])
    lines = (tmp_path / "bol_lightcurvedata_TEST MODEL.txt").read_text(encoding="utf-8").splitlines()
    dflc = at.lightcurve.scan_lightcurve(modelpath / "light_curve.out")[-1].collect()
    assert lines[0].startswith("# 1st col is time in days")
    values = np.array([[float(value) for value in line.split()] for line in lines[1:]])
    assert np.allclose(values[:, 0], dflc["time_days"].to_numpy())
    assert np.allclose(values[:, 1], dflc["luminosity_erg/s"].to_numpy())

    writebollightcurvedata.main(argsraw=[str(modelpath_classic_3d), "--fromspectra", "-o", str(tmp_path)])
    header, *rows = (
        (tmp_path / "bol_lightcurvedata_test-classicmode_3d_fromspectra.txt").read_text(encoding="utf-8").splitlines()
    )
    assert header.startswith("# 1st col is time in days")
    assert {len(row.split()) for row in rows} == {101}


def test_viewer_direction_removes_the_packets_that_it_gave_to_the_observers() -> None:
    """A direction after the observers of the virtual packets removes the --frompackets that the window gave.

    The window gave --frompackets to the observers, and the flag stayed for all the directions.
    """
    vpktmodelpath = at.get_path("testdata") / "vpktcontrib"
    observers = interactive.DirectionChoice(kind="vpkt", bins=(0,), usedegrees=False)
    alldirections = interactive.DirectionChoice(kind="", bins=(), usedegrees=False)
    viewer = interactive.LightCurveViewer([str(vpktmodelpath), "--interactive"], mplfig.Figure())
    assert not viewer.values.frompackets
    observervalues = viewer.set_direction(viewer.values, observers)
    assert observervalues.frompackets
    assert not viewer.set_direction(observervalues, alldirections).frompackets

    viewer = interactive.LightCurveViewer([str(vpktmodelpath), "--frompackets", "--interactive"], mplfig.Figure())
    observervalues = viewer.set_direction(viewer.values, observers)
    assert viewer.set_direction(observervalues, alldirections).frompackets

    # a command with an observer and no --frompackets gets the flag from the window
    viewer = interactive.LightCurveViewer([str(vpktmodelpath), "-plotvspecpol", "0", "--interactive"], mplfig.Figure())
    assert viewer.values.frompackets
    assert not viewer.set_direction(viewer.values, alldirections).frompackets
