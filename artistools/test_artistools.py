import argparse
import dataclasses as dc
import functools
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
import polars as pl
import polars.testing as pltest
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.viewertools.window import draw_values

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


def test_timesteps_command_answers_a_reverse_lookup(capsys: pytest.CaptureFixture[str]) -> None:
    """-timedays names the timestep that covers a time, and -timestep gives the days of one timestep."""
    at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-t", "300"])
    assert capsys.readouterr().out.strip() == "300 days falls in timestep 54, which covers 299.812 to 300.823 days"

    at.listtimesteps.main(argsraw=["-modelpath", str(modelpath), "-ts", "last"])
    assert capsys.readouterr().out.strip() == "Timestep 99 covers 348.824 to 350.000 days"


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


def read_data_file(path: Path) -> pl.DataFrame:
    """Read a table that --write_data wrote."""
    return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")


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
