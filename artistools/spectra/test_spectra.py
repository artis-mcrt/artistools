import argparse
import math
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.spectra import spectra as atspectra

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraplot(mockplot: mock.MagicMock) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=outputpath,
        timemin=290,
        timemax=320,
        distmpc=1.0,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    integral = np.trapezoid(y=arr_f_lambda, x=arr_lambda)
    assert np.isclose(integral, 5.870730903198916e-11, atol=1e-14)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_spectra_frompackets(mockplot: mock.MagicMock) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=Path(outputpath, "spectrum_from_packets.pdf"),
        timemin=290,
        timemax=320,
        frompackets=True,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    integral = np.trapezoid(y=arr_f_lambda, x=arr_lambda)

    assert np.isclose(integral, 7.7888e-12, rtol=1e-3)


def test_spectra_outputtext() -> None:
    at.spectra.plot(argsraw=[], specpath=modelpath, output_spectra=True)


@pytest.mark.benchmark
def test_spectraemissionplot() -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=outputpath,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )


@pytest.mark.benchmark
def test_spectraemissionplot_nostack() -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=outputpath,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        nostack=True,
        use_thermalemissiontype=True,
    )


def test_spectra_get_spectrum() -> None:
    def check_spectrum(dfspectrumpkts: pl.DataFrame) -> None:
        assert math.isclose(max(dfspectrumpkts["f_lambda"]), 2.548532804918824e-13, abs_tol=1e-5)
        assert min(dfspectrumpkts["f_lambda"]) < 1e-9
        flambdamean = dfspectrumpkts["f_lambda"].mean()
        assert isinstance(flambdamean, float)
        assert math.isclose(flambdamean, 1.0314682640070206e-14, abs_tol=1e-5)

    dfspectrum = at.spectra.get_spectra(modelpath, 55, 65, fluxfilterfunc=None)[-1].collect()

    assert len(dfspectrum["lambda_angstroms"]) == 1000
    assert len(dfspectrum["f_lambda"]) == 1000
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[-1] - 29920.601421214415) < 1e-5
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[0] - 600.75759482509852) < 1e-5

    check_spectrum(dfspectrum)

    timelowdays = at.get_timestep_times(modelpath)[55]
    timehighdays = at.get_timestep_times(modelpath)[65]

    dfspectrumpkts = at.spectra.get_from_packets(modelpath, timelowdays=timelowdays, timehighdays=timehighdays)[
        -1
    ].collect()

    check_spectrum(dfspectrumpkts)


@pytest.mark.benchmark
def test_spectra_get_spectrum_polar_angles() -> None:
    spectra = at.spectra.get_spectra(
        modelpath=modelpath_classic_3d, average_over_phi=True, timestepmin=20, timestepmax=25
    )

    assert all(
        math.isclose(dirspec.select(pl.col("lambda_angstroms").mean()).collect().item(), 7510.074, rel_tol=1e-3)
        for dirspec in spectra.values()
    )
    assert all(
        math.isclose(dirspec.select(pl.col("lambda_angstroms").std()).collect().item(), 7647.317, rel_tol=1e-3)
        for dirspec in spectra.values()
    )

    results = {
        dirbin: (dfspecdir.select(mean=pl.col("f_lambda").mean(), std=pl.col("f_lambda").std()).collect().row(0))
        for dirbin, dfspecdir in spectra.items()
    }

    print(f"expected_results = {results!r}")

    expected_results = {
        0: (8.944885683622777e-12, 2.5390561316336613e-11),
        10: (7.192449910173842e-12, 2.0713405870496142e-11),
        20: (8.963182635824623e-12, 2.4720178744713477e-11),
        30: (8.06805028771611e-12, 2.2672897557383406e-11),
        40: (7.8306536944195e-12, 2.2812958326863807e-11),
        50: (8.259135507460651e-12, 2.2795973908331984e-11),
        60: (7.964029031817186e-12, 2.637892822134082e-11),
        70: (7.691392868658026e-12, 2.1262113332060223e-11),
        80: (8.450665096838155e-12, 2.352725654000879e-11),
        90: (8.828105146277665e-12, 2.534549767123003e-11),
    }

    for dirbin in expected_results:
        result_mean = results[dirbin][0]
        assert isinstance(result_mean, float)
        assert math.isclose(result_mean, expected_results[dirbin][0], rel_tol=1e-3)
        result_std = results[dirbin][1]
        assert isinstance(result_std, float)
        assert math.isclose(result_std, expected_results[dirbin][1], rel_tol=1e-3)


@pytest.mark.benchmark
def test_spectra_get_spectrum_polar_angles_frompackets(benchmark: BenchmarkFixture) -> None:
    timelowdays = at.get_timestep_times(modelpath_classic_3d, loc="start")[0]
    timehighdays = at.get_timestep_times(modelpath_classic_3d, loc="end")[25]

    lambda_bin_edges = np.arange(100.0, 50000.0 + 100.0, 100.0)

    spectrafrompkts = benchmark(
        lambda: at.spectra.get_from_packets(
            modelpath=modelpath_classic_3d,
            average_over_phi=True,
            timelowdays=timelowdays,
            timehighdays=timehighdays,
            lambda_bin_edges=lambda_bin_edges,
        )
    )

    results_pkts = {
        dirbin: (
            dfspecdir.select(pl.col("f_lambda").mean()).collect().item(),
            dfspecdir.select(pl.col("f_lambda").std()).collect().item(),
        )
        for dirbin, dfspecdir in spectrafrompkts.items()
    }

    expected_results = {
        0: (1.797586165792736e-12, 5.590298324735384e-12),
        10: (1.5663080169205979e-12, 4.939024069693974e-12),
        20: (1.774102666296695e-12, 5.520228221791345e-12),
        30: (1.5995255247265518e-12, 4.9241711406614e-12),
        40: (1.639362478438235e-12, 4.9895610982735244e-12),
        50: (1.675756083487473e-12, 5.1182931882077005e-12),
        60: (1.5760353230761404e-12, 4.736330612172389e-12),
        70: (1.6263369485761792e-12, 4.913505318082798e-12),
        80: (1.7403350676338995e-12, 5.223319837133006e-12),
        90: (1.7311116087677568e-12, 5.333605673696894e-12),
    }
    actual_results = {
        dirbin: (float(results_pkts[dirbin][0]), float(results_pkts[dirbin][1])) for dirbin in expected_results
    }
    print(f"results: {actual_results}")

    for dirbin in expected_results:
        expected_mean = expected_results[dirbin][0]
        actual_mean = actual_results[dirbin][0]
        assert math.isclose(actual_mean, expected_mean, rel_tol=1e-3)
        expected_std = expected_results[dirbin][1]
        actual_std = actual_results[dirbin][1]
        assert math.isclose(actual_std, expected_std, rel_tol=1e-3)


def get_contributions_classic_3d(
    **kwargs: t.Any,
) -> tuple[list[atspectra.FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Bin the packets of the classic 3d model. The time window and the wavelength window are constant.

    Thus the results of the different tests are comparable.
    """
    kwargs.setdefault("emtypecolumn", "emissiontype")
    return atspectra.get_flux_contributions_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=3.0,
        timehighdays=8.0,
        lambda_bin_edges=np.linspace(3500.0, 8000.0, 101),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("groupby", "expected_contribs", "expected_top"),
    [
        (
            "ion",
            10,
            [
                ("Co II", 1.541452039348011e-08),
                ("free-free", 5.6865311711303595e-09),
                ("Co III", 6.196070067481124e-10),
            ],
        ),
        (
            "line",
            776,
            [
                ("free-free", 5.6865311711303595e-09),
                ("Co II λ3754 69-25", 4.2743896083241423e-10),
                ("Co II λ4160 69-31", 3.974052796767683e-10),
            ],
        ),
    ],
)
def test_spectra_flux_contribution_labels_from_packets(
    groupby: str, expected_contribs: int, expected_top: list[tuple[str, float]]
) -> None:
    """The emission group labels and the absorption group labels come from the linelist.

    This test makes sure that the labels and their fluxes do not change.
    """
    contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(groupby=groupby)

    assert len(contributions) == expected_contribs
    for contrib, (linelabel, fluxcontrib) in zip(contributions[: len(expected_top)], expected_top, strict=True):
        assert contrib.linelabel == linelabel
        assert np.isclose(contrib.fluxcontrib, fluxcontrib, rtol=1e-4)

    assert np.isclose(np.trapezoid(array_flambda_emission_total, x=array_lambda), 2.019784984151385e-08, rtol=1e-4)


# The "ion" group makes an ion label for the absorption. The "nucmass" group makes a line label.
# These two groups thus test the two conditions. The "line" group gives the same result as the "nucmass" group.
@pytest.mark.parametrize("groupby", ["ion", "nucmass"])
def test_spectra_absorption_contributions_from_packets(groupby: str) -> None:
    """Each absorption label is a line label. The group of the emission contributions does not change this."""
    contributions, _, _ = get_contributions_classic_3d(groupby=groupby, getemission=False)

    assert contributions
    assert all(contrib.linelabel for contrib in contributions)


def test_spectra_absorption_contributions_reject_nuclide_groupby() -> None:
    with pytest.raises(ValueError, match="cannot be grouped by nuclide"):
        get_contributions_classic_3d(groupby="nuc", emtypecolumn="pellet_nucindex")


def test_spectra_get_flux_contributions(benchmark: BenchmarkFixture) -> None:
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.get_spectra(
        modelpath=modelpath, timestepmin=timestepmin, timestepmax=timestepmax, fluxfilterfunc=None
    )[-1].collect()

    integrated_flux_specout = np.trapezoid(dfspectrum["f_lambda"], x=dfspectrum["lambda_angstroms"])

    _contribution_list, array_flambda_emission_total, arraylambda_angstroms = benchmark(
        lambda: at.spectra.get_flux_contributions(
            modelpath, timestepmin=timestepmin, timestepmax=timestepmax, use_lastemissiontype=False
        )
    )

    integrated_flux_emission = -np.trapezoid(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f"Integrated flux from spec.out:     {integrated_flux_specout}")
    print(f"Integrated flux from emission sum: {integrated_flux_emission}")
    assert math.isclose(integrated_flux_specout, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [
        abs(x - y)
        for x, y in zip(reversed(array_flambda_emission_total), dfspectrum["f_lambda"].to_numpy(), strict=False)
    ]
    print(f"Max f_lambda difference {max(diff) / integrated_flux_specout}")
    assert max(diff) / integrated_flux_specout < 1e-9


def test_spectra_get_flux_contributions_wavelength_window() -> None:
    """A wavelength window restricts the spectra and the flux contributions used for ranking."""
    timestepmin = 40
    timestepmax = 80
    lambda_min = 3500.0
    lambda_max = 7000.0

    contributions_full, flambda_total_full, arraylambda_full = at.spectra.get_flux_contributions(
        modelpath, timestepmin=timestepmin, timestepmax=timestepmax, use_lastemissiontype=False
    )

    contributions_window, flambda_total_window, arraylambda_window = at.spectra.get_flux_contributions(
        modelpath,
        timestepmin=timestepmin,
        timestepmax=timestepmax,
        use_lastemissiontype=False,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    nu_select = (arraylambda_full >= lambda_min) & (arraylambda_full <= lambda_max)
    assert np.array_equal(arraylambda_window, arraylambda_full[nu_select])
    assert np.allclose(flambda_total_window, flambda_total_full[nu_select], rtol=1e-12)

    contrib_full_bylabel = {c.linelabel: c for c in contributions_full}
    assert any(c.fluxcontrib < 0.999 * contrib_full_bylabel[c.linelabel].fluxcontrib for c in contributions_window), (
        "expected some flux contribution outside the window to be excluded from the ranking integral"
    )
    for contrib in contributions_window:
        full = contrib_full_bylabel[contrib.linelabel]
        assert contrib.fluxcontrib <= full.fluxcontrib * (1 + 1e-12)
        assert np.allclose(contrib.array_flambda_emission, full.array_flambda_emission[nu_select], rtol=1e-12)
        assert np.allclose(contrib.array_flambda_absorption, full.array_flambda_absorption[nu_select], rtol=1e-12)


def test_spectra_get_flux_contributions_from_packets(benchmark: BenchmarkFixture) -> None:
    lambdamin = 200.0
    lambdamax = 20000.0
    deltalambda = 100.0
    timelowdays = 4.0
    timehighdays = 7.0
    lambda_bin_edges = np.arange(lambdamin, lambdamax + deltalambda, deltalambda)
    dfspectrum = at.spectra.get_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=timelowdays,
        timehighdays=timehighdays,
        lambda_bin_edges=lambda_bin_edges,
    )[-1].collect()

    integrated_flux_specout = np.trapezoid(dfspectrum["f_lambda"], x=dfspectrum["lambda_angstroms"])
    _contribution_list, array_flambda_emission_total, arraylambda_angstroms = benchmark(
        lambda: at.spectra.get_flux_contributions_from_packets(
            modelpath_classic_3d,
            timelowdays=timelowdays,
            timehighdays=timehighdays,
            emtypecolumn="emissiontype",
            lambda_bin_edges=lambda_bin_edges,
        )
    )

    integrated_flux_emission = np.trapezoid(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f"Integrated flux from spec.out:     {integrated_flux_specout}")
    print(f"Integrated flux from emission sum: {integrated_flux_emission}")
    assert math.isclose(integrated_flux_specout, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [abs(x - y) for x, y in zip(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy(), strict=False)]
    print(f"Max f_lambda difference {max(diff) / integrated_flux_specout}")
    assert max(diff) / integrated_flux_specout < 1e-10


@pytest.mark.parametrize(
    ("use_emissiontime", "use_escapetime", "expected_use_time"), [(True, False, "emission"), (False, True, "escape")]
)
@mock.patch("artistools.spectra.plotspectra.atspectra.get_flux_contributions_from_packets")
def test_spectra_contribution_plot_forwards_packet_time(
    mockgetcontributions: mock.MagicMock,
    tmp_path: Path,
    use_emissiontime: bool,
    use_escapetime: bool,
    expected_use_time: str,
) -> None:
    """The contribution plot must use the packet time that the command selects."""
    contribution_list: list[atspectra.FluxContributionTuple] = []
    mockgetcontributions.return_value = (contribution_list, np.zeros(2), np.array([4000.0, 5000.0]))

    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / f"contributions_{expected_use_time}.pdf",
        timemin=290.0,
        timemax=320.0,
        emissionabsorption=True,
        use_emissiontime=use_emissiontime,
        use_escapetime=use_escapetime,
    )

    assert mockgetcontributions.call_args.kwargs["use_time"] == expected_use_time


def test_spectra_gamma_emission_time_uses_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gamma packet selection must use the decay time and not the radiation packet emission time."""
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0],
        "t_arrive_d": [1.0, 3.0],
        "tdecay": [1.0 * at.constants.day_to_s, 3.0 * at.constants.day_to_s],
        "em_time": [3.0 * at.constants.day_to_s, 1.0 * at.constants.day_to_s],
        "nu_rf": [at.constants.c_ang_per_s / 5000.0] * 2,
    })

    def get_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
        return 1, dfpackets.lazy()

    monkeypatch.setattr(atspectra.atpackets, "get_packets", get_packets)
    dfspectrum = atspectra.get_from_packets(
        modelpath=Path(),
        timelowdays=0.5,
        timehighdays=1.5,
        lambda_bin_edges=np.array([4000.0, 5000.0, 6000.0]),
        use_time="emission",
        gamma=True,
        directionbins=[-1],
    )[-1].collect()

    integrated_flux = dfspectrum.select((pl.col("f_lambda") * pl.col("delta_lambda")).sum()).item()
    expected_flux = 1.0 / at.constants.day_to_s / (4 * math.pi * at.constants.megaparsec_to_cm**2)
    assert np.isclose(integrated_flux, expected_flux)


def test_spectra_contributions_use_escape_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contribution spectra must use the same escape-time packet set and energy as the total spectrum."""
    nu_rf = at.constants.c_ang_per_s / 5000.0
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0],
        "e_cmf": [10.0, 20.0],
        "t_arrive_d": [1.5, 3.0],
        "escape_time": [3.0 * at.constants.day_to_s, 1.5 * at.constants.day_to_s],
        "pellet_nucindex": [0, 0],
        "nu_rf": [nu_rf, nu_rf],
    })
    lambda_bin_edges = np.array([4000.0, 5000.0, 6000.0])

    def get_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
        return 1, dfpackets.lazy()

    def get_escape_surface_gamma(_modelpath: Path | str) -> float:
        return 1.0

    def get_nuclides(modelpath: Path | str) -> pl.LazyFrame:
        del modelpath
        return pl.LazyFrame({"pellet_nucindex": [0], "nucname": ["Ni56"]})

    monkeypatch.setattr(atspectra.atpackets, "get_packets", get_packets)
    monkeypatch.setattr(atspectra, "get_escape_surface_gamma", get_escape_surface_gamma)
    monkeypatch.setattr(atspectra, "get_nuclides", get_nuclides)

    dfspectrum = atspectra.get_from_packets(
        modelpath=Path(),
        timelowdays=1.0,
        timehighdays=2.0,
        lambda_bin_edges=lambda_bin_edges,
        use_time="escape",
        directionbins=[-1],
    )[-1].collect()
    contributions, array_flambda_emission_total, array_lambda = atspectra.get_flux_contributions_from_packets(
        modelpath=Path(),
        timelowdays=1.0,
        timehighdays=2.0,
        lambda_bin_edges=lambda_bin_edges,
        getabsorption=False,
        groupby="nuc",
        use_time="escape",
        emtypecolumn="pellet_nucindex",
    )

    assert [contribution.linelabel for contribution in contributions] == ["Ni56"]
    assert np.array_equal(array_lambda, dfspectrum["lambda_angstroms"].to_numpy())
    assert np.allclose(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy())


def test_spectra_escape_time_with_3d_model() -> None:
    """Escape-time spectra must get the outer velocity from metadata for a 3D model."""
    dfspectrum = atspectra.get_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=3.0,
        timehighdays=8.0,
        lambda_bin_edges=np.linspace(3500.0, 8000.0, 101),
        use_time="escape",
        directionbins=[-1],
    )[-1].collect()

    assert dfspectrum["f_lambda"].is_finite().all()
    assert dfspectrum["f_lambda"].sum() > 0.0


@pytest.mark.benchmark
def test_spectra_timeseries_subplots() -> None:
    timedayslist = [295, 300]
    at.spectra.plot(
        argsraw=[], specpath=modelpath, outputfile=outputpath, timedayslist=timedayslist, multispecplot=True
    )


def test_spectra_multispec_outputfile(tmp_path: Path) -> None:
    """A custom -outputfile must be honoured in multispecplot mode instead of falling back to plotspec.pdf in CWD."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "custom_multispec.pdf",
        timedayslist=[295, 300],
        multispecplot=True,
    )
    assert (tmp_path / "custom_multispec.pdf").is_file()


def test_write_data_skips_reference_spectra(tmp_path: Path) -> None:
    """--write_data must write one column per ARTIS model and nothing for a reference spectrum.

    Only the ARTIS branch produces series data, so an unset variable used to raise UnboundLocalError for a
    reference spectrum alone, and to re-write the preceding model's spectrum as a bogus extra column after it.
    """
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "spec.pdf",
        timedays="300",
        write_data=True,
    )

    dfout = pl.read_csv(tmp_path / "spec.txt", separator=" ")
    assert [col for col in dfout.columns if col.startswith("f_lambda")] == ["f_lambda.TEST MODEL"]

    # a reference spectrum on its own has nothing to write, but must still plot without error
    at.spectra.plot(
        argsraw=[],
        specpath=["2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "refonly.pdf",
        timedays="300",
        write_data=True,
    )
    assert (tmp_path / "refonly.pdf").is_file()


def test_plotspectra_title_arg() -> None:
    parser = argparse.ArgumentParser()
    at.spectra.plotspectra.addargs(parser)
    assert parser.parse_args(["-title", "Custom title"]).title == "Custom title"
    assert parser.parse_args([]).title is None


@mock.patch.object(mplax.Axes, "set_title", side_effect=mplax.Axes.set_title, autospec=True)
def test_spectraplot_custom_title(mocksettitle: mock.MagicMock) -> None:
    """-title text must be passed through to the axis title (previously store_true produced a title of 'True')."""
    at.spectra.plot(
        argsraw=[], specpath=modelpath, outputfile=outputpath, timemin=290, timemax=320, title="Custom title"
    )
    titles = [call[0][1] for call in mocksettitle.call_args_list]
    assert "Custom title" in titles
    assert all(isinstance(title, str) for title in titles)


def test_writespectra() -> None:
    at.spectra.writespectra.main(argsraw=[], modelpath=modelpath)


def test_a_unit_that_no_spectrum_takes_stops_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    """-xunit reads the name while argparse parses, thus a mistake stops the command before it reads a file.

    The command took every name and stopped later with "Unknown xunit", after it read the spectra.
    """
    import artistools.spectra.plotspectra

    parser = at.commands.SuggestingArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([".", "-xunit", "micrometre"])

    assert "Did you mean micron?" in capsys.readouterr().err

    # a name of a unit is not case sensitive, and each spelling gives the canonical one
    assert parser.parse_args([".", "-xunit", "Angstrom"]).xunit == "angstroms"
    assert parser.parse_args([".", "-x", "MU"]).xunit == "micron"


@pytest.mark.parametrize("xunit", ["angstroms", "nm", "micron", "hz", "ev", "kev", "mev", "erg"])
def test_xunit_conversion_roundtrip(xunit: str) -> None:
    """Every unit accepted by convert_angstroms_to_unit must also be invertible by convert_unit_to_angstroms."""
    arr_lambda = np.array([1.0, 100.0, 5000.0, 1e5])

    arr_converted = at.spectra.convert_angstroms_to_unit(arr_lambda, xunit)
    arr_roundtrip = at.spectra.convert_unit_to_angstroms(arr_converted, xunit)

    assert arr_roundtrip == pytest.approx(arr_lambda, rel=1e-9)


def test_get_emabs_timeblock_count() -> None:
    """The row stride must come from each file's own size, not from shared state."""
    n_nu, n_timesteps = 5, 4

    # a normal file has one time block per timestep, a polarisation file has three (I, Q, U)
    dfplain = pl.DataFrame({"col": range(n_nu * n_timesteps)})
    dfpol = pl.DataFrame({"col": range(n_nu * n_timesteps * 3)})

    assert atspectra.get_emabs_timeblock_count(dfplain, n_nu, n_timesteps, "emission.out") == n_timesteps
    assert atspectra.get_emabs_timeblock_count(dfpol, n_nu, n_timesteps, "emissionpol.out") == 3 * n_timesteps

    # a row count that is not a whole number of frequency bins is rejected
    dfragged = pl.DataFrame({"col": range(n_nu * n_timesteps + 1)})
    with pytest.raises(AssertionError, match="not a multiple"):
        atspectra.get_emabs_timeblock_count(dfragged, n_nu, n_timesteps, "emission.out")

    # so is a whole-multiple count that matches neither the plain nor the polarisation layout
    dfwrong = pl.DataFrame({"col": range(n_nu * n_timesteps * 2)})
    with pytest.raises(AssertionError, match="time blocks per frequency bin"):
        atspectra.get_emabs_timeblock_count(dfwrong, n_nu, n_timesteps, "emission.out")


def test_get_spectra_gamma_does_not_mix_in_uvoir_dirbins() -> None:
    """Gamma spectra must not pick up the UVOIR direction-resolved file.

    test-classicmode_3d has spec_res.out but no gamma_spec_res.out, so a gamma request must return only the
    spherically averaged bin rather than silently pairing gamma_spec.out with the UVOIR direction bins.
    """
    dirbin_spectra_uvoir = atspectra.get_spectra(modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10)
    assert -1 in dirbin_spectra_uvoir
    assert len(dirbin_spectra_uvoir) > 1, "expected direction-resolved UVOIR spectra in this test model"

    dirbin_spectra_gamma = atspectra.get_spectra(
        modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10, gamma=True
    )
    assert set(dirbin_spectra_gamma) == {-1}

    # the gamma spectrum must actually differ from the UVOIR one, i.e. it came from gamma_spec.out
    nu_gamma = dirbin_spectra_gamma[-1].select(pl.col("nu").max()).collect().item()
    nu_uvoir = dirbin_spectra_uvoir[-1].select(pl.col("nu").max()).collect().item()
    assert nu_gamma > nu_uvoir


def test_get_spectra_rejects_averaging_over_both_angles() -> None:
    """Averaging over phi and theta at once leaves too few bins, so it must be rejected up front."""
    with pytest.raises(ValueError, match="both the phi and theta"):
        atspectra.get_spectra(
            modelpath=modelpath_classic_3d,
            timestepmin=10,
            timestepmax=10,
            average_over_phi=True,
            average_over_theta=True,
        )


def test_reference_spectrum_named_out_is_not_read_as_artis(tmp_path: Path) -> None:
    """A user can name reference data with the .out suffix that ARTIS uses for its own output.

    path_is_artis_model reads that suffix, thus the predicate sent such a file to the ARTIS reader and
    the command drew nothing. The folder decides: an ARTIS run holds input.txt beside its output.
    """
    import shutil

    from artistools.spectra.plotspectra import path_is_reference_spectrum

    reference = at.get_path("artistools_dir") / "data" / "refspectra" / "2010lp_20110928_fors2.txt"
    shutil.copy(reference, tmp_path / "myref.out")
    shutil.copy(reference, tmp_path / "myref.txt")

    assert path_is_reference_spectrum(tmp_path / "myref.out")
    assert path_is_reference_spectrum(tmp_path / "myref.txt")

    # the output of a real run keeps its own reading, whether the user names the folder or the file
    assert not path_is_reference_spectrum(modelpath)
    assert not path_is_reference_spectrum(modelpath / "spec.out")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraemissionplot_draws_a_reference_named_out(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The emission plot must draw a reference spectrum whose name ends in the .out suffix of ARTIS.

    The overlay loop read the suffix alone, thus it skipped such a file without a word.
    """
    import shutil

    source = at.get_path("artistools_dir") / "data" / "refspectra" / "2003du_20031213_3219_8822_00.txt"
    shutil.copy(source, tmp_path / "myref.out")
    shutil.copy(f"{source}.meta.yml", tmp_path / "myref.out.meta.yml")

    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, tmp_path / "myref.out"],
        outputfile=tmp_path,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )

    labels = {callargs.kwargs.get("label") for callargs in mockplot.call_args_list}
    assert any(label and "2003du" in str(label) for label in labels), labels
