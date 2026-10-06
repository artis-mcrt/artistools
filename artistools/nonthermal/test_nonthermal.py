import argparse
import itertools
import math
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testartismodel")


def read_ntstats(ostatfile: Path) -> pl.DataFrame:
    return at.misc.read_wsv(ostatfile, comment_prefix="#", header_from_comment=True)


def test_spencerfano(tmp_path: Path) -> None:
    """Solve the cell of the test model at 300 days, and save the plot of the solution."""
    ostatfile = tmp_path / "ntstats.txt"
    at.nonthermal.spencerfano.main(
        argsraw=[],
        modelpath=modelpath,
        timedays=300,
        makeplot=True,
        npts=200,
        noexcitation=True,
        outputfile=tmp_path,
        ostat=str(ostatfile),
    )

    # 300 days is in timestep 54
    assert len(list(tmp_path.glob("spencerfano_cell00000_ts054_*.pdf"))) == 1
    dfstats = read_ntstats(ostatfile)
    assert dfstats["frac_excitation"].item() == 0.0
    assert dfstats["frac_sum"].item() == pytest.approx(1.0, abs=0.1)


def test_spencerfano_excitation(tmp_path: Path) -> None:
    """Solve with the excitation path. The solver reads the derived transition columns, e.g. epsilon_trans_ev."""
    ostatfile = tmp_path / "ntstats.txt"
    at.nonthermal.spencerfano.main(argsraw=[], modelpath=modelpath, timedays=300, npts=200, ostat=str(ostatfile))

    dfstats = read_ntstats(ostatfile)
    assert dfstats["frac_excitation"].item() > 0.0
    assert dfstats["frac_sum"].item() == pytest.approx(1.0, abs=0.1)


def test_spencerfano_ostat_takes_a_changing_ion_list(tmp_path: Path) -> None:
    """A -vary x_e sweep changes the list of ions from one step to the next, thus -ostat gives a column to each ion.

    The file named the ions of the first step, and a later step with another list stopped the command.
    """
    ostatfile = tmp_path / "ntstats.txt"
    at.nonthermal.spencerfano.main(
        argsraw=[], composition="Fe", x_e=0.001, vary="x_e", npts=50, noexcitation=True, ostat=str(ostatfile)
    )

    dfstats = at.misc.read_wsv(ostatfile, comment_prefix="#", header_from_comment=True)
    assert dfstats.height == 9
    assert {"frac_ionization_FeI", "frac_ionization_FeII", "frac_ionization_FeXI"} <= set(dfstats.columns)
    # an ion that a step does not hold takes zero, e.g. Fe I at the highest electron fraction
    assert dfstats["frac_ionization_FeI"][-1] == 0.0
    assert dfstats["frac_ionization_FeXI"][-1] > 0.0

    # the heating, the ionisation, and the excitation take all of the deposited energy. The grid of this
    # test holds 50 points only, thus the sum of the fractions differs from 1.0 by up to ten percent
    assert dfstats["frac_sum"].to_numpy() == pytest.approx(1.0, abs=0.1)


def test_spencerfano_vary_x_e_stays_below_the_atomic_number() -> None:
    """The sweep must not ask for more free electrons than a nucleus can supply.

    An electron fraction above atomic_number - 1 gives only the bare nucleus. The sweep multiplied the
    start value by ten at each half step. Thus the default start of 2 went above the atomic number of iron.
    """
    from artistools.nonthermal.spencerfano import x_e_of_sweep_step

    stepcount = 9
    x_e_sweep = [x_e_of_sweep_step(2.0, 26, step, stepcount) for step in range(stepcount)]
    assert x_e_sweep[0] == pytest.approx(2.0)
    assert x_e_sweep[-1] == pytest.approx(25.0)

    # the sweep is uniform in log10(x_e), thus each pair of steps has the same ratio
    stepratios = [x_e_next / x_e for x_e, x_e_next in itertools.pairwise(x_e_sweep)]
    assert stepratios == pytest.approx([stepratios[0]] * len(stepratios))

    with pytest.raises(ValueError, match="gives no sweep"):
        x_e_of_sweep_step(25.0, 26, 0, stepcount)


@pytest.mark.parametrize("x_e", [0.0, 0.01, 0.5, 1.0, 1.5, 2.0, 3.7, 26.0])
def test_ionpops_for_electronfraction(x_e: float) -> None:
    """The ion populations must average to x_e free electrons per nucleus, for x_e above one as well as below.

    x_e = N_e / N_ions is not capped at one: a nucleus ionised k times releases k electrons, so x_e = 2 is a
    doubly-ionised plasma. Splitting the nuclei only between the neutral and singly-ionised stages could not
    represent that, and gave a negative neutral population for x_e > 1.
    """
    from artistools.nonthermal.spencerfano import ionpops_for_electronfraction

    atomic_number = 26
    nntot = 3.0
    ionpopdict = ionpops_for_electronfraction(atomic_number, x_e, nntot)

    assert all(pop >= 0.0 for pop in ionpopdict.values()), "a population cannot be negative"
    assert sum(ionpopdict.values()) == pytest.approx(nntot), "every nucleus must be in some ion stage"

    # ion stage 1 is neutral, so a nucleus in stage n has released n - 1 electrons
    n_e = sum((ion_stage - 1) * pop for (_, ion_stage), pop in ionpopdict.items())
    assert n_e / nntot == pytest.approx(x_e)


def test_ionpops_for_electronfraction_rejects_impossible_values() -> None:
    """An element cannot release more electrons than it has, nor a negative number."""
    from artistools.nonthermal.spencerfano import ionpops_for_electronfraction

    with pytest.raises(ValueError, match="negative"):
        ionpops_for_electronfraction(26, -0.1, 1.0)

    with pytest.raises(ValueError, match="exceeds the atomic number"):
        ionpops_for_electronfraction(26, 26.5, 1.0)


def test_spencerfano_makeplot_with_element_composition(tmp_path: Path) -> None:
    """--makeplot with a non-ARTIS -composition names the plot file after the element.

    The default file name holds a timestep field and a time field, which stay None without an
    ARTIS model. Thus the format stopped with TypeError before this test existed.
    """
    at.nonthermal.spencerfano.main(
        argsraw=[], composition="He", x_e=0.5, makeplot=True, npts=200, noexcitation=True, outputfile=tmp_path
    )

    assert (tmp_path / "spencerfano_He.pdf").is_file()


def test_spencerfano_solves_an_ion_stage_above_xx(tmp_path: Path) -> None:
    """An electron fraction above 19 gives an ion stage above XX, and the table of roman numerals stops at XX.

    The statistics row named each ion with a roman numeral, thus the command stopped with an IndexError after
    the solution. The command built that row also without -ostat.
    """
    # the default sweep of iron runs from x_e 2 to 25, thus its last step holds Fe XXVI alone
    at.nonthermal.spencerfano.main(argsraw=[], composition="Fe", vary="x_e", npts=50, noexcitation=True)

    ostatfile = tmp_path / "ntstats.txt"
    at.nonthermal.spencerfano.main(
        argsraw=[], composition="Fe", x_e=21, npts=50, noexcitation=True, ostat=str(ostatfile)
    )
    # x_e 21 puts every nucleus of iron in ion stage 22, which has the charge 21+
    dfstats = read_ntstats(ostatfile)
    assert dfstats["frac_ionization_Fe21+"].item() > 0.0
    assert at.nonthermal.spencerfano.get_ion_column_name(26, 20) == "FeXX"


def test_ntlepton_deposition_rate_density_follows_artis() -> None:
    """ARTIS normalises the solution with the deposition of the gamma rays, the positrons, and the electrons.

    The command took heating_dep, which holds only the heating part of that rate, and the alpha deposition.
    """
    from artistools.nonthermal.spencerfano import get_ntlepton_deposition_rate_density

    newerrun = {
        "deposition_gamma": 2.0e-10,
        "deposition_positron": 1.0e-10,
        "deposition_electron": 5.0e-11,
        "deposition_alpha": 2.5e-11,
        "total_dep": 3.75e-10,
        "heating_dep": 3.0e-10,
    }
    assert get_ntlepton_deposition_rate_density(newerrun) == pytest.approx(3.5e-10, rel=1e-12)

    # an older run gives no rate for each particle, and its total_dep is the rate that ARTIS used
    olderrun = {"total_dep": 7.7e-10, "heating_dep": 6.5e-10}
    assert get_ntlepton_deposition_rate_density(olderrun) == pytest.approx(7.7e-10, rel=1e-12)
    assert get_ntlepton_deposition_rate_density({"heating_dep": 6.5e-10}) is None


def test_spencerfano_takes_the_deposition_of_the_cell() -> None:
    """The solution of an ARTIS cell takes total_dep of an older run, and not the smaller heating_dep."""
    from artistools.nonthermal import spencerfano

    parser = argparse.ArgumentParser()
    spencerfano.addargs(parser)
    args = parser.parse_args(["-timedays", "300"])
    conditions = spencerfano.get_artis_conditions(args, modelpath)

    estim = at.estimators.read_estimators(modelpath, timestep=54, modelgridindex=0)[54, 0]
    assert estim["heating_dep"] < 0.9 * estim["total_dep"]
    assert conditions.deposition_density_ev == pytest.approx(estim["total_dep"] / at.constants.EV_to_erg, rel=1e-9)


def test_spencerfano_stops_with_a_message_for_a_cell_of_no_deposition(capsys: pytest.CaptureFixture[str]) -> None:
    """The first timestep of the test model has no deposition, and the solver needs a positive rate."""
    with pytest.raises(SystemExit):
        at.nonthermal.spencerfano.main(argsraw=[], modelpath=modelpath, timestep=0, npts=50, noexcitation=True)

    assert "deposition rate of 0.0 erg/s/cm3 at timestep 0" in capsys.readouterr().err


def test_spencerfano_needs_no_nlte_populations(tmp_path: Path) -> None:
    """The solution takes the ion populations of the estimators, thus a run with no NLTE output can be solved.

    The command stopped with FileNotFoundError when the model folder held no nlte_*.out file.
    """
    for filepath in modelpath.iterdir():
        if not filepath.name.startswith("nlte_"):
            (tmp_path / filepath.name).symlink_to(filepath)

    ostatfile = tmp_path / "ntstats.txt"
    at.nonthermal.spencerfano.main(
        argsraw=[], modelpath=tmp_path, timedays=300, npts=50, noexcitation=True, ostat=str(ostatfile)
    )

    assert read_ntstats(ostatfile).height == 1


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_ntstats_plot_shows_the_parameter_of_the_sweep(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A sweep of npts keeps x_e, thus the plot takes npts as the horizontal axis.

    The axis was always x_e, thus every step of such a sweep took one position.
    """
    from artistools.nonthermal.spencerfano import write_ntstats_file

    nptsvalues = [50, 100, 200]
    ostatfile = tmp_path / "ntstats.txt"
    write_ntstats_file(
        ostatfile,
        [
            {
                "emin": 0.1,
                "emax": 16000.0,
                "npts": npts,
                "x_e": 0.5,
                "frac_sum": 1.0,
                "frac_excitation": 0.0,
                "frac_ionization": 0.2,
                "frac_heating": 0.8,
            }
            for npts in nptsvalues
        ],
    )

    at.nonthermal.spencerfano.main(argsraw=[], plotstats=str(ostatfile))

    xarr = np.asarray(mockplot.call_args_list[0][0][1], dtype=float)
    assert np.allclose(xarr, [math.log10(npts) for npts in nptsvalues])
    assert ostatfile.with_suffix(".pdf").is_file()


def test_spencerfano_quiet_keeps_the_energy_fractions(capsys: pytest.CaptureFixture[str]) -> None:
    """--quiet hides the progress messages, and the energy fractions are the product of the command.

    The command printed the fractions with print, thus --quiet removed them.
    """
    import artistools.__main__

    artistools.__main__.main(
        argsraw=["spencerfano", "-composition", "Fe", "-x_e", "0.5", "-npts", "50", "--noexcitation", "--quiet"]
    )

    stdout = capsys.readouterr().out
    assert "frac_heating" in stdout
    assert "frac_sum" in stdout
    # the analysis of the solver is a progress message
    assert "Reading" not in stdout
    assert len(stdout.strip().splitlines()) == 1
