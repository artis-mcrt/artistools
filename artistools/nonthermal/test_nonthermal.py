from pathlib import Path

import pytest

import artistools as at

modelpath = at.get_path("testartismodel")
outputpath = at.get_path("testoutput")


def test_spencerfano() -> None:
    at.nonthermal.solvespencerfanocmd.main(
        argsraw=[], modelpath=modelpath, timedays=300, makeplot=True, npts=200, noexcitation=True, outputfile=outputpath
    )


@pytest.mark.parametrize("x_e", [0.0, 0.01, 0.5, 1.0, 1.5, 2.0, 3.7, 26.0])
def test_ionpops_for_electronfraction(x_e: float) -> None:
    """The ion populations must average to x_e free electrons per nucleus, for x_e above one as well as below.

    x_e = N_e / N_ions is not capped at one: a nucleus ionised k times releases k electrons, so x_e = 2 is a
    doubly-ionised plasma. Splitting the nuclei only between the neutral and singly-ionised stages could not
    represent that, and gave a negative neutral population for x_e > 1.
    """
    from artistools.nonthermal.solvespencerfanocmd import ionpops_for_electronfraction

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
    from artistools.nonthermal.solvespencerfanocmd import ionpops_for_electronfraction

    with pytest.raises(ValueError, match="negative"):
        ionpops_for_electronfraction(26, -0.1, 1.0)

    with pytest.raises(ValueError, match="exceeds the atomic number"):
        ionpops_for_electronfraction(26, 26.5, 1.0)


def test_leptontransport_fully_ionised(tmp_path: Path) -> None:
    """A fully ionised plasma has no bound electrons, so only the plasma loss term stops the lepton."""
    from artistools.nonthermal.leptontransport import calculate_dE_on_dx_ionexc
    from artistools.nonthermal.leptontransport import calculate_dE_on_dx_plasma

    energy = 1e3 * 1.602176634e-19  # [J]
    assert calculate_dE_on_dx_ionexc(energy, 0.0) == 0.0
    assert calculate_dE_on_dx_plasma(energy, 1e11) < 0.0

    # propagating on the ion/exc term alone would divide by zero here
    outputfile = tmp_path / "leptontransport.pdf"
    at.nonthermal.leptontransport.main(argsraw=[], energy=1e3, nnebound=0.0, nnefree=1e5, outputfile=outputfile)
    assert outputfile.is_file()


def test_leptontransport_rejects_empty_plasma() -> None:
    """With neither bound nor free electrons the lepton never loses energy, so the integration cannot terminate."""
    with pytest.raises(ValueError, match="must be positive"):
        at.nonthermal.leptontransport.main(argsraw=[], nnebound=0.0, nnefree=0.0)


@pytest.mark.parametrize(("nnebound", "nnefree"), [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_leptontransport_rejects_negative_density(nnebound: float, nnefree: float) -> None:
    """A negative density would contribute as a positive one, since the helpers return a loss magnitude."""
    with pytest.raises(ValueError, match="cannot be negative"):
        at.nonthermal.leptontransport.main(argsraw=[], nnebound=nnebound, nnefree=nnefree)
