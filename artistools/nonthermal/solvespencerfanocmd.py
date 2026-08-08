"""Solve the Spencer-Fano equation for a cell's nonthermal electron spectrum and plot the energy deposition."""

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at
from artistools.constants import EV_to_erg
from artistools.misc import add_modelpath_arg
from artistools.misc import add_timedays_arg
from artistools.misc import add_timestep_arg

minionfraction = 0.0  # minimum number fraction of the total population to include in SF solution

defaultoutputfile = "spencerfano_cell{cell:03d}_ts{timestep:02d}_{timedays:.0f}d.pdf"


def make_ntstats_plot(ntstatfile: str | Path) -> None:
    """Plot the fractions of nonthermal energy going to heating, ionisation, and excitation over time."""
    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(4, 3), tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3}
    )

    # the header line was written as a "#" comment
    dfstats = at.read_wsv(ntstatfile, comment_prefix="#", header_from_comment=True).fill_null(0)

    with pl.Config(tbl_cols=-1, tbl_rows=50):
        print(dfstats)

    xarr = np.log10(dfstats["x_e"])
    ax.plot(xarr, dfstats["frac_ionization"], label="Ionisation")
    max_frac_excitation = dfstats["frac_excitation"].max()
    assert isinstance(max_frac_excitation, int | float)
    if max_frac_excitation > 0.0:
        ax.plot(xarr, dfstats["frac_excitation"], label="Excitation")
    ax.plot(xarr, dfstats["frac_heating"], label="Heating")
    ioncols = [col for col in dfstats.columns if col.startswith("frac_ionization_")]
    for ioncol in ioncols:
        ion = ioncol.replace("frac_ionization_", "")
        ax.plot(xarr, dfstats[ioncol], label=f"{ion} ionisation")

    ax.set_ylabel(r"Energy fraction")
    ax.set_xlabel(r"log x$_e$")
    ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1)
    ax.autoscale(enable=True, axis="both", tight=True)
    outputfilename = Path(ntstatfile).with_suffix(".pdf")
    fig.savefig(outputfilename, format="pdf")
    at.print_saved(outputfilename)
    plt.close(fig)


def ionpops_for_electronfraction(atomic_number: int, x_e: float, nntot: float) -> dict[tuple[int, int], float]:
    """Distribute nntot nuclei of one element over ion stages whose mean charge is x_e free electrons per nucleus.

    x_e = N_e / N_ions is not bounded by one: a nucleus ionised k times contributes k free electrons, so a
    doubly-ionised plasma has x_e = 2. The nuclei are split between the two ion stages either side of x_e, which
    for x_e <= 1 reduces to the neutral/singly-ionised pair (1 - x_e, x_e).
    """
    if x_e < 0.0:
        msg = f"Electron fraction x_e must not be negative, got {x_e}"
        raise ValueError(msg)

    # a nucleus cannot give up more electrons than it has
    if x_e > atomic_number:
        msg = (
            f"Electron fraction x_e={x_e} exceeds the atomic number {atomic_number} of"
            f" {at.get_elsymbol(atomic_number)}, which cannot supply that many free electrons"
        )
        raise ValueError(msg)

    charge_lower = math.floor(x_e)
    frac_upper = x_e - charge_lower  # fraction of nuclei carrying one more charge than charge_lower

    # ion stage 1 is neutral, so a charge of k is ion stage k + 1
    ionpopdict: dict[tuple[int, int], float] = {(atomic_number, charge_lower + 1): nntot * (1.0 - frac_upper)}
    if frac_upper > 0.0:
        ionpopdict[atomic_number, charge_lower + 2] = nntot * frac_upper

    return ionpopdict


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    add_modelpath_arg(parser, default=".")

    add_timedays_arg(parser, kind="str")

    add_timestep_arg(parser, kind="int")

    parser.add_argument("-modelgridindex", "-cell", type=int, default=0, help="Modelgridindex to plot")

    parser.add_argument("-velocity", "-v", type=float, default=-1, help="Specify cell by velocity")

    parser.add_argument("-npts", type=int, default=4096, help="Number of points in the energy grid")

    parser.add_argument("-emin", type=float, default=0.1, help="Minimum energy in eV of Spencer-Fano solution")

    parser.add_argument(
        "-emax",
        type=float,
        default=16000,
        help="Maximum energy in eV of Spencer-Fano solution (approx where energy is injected)",
    )

    parser.add_argument(
        "-vary", action="store", choices=["emin", "emax", "npts", "emax,npts", "x_e"], help="Which parameter to vary"
    )

    parser.add_argument(
        "-composition",
        action="store",
        default="artis",
        choices=["artis", *at.get_elsymbolslist()[1:]],
        help="Composition comes from artis or specific an element to use",
    )

    parser.add_argument(
        "-x_e",
        type=float,
        default=2,
        help=(
            "If not using artis composition, specify the electron fraction = N_e / N_ions. Values above one mean"
            " multiply-ionised nuclei, e.g. 2 for a doubly-ionised plasma, up to the atomic number"
        ),
    )

    parser.add_argument("--makeplot", action="store_true", help="Save a plot of the non-thermal spectrum")

    parser.add_argument(
        "--differentialform",
        action="store_true",
        help="Solve differential form (KF92 Equation 6) instead ofintegral form (KF92 Equation 7)",
    )

    parser.add_argument("--noexcitation", action="store_true", help="Do not include collisional excitation transitions")

    parser.add_argument(
        "--ar1985",
        action="store_true",
        help="Use Arnaud & Rothenflug (1985, A&AS, 60, 425) for Fe ionization cross sections",
    )

    parser.add_argument(
        "-o",
        action="store",
        dest="outputfile",
        default=defaultoutputfile,
        help="Path/filename for PDF file if --makeplot is enabled",
    )

    parser.add_argument("-ostat", action="store", help="Path/filename for stats output")

    parser.add_argument(
        "-plotstats",
        action="store",
        default=None,
        help="Path/filename for NT stats input (no solution, only plotting stat file)",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Solve Spencer-Fano equation using data from ARTIS cell at some timestep."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.plotstats:
        make_ntstats_plot(args.plotstats)
        return

    modelpath = Path(args.modelpath)

    args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)
    dfpops: pl.DataFrame | None
    ionpopdict: dict[tuple[int, int] | int, float]
    if args.composition == "artis":
        if args.timedays:
            args.timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
        elif args.timestep is None:
            print("A time or timestep must be specified.")
            sys.exit()

        modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
        if args.velocity >= 0.0:
            args.modelgridindex = at.inputmodel.get_mgi_of_velocity_kms(modelpath, args.velocity)
        else:
            args.modelgridindex = args.modelgridindex
        assert isinstance(args.modelgridindex, int)
        estimators = at.estimators.read_estimators(
            modelpath, timestep=args.timestep, modelgridindex=args.modelgridindex
        )
        assert isinstance(args.timestep, int)
        assert isinstance(args.modelgridindex, int)
        estim = estimators[args.timestep, args.modelgridindex]

        dfpops = at.nltepops.read_files(modelpath, modelgridindex=args.modelgridindex, timestep=args.timestep)

        if dfpops.is_empty():
            print(f"ERROR: no NLTE populations for cell {args.modelgridindex} at timestep {args.timestep}")
            raise AssertionError

        nntot = estim["nntot"]
        x_e = estim["nne"] / nntot
        T_e = estim["Te"]
        print("WARNING: Use LTE pops at Te for now")
        deposition_density_ev = estim["heating_dep"] / EV_to_erg
        ionpopdict = {at.get_ion_tuple(k): v for k, v in estim.items() if k.startswith(("nnion_", "nnelement_"))}

        velocity = modeldata["vel_r_max_kmps"][args.modelgridindex]
        args.timedays = at.get_timestep_time(modelpath, args.timestep)
        print(f"timestep {args.timestep} cell {args.modelgridindex} (v={velocity} km/s at {args.timedays:.1f}d)")

    # ionpopdict = {}
    # deposition_density_ev = 327
    # nne = 6.7e5
    #
    # ionpopdict[(26, 1)] = ionpopdict[26] * 1e-4
    # ionpopdict[(26, 2)] = ionpopdict[26] * 0.20
    # ionpopdict[(26, 3)] = ionpopdict[26] * 0.80
    # ionpopdict[(26, 4)] = ionpopdict[26] * 0.
    # ionpopdict[(26, 5)] = ionpopdict[26] * 0.
    # ionpopdict[(27, 2)] = ionpopdict[27] * 0.20
    # ionpopdict[(27, 3)] = ionpopdict[27] * 0.80
    # ionpopdict[(27, 4)] = 0.
    # # ionpopdict[(28, 1)] = ionpopdict[28] * 6e-3
    # ionpopdict[(28, 2)] = ionpopdict[28] * 0.18
    # ionpopdict[(28, 3)] = ionpopdict[28] * 0.82
    # ionpopdict[(28, 4)] = ionpopdict[28] * 0.
    # ionpopdict[(28, 5)] = ionpopdict[28] * 0.

    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # # nne = nntot * x_e
    # # nne = .1
    # dfpops = None

    # ionpopdict[(at.get_atomic_number('Fe'), 2)] = nntot * 1.
    # ionpopdict[(at.get_atomic_number('Fe'), 3)] = nntot * 0.5
    # ionpopdict[(at.get_atomic_number('Fe'), 4)] = nntot * 0.3

    # KF1992 Figure 2. Pure-Oxygen Plasma
    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = None
    # ionpopdict[(at.get_atomic_number('O'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('O'), 2)] = nntot * x_e

    # KF1992 Figure 3. Pure-Helium Plasma
    # compelement = args.composition
    # compelement_atomicnumber = at.get_atomic_number(compelement)
    # x_e = args.x_e
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = None
    # ionpopdict[(at.get_atomic_number('He'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('He'), 2)] = nntot * x_e

    # KF1992 Figure 5. Pure-Iron Plasma
    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = None
    # ionpopdict[(at.get_atomic_number('Fe'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('Fe'), 2)] = nntot * x_e

    # KF1992 D. The Oxygen-Carbon Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.16 * nntot
    # ionpopdict[(at.get_atomic_number('C'), 2)] = 0.16 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.82 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 2)] = 0.82 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ne'), 1)] = 0.016 * nntot

    # # KF1992 G. The Silicon-Calcium Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.38e-5 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.94e-4 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 1)] = 0.63 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 2)] = 0.63 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('S'), 1)] = 0.29 * nntot
    # ionpopdict[(at.get_atomic_number('S'), 2)] = 0.29 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ar'), 1)] = 0.041 * nntot
    # ionpopdict[(at.get_atomic_number('Ca'), 1)] = 0.026 * nntot
    # ionpopdict[(at.get_atomic_number('Fe'), 1)] = 0.012 * nntot

    stepcount = 9 if args.vary else 1
    for step in range(stepcount):
        emin = args.emin
        emax = args.emax
        npts = args.npts
        if args.vary == "emax":
            emax *= 2**step
        elif args.vary == "emax,npts":
            npts *= 2**step
            emax *= 2**step

        elif args.vary == "emin":
            emin *= 2**step
        elif args.vary == "npts":
            npts *= 2**step
        elif args.vary == "x_e":
            assert args.composition != "artis"
        if args.composition != "artis":
            compelement = args.composition
            compelement_atomicnumber = at.get_atomic_number(compelement)
            deposition_density_ev = 5.0e3
            nntot = 1.0
            x_e = (args.x_e * 10 ** (0.5 * step)) if args.vary == "x_e" else args.x_e
            ionpopdict = {}
            dfpops = None
            T_e = 3000
            ionpopdict |= ionpops_for_electronfraction(compelement_atomicnumber, x_e, nntot)

        # keep only the ion populations, not element or total populations
        ions = [key for key in ionpopdict if isinstance(key, tuple) and ionpopdict[key] / nntot >= minionfraction]
        ions.sort()

        if args.noexcitation:
            adata = None
            dfpops = None
        else:
            adata = at.atomic.get_levels(modelpath, get_transitions=True, ionlist=tuple(ions))

        if step == 0 and args.ostat:
            strheader = "#emin emax npts x_e frac_sum frac_excitation frac_ionization frac_heating"
            for atomic_number, ion_stage in ions:
                strheader += " frac_ionization_" + at.get_ionstring(atomic_number, ion_stage, sep="")
            Path(args.ostat).write_text(strheader + "\n", encoding="utf-8")

        import pynonthermal as pynt

        with pynt.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts, verbose=True) as sf:
            for Z, ion_stage in ions:
                nnion = ionpopdict[Z, ion_stage]
                if nnion == 0.0:
                    print(f"   skipping Z={Z} ion_stage {ion_stage} due to nnion={nnion:.1e}")
                    continue

                sf.add_ionisation(Z, ion_stage, nnion)
                if not args.noexcitation:
                    sf.add_ion_ltepopexcitation(Z, ion_stage, nnion, adata_polars=adata, temperature=T_e)

            sf.solve(depositionratedensity_ev=deposition_density_ev)

            sf.analyse_ntspectrum()

            if args.makeplot:
                outputfilename = str(args.outputfile).format(
                    cell=args.modelgridindex, timestep=args.timestep, timedays=args.timedays
                )
                # outputfilename = "spencerfano.pdf"
                sf.plot_spec_channels(outputfilename=outputfilename)

            if args.ostat:
                with Path(args.ostat).open("a", encoding="utf-8") as fstat:
                    strlineout = (
                        f"{emin} {emax} {npts} {x_e:7.2e} {sf.get_frac_sum():6.3f} "
                        f"{sf.get_frac_excitation_tot():6.3f} {sf.get_frac_ionisation_tot():6.3f} "
                        f" {sf.get_frac_heating():6.3f}"
                    )
                    for atomic_number, ion_stage in ions:
                        nnion = ionpopdict[atomic_number, ion_stage]
                        frac_ionis_ion = sf.get_frac_ionisation_ion(atomic_number, ion_stage) if nnion > 0.0 else 0.0
                        strlineout += f" {frac_ionis_ion:.4f}"
                    fstat.write(strlineout + "\n")

    if args.ostat:
        make_ntstats_plot(args.ostat)


if __name__ == "__main__":
    main()
