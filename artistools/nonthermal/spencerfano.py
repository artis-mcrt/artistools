"""Solve the Spencer-Fano equation for a cell's nonthermal electron spectrum and plot the energy deposition."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import get_elsymbolslist
from artistools.atomic import get_ion_tuple
from artistools.atomic import get_ionstring
from artistools.atomic import get_levels
from artistools.constants import EV_to_erg
from artistools.constants import km_to_cm
from artistools.estimators import read_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import exit_with_error
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_single_timestep
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.nltepops import read_nltepops
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend

minionfraction = 0.0  # minimum number fraction of the total population to include in SF solution

defaultoutputfile = "spencerfano_cell{cell:05d}_ts{timestep:03d}_{timedays:.2f}d.pdf"


def write_ntstats_file(ntstatfile: str | Path, rows: Sequence[dict[str, float]]) -> None:
    """Write one row for each solver step, with a column for each ion of any step.

    A -vary x_e sweep changes the list of ions from one step to the next, thus an ion that a step does not hold
    takes zero in that row.
    """
    ioncolumns = list(dict.fromkeys(col for row in rows for col in row if col.startswith("frac_ionization_")))
    with Path(ntstatfile).open("w", encoding="utf-8") as fstat:
        fstat.write(
            " ".join(["#emin emax npts x_e frac_sum frac_excitation frac_ionization frac_heating", *ioncolumns]) + "\n"
        )
        fstat.writelines(
            f"{row['emin']} {row['emax']} {row['npts']} {row['x_e']:7.2e} {row['frac_sum']:6.3f} "
            f"{row['frac_excitation']:6.3f} {row['frac_ionization']:6.3f}  {row['frac_heating']:6.3f}"
            + "".join(f" {row.get(col, 0.0):.4f}" for col in ioncolumns)
            + "\n"
            for row in rows
        )


def make_ntstats_plot(ntstatfile: str | Path) -> None:
    """Plot the fractions of nonthermal energy going to heating, ionisation, and excitation over time."""
    fig, axesgrid = make_frame_figure(fullwidth=False)
    ax = axesgrid[0][0]

    # the header line was written as a "#" comment
    dfstats = read_wsv(ntstatfile, comment_prefix="#", header_from_comment=True).fill_null(0)

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
    set_legend(ax)
    ax.autoscale(enable=True, axis="both", tight=True)
    outputfilename = Path(ntstatfile).with_suffix(".pdf")
    save_figure(fig, outputfilename, format="pdf")


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
            f" {get_elsymbol(atomic_number)}, which cannot supply that many free electrons"
        )
        raise ValueError(msg)

    charge_lower = math.floor(x_e)
    frac_upper = x_e - charge_lower  # fraction of nuclei carrying one more charge than charge_lower

    # ion stage 1 is neutral, so a charge of k is ion stage k + 1
    ionpopdict: dict[tuple[int, int], float] = {(atomic_number, charge_lower + 1): nntot * (1.0 - frac_upper)}
    if frac_upper > 0.0:
        ionpopdict[atomic_number, charge_lower + 2] = nntot * frac_upper

    return ionpopdict


def x_e_of_sweep_step(x_e_start: float, atomic_number: int, step: int, stepcount: int) -> float:
    """Return the electron fraction of one step of a -vary x_e sweep.

    The sweep runs in equal steps of log10(x_e) from x_e_start to four decades above it. An x_e above
    atomic_number - 1 gives only the bare nucleus, thus atomic_number - 1 is the upper limit of the sweep.
    """
    x_e_max = min(x_e_start * 1e4, float(atomic_number - 1))
    if x_e_start <= 0.0 or x_e_start >= x_e_max:
        msg = f"x_e {x_e_start} gives no sweep. Give an x_e above 0 and below {atomic_number - 1}."
        raise ValueError(msg)

    return float(x_e_start * (x_e_max / x_e_start) ** (step / (stepcount - 1)))


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser)

    addarg_modelgridindex(parser, default=0)

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
        choices=["artis", *get_elsymbolslist()[1:]],
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

    # pynonthermal solves only the integral form, thus this flag does nothing. A script can still hold it
    parser.add_argument("--differentialform", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--noexcitation", action="store_true", help="Do not include collisional excitation transitions")

    parser.add_argument(
        "--ar1985",
        action="store_true",
        help="Use Arnaud & Rothenflug (1985, A&AS, 60, 425) for Fe ionization cross sections",
    )

    addarg_output(
        parser,
        kind="file",
        defaultname=defaultoutputfile,
        helptext="Path/filename for PDF file if --makeplot is enabled",
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
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.plotstats:
        # this plot reads a stats file that a former run wrote, thus it calls no solver
        make_ntstats_plot(args.plotstats)
        return

    if args.differentialform:
        print_warning("--differentialform has no effect. The solver gives only the integral form")

    if args.vary == "x_e":
        if args.composition == "artis":
            exit_with_error("-vary x_e needs an element", "Give an element with -composition, e.g. -composition Fe")
        x_e_limit = get_atomic_number(args.composition) - 1
        if args.x_e <= 0.0 or args.x_e >= x_e_limit:
            exit_with_error(f"-x_e {args.x_e} gives no sweep", f"Give -x_e above 0 and below {x_e_limit}")

    # the import stands in front of the work, thus a missing module stops the command at once
    pynt = import_optional("pynonthermal")

    modelpath = Path(args.modelpath)

    ionpopdict: dict[tuple[int, int] | int, float]
    if args.composition == "artis":
        if args.timedays:
            args.timestep = get_timestep_of_timedays(modelpath, args.timedays)
        else:
            args.timestep = get_single_timestep(args.timestep, modelpath)
            if args.timestep is None:
                exit_with_error("no time was given", "Give a time or a timestep, e.g. -timedays 250 or -timestep last")

        dfmodel, modelmeta = get_modeldata(modelpath)
        # vel_r_mid is the mid-point radial velocity of a cell in a model of any dimension, in cm/s
        modeldata = add_derived_cols_to_modeldata(dfmodel, modelmeta).select("vel_r_mid").collect()
        if args.velocity >= 0.0:
            args.modelgridindex = get_mgi_of_velocity_kms(modelpath, args.velocity)
        else:
            args.modelgridindex = get_single_modelgridindex(args.modelgridindex)
        assert isinstance(args.modelgridindex, int)
        estimators = read_estimators(modelpath, timestep=args.timestep, modelgridindex=args.modelgridindex)
        assert isinstance(args.timestep, int)
        assert isinstance(args.modelgridindex, int)
        estim = estimators[args.timestep, args.modelgridindex]

        if read_nltepops(modelpath, modelgridindex=args.modelgridindex, timestep=args.timestep).is_empty():
            exit_with_error(f"no NLTE populations for cell {args.modelgridindex} at timestep {args.timestep}")

        nntot = estim["nntot"]
        x_e = estim["nne"] / nntot
        T_e = estim["Te"]
        print_warning("Use LTE pops at Te for now")
        deposition_density_ev = estim["heating_dep"] / EV_to_erg
        ionpopdict = {get_ion_tuple(k): v for k, v in estim.items() if k.startswith(("nnion_", "nnelement_"))}

        velocity_kmps = modeldata["vel_r_mid"][args.modelgridindex] / km_to_cm
        args.timedays = get_timestep_time(modelpath, args.timestep)
        print(
            f"timestep {args.timestep} cell {args.modelgridindex} (v={velocity_kmps:.1f} km/s at {args.timedays:.1f}d)"
        )

    stepcount = 9 if args.vary else 1
    ostatrows: list[dict[str, float]] = []
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
        if args.composition != "artis":
            compelement = args.composition
            compelement_atomicnumber = get_atomic_number(compelement)
            deposition_density_ev = 5.0e3
            nntot = 1.0
            x_e = (
                x_e_of_sweep_step(args.x_e, compelement_atomicnumber, step, stepcount)
                if args.vary == "x_e"
                else args.x_e
            )
            ionpopdict = {}
            T_e = 3000
            ionpopdict |= ionpops_for_electronfraction(compelement_atomicnumber, x_e, nntot)

        # keep only the ion populations, not element or total populations
        ions = [key for key in ionpopdict if isinstance(key, tuple) and ionpopdict[key] / nntot >= minionfraction]
        ions.sort()

        if args.noexcitation:
            adata = None
        else:
            # the excitation cross sections read epsilon_trans_ev, lower_g, and upper_g from each transition
            adata = get_levels(
                modelpath,
                get_transitions=True,
                ionlist=tuple(ions),
                derived_transitions_columns=("epsilon_trans_ev", "lower_g", "upper_g"),
            )

        with pynt.SpencerFanoSolver(emin_ev=emin, emax_ev=emax, npts=npts, verbose=True, use_ar1985=args.ar1985) as sf:
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
                if args.timestep is not None and args.timedays is not None:
                    outputfilename = str(args.outputfile).format(
                        cell=args.modelgridindex, timestep=args.timestep, timedays=args.timedays
                    )
                else:
                    # a non-ARTIS composition has no cell and no timestep, thus the default template
                    # does not apply, and the element names the file instead
                    outputfilename = str(args.outputfile).replace(
                        defaultoutputfile, f"spencerfano_{args.composition}.pdf"
                    )
                if args.vary:
                    # each step of a sweep writes its own file, because one name would keep the last step only
                    outputpath = Path(outputfilename)
                    outputfilename = str(outputpath.with_name(f"{outputpath.stem}_step{step:02d}{outputpath.suffix}"))
                sf.plot_spec_channels(outputfilename=outputfilename)

            if args.ostat:
                ostatrows.append(
                    {
                        "emin": emin,
                        "emax": emax,
                        "npts": npts,
                        "x_e": x_e,
                        "frac_sum": sf.get_frac_sum(),
                        "frac_excitation": sf.get_frac_excitation_tot(),
                        "frac_ionization": sf.get_frac_ionisation_tot(),
                        "frac_heating": sf.get_frac_heating(),
                    }
                    | {
                        f"frac_ionization_{get_ionstring(atomic_number, ion_stage, sep='')}": (
                            sf.get_frac_ionisation_ion(atomic_number, ion_stage)
                            if ionpopdict[atomic_number, ion_stage] > 0.0
                            else 0.0
                        )
                        for atomic_number, ion_stage in ions
                    }
                )

    if args.ostat:
        write_ntstats_file(args.ostat, ostatrows)
        make_ntstats_plot(args.ostat)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
