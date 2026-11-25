"""Solve the Spencer-Fano equation for a cell's nonthermal electron spectrum and plot the energy deposition."""

import argparse
import math
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from itertools import starmap
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
from artistools.constants import K_B_ev_per_K
from artistools.constants import km_to_cm
from artistools.estimators import get_deposition_channels
from artistools.estimators import read_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import scan_modeldata
from artistools.lines import get_opacity_atomic_data
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import exit_with_error
from artistools.misc import format_frame_path
from artistools.misc import get_artis_options
from artistools.misc import get_selected_timestep
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_timestep_time
from artistools.misc import import_optional
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc import read_rank_outputfiles
from artistools.misc import read_wsv
from artistools.nltepops import read_nltepops
from artistools.plottools import addarg_cellvelocity
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend

if t.TYPE_CHECKING:
    import pynonthermal as pynt

minionfraction = 0.0  # minimum number fraction of the total population to include in SF solution

# the values of ARTIS: SF_EMIN and SF_EMAX in nonthermal.cc, SFPTS in nonthermal.h, and MIN_ION_OVER_NNTOT
ARTIS_SF_GRID = (0.1, 16000.0, 4096)
ARTIS_MIN_ION_OVER_NNTOT = 1e-8

defaultoutputfile = "spencerfano_cell{cell:05d}_ts{timestep:03d}_{timedays:.2f}d.pdf"

# ARTIS normalises the Spencer-Fano solution with the deposition of these particles (nonthermal.cc)
NTLEPTON_DEPOSITION_CHANNELS = ("gamma", "positron", "electron")


def write_ntstats_file(ntstatfile: str | Path, rows: Sequence[dict[str, float]]) -> None:
    """Write one row for each solver step, with a column for each ion of any step.

    A --vary x_e sweep changes the list of ions from one step to the next, thus an ion that a step does not hold
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


def get_ion_column_name(atomic_number: int, ion_stage: int) -> str:
    """Return the name of an ion in a column of the statistics file, e.g. FeII."""
    return get_ionstring(atomic_number, ion_stage, sep="")


def get_sweep_column(dfstats: pl.DataFrame) -> str:
    """Return the column of the parameter that a --vary sweep changes, or x_e if no parameter changes.

    --vary emax,npts changes two columns, and emax then gives the horizontal axis.
    """
    return next((col for col in ("x_e", "emax", "emin", "npts") if dfstats[col].n_unique() > 1), "x_e")


def make_ntstats_plot(ntstatfile: str | Path, args: argparse.Namespace) -> None:
    """Plot the fractions of the non-thermal energy that go to heating, ionisation, and excitation.

    The horizontal axis shows the parameter that the sweep changes. A sweep of the energy grid keeps x_e, thus an
    axis of x_e put every step at one position.
    """
    fig, axesgrid = make_frame_figure(fullwidth=False)
    ax = axesgrid[0][0]

    # the header line was written as a "#" comment
    dfstats = read_wsv(ntstatfile, comment_prefix="#", header_from_comment=True).fill_null(0)

    # the table is the product of -plotstats, thus --quiet keeps it
    with pl.Config(tbl_cols=-1, tbl_rows=50):
        print_product(args, dfstats)

    sweepcolumn = get_sweep_column(dfstats)
    xarr = np.log10(dfstats[sweepcolumn])
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

    xlabels = {"x_e": r"log x$_e$", "emax": r"log E$_{max}$ [eV]", "emin": r"log E$_{min}$ [eV]", "npts": "log npts"}
    ax.set_ylabel(r"Energy fraction")
    ax.set_xlabel(xlabels[sweepcolumn])
    set_legend(ax, args)
    ax.autoscale(enable=True, axis="both", tight=True)
    outputfilename = Path(ntstatfile).with_suffix(".pdf")
    save_figure(fig, outputfilename, format="pdf", args=args)
    # the usual name of the data file is the name of the statistics file that this plot reads
    output_data(args, {"_plotted": dfstats}, outputfilename)


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
    """Return the electron fraction of one step of a --vary x_e sweep.

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
    addarg_legend(parser)
    addarg_darkmode(parser)
    addarg_show(parser)
    addarg_data_output(parser)
    addarg_modelpath(parser, positional=True, default=Path())

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser)

    # the command solves one cell, thus the help names no range
    addarg_modelgridindex(parser, default=[0], helptext="Model grid cell to solve, e.g. 12")

    addarg_cellvelocity(parser)

    gridhelp = " Default: the value in nonthermalspec_*.out for the cell of an ARTIS model, else the ARTIS value of"
    parser.add_argument(
        "-npts", type=int, default=None, help=f"The number of points in the energy grid.{gridhelp} 4096"
    )

    parser.add_argument(
        "-emin", type=float, default=None, help=f"The minimum energy [eV] of the Spencer-Fano solution.{gridhelp} 0.1"
    )

    parser.add_argument(
        "-emax",
        type=float,
        default=None,
        help=(
            "The maximum energy [eV] of the Spencer-Fano solution, which is approximately the injection energy."
            f"{gridhelp} 16000"
        ),
    )

    # the help lists the choices, because the comma of emax,npts made the default list of choices unclear
    parser.add_argument(
        "-vary",
        action="store",
        choices=["emin", "emax", "npts", "emax,npts", "x_e"],
        metavar="PARAMETER",
        help="Which parameter to vary: emin, emax, npts, x_e, or emax,npts for emax and npts together",
    )

    parser.add_argument(
        "-composition",
        action="store",
        default="artis",
        choices=["artis", *get_elsymbolslist()[1:]],
        help="The composition: artis for the ARTIS cell, or the symbol of one element, e.g. Fe",
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


class ArtisCellData(t.NamedTuple):
    """The data of an ARTIS cell that the solver needs to repeat the Spencer-Fano solution of ARTIS."""

    topionstages: dict[int, int]
    """The ion stage of the top ion of each element, which ARTIS does not ionise."""

    dfnltepops: pl.DataFrame | None
    """The rows of nlte_*.out for the cell and the timestep, or None if the model has no such file."""

    T_exc: float
    """The excitation temperature of a level with a Boltzmann population [K]."""

    maxnlevelslower: int
    """The value of NTEXCITATION_MAXNLEVELS_LOWER. An excitation has a lower level below this number."""

    maxnlevelsupper: int
    """The value of NTEXCITATION_MAXNLEVELS_UPPER. An excitation has an upper level below this number."""

    use_km1993_auger: bool
    """True if ARTIS adds the Auger electrons of an inner-shell ionisation to the source term.

    get_artis_auger_treatment gives the source of the Auger data.
    """

    max_auger_electrons: int
    """The value of NT_MAX_AUGER_ELECTRONS, which is the largest number of Auger electrons of one ionisation."""


class PlasmaConditions(t.NamedTuple):
    """The conditions of one Spencer-Fano solution."""

    nntot: float
    nne: float
    T_e: float
    deposition_density_ev: float
    ionpopdict: dict[tuple[int, int] | int, float]
    artiscell: ArtisCellData | None


def get_ntlepton_deposition_rate_density(estim: Mapping[str, t.Any]) -> float | None:
    """Return the deposition rate in erg/s/cm3 that ARTIS uses for the Spencer-Fano solution of a cell.

    A model from a newer ARTIS version gives the deposition of each particle, and ARTIS takes the sum of
    NTLEPTON_DEPOSITION_CHANNELS. A model from an older version gives total_dep, which is the rate that ARTIS used for
    it. heating_dep holds only the heating part of the rate, and a newer version adds the alpha and the fission
    deposition to it. Return None if the estimators give
    no deposition.
    """
    if get_deposition_channels(estim):
        return sum(float(estim.get(f"deposition_{channel}", 0.0)) for channel in NTLEPTON_DEPOSITION_CHANNELS)

    return estim.get("total_dep")


def get_artis_conditions(args: argparse.Namespace, modelpath: Path) -> PlasmaConditions:
    """Return the conditions of the ARTIS cell and timestep that the arguments name.

    The function sets args.timestep, args.modelgridindex, and args.timedays to the resolved values, because
    the name of the plot file holds them.
    """
    args.timestep = get_selected_timestep(modelpath, args.timestep, args.timedays)

    dfmodel, modelmeta = scan_modeldata(modelpath)
    # vel_r_mid is the mid-point radial velocity of a cell in a model of any dimension, in cm/s
    modeldata = add_derived_cols_to_modeldata(dfmodel, modelmeta).select("vel_r_mid").collect()
    if args.velocity_kmps is not None:
        args.modelgridindex = get_mgi_of_velocity_kms(modelpath, args.velocity_kmps)
    else:
        args.modelgridindex = get_single_modelgridindex(args.modelgridindex)
    assert isinstance(args.modelgridindex, int)
    estimators = read_estimators(modelpath, timestep=args.timestep, modelgridindex=args.modelgridindex)
    assert isinstance(args.timestep, int)
    if (args.timestep, args.modelgridindex) not in estimators:
        exit_with_error(
            f"The estimators hold no row for cell {args.modelgridindex} at timestep {args.timestep}",
            "ARTIS writes no estimators for an empty cell. Give a different cell or timestep",
        )
    estim = estimators[args.timestep, args.modelgridindex]

    deposition_erg = get_ntlepton_deposition_rate_density(estim)
    if deposition_erg is None:
        exit_with_error(f"The estimators of cell {args.modelgridindex} at timestep {args.timestep} hold no deposition")
    if not 0.0 < deposition_erg < math.inf:
        exit_with_error(
            f"Cell {args.modelgridindex} has a deposition rate of {deposition_erg} erg/s/cm3 at timestep"
            f" {args.timestep}",
            "ARTIS solves the Spencer-Fano equation only for a positive rate. Give a later time or a different cell",
        )

    nntot = estim["nntot"]

    velocity_kmps = modeldata["vel_r_mid"][args.modelgridindex] / km_to_cm
    args.timedays = get_timestep_time(modelpath, args.timestep)
    print(f"Timestep {args.timestep} cell {args.modelgridindex} (v={velocity_kmps:.1f} km/s at {args.timedays:.1f}d)")

    nne = estim["nne"]
    print(
        f"  The free electron density is the nne of the ARTIS estimators, {nne:.4e} cm^-3. ARTIS takes this value for"
        " its Spencer-Fano solution of the cell (nonthermal.cc)"
    )

    ionpopdict: dict[tuple[int, int] | int, float] = {
        get_ion_tuple(k): v for k, v in estim.items() if k.startswith(("nnion_", "nnelement_"))
    }
    return PlasmaConditions(
        nntot=nntot,
        nne=nne,
        T_e=estim["Te"],
        deposition_density_ev=deposition_erg / EV_to_erg,
        ionpopdict=ionpopdict,
        artiscell=get_artis_cell_data(modelpath, args.timestep, args.modelgridindex, estim, ionpopdict),
    )


def get_artis_cell_data(
    modelpath: Path,
    timestep: int,
    modelgridindex: int,
    estim: Mapping[str, t.Any],
    ionpopdict: Mapping[tuple[int, int] | int, float],
) -> ArtisCellData:
    """Return the data that the solver needs to repeat the Spencer-Fano solution of ARTIS for a cell.

    The estimators list each ion of the run, thus the last ion of each element is the top ion of the run.
    compositiondata.txt can hold a different list from the run.
    """
    topionstages: dict[int, int] = {}
    for key in ionpopdict:
        if isinstance(key, tuple):
            topionstages[key[0]] = max(key[1], topionstages.get(key[0], key[1]))
    print(
        "  ARTIS does not ionise the top ion of each element (nonthermal.cc): "
        + ", ".join(starmap(get_ionstring, sorted(topionstages.items())))
    )

    try:
        dfnltepops: pl.DataFrame | None = read_nltepops(modelpath, timestep=timestep, modelgridindex=modelgridindex)
    except FileNotFoundError:
        dfnltepops = None
    if dfnltepops is not None and dfnltepops.is_empty():
        dfnltepops = None

    artisoptions = get_artis_options(modelpath)
    usetj = artisoptions.get("LTEPOP_EXCITATION_USE_TJ")
    T_exc = float(estim["TJ"]) if usetj == "true" else float(estim["Te"])
    print(
        f"  A level with a Boltzmann population has the excitation temperature {T_exc:.1f} K, which is "
        + ("TJ" if usetj == "true" else "Te")
        + (
            " (LTEPOP_EXCITATION_USE_TJ in artisoptions.h)"
            if usetj is not None
            else ", because the model has no artisoptions.h with LTEPOP_EXCITATION_USE_TJ"
        )
    )
    if dfnltepops is None:
        print_warning(
            f"The model has no nlte_*.out row for cell {modelgridindex} at timestep {timestep}, thus the"
            " excitation takes the LTE populations at the excitation temperature"
        )
    else:
        print(
            "  The excitation takes the level populations of nlte_*.out (ltepop.cc). An ion with no row there has LTE"
            " populations at the excitation temperature"
        )

    maxnlevelsoptions = (
        artisoptions.get("NTEXCITATION_MAXNLEVELS_LOWER"),
        artisoptions.get("NTEXCITATION_MAXNLEVELS_UPPER"),
    )
    maxnlevelslower = int(maxnlevelsoptions[0] or "5")
    maxnlevelsupper = int(maxnlevelsoptions[1] or "250")
    maxnlevelssource = (
        "NTEXCITATION_MAXNLEVELS_LOWER and NTEXCITATION_MAXNLEVELS_UPPER in artisoptions.h"
        if None not in maxnlevelsoptions
        else "the ARTIS defaults, because artisoptions.h does not define NTEXCITATION_MAXNLEVELS_LOWER and"
        " NTEXCITATION_MAXNLEVELS_UPPER"
    )
    print(
        f"  An excitation has a lower level below {maxnlevelslower} and an upper level below {maxnlevelsupper}"
        f" ({maxnlevelssource})"
    )

    use_km1993_auger, max_auger_electrons = get_artis_auger_treatment(artisoptions)
    return ArtisCellData(
        topionstages=topionstages,
        dfnltepops=dfnltepops,
        T_exc=T_exc,
        maxnlevelslower=maxnlevelslower,
        maxnlevelsupper=maxnlevelsupper,
        use_km1993_auger=use_km1993_auger,
        max_auger_electrons=max_auger_electrons,
    )


def get_artis_auger_treatment(artisoptions: Mapping[str, str]) -> tuple[bool, int]:
    """Return use_km1993_auger and max_auger_electrons of SpencerFanoSolver that agree with artisoptions.h.

    ARTIS reads the Auger data only if NT_MAX_AUGER_ELECTRONS is above zero (nonthermal.cc). It adds the Auger
    electrons to the source term only if SF_AUGER_CONTRIBUTION_ON is also true. Without that source, the ionisation
    keeps the full shell potential, which is the result of the solver without the Auger data. The data are table 2 of
    Kaastra & Mewe (1993, A&AS, 97, 443-482, ADS 1993A&AS...97..443K).
    """
    # an older artisoptions.h has no such option, thus the value is the ARTIS default
    optiontexts = {
        name: (
            f"{artisoptions[name]} (artisoptions.h)"
            if name in artisoptions
            else f"{default} (the ARTIS default, because artisoptions.h does not define it)"
        )
        for name, default in (("NT_MAX_AUGER_ELECTRONS", "2"), ("SF_AUGER_CONTRIBUTION_ON", "true"))
    }
    max_auger_electrons = int(artisoptions.get("NT_MAX_AUGER_ELECTRONS", "2"))
    use_km1993_auger = max_auger_electrons > 0 and artisoptions.get("SF_AUGER_CONTRIBUTION_ON") != "false"
    print(
        f"  NT_MAX_AUGER_ELECTRONS is {optiontexts['NT_MAX_AUGER_ELECTRONS']} and SF_AUGER_CONTRIBUTION_ON is"
        f" {optiontexts['SF_AUGER_CONTRIBUTION_ON']}"
    )
    if use_km1993_auger:
        print(
            "  An ionisation of an inner subshell adds one electron with the mean Auger energy to the source term, as"
            " in ARTIS (Shingles et al. 2020, MNRAS, 492, 2029-2043, doi:10.1093/mnras/stz3412). The Auger data are"
            " table 2 of Kaastra & Mewe (1993, A&AS, 97, 443-482, ADS 1993A&AS...97..443K). ARTIS versions before"
            " artis-mcrt/artis#813 give the Auger electron the energy of the next grid point above it"
        )
    else:
        print("  The solver adds no Auger electrons, because ARTIS with these options adds none")
    # artis-mcrt/artis#385 moved this option from artisoptions.h to nonthermal.cc, where it is false
    if use_km1993_auger and artisoptions.get("SF_AUGER_CONTRIBUTION_DISTRIBUTE_EN") == "true":
        print_warning(
            "artisoptions.h sets SF_AUGER_CONTRIBUTION_DISTRIBUTE_EN to true. ARTIS then gives the Auger energy only to"
            " the ionisations with one or more Auger electrons. The solver gives the mean Auger energy to each"
            " ionisation, thus its result differs from the run"
        )
    return use_km1993_auger, max_auger_electrons


def get_element_conditions(args: argparse.Namespace, step: int, stepcount: int) -> PlasmaConditions:
    """Return the conditions of one element with the electron fraction of one step of a --vary x_e sweep."""
    compelement_atomicnumber = get_atomic_number(args.composition)
    nntot = 1.0
    x_e = x_e_of_sweep_step(args.x_e, compelement_atomicnumber, step, stepcount) if args.vary == "x_e" else args.x_e
    ionpopdict: dict[tuple[int, int] | int, float] = dict(
        ionpops_for_electronfraction(compelement_atomicnumber, x_e, nntot)
    )

    print(f"  The free electron density is x_e n_tot, with x_e = {x_e:.4g}")

    return PlasmaConditions(
        nntot=nntot, nne=x_e * nntot, T_e=3000, deposition_density_ev=5.0e3, ionpopdict=ionpopdict, artiscell=None
    )


def get_sweep_parameters(args: argparse.Namespace, step: int) -> tuple[float, float, int]:
    """Return the emin, the emax, and the npts of the energy grid at one step of a --vary sweep."""
    # --vary emax,npts doubles the two values together
    varied = str(args.vary).split(",")
    emin = args.emin * 2**step if "emin" in varied else args.emin
    emax = args.emax * 2**step if "emax" in varied else args.emax
    npts = args.npts * 2**step if "npts" in varied else args.npts
    return emin, emax, npts


def get_plot_filename(args: argparse.Namespace, step: int) -> str:
    """Return the name of the plot file of one step."""
    if args.timestep is not None and args.timedays is not None:
        outputfilename = format_frame_path(
            args.outputfile, cell=args.modelgridindex, timestep=args.timestep, timedays=args.timedays
        )
    else:
        # a non-ARTIS composition has no cell and no timestep, thus the default template
        # does not apply, and the element names the file instead
        outputfilename = str(args.outputfile).replace(defaultoutputfile, f"spencerfano_{args.composition}.pdf")
    if args.vary:
        # each step of a sweep writes its own file, because one name would keep the last step only
        outputpath = Path(outputfilename)
        outputfilename = str(outputpath.with_name(f"{outputpath.stem}_step{step:02d}{outputpath.suffix}"))

    return outputfilename


def add_ions_to_solver(
    sf: "pynt.SpencerFanoSolver",
    args: argparse.Namespace,
    modelpath: Path,
    conditions: PlasmaConditions,
    ions: Sequence[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Add the ionisation and the excitation of each ion to the solver, and return the ions with an ionisation.

    For an ARTIS cell, the solver takes the atomic data, the level populations, and the ions as ARTIS does.
    """
    artiscell = conditions.artiscell
    if args.noexcitation:
        adata = None
    elif artiscell is not None:
        adata = get_artis_excitation_atomic_data(modelpath, ions)
    else:
        # the excitation cross sections read epsilon_trans_ev, lower_g, and upper_g from each transition
        adata = get_levels(
            modelpath,
            get_transitions=True,
            ionlist=tuple(ions),
            derived_transitions_columns=("epsilon_trans_ev", "lower_g", "upper_g"),
        )

    if adata is not None:
        if artiscell is not None:
            sf.set_temperature(artiscell.T_exc)
            sf.set_atomic_data(
                adata_polars=adata, maxnlevelslower=artiscell.maxnlevelslower, maxnlevelsupper=artiscell.maxnlevelsupper
            )
        else:
            # the excitation takes the LTE populations of the levels at T_e, and the level data of the model
            sf.set_temperature(conditions.T_e)
            sf.set_atomic_data(adata_polars=adata)

    ionisedions: set[tuple[int, int]] = set()
    for Z, ion_stage in ions:
        nnion = conditions.ionpopdict[Z, ion_stage]
        if nnion == 0.0:
            print(f"   Z={Z} ion_stage {ion_stage} has nnion={nnion:.1e}, thus the solver skips it")
            continue
        if ion_stage > Z:
            # a bare nucleus has no bound electron to ionise or to excite, and ARTIS skips it (nonthermal.cc)
            print(f"   Z={Z} ion_stage {ion_stage} is a bare nucleus with no bound electron, thus the solver skips it")
            continue

        if artiscell is None or artiscell.topionstages.get(Z) != ion_stage:
            sf.add_ionisation(Z, ion_stage, nnion)
            ionisedions.add((Z, ion_stage))

        if adata is not None:
            ionrow = adata.filter(pl.col("Z") == Z, pl.col("ion_stage") == ion_stage)
            levelpops = (
                get_artis_lower_level_pops(artiscell, ionrow["levels"].item(), Z, ion_stage)
                if artiscell is not None and not ionrow.is_empty()
                else None
            )
            if levelpops is None:
                sf.add_ion_excitation(Z, ion_stage, nnion)
            else:
                assert artiscell is not None
                add_ion_excitation_with_level_pops(
                    sf, ionrow["transitions"].item(), Z, ion_stage, levelpops, artiscell.maxnlevelsupper
                )

    return ionisedions


def get_artis_excitation_atomic_data(modelpath: Path, ions: Sequence[tuple[int, int]]) -> pl.DataFrame:
    """Return the levels and the transitions of each ion as ARTIS holds them, with the columns of the excitation.

    ARTIS keeps some of the levels, and it merges the transitions of one pair of levels (input.cc). A transition to a
    level that ARTIS does not keep has no excitation.
    """
    adata = get_opacity_atomic_data(
        modelpath, tuple(ions), derived_transitions_columns=("epsilon_trans_ev", "lower_g", "upper_g")
    )
    kepttransitions = [
        dftransitions.lazy().filter(pl.col("upper") < dflevels.height)
        for dflevels, dftransitions in zip(adata["levels"], adata["transitions"], strict=True)
    ]
    return adata.with_columns(pl.Series("transitions", kepttransitions, dtype=pl.Object))


def get_artis_lower_level_pops(
    artiscell: ArtisCellData, dflevels: pl.DataFrame, Z: int, ion_stage: int
) -> dict[int, float] | None:
    """Return the population [cm^-3] of each lower level of the excitations of an ion, as ARTIS calculates it.

    calculate_levelpop_nominpop in ltepop.cc gives a level of nlte_*.out its NLTE population. A level in the
    superlevel gets the part of the superlevel population that its Boltzmann factor at T_exc gives
    (superlevel_boltzmann in nltepop.cc). A different level gets the Boltzmann population relative to the ground level.
    Return None for an ion with no row in nlte_*.out. ARTIS then gives the ion LTE populations at T_exc.
    """
    if artiscell.dfnltepops is None:
        return None
    dfion = artiscell.dfnltepops.filter(pl.col("Z") == Z, pl.col("ion_stage") == ion_stage)
    if dfion.is_empty():
        return None

    nltepops = dict(zip(dfion["level"].to_list(), dfion["n_NLTE"].to_list(), strict=True))
    # nlte_*.out gives the superlevel the level number -1, after the NLTE levels
    lastnltelevel = max(level for level in nltepops if level >= 0)
    energies_ev = dflevels["energy_ev"].to_numpy()
    statweights = dflevels["g"].to_numpy()

    def get_boltzmann_factor(level: int, reflevel: int) -> float:
        return float(
            statweights[level]
            / statweights[reflevel]
            * np.exp(-(energies_ev[level] - energies_ev[reflevel]) / K_B_ev_per_K / artiscell.T_exc)
        )

    superlevelstart = lastnltelevel + 1
    superlevelpartfunc = sum(
        get_boltzmann_factor(level, superlevelstart) for level in range(superlevelstart, len(energies_ev))
    )
    levelpops: dict[int, float] = {}
    for level in range(min(artiscell.maxnlevelslower, len(energies_ev))):
        if level in nltepops:
            levelpops[level] = nltepops[level]
        elif -1 in nltepops and superlevelpartfunc > 0.0:
            levelpops[level] = nltepops[-1] * get_boltzmann_factor(level, superlevelstart) / superlevelpartfunc
        else:
            levelpops[level] = nltepops[0] * get_boltzmann_factor(level, 0)
    return levelpops


def add_ion_excitation_with_level_pops(
    sf: "pynt.SpencerFanoSolver",
    dftransitions: pl.LazyFrame,
    Z: int,
    ion_stage: int,
    levelpops: Mapping[int, float],
    maxnlevelsupper: int,
) -> None:
    """Add the excitations of an ion with the given populations of the lower levels, as ARTIS does (nonthermal.cc).

    The transitions and the cross sections are those of add_ion_excitation in pynonthermal, which agree with
    has_xs_excitation and xs_excitation of ARTIS.
    """
    pynt = import_optional("pynonthermal")
    dfexcitations = (
        dftransitions
        .filter(
            pl.col("lower").is_in(list(levelpops)),
            pl.col("upper") < maxnlevelsupper,
            (pl.col("collstr") >= 0) | (pl.col("forbidden") == 0),
            pl.col("epsilon_trans_ev").is_between(float(sf.engrid[0]), float(sf.engrid[-1])),
        )
        .sort("lower", "upper", maintain_order=True)
        .collect()
    )
    for transition in dfexcitations.iter_rows(named=True):
        sf.add_excitation(
            Z,
            ion_stage,
            levelpops[transition["lower"]],
            partial(pynt.excitation.get_xs_excitation_vector, row=transition),
            float(transition["epsilon_trans_ev"]),
            transitionkey=(transition["lower"], transition["upper"]),
        )


def get_run_energy_grid(modelpath: Path, timestep: int, modelgridindex: int) -> tuple[float, float, int] | None:
    """Return the emin, the emax, and the npts of the energy grid in nonthermalspec_*.out, or None with no such file.

    An older ARTIS wrote the solution of each cell in nonthermalspec_*.out. Its grid can differ from the grid of the
    current ARTIS.
    """
    try:
        dfspec = read_rank_outputfiles(
            modelpath, "nonthermalspec_{mpirank:04d}.out", timestep=timestep, modelgridindex=modelgridindex
        )
    except FileNotFoundError:
        return None
    if dfspec.is_empty():
        return None
    energies_ev = dfspec["energy_ev"].to_numpy()
    return float(energies_ev.min()), float(energies_ev.max()), len(energies_ev)


def resolve_energy_grid(args: argparse.Namespace, modelpath: Path) -> None:
    """Set each value of the energy grid that the user does not give, and log the source of the grid."""
    if None not in {args.emin, args.emax, args.npts}:
        return
    rungrid = (
        get_run_energy_grid(modelpath, args.timestep, args.modelgridindex) if args.composition == "artis" else None
    )
    defaultgrid = rungrid or ARTIS_SF_GRID
    args.emin = defaultgrid[0] if args.emin is None else args.emin
    args.emax = defaultgrid[1] if args.emax is None else args.emax
    args.npts = defaultgrid[2] if args.npts is None else args.npts
    gridsource = (
        "the grid of nonthermalspec_*.out of the run"
        if rungrid is not None
        else "the grid of ARTIS (SF_EMIN and SF_EMAX in nonthermal.cc, SFPTS in nonthermal.h)"
    )
    print(
        f"  The energy grid has {args.npts} points from {args.emin} to {args.emax} eV. A value that the command does"
        f" not give comes from {gridsource}"
    )


def solve_step(
    args: argparse.Namespace, modelpath: Path, conditions: PlasmaConditions, step: int
) -> dict[str, float] | None:
    """Solve the Spencer-Fano equation at one step of a sweep, and plot the solution.

    Return the statistics of the step for --ostat, or None without --ostat.
    """
    pynt = import_optional("pynonthermal")
    emin, emax, npts = get_sweep_parameters(args, step)
    ionpopdict = conditions.ionpopdict

    # ARTIS does not include an ion with a population below MIN_ION_OVER_NNTOT times nntot (nonthermal.cc)
    ionfractionlimit = ARTIS_MIN_ION_OVER_NNTOT if conditions.artiscell is not None else minionfraction
    # keep only the populations of the ions, and not the populations of the elements or the total population
    ions = sorted(
        key for key in ionpopdict if isinstance(key, tuple) and ionpopdict[key] / conditions.nntot >= ionfractionlimit
    )

    # an element composition has no ARTIS options, thus it has no Auger electrons, which is the default of the solver
    artiscell = conditions.artiscell
    with pynt.SpencerFanoSolver(
        emin_ev=emin,
        emax_ev=emax,
        npts=npts,
        verbose=True,
        use_ar1985=args.ar1985,
        use_km1993_auger=artiscell is not None and artiscell.use_km1993_auger,
        max_auger_electrons=artiscell.max_auger_electrons if artiscell is not None else 0,
    ) as sf:
        ionisedions = add_ions_to_solver(sf, args, modelpath, conditions, ions)

        # the solver otherwise sums n_e from the charges of the ions that it holds, which leaves out a skipped ion
        sf.override_n_e(conditions.nne)

        sf.solve(deposition_ev_per_s_per_cm3=conditions.deposition_density_ev)

        sf.analyse_ntspectrum()
        x_e = conditions.nne / conditions.nntot

        if args.makeplot:
            sf.plot_spec_channels(outputfilename=get_plot_filename(args, step))

        # the energy fractions are the product of the command, thus --quiet keeps them. The solver prints its
        # analysis as progress messages
        print_product(
            args,
            f"emin {emin} emax {emax} npts {npts} x_e {x_e:.4g}:"
            f" frac_heating {sf.get_frac_heating():.4f} frac_ionization {sf.get_frac_ionisation_tot():.4f}"
            f" frac_excitation {sf.get_frac_excitation_tot():.4f} frac_sum {sf.get_frac_sum():.4f}",
        )

        if not args.ostat:
            return None

        return {
            "emin": emin,
            "emax": emax,
            "npts": npts,
            "x_e": x_e,
            "frac_sum": sf.get_frac_sum(),
            "frac_excitation": sf.get_frac_excitation_tot(),
            "frac_ionization": sf.get_frac_ionisation_tot(),
            "frac_heating": sf.get_frac_heating(),
        } | {
            f"frac_ionization_{get_ion_column_name(atomic_number, ion_stage)}": (
                sf.get_frac_ionisation_ion(atomic_number, ion_stage)
                if (atomic_number, ion_stage) in ionisedions
                else 0.0
            )
            for atomic_number, ion_stage in ions
        }


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Solve Spencer-Fano equation using data from ARTIS cell at some timestep."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.plotstats:
        # this plot reads a stats file that an earlier command with -ostat wrote, thus it calls no solver
        make_ntstats_plot(args.plotstats, args)
        return

    if args.differentialform:
        print_warning("--differentialform has no effect. The solver gives only the integral form")

    if args.vary == "x_e":
        if args.composition == "artis":
            exit_with_error("--vary x_e needs an element", "Give an element with --composition, e.g. --composition Fe")
        x_e_limit = get_atomic_number(args.composition) - 1
        if args.x_e <= 0.0 or args.x_e >= x_e_limit:
            exit_with_error(f"--x-e {args.x_e} gives no sweep", f"Give --x-e above 0 and below {x_e_limit}")

    # the import stands in front of the work, thus a missing module stops the command at once
    import_optional("pynonthermal")

    modelpath = Path(args.modelpath)
    artisconditions = get_artis_conditions(args, modelpath) if args.composition == "artis" else None
    resolve_energy_grid(args, modelpath)

    stepcount = 9 if args.vary else 1
    ostatrows: list[dict[str, float]] = []
    for step in range(stepcount):
        conditions = artisconditions if artisconditions is not None else get_element_conditions(args, step, stepcount)
        if (ostatrow := solve_step(args, modelpath, conditions, step)) is not None:
            ostatrows.append(ostatrow)

    if args.ostat:
        write_ntstats_file(args.ostat, ostatrows)
        make_ntstats_plot(args.ostat, args)
