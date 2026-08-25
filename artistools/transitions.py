"""Plot synthetic emission spectra of individual ions from Kurucz, NIST, or ARTIS transition data."""

import argparse
import dataclasses as dc
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import hc_in_ev_cm
from artistools.constants import K_B_ev_per_K
from artistools.constants import km_to_cm
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_outputfile
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.plottools import save_figure
from artistools.plottools import set_plot_title

if t.TYPE_CHECKING:
    from collections.abc import Iterable

defaultoutputfile = "plottransitions_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf"


class IonTuple(t.NamedTuple):
    """An ion, identified by its atomic number and ion stage."""

    Z: int
    ion_stage: int


def get_kurucz_transitions() -> tuple[pl.DataFrame, list[IonTuple]]:
    """Return the transitions from the bundled Kurucz gfall line list, and the ions they cover."""

    class KuruczTransitionTuple(t.NamedTuple):
        Z: int
        ion_stage: int
        lambda_angstroms: float
        A: float
        lower_energy_ev: float
        upper_energy_ev: float
        lower_statweight: float
        upper_statweight: float

    translist = []
    ionlist: list[IonTuple] = []
    with Path("gfall.dat").open(encoding="utf-8") as fnist:
        for line in fnist:
            row = line.split()
            if len(row) >= 24:
                Z, ion_stage = int(row[2].split(".")[0]), int(row[2].split(".")[1]) + 1
                if Z < 44 or ion_stage >= 2:  # and Z not in [26, 27]
                    continue
                # gfall.dat is fixed-width: wavelength in nm is F11.4 (columns 0-10) and loggf is F7.3 (columns 11-17)
                lambda_angstroms = float(line[:11]) * 10
                loggf = float(line[11:18])
                lower_energy_ev, upper_energy_ev = hc_in_ev_cm * float(line[24:36]), hc_in_ev_cm * float(line[52:64])
                lower_statweight, upper_statweight = 2 * float(line[36:42]) + 1, 2 * float(line[64:70]) + 1
                fij = (10**loggf) / lower_statweight
                A = fij / (1.49919e-16 * upper_statweight / lower_statweight * lambda_angstroms**2)
                translist.append(
                    KuruczTransitionTuple(
                        Z,
                        ion_stage,
                        lambda_angstroms,
                        A,
                        lower_energy_ev,
                        upper_energy_ev,
                        lower_statweight,
                        upper_statweight,
                    )
                )

                if IonTuple(Z, ion_stage) not in ionlist:
                    ionlist.append(IonTuple(Z, ion_stage))

    dftransitions = pl.DataFrame(translist, orient="row", schema=list(KuruczTransitionTuple._fields))
    return dftransitions, ionlist


def get_nist_transitions(filename: Path | str) -> pl.DataFrame:
    """Return the transitions read from a NIST Atomic Spectra Database line list export."""

    class NISTTransitionTuple(t.NamedTuple):
        lambda_angstroms: float
        A: float
        lower_energy_ev: float
        upper_energy_ev: float
        lower_statweight: float
        upper_statweight: float

    translist = []
    with Path(filename).open(encoding="utf-8") as fnist:
        for line in fnist:
            row = line.split("|")
            if len(row) == 17 and "-" in row[5]:
                if row[0].strip():
                    lambda_angstroms = float(row[0])
                elif row[1].strip():
                    lambda_angstroms = float(row[1])
                else:
                    continue
                A = float(row[3]) if row[3].strip() else 1e8
                lower_energy_ev, upper_energy_ev = (float(x.strip(" []")) for x in row[5].split("-"))
                lower_statweight, upper_statweight = (float(x.strip()) for x in row[12].split("-"))
                translist.append(
                    NISTTransitionTuple(
                        lambda_angstroms, A, lower_energy_ev, upper_energy_ev, lower_statweight, upper_statweight
                    )
                )

    return pl.DataFrame(translist, orient="row", schema=list(NISTTransitionTuple._fields))


def generate_ion_spectrum(
    transitions: pl.DataFrame,
    xvalues: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    popcolumn: str,
    plot_resolution: float,
    args: argparse.Namespace,
) -> npt.NDArray[np.floating]:
    """Return the emission spectrum of one ion, summing a Gaussian profile for each line.

    Each line is accumulated over its own window of grid points, so the work stays proportional to the total
    window width rather than to the number of lines times the whole grid.
    """
    npoints = len(xvalues)
    yvalues = np.zeros(npoints)
    if transitions.is_empty():
        return yvalues

    lambda_angstroms = transitions["lambda_angstroms"].cast(pl.Float64).to_numpy()
    flux = (transitions["flux_factor"] * transitions[popcolumn]).cast(pl.Float64).to_numpy()

    centre_index = np.round((lambda_angstroms - args.xmin) / plot_resolution).astype(np.int64)
    sigma_angstroms = lambda_angstroms * args.sigma_v * km_to_cm / C_cm_per_s
    sigma_gridpoints = np.ceil(sigma_angstroms / plot_resolution).astype(np.int64)
    halfwidth = (args.gaussian_window * sigma_gridpoints).astype(np.int64)

    window_left = np.clip(centre_index - halfwidth, 0, npoints)
    window_right = np.clip(centre_index + halfwidth, 0, npoints)

    for lineindex in np.flatnonzero(window_right > window_left):
        left = int(window_left[lineindex])
        right = int(window_right[lineindex])
        offsets = np.arange(left, right) - centre_index[lineindex]
        yvalues[left:right] += (
            flux[lineindex]
            * np.exp(-((offsets * plot_resolution / sigma_angstroms[lineindex]) ** 2))
            / sigma_angstroms[lineindex]
        )

    return yvalues


def make_plot(
    xvalues: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    yvalues: npt.NDArray[np.floating],
    temperature_list: Sequence[str],
    vardict: Mapping[str, float],
    ionlist: Sequence[IonTuple],
    ionpopdict: Mapping[IonTuple, float],
    xmin: float,
    xmax: float,
    figure_title: str,
    outputfilename: str,
    args: argparse.Namespace,
) -> None:
    """Plot one panel per ion plus a combined panel, and save the figure."""
    npanels = len(ionlist)

    fig, axes = plt.subplots(
        nrows=npanels,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 2 * (len(ionlist) + 1)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if len(ionlist) == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    if figure_title:
        print(figure_title)
    set_plot_title(axes[0], figure_title, args)

    yvalues_combined = np.zeros((len(temperature_list), len(xvalues)))
    for seriesindex, temperature in enumerate(temperature_list):
        serieslabel = "NLTE" if temperature == "NOTEMPNLTE" else f"LTE {temperature} = {vardict[temperature]:.0f} K"
        for ion_index, axis in enumerate(axes[: len(ionlist)]):
            # an ion subplot
            yvalues_combined[seriesindex] += yvalues[seriesindex][ion_index]

            axis.plot(xvalues, yvalues[seriesindex][ion_index], linewidth=1.5, label=serieslabel)

            axis.legend(loc="upper left", handlelength=1, frameon=False, numpoints=1, prop={"size": 8})

        if len(axes) > len(ionlist):
            axes[len(ionlist)].plot(xvalues, yvalues_combined[seriesindex], linewidth=1.5, label=serieslabel)

    axislabels = [
        f"{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]}\n(pop={ionpopdict[IonTuple(Z, ion_stage)]:.1e}/cm³)"
        for (Z, ion_stage) in ionlist
    ]
    axislabels += ["Total"]

    for axis, axislabel in zip(axes, axislabels, strict=False):
        axis.annotate(
            axislabel,
            xy=(0.99, 0.96),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=10,
        )

    # at.spectra.plot_reference_spectrum(
    #     'dop_dered_SN2013aa_20140208_fc_final.txt', axes[-1], xmin, xmax, True,
    #     scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')
    #
    # at.spectra.plot_reference_spectrum(
    #     '2003du_20031213_3219_8822_00.txt', axes[-1], xmin, xmax,
    #     scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')

    axes[-1].set_xlabel(r"Wavelength ($\AA$)")

    for axis in axes:
        axis.set_xlim(xmin, xmax)
        axis.set_ylabel(r"$\propto$ F$_\lambda$")

    save_figure(fig, outputfilename, format="pdf")


def get_lte_partfunc(pldflevels: pl.DataFrame, T_exc: float) -> float:
    """Return the LTE partition function of the ion at the excitation temperature."""
    return float(pldflevels.select(pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / T_exc).exp()).sum().item())


def add_upper_lte_pop(
    dftransitions: pl.DataFrame, T_exc: float, ionpop: float, ltepartfunc: float, columnname: str | None = None
) -> pl.DataFrame:
    """Add a column of upper level populations in LTE at T_exc."""
    scalefactor = ionpop / ltepartfunc
    if columnname is None:
        columnname = f"upper_pop_lte_{T_exc:.0f}K"

    return dftransitions.with_columns(
        (scalefactor * pl.col("upper_statweight") * (-pl.col("upper_energy_ev") / K_B_ev_per_K / T_exc).exp()).alias(
            columnname
        )
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=None)

    addarg_notitle(parser)

    addarg_axislimits(
        parser,
        xmindefault=3500,
        xmaxdefault=8000,
        xminhelp="Plot range: minimum wavelength in Angstroms",
        xmaxhelp="Plot range: maximum wavelength in Angstroms",
        include_y=False,
        wavelength_aliases=True,
    )

    parser.add_argument("-T", type=float, dest="T", default=[], nargs="*", help="Temperature in Kelvin")

    parser.add_argument("-sigma_v", type=float, default=5500.0, help="Gaussian width in km/s")

    parser.add_argument(
        "-gaussian_window", type=float, default=3, help="Truncate Gaussian line profiles n sigmas from the centre"
    )

    parser.add_argument("--include-permitted", action="store_true", help="Also consider permitted lines")

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser, kind="int", default=70)

    addarg_modelgridindex(parser, default=0)

    parser.add_argument("--normalised", action="store_true", help="Normalise all spectra to their peak values")

    parser.add_argument("--print-lines", action="store_true", help="Output details of matching lines to standard out")

    parser.add_argument(
        "-atomicdatabase",
        default="artis",
        choices=["artis", "kurucz", "nist"],
        help="Source of atomic data for excitation transitions",
    )
    # deprecated double-dash spelling kept as a hidden alias
    parser.add_argument(
        "--atomicdatabase", dest="atomicdatabase", choices=["artis", "kurucz", "nist"], help=argparse.SUPPRESS
    )

    addarg_outputfile(parser, default=defaultoutputfile, astype=None, helptext="path/filename for PDF file")


@dc.dataclass(frozen=True, slots=True)
class CellConditions:
    """The state of one cell of a model at one timestep."""

    modelgridindex: int
    timestep: int
    time_days: float
    velocity: float
    estimators: Mapping[str, t.Any]


@dc.dataclass(frozen=True, slots=True)
class PlotConditions:
    """The populations, the temperatures, and the title that one plot of transitions needs.

    A temperature name of "NOTEMPNLTE" selects the NLTE populations in place of an LTE calculation.
    """

    ionpopdict: Mapping[IonTuple, float]
    temperature_list: Sequence[str]
    vardict: Mapping[str, float]
    figure_title: str
    dfnltepops: pl.DataFrame | None = None


def get_ionlist() -> list[IonTuple]:
    """Return the ions that the plot shows.

    The commented lines are further ions that a user can select in place of these.
    """
    return [
        IonTuple(26, 1),
        IonTuple(26, 2),
        IonTuple(26, 3),
        IonTuple(27, 2),
        IonTuple(27, 3),
        IonTuple(28, 2),
        IonTuple(28, 3),
        # IonTuple(45, 1),
        # IonTuple(54, 1),
        # IonTuple(54, 2),
        # IonTuple(55, 1),
        # IonTuple(55, 2),
        # IonTuple(58, 1),
        # IonTuple(79, 1),
        # IonTuple(83, 1),
    ]


def get_cell_conditions(modelpath: Path, args: argparse.Namespace) -> CellConditions:
    """Read the time, the velocity, and the estimators of the selected cell and timestep."""
    timestep = at.get_timestep_of_timedays(modelpath, args.timedays) if args.timedays else args.timestep

    modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
    estimators_all = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=args.modelgridindex)
    if not estimators_all:
        at.exit_with_error("no estimators")

    return CellConditions(
        modelgridindex=args.modelgridindex,
        timestep=timestep,
        time_days=at.get_timestep_time(modelpath, timestep),
        velocity=modeldata["vel_r_max_kmps"][args.modelgridindex],
        estimators=estimators_all[timestep, args.modelgridindex],
    )


def get_model_conditions(modelpath: Path, cell: CellConditions, ionlist: Sequence[IonTuple]) -> PlotConditions:
    """Return the NLTE populations and the temperatures of one cell of a model."""
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=cell.modelgridindex, timestep=cell.timestep)

    if dfnltepops.is_empty():
        at.exit_with_error(f"no NLTE populations for cell {cell.modelgridindex} at timestep {cell.timestep}")

    T_e = float(cell.estimators["Te"])
    T_R = float(cell.estimators["TR"])
    figure_title = (
        f"{at.get_model_name(modelpath)}\n"
        f"Cell {cell.modelgridindex} ({cell.velocity} km/s) with Te = {T_e:.1f} K, "
        f"TR = {T_R:.1f} K at timestep {cell.timestep}"
    )
    if cell.time_days != -1:
        figure_title += f" ({cell.time_days:.1f}d)"

    return PlotConditions(
        ionpopdict={
            IonTuple(Z, ion_stage): float(
                dfnltepops.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["n_NLTE"].sum()
            )
            for Z, ion_stage in ionlist
        },
        temperature_list=["NOTEMPNLTE"],
        vardict={"Te": T_e, "TR": T_R},
        figure_title=figure_title,
        dfnltepops=dfnltepops,
    )


def get_fixed_temperature_conditions(args: argparse.Namespace, ionlist: Sequence[IonTuple]) -> PlotConditions:
    """Return one series for each temperature that -T names, with the same population for every ion."""
    if not args.T:
        args.T = [2000]

    vardict = {}
    temperature_list = []
    for index, temperature in enumerate(args.T):
        tlabel = "Te" if index == 0 else f"Te_{index + 1}"
        vardict[tlabel] = float(temperature)
        temperature_list.append(tlabel)

    return PlotConditions(
        ionpopdict={IonTuple(Z, ionstage): 1.0 for Z, ionstage in ionlist},
        temperature_list=temperature_list,
        vardict=vardict,
        figure_title=f"Te = {args.T[0]:.1f}" if len(args.T) == 1 else "",
    )


def get_ion_transitions(
    ion: Mapping[str, t.Any], dftransgfall: pl.DataFrame | None, args: argparse.Namespace
) -> pl.DataFrame:
    """Return the transitions of one ion from the selected atomic database."""
    if args.atomicdatabase == "kurucz":
        assert dftransgfall is not None
        return dftransgfall.filter((pl.col("Z") == ion["Z"]) & (pl.col("ion_stage") == ion["ion_stage"]))

    if args.atomicdatabase == "nist":
        return get_nist_transitions(f"nist/nist-{ion['Z']:02d}-{ion['ion_stage']:02d}.txt")

    pldftransitions = ion["transitions"]
    assert isinstance(pldftransitions, pl.DataFrame | pl.LazyFrame)

    return pldftransitions.lazy().collect()


def add_artis_transition_columns(pldftransitions: pl.DataFrame, pldflevels: pl.DataFrame) -> pl.DataFrame:
    """Add the level energies, the wavelengths, and the statistical weights of an ARTIS transition list."""
    return (
        at.atomic
        .add_transition_columns(
            pldftransitions,
            pldflevels,
            [
                "lower_energy_ev",
                "upper_energy_ev",
                "lambda_angstroms",
                "lower_level",
                "upper_level",
                "lower_g",
                "upper_g",
            ],
        )
        .rename({"lower_g": "lower_statweight", "upper_g": "upper_statweight"})
        .collect()
    )


def get_ion_spectra(
    xvalues: npt.NDArray[np.floating],
    ionlist: Sequence[IonTuple],
    conditions: PlotConditions,
    adata: pl.DataFrame | None,
    dftransgfall: pl.DataFrame | None,
    plot_resolution: int,
    args: argparse.Namespace,
) -> tuple[npt.NDArray[np.floating], dict[IonTuple, float]]:
    """Return the spectrum of each ion at each temperature, and the departure coefficient of two lines.

    The plot marks the departure coefficient of the Fe II 7155 line and of the Ni II 7378 line, thus
    this function gives back both.
    """
    yvalues = np.zeros((len(conditions.temperature_list) + 1, len(ionlist), len(xvalues)))
    depcoeffs: dict[IonTuple, float] = {}

    iterdict: Iterable[Mapping[str, t.Any]] = (
        adata.iter_rows(named=True)
        if adata is not None
        else ({"Z": Z, "ion_stage": ion_stage, "levels": None} for Z, ion_stage in ionlist)
    )
    for ion in iterdict:
        assert isinstance(ion["Z"], int)
        assert isinstance(ion["ion_stage"], int)
        ionid = IonTuple(ion["Z"], ion["ion_stage"])
        if ionid not in ionlist:
            continue

        ionindex = ionlist.index(ionid)
        pldftransitions = get_ion_transitions(ion, dftransgfall, args)

        print(
            f"\n======> {at.get_elsymbol(ionid.Z)} {at.roman_numerals[ionid.ion_stage]:3s} "
            f"(pop={conditions.ionpopdict[ionid]:.2e} / cm3, {pldftransitions.height:6d} transitions)"
        )

        if not args.include_permitted and not pldftransitions.is_empty():
            pldftransitions = pldftransitions.filter(pl.col("forbidden") != 0)
            print(f"  ({pldftransitions.height:6d} forbidden)")

        if pldftransitions.is_empty():
            continue

        pldflevels = None
        if args.atomicdatabase == "artis":
            assert isinstance(ion["levels"], pl.DataFrame | pl.LazyFrame)
            pldflevels = ion["levels"].lazy().collect()
            pldftransitions = add_artis_transition_columns(pldftransitions, pldflevels)

        pldftransitions = pldftransitions.sort(by="lambda_angstroms")
        print(f"  {pldftransitions.height} plottable transitions")

        ltepartfunc = get_lte_partfunc(pldflevels, conditions.vardict["Te"]) if pldflevels is not None else 1.0
        pldftransitions = add_upper_lte_pop(
            pldftransitions.with_columns(
                flux_factor=(pl.col("upper_energy_ev") - pl.col("lower_energy_ev")) * pl.col("A")
            ),
            conditions.vardict["Te"],
            conditions.ionpopdict[ionid],
            ltepartfunc,
            columnname="upper_pop_Te",
        )

        for seriesindex, temperature in enumerate(conditions.temperature_list):
            if temperature == "NOTEMPNLTE":
                popcolumnname = "upper_pop_nlte"
                dftransitions = add_nlte_pop(pldftransitions, conditions, ionid)
                depcoeffs |= get_line_departure_coeffs(dftransitions, ionid)

                with pl.Config(tbl_cols=-1):
                    print(dftransitions.top_k(1, by="flux_factor_nlte"))
            else:
                T_exc = conditions.vardict[temperature]
                popcolumnname = f"upper_pop_lte_{T_exc:.0f}K"
                dftransitions = add_upper_lte_pop(
                    pldftransitions,
                    T_exc,
                    conditions.ionpopdict[ionid],
                    get_lte_partfunc(pldflevels, T_exc) if pldflevels is not None else 1.0,
                    columnname=popcolumnname,
                )

            if args.print_lines:
                dftransitions = dftransitions.with_columns(
                    (pl.col("flux_factor") * pl.col(popcolumnname)).alias(f"flux_factor_{popcolumnname}")
                )

            yvalues[seriesindex][ionindex] = generate_ion_spectrum(
                dftransitions, xvalues, popcolumnname, plot_resolution, args
            )
            if args.normalised:
                yvalues[seriesindex][ionindex] /= max(yvalues[seriesindex][ionindex])

        if args.print_lines:
            print(dftransitions.columns)
            print(dftransitions.select("lower", "upper", "forbidden", "A", "lambda_angstroms"))

    print()

    return yvalues, depcoeffs


def add_nlte_pop(pldftransitions: pl.DataFrame, conditions: PlotConditions, ionid: IonTuple) -> pl.DataFrame:
    """Add the NLTE population of the upper level, its flux factor, and its departure coefficient."""
    assert conditions.dfnltepops is not None
    dfnltepops_thision = conditions.dfnltepops.filter(
        (pl.col("Z") == ionid.Z) & (pl.col("ion_stage") == ionid.ion_stage)
    )
    nltepopdict = dict(zip(dfnltepops_thision["level"], dfnltepops_thision["n_NLTE"], strict=True))

    return pldftransitions.with_columns(
        upper_pop_nlte=pl.col("upper").replace_strict(nltepopdict, default=0.0, return_dtype=pl.Float64)
    ).with_columns(
        flux_factor_nlte=pl.col("flux_factor") * pl.col("upper_pop_nlte"),
        upper_departure=pl.col("upper_pop_nlte") / pl.col("upper_pop_Te"),
    )


def get_line_departure_coeffs(dftransitions: pl.DataFrame, ionid: IonTuple) -> dict[IonTuple, float]:
    """Return the departure coefficient of the Fe II 7155 line or of the Ni II 7378 line."""
    upperlower = {IonTuple(26, 2): (16, 5), IonTuple(28, 2): (6, 0)}.get(ionid)
    if upperlower is None:
        return {}

    upper, lower = upperlower
    departure = dftransitions.filter((pl.col("upper") == upper) & (pl.col("lower") == lower))["upper_departure"]

    return {ionid: float(departure.item(0))}


def print_ionisation_table(cell: CellConditions, depcoeffs: Mapping[IonTuple, float]) -> None:
    """Print the ionisation fractions of iron and of nickel, beside two departure coefficients."""
    estimators = cell.estimators

    def get_strionfracs(atomic_number: int, ion_stages: Sequence[int]) -> tuple[str, str]:
        elsym = at.get_elsymbol(atomic_number)
        est_ionfracs = [
            estimators[f"nnion_{at.get_ionstring(atomic_number, ion_stage, sep='_', style='spectral')}"]
            / estimators[f"nnelement_{elsym}"]
            for ion_stage in ion_stages
        ]
        ionfracs_str = " ".join([f"{pop:6.0e}" if pop < 0.01 else f"{pop:6.2f}" for pop in est_ionfracs])
        strions = " ".join([f"{elsym}{at.roman_numerals[ion_stage]}".rjust(6) for ion_stage in ion_stages])

        return strions, ionfracs_str

    strfeions, est_fe_ionfracs_str = get_strionfracs(26, [2, 3])
    strniions, est_ni_ionfracs_str = get_strionfracs(28, [2, 3])

    print(
        f"                     Fe II 7155             Ni II 7378  {strfeions}   /  {strniions}"
        "      T_e    Fe III/II       Ni III/II"
    )
    print(
        f"{cell.velocity:5.0f} km/s({cell.modelgridindex})      {depcoeffs[IonTuple(26, 2)]:5.2f}                   "
        f"{depcoeffs[IonTuple(28, 2)]:.2f}        "
        f"{est_fe_ionfracs_str}   /  {est_ni_ionfracs_str}      {estimators['Te']:.0f}    "
        f"{estimators['nnion_Fe_III'] / estimators['nnion_Fe_II']:.2f}          "
        f"{estimators['nnion_Ni_III'] / estimators['nnion_Ni_II']:5.2f}"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot estimated spectra from bound-bound transitions."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

    # with no model path the plot shows one fixed temperature, thus it reads the atomic data of the working folder
    from_model = bool(args.modelpath)
    modelpath = Path(args.modelpath) if from_model else Path()
    args.modelpath = modelpath

    cell = get_cell_conditions(modelpath, args) if from_model else None

    ionlist = get_ionlist()
    dftransgfall = None
    if args.atomicdatabase == "kurucz":
        dftransgfall, ionlist = get_kurucz_transitions()

    ionlist.sort()

    # resolution of the plot in Angstroms
    plot_resolution = max(1, int((args.xmax - args.xmin) / 1000))

    adata = (
        at.atomic.get_levels(modelpath, tuple(ionlist), get_transitions=True)
        if args.atomicdatabase == "artis"
        else None
    )

    conditions = (
        get_model_conditions(modelpath, cell, ionlist)
        if cell is not None
        else get_fixed_temperature_conditions(args, ionlist)
    )

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues, depcoeffs = get_ion_spectra(xvalues, ionlist, conditions, adata, dftransgfall, plot_resolution, args)

    if cell is not None:
        print_ionisation_table(cell, depcoeffs)
        outputfilename = str(args.outputfile).format(
            cell=cell.modelgridindex, timestep=cell.timestep, time_days=cell.time_days
        )
    else:
        outputfilename = "plottransitions.pdf"

    make_plot(
        xvalues,
        yvalues,
        conditions.temperature_list,
        conditions.vardict,
        ionlist,
        conditions.ionpopdict,
        args.xmin,
        args.xmax,
        conditions.figure_title,
        outputfilename,
        args,
    )
