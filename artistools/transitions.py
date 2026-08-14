"""Plot synthetic emission spectra of individual ions from Kurucz, NIST, or ARTIS transition data."""

import argparse
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import K_B_ev_per_K
from artistools.misc import add_axis_limit_args
from artistools.misc import add_modelpath_arg
from artistools.misc import add_outputfile_arg
from artistools.misc import add_timedays_arg
from artistools.misc import add_timestep_arg
from artistools.plottools import save_figure

if t.TYPE_CHECKING:
    from collections.abc import Iterable

defaultoutputfile = "plottransitions_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf"


class IonTuple(t.NamedTuple):
    """An ion, identified by its atomic number and ion stage."""

    Z: int
    ion_stage: int


def get_kurucz_transitions() -> tuple[pl.DataFrame, list[IonTuple]]:
    """Return the transitions from the bundled Kurucz gfall line list, and the ions they cover."""
    hc_in_ev_cm = 0.0001239841984332003

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
    sigma_angstroms = lambda_angstroms * args.sigma_v * 1e5 / C_cm_per_s
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
    temperature_list: list[str],
    vardict: dict[str, float],
    ionlist: Sequence[IonTuple],
    ionpopdict: dict[IonTuple, float],
    xmin: float,
    xmax: float,
    figure_title: str,
    outputfilename: str,
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
        axes[0].set_title(figure_title, fontsize=10)

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
    add_modelpath_arg(parser, default=None)

    add_axis_limit_args(
        parser,
        xlimtype=int,
        xmindefault=3500,
        xmaxdefault=8000,
        xminhelp="Plot range: minimum wavelength in Angstroms",
        xmaxhelp="Plot range: maximum wavelength in Angstroms",
        include_y=False,
    )

    parser.add_argument("-T", type=float, dest="T", default=[], nargs="*", help="Temperature in Kelvin")

    parser.add_argument("-sigma_v", type=float, default=5500.0, help="Gaussian width in km/s")

    parser.add_argument(
        "-gaussian_window", type=float, default=3, help="Truncate Gaussian line profiles n sigmas from the centre"
    )

    parser.add_argument("--include-permitted", action="store_true", help="Also consider permitted lines")

    add_timedays_arg(parser, kind="str")

    add_timestep_arg(parser, kind="int", default=70)

    parser.add_argument("-modelgridindex", "-cell", type=int, default=0, help="Modelgridindex to plot")

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

    add_outputfile_arg(parser, default=defaultoutputfile, astype=None, helptext="path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot estimated spectra from bound-bound transitions."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

    if args.modelpath:
        from_model = True
    else:
        from_model = False
        args.modelpath = Path()

    modelpath = args.modelpath
    if from_model:
        modelgridindex = args.modelgridindex

        timestep = at.get_timestep_of_timedays(modelpath, args.timedays) if args.timedays else args.timestep

        modeldata = at.inputmodel.get_modeldata(Path(modelpath, "model.txt"))[0].collect()
        estimators_all = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        if not estimators_all:
            print("no estimators")
            sys.exit(1)

        estimators = estimators_all[timestep, modelgridindex]

    ionlist: list[IonTuple] = [
        IonTuple(26, 1),
        IonTuple(26, 2),
        IonTuple(26, 3),
        IonTuple(27, 2),
        IonTuple(27, 3),
        IonTuple(28, 2),
        IonTuple(28, 3),
        # iontuple(28, 2),
        # iontuple(45, 1),
        # iontuple(54, 1),
        # iontuple(54, 2),
        # iontuple(55, 1),
        # iontuple(55, 2),
        # iontuple(58, 1),
        # iontuple(79, 1),
        # iontuple(83, 1),
        # iontuple(26, 2),
        # iontuple(26, 3),
    ]

    if args.atomicdatabase == "kurucz":
        dftransgfall, ionlist = get_kurucz_transitions()

    ionlist.sort()

    # resolution of the plot in Angstroms
    plot_resolution = max(1, int((args.xmax - args.xmin) / 1000))

    if args.atomicdatabase == "artis":
        adata = at.atomic.get_levels(modelpath, tuple(ionlist), get_transitions=True)
    ionpopdict: dict[IonTuple, float] = {}
    if from_model:
        dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)

        if dfnltepops.is_empty():
            print(f"ERROR: no NLTE populations for cell {modelgridindex} at timestep {timestep}")
            sys.exit(1)

        ionpopdict = {
            IonTuple(Z, ion_stage): float(
                dfnltepops.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["n_NLTE"].sum()
            )
            for Z, ion_stage in ionlist
        }

        modelname = at.get_model_name(modelpath)
        velocity = modeldata["vel_r_max_kmps"][modelgridindex]

        Te = estimators["Te"]
        TR = estimators["TR"]
        figure_title = f"{modelname}\n"
        figure_title += (
            f"Cell {modelgridindex} ({velocity} km/s) with Te = {Te:.1f} K, TR = {TR:.1f} K at timestep {timestep}"
        )
        time_days = at.get_timestep_time(modelpath, timestep)
        if time_days != -1:
            figure_title += f" ({time_days:.1f}d)"

        # NOTEMPNLTE means use NLTE populations
        temperature_list = ["NOTEMPNLTE"]
        vardict = {"Te": Te, "TR": TR}
    else:
        if not args.T:
            args.T = [2000]
        figure_title = f"Te = {args.T[0]:.1f}" if len(args.T) == 1 else ""

        temperature_list = []
        vardict = {}
        for index, temperature in enumerate(args.T):
            tlabel = "Te"
            if index > 0:
                tlabel += f"_{index + 1}"
            vardict[tlabel] = temperature
            temperature_list.append(tlabel)

        # Fe3overFe2 = 8  # number ratio
        # ionpopdict = {
        #     IonTuple(26, 2): 1 / (1 + Fe3overFe2),
        #     IonTuple(26, 3): Fe3overFe2 / (1 + Fe3overFe2),
        #     IonTuple(28, 2): 1.0e-2,
        # }
        ionpopdict = {IonTuple(Z, ionstage): 1.0 for Z, ionstage in ionlist}

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(temperature_list) + 1, len(ionlist), len(xvalues)))
    fe2depcoeff, ni2depcoeff = None, None
    iterdict: Iterable[dict[str, t.Any]] = (
        adata.iter_rows(named=True)
        if args.atomicdatabase == "artis"
        else ({"Z": Z, "ion_stage": ion_stage, "levels": None} for Z, ion_stage in ionlist)
    )
    for ion in iterdict:
        assert isinstance(ion["Z"], int)
        assert isinstance(ion["ion_stage"], int)
        ionid = IonTuple(ion["Z"], ion["ion_stage"])
        if ionid not in ionlist:
            continue

        ionindex = ionlist.index(ionid)

        if args.atomicdatabase == "kurucz":
            pldftransitions = dftransgfall.filter((pl.col("Z") == ion["Z"]) & (pl.col("ion_stage") == ion["ion_stage"]))
        elif args.atomicdatabase == "nist":
            pldftransitions = get_nist_transitions(f"nist/nist-{ion['Z']:02d}-{ion['ion_stage']:02d}.txt")
        else:
            pldftransitions = ion["transitions"]
            assert isinstance(pldftransitions, pl.DataFrame | pl.LazyFrame)
            pldftransitions = pldftransitions.lazy().collect()

        print(
            f"\n======> {at.get_elsymbol(ionid.Z)} {at.roman_numerals[ionid.ion_stage]:3s} "
            f"(pop={ionpopdict[ionid]:.2e} / cm3, {pldftransitions.height:6d} transitions)"
        )

        if not args.include_permitted and not pldftransitions.is_empty():
            pldftransitions = pldftransitions.filter(pl.col("forbidden") != 0)
            print(f"  ({pldftransitions.height:6d} forbidden)")

        if not pldftransitions.is_empty():
            if args.atomicdatabase == "artis":
                assert isinstance(ion["levels"], pl.DataFrame | pl.LazyFrame)
                pldftransitions = (
                    at.atomic
                    .add_transition_columns(
                        pldftransitions,
                        ion["levels"],
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

            pldftransitions = pldftransitions.sort(by="lambda_angstroms")

            print(f"  {pldftransitions.height} plottable transitions")

            if args.atomicdatabase == "artis":
                T_exc = vardict["Te"]
                pldflevels = ion["levels"]
                assert isinstance(pldflevels, pl.DataFrame | pl.LazyFrame)
                pldflevels = pldflevels.lazy().collect()
                ltepartfunc = (
                    pldflevels.select(pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / T_exc).exp()).sum().item()
                )

            else:
                ltepartfunc = 1.0

            pldftransitions = pldftransitions.with_columns(
                flux_factor=(pl.col("upper_energy_ev") - pl.col("lower_energy_ev")) * pl.col("A")
            )

            pldftransitions = add_upper_lte_pop(
                pldftransitions, vardict["Te"], ionpopdict[ionid], ltepartfunc, columnname="upper_pop_Te"
            )

            for seriesindex, temperature in enumerate(temperature_list):
                if temperature == "NOTEMPNLTE":
                    dfnltepops_thision = dfnltepops.filter(
                        (pl.col("Z") == ionid.Z) & (pl.col("ion_stage") == ionid.ion_stage)
                    )

                    nltepopdict = dict(zip(dfnltepops_thision["level"], dfnltepops_thision["n_NLTE"], strict=True))

                    popcolumnname = "upper_pop_nlte"
                    dftransitions = pldftransitions.with_columns(
                        upper_pop_nlte=pl.col("upper").replace_strict(nltepopdict, default=0.0, return_dtype=pl.Float64)
                    ).with_columns(
                        flux_factor_nlte=pl.col("flux_factor") * pl.col(popcolumnname),
                        upper_departure=pl.col("upper_pop_nlte") / pl.col("upper_pop_Te"),
                    )
                    if ionid == IonTuple(26, 2):
                        fe2depcoeff = dftransitions.filter((pl.col("upper") == 16) & (pl.col("lower") == 5))[
                            "upper_departure"
                        ].item(0)
                    elif ionid == IonTuple(28, 2):
                        ni2depcoeff = dftransitions.filter((pl.col("upper") == 6) & (pl.col("lower") == 0))[
                            "upper_departure"
                        ].item(0)

                    with pl.Config(tbl_cols=-1):
                        print(dftransitions.top_k(1, by="flux_factor_nlte"))
                else:
                    T_exc = vardict[temperature]
                    popcolumnname = f"upper_pop_lte_{T_exc:.0f}K"
                    if args.atomicdatabase == "artis":
                        ltepartfunc = (
                            pldflevels
                            .select(pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / T_exc).exp())
                            .sum()
                            .item()
                        )
                    else:
                        ltepartfunc = 1.0
                    dftransitions = add_upper_lte_pop(
                        pldftransitions, T_exc, ionpopdict[ionid], ltepartfunc, columnname=popcolumnname
                    )

                if args.print_lines:
                    dftransitions = dftransitions.with_columns(
                        (pl.col("flux_factor") * pl.col(popcolumnname)).alias(f"flux_factor_{popcolumnname}")
                    )

                yvalues[seriesindex][ionindex] = generate_ion_spectrum(
                    dftransitions, xvalues, popcolumnname, plot_resolution, args
                )
                if args.normalised:
                    yvalues[seriesindex][ionindex] /= max(yvalues[seriesindex][ionindex])  # TODO: move to ax.plot line

        if args.print_lines:
            print(dftransitions.columns)
            print(dftransitions.select("lower", "upper", "forbidden", "A", "lambda_angstroms"))
    print()

    if from_model:

        def get_strionfracs(atomic_number: int, ion_stages: Sequence[int]) -> tuple[str, str]:
            elsym = at.get_elsymbol(atomic_number)
            est_ionfracs = [
                estimators[f"nnion_{at.get_ionstring(atomic_number, ion_stage, sep='_', style='spectral')}"]
                / estimators[f"nnelement_{elsym}"]
                for ion_stage in ion_stages
            ]
            ionfracs_str = " ".join([f"{pop:6.0e}" if pop < 0.01 else f"{pop:6.2f}" for pop in est_ionfracs])
            strions = " ".join([
                f"{at.get_elsymbol(atomic_number)}{at.roman_numerals[ion_stage]}".rjust(6) for ion_stage in ion_stages
            ])
            return strions, ionfracs_str

        strfeions, est_fe_ionfracs_str = get_strionfracs(26, [2, 3])

        strniions, est_ni_ionfracs_str = get_strionfracs(28, [2, 3])

        print(
            f"                     Fe II 7155             Ni II 7378  {strfeions}   /  {strniions}"
            "      T_e    Fe III/II       Ni III/II"
        )

        print(
            f"{velocity:5.0f} km/s({modelgridindex})      {fe2depcoeff:5.2f}                   "
            f"{ni2depcoeff:.2f}        "
            f"{est_fe_ionfracs_str}   /  {est_ni_ionfracs_str}      {Te:.0f}    "
            f"{estimators['nnion_Fe_III'] / estimators['nnion_Fe_II']:.2f}          "
            f"{estimators['nnion_Ni_III'] / estimators['nnion_Ni_II']:5.2f}"
        )

    outputfilename = (
        str(args.outputfile).format(cell=modelgridindex, timestep=timestep, time_days=time_days)
        if from_model
        else "plottransitions.pdf"
    )

    make_plot(
        xvalues,
        yvalues,
        temperature_list,
        {k: float(v) for k, v in vardict.items()},
        ionlist,
        ionpopdict,
        args.xmin,
        args.xmax,
        figure_title,
        outputfilename,
    )


if __name__ == "__main__":
    main()
