import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at

defaultoutputfile = "plottransitions_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf"


class IonTuple(t.NamedTuple):
    Z: int
    ion_stage: int


def get_kurucz_transitions() -> tuple[pd.DataFrame, list[IonTuple]]:
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
    ionlist = []
    with Path("gfall.dat").open(encoding="utf-8") as fnist:
        for line in fnist:
            row = line.split()
            if len(row) >= 24:
                Z, ion_stage = int(row[2].split(".")[0]), int(row[2].split(".")[1]) + 1
                if Z < 44 or ion_stage >= 2:  # and Z not in [26, 27]
                    continue
                lambda_angstroms = float(line[:12]) * 10
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

    dftransitions = pd.DataFrame(translist, columns=KuruczTransitionTuple._fields)
    return dftransitions, ionlist


def get_nist_transitions(filename: Path | str) -> pd.DataFrame:
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

    return pd.DataFrame(translist, columns=NISTTransitionTuple._fields)


def generate_ion_spectrum(
    transitions, xvalues, popcolumn, plot_resolution, args: argparse.Namespace
) -> npt.NDArray[np.floating[t.Any]]:
    yvalues = np.zeros(len(xvalues))

    # iterate over lines
    for _, line in transitions.iterrows():
        flux = line["flux_factor"] * line[popcolumn]

        # contribute the Gaussian line profile to the discrete flux bins

        centre_index = round((line["lambda_angstroms"] - args.xmin) / plot_resolution)
        sigma_angstroms = line["lambda_angstroms"] * args.sigma_v / 299792.458
        sigma_gridpoints = math.ceil(sigma_angstroms / plot_resolution)
        window_left_index = max(int(centre_index - args.gaussian_window * sigma_gridpoints), 0)
        window_right_index = min(int(centre_index + args.gaussian_window * sigma_gridpoints), len(xvalues))

        for x in range(max(0, window_left_index), min(len(xvalues), window_right_index)):
            yvalues[x] += (
                flux * math.exp(-(((x - centre_index) * plot_resolution / sigma_angstroms) ** 2)) / sigma_angstroms
            )

    return yvalues


def make_plot(
    xvalues,
    yvalues,
    temperature_list: list[str],
    vardict,
    ionlist: Sequence[IonTuple],
    ionpopdict,
    xmin: float,
    xmax: float,
    figure_title: str,
    outputfilename: str,
) -> None:
    # npanels = len(ionlist) + 1
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

    peak_y_value = -1
    yvalues_combined = np.zeros((len(temperature_list), len(xvalues)))
    for seriesindex, temperature in enumerate(temperature_list):
        serieslabel = "NLTE" if temperature == "NOTEMPNLTE" else f"LTE {temperature} = {vardict[temperature]:.0f} K"
        for ion_index, axis in enumerate(axes[: len(ionlist)]):
            # an ion subplot
            yvalues_combined[seriesindex] += yvalues[seriesindex][ion_index]

            axis.plot(xvalues, yvalues[seriesindex][ion_index], linewidth=1.5, label=serieslabel)

            peak_y_value = max(yvalues[seriesindex][ion_index])

            axis.legend(loc="upper left", handlelength=1, frameon=False, numpoints=1, prop={"size": 8})

        if len(axes) > len(ionlist):
            axes[len(ionlist)].plot(xvalues, yvalues_combined[seriesindex], linewidth=1.5, label=serieslabel)
            peak_y_value = max([peak_y_value] + yvalues_combined[seriesindex])

    axislabels = [
        f"{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]}\n(pop={ionpopdict[Z, ion_stage]:.1e}/cm³)"
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

    print(f"Saving '{outputfilename}'")
    fig.savefig(outputfilename, format="pdf")
    plt.close()


def add_upper_lte_pop(
    dftransitions: pl.DataFrame, T_exc: float, ionpop: float, ltepartfunc: float, columnname: str | None = None
) -> pl.DataFrame:
    K_B = 8.617333262145179e-05  # eV / K
    scalefactor = ionpop / ltepartfunc
    if columnname is None:
        columnname = f"upper_pop_lte_{T_exc:.0f}K"

    return dftransitions.with_columns(
        (scalefactor * pl.col("upper_statweight") * (-pl.col("upper_energy_ev") / K_B / T_exc).exp()).alias(columnname)
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=None, type=Path, help="Path to ARTIS folder")

    parser.add_argument("-xmin", type=int, default=3500, help="Plot range: minimum wavelength in Angstroms")

    parser.add_argument("-xmax", type=int, default=8000, help="Plot range: maximum wavelength in Angstroms")

    parser.add_argument("-T", type=float, dest="T", default=[], nargs="*", help="Temperature in Kelvin")

    parser.add_argument("-sigma_v", type=float, default=5500.0, help="Gaussian width in km/s")

    parser.add_argument(
        "-gaussian_window", type=float, default=3, help="Truncate Gaussian line profiles n sigmas from the centre"
    )

    parser.add_argument("--include-permitted", action="store_true", help="Also consider permitted lines")

    parser.add_argument("-timedays", "-time", "-t", help="Time in days to plot")

    parser.add_argument("-timestep", "-ts", type=int, default=70, help="Timestep number to plot")

    parser.add_argument("-modelgridindex", "-cell", type=int, default=0, help="Modelgridindex to plot")

    parser.add_argument("--normalised", action="store_true", help="Normalise all spectra to their peak values")

    parser.add_argument("--print-lines", action="store_true", help="Output details of matching lines to standard out")

    parser.add_argument("--save-lines", action="store_true", help="Output details of all lines to transitionlines.txt")

    parser.add_argument(
        "--atomicdatabase",
        default="artis",
        choices=["artis", "kurucz", "nist"],
        help="Source of atomic data for excitation transitions",
    )

    parser.add_argument(
        "-o", action="store", dest="outputfile", default=defaultoutputfile, help="path/filename for PDF file"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot estimated spectra from bound-bound transitions."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    if Path(args.outputfile).is_dir():
        args.outputfile = Path(args.outputfile, defaultoutputfile)

    if args.modelpath:
        from_model = True
    else:
        from_model = False
        args.modelpath = Path()

    modelpath = args.modelpath
    if from_model:
        modelgridindex = args.modelgridindex

        timestep = at.get_timestep_of_timedays(modelpath, args.timedays) if args.timedays else args.timestep

        modeldata, _ = at.inputmodel.get_modeldata_pandas(Path(modelpath, "model.txt"))
        estimators_all = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        if not estimators_all:
            print("no estimators")
            sys.exit(1)

        estimators = estimators_all[timestep, modelgridindex]

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    c_in_km_per_s = 299792.458
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / c_in_km_per_s)  # noqa: F841
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / c_in_km_per_s)  # noqa: F841

    ionlist = [
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
        adata = at.atomic.get_levels_polars(modelpath, tuple(ionlist), get_transitions=True)

    if from_model:
        dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)

        if dfnltepops.empty:
            print(f"ERROR: no NLTE populations for cell {modelgridindex} at timestep {timestep}")
            sys.exit(1)

        ionpopdict = {
            (Z, ion_stage): dfnltepops.query("Z==@Z and ion_stage==@ion_stage")["n_NLTE"].sum()
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

        # -1 means use NLTE populations
        temperature_list = ["Te", "TR", "NOTEMPNLTE"]
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
        #     (26, 2): 1 / (1 + Fe3overFe2),
        #     (26, 3): Fe3overFe2 / (1 + Fe3overFe2),
        #     (28, 2): 1.0e-2,
        # }
        ionpopdict = dict.fromkeys(ionlist, 1)

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(temperature_list) + 1, len(ionlist), len(xvalues)))
    dftransitions_all = None
    fe2depcoeff, ni2depcoeff = None, None
    iterdict = (
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
            pldftransitions = pl.from_pandas(
                dftransgfall.query("Z == @ion.Z and ion_stage == @ion.ion_stage", inplace=False)
            )
        elif args.atomicdatabase == "nist":
            pldftransitions = pl.from_pandas(
                get_nist_transitions(f"nist/nist-{ion['Z']:02d}-{ion['ion_stage']:02d}.txt")
            )
        else:
            assert isinstance(ion["transitions"], pl.DataFrame | pl.LazyFrame)
            pldftransitions = ion["transitions"].lazy().collect()

        print(
            f"\n======> {at.get_elsymbol(ionid.Z)} {at.roman_numerals[ionid.ion_stage]:3s} "
            f"(pop={ionpopdict[ionid]:.2e} / cm3, {len(pldftransitions):6d} transitions)"
        )

        if not args.include_permitted and not pldftransitions.is_empty():
            pldftransitions = pldftransitions.filter(pl.col("forbidden") != 0)
            print(f"  ({len(pldftransitions):6d} forbidden)")

        if not pldftransitions.is_empty():
            if args.atomicdatabase == "artis":
                assert isinstance(ion["levels"], pl.DataFrame)
                pldftransitions = (
                    at.atomic.add_transition_columns(
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
            # dftransitions = dftransitions.query(
            #     "lambda_angstroms >= @plot_xmin_wide & lambda_angstroms <= @plot_xmax_wide"
            # )

            pldftransitions = pldftransitions.sort(by="lambda_angstroms")

            print(f"  {len(pldftransitions)} plottable transitions")

            if args.atomicdatabase == "artis":
                K_B = 8.617333262145179e-05  # eV / K
                T_exc = vardict["Te"]
                ltepartfunc = (
                    ion["levels"].select(pl.col("g") * (-pl.col("energy_ev") / K_B / T_exc).exp()).sum().item()
                )

            else:
                ltepartfunc = 1.0

            if args.save_lines:
                pldftransitions = pldftransitions.with_columns(Z=ion["Z"], ion_stage=ion["ion_stage"])

                if dftransitions_all is None:
                    dftransitions_all = pldftransitions
                else:
                    dftransitions_all = pl.concat([dftransitions_all, pldftransitions])

            pldftransitions = pldftransitions.with_columns(
                flux_factor=(pl.col("upper_energy_ev") - pl.col("lower_energy_ev")) * pl.col("A")
            )

            pldftransitions = add_upper_lte_pop(
                pldftransitions, vardict["Te"], ionpopdict[ionid], ltepartfunc, columnname="upper_pop_Te"
            )

            for seriesindex, temperature in enumerate(temperature_list):
                if temperature == "NOTEMPNLTE":
                    dftransitions: pd.DataFrame = pldftransitions.to_pandas(use_pyarrow_extension_array=False)
                    dfnltepops_thision = dfnltepops.query("Z==@ionid.Z & ion_stage==@ionid.ion_stage")

                    nltepopdict = {row["level"]: row["n_NLTE"] for _, row in dfnltepops_thision.iterrows()}

                    assert isinstance(dftransitions, pd.DataFrame)
                    dftransitions["upper_pop_nlte"] = dftransitions.apply(
                        lambda x: nltepopdict.get(x["upper"], 0.0),  # noqa: B023 # pylint: disable=cell-var-from-loop
                        axis=1,
                    )

                    # dftransitions['lower_pop_nlte'] = dftransitions.apply(
                    #     lambda x: nltepopdict.get(x.lower, 0.), axis=1)

                    popcolumnname = "upper_pop_nlte"
                    dftransitions["flux_factor_nlte"] = dftransitions["flux_factor"] * dftransitions[popcolumnname]
                    dftransitions["upper_departure"] = dftransitions["upper_pop_nlte"] / dftransitions["upper_pop_Te"]
                    if ionid == (26, 2):
                        fe2depcoeff = dftransitions.query("upper == 16 and lower == 5").iloc[0]["upper_departure"]
                    elif ionid == (28, 2):
                        ni2depcoeff = dftransitions.query("upper == 6 and lower == 0").iloc[0]["upper_departure"]

                    with pd.option_context("display.width", 200):
                        print(dftransitions.nlargest(1, "flux_factor_nlte"))
                else:
                    T_exc = vardict[temperature]
                    popcolumnname = f"upper_pop_lte_{T_exc:.0f}K"
                    if args.atomicdatabase == "artis":
                        K_B = 8.617333262145179e-05  # eV / K
                        ltepartfunc = (
                            ion["levels"].select(pl.col("g") * (-pl.col("energy_ev") / K_B / T_exc).exp()).sum().item()
                        )
                    else:
                        ltepartfunc = 1.0
                    dftransitions = add_upper_lte_pop(
                        pldftransitions, T_exc, ionpopdict[ionid], ltepartfunc, columnname=popcolumnname
                    ).to_pandas(use_pyarrow_extension_array=True)

                if args.print_lines:
                    dftransitions[f"flux_factor_{popcolumnname}"] = (
                        dftransitions["flux_factor"] * dftransitions[popcolumnname]
                    )

                yvalues[seriesindex][ionindex] = generate_ion_spectrum(
                    dftransitions, xvalues, popcolumnname, plot_resolution, args
                )
                if args.normalised:
                    yvalues[seriesindex][ionindex] /= max(yvalues[seriesindex][ionindex])  # TODO: move to ax.plot line

        if args.print_lines:
            print(dftransitions.columns)
            print(dftransitions[["lower", "upper", "forbidden", "A", "lambda_angstroms"]].to_string(index=False))
    print()

    if args.save_lines:
        from tabulate import tabulate

        assert dftransitions_all is not None
        dftransitions_all = dftransitions_all[dftransitions_all["A"] > 0]
        dftransitions_all = dftransitions_all.rename({
            "lower_energy_ev": "lower_energy_Ev",
            "upper_energy_ev": "upper_energy_Ev",
        })
        dftransitions_all = dftransitions_all.with_columns(pl.col("forbidden").cast(pl.Int32))
        dftransitions_all["lambda_angstroms"] /= 1.0003
        dftransitions_all = dftransitions_all.sort(by=["Z", "ion_stage", "lower", "upper"], descending=False)
        dftransitions_all = dftransitions_all[
            [
                "lambda_angstroms",
                "A",
                "Z",
                "ion_stage",
                "lower_energy_Ev",
                "lower_statweight",
                "forbidden",
                "lower_level",
                "upper_level",
                "upper_statweight",
                "upper_energy_Ev",
            ]
        ]
        print(dftransitions_all)
        # dftransitions_all.to_csv("transitions.txt", index=False, sep=" ")
        content = tabulate(dftransitions_all.to_numpy().tolist(), list(dftransitions_all.columns), tablefmt="plain")
        # print(content)

        outpath = (
            Path(args.outputfile).parent if Path(args.outputfile).suffix else Path(args.outputfile)
        ) / "transitionlines.txt"
        print(f"Writing {outpath}")
        with outpath.open("w", encoding="utf-8") as f:
            f.write(content)

    if from_model:
        feions = [2, 3]

        def get_strionfracs(atomic_number, ion_stages):
            elsym = at.get_elsymbol(atomic_number)
            est_ionfracs = [
                estimators[f"nnion_{at.get_ionstring(atomic_number, ion_stage, sep='_', style='spectral')}"]
                / estimators[f"nnelement_{elsym}"]
                for ion_stage in ion_stages
            ]
            ionfracs_str = " ".join([f"{pop:6.0e}" if pop < 0.01 else f"{pop:6.2f}" for pop in est_ionfracs])
            strions = " ".join([
                f"{at.get_elsymbol(atomic_number)}{at.roman_numerals[ion_stage]}".rjust(6) for ion_stage in feions
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
        vardict,
        ionlist,
        ionpopdict,
        args.xmin,
        args.xmax,
        figure_title,
        outputfilename,
    )


if __name__ == "__main__":
    main()
