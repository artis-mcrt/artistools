"""Artistools - spectra related functions."""

import argparse
import contextlib
import json
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from matplotlib import markers as mplmarkers
from matplotlib.typing import MarkerType

import artistools as at


class FeatureTuple(t.NamedTuple):
    colname: str
    featurelabel: str
    approxlambda: float | str
    linelistindices: Sequence[int]
    lowestlambda: float
    highestlambda: float
    atomic_number: int
    ion_stage: int
    upperlevelindicies: Sequence[int]
    lowerlevelindicies: Sequence[int]


def get_packets_with_emtype(
    modelpath: Path | str, emtypecolumn: str, lineindices: Sequence[int], maxpacketfiles: int | None = None
) -> tuple[pl.LazyFrame, int]:
    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath=modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    dfpackets = dfpackets.filter(pl.col(emtypecolumn).is_in(lineindices))
    dfmatchingpackets = dfpackets

    return dfmatchingpackets, nprocs_read


def get_line_fluxes_from_packets(
    emtypecolumn: str,
    emfeatures: Sequence[FeatureTuple],
    modelpath: Path | str,
    maxpacketfiles: int | None = None,
    arr_tstart: Sequence[float] | None = None,
    arr_tend: Sequence[float] | None = None,
) -> pl.DataFrame:
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times(modelpath, loc="start")
    if arr_tend is None:
        arr_tend = at.get_timestep_times(modelpath, loc="end")

    arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = (np.array(arr_tstart) + np.array(arr_tend)) / 2.0
    timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]]).tolist()

    linelistindices_allfeatures = tuple(lineindex for feature in emfeatures for lineindex in feature.linelistindices)

    dfpackets, nprocs_read = get_packets_with_emtype(
        modelpath, emtypecolumn, linelistindices_allfeatures, maxpacketfiles=maxpacketfiles
    )

    dictlcdata = {
        "time": arr_tmid,
        **{
            feature.colname: (
                at.packets
                .bin_and_sum(
                    dfpackets.filter(pl.col(emtypecolumn).is_in(feature.linelistindices)),
                    bincol="t_arrive_d",
                    bins=timearrayplusend,
                    sumcols=["e_rf"],
                )
                .with_columns(pl.Series("timedelta_days", arr_timedelta))
                .select(pl.col("e_rf_sum") / nprocs_read / (86400.0 * pl.col("timedelta_days")))
                .collect()
                .to_series()
                .to_numpy()
            )
            for feature in emfeatures
        },
    }

    return pl.DataFrame(dictlcdata)


def get_line_fluxes_from_pops(
    emfeatures: Sequence[FeatureTuple],
    modelpath: Path | str,
    arr_tstart: Sequence[float] | None = None,
    arr_tend: Sequence[float] | None = None,
) -> pl.DataFrame:
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times(modelpath, loc="start")
    if arr_tend is None:
        arr_tend = at.get_timestep_times(modelpath, loc="end")

    # arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = list((np.array(arr_tstart) + np.array(arr_tend)) / 2.0)

    modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect().to_pandas(use_pyarrow_extension_array=True)

    ionlist = [(feature.atomic_number, feature.ion_stage) for feature in emfeatures]
    adata = (
        at.atomic
        .get_levels(modelpath, ionlist=tuple(ionlist), get_transitions=True, get_photoionisations=False)
        .with_columns(
            levels=pl.col("levels").map_elements(
                lambda x: x.to_pandas(use_pyarrow_extension_array=True), return_dtype=pl.Object
            ),
            transitions=pl.col("transitions").map_elements(
                lambda x: x.collect().to_pandas(use_pyarrow_extension_array=True), return_dtype=pl.Object
            ),
        )
        .to_pandas(use_pyarrow_extension_array=True)
    )

    # timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]])

    dictlcdata = {"time": arr_tmid}
    ev_to_erg = 1.60218e-12
    for feature in emfeatures:
        fluxdata = np.zeros_like(arr_tmid, dtype=float)

        dfnltepops = (
            at.nltepops
            .read_files(modelpath)
            .filter(pl.col("Z") == feature.atomic_number)
            .filter(pl.col("ion_stage") == feature.ion_stage)
            .filter(pl.col("level").is_in(feature.upperlevelindicies))
        )

        ion = adata.query("Z == @feature.atomic_number and ion_stage == @feature.ion_stage").iloc[0]

        for timeindex, timedays in enumerate(arr_tmid):
            v_inner = modeldata.vel_r_min_kmps.to_numpy(dtype=float) * 1e5
            v_outer = modeldata.vel_r_max_kmps.to_numpy(dtype=float) * 1e5

            t_sec = timedays * 86400
            shell_volumes = (4 * math.pi / 3) * ((v_outer * t_sec) ** 3 - (v_inner * t_sec) ** 3)

            timestep = at.get_timestep_of_timedays(modelpath, float(timedays))
            print(f"{feature.approxlambda}A {timedays}d (ts {timestep})")

            for upperlevelindex, lowerlevelindex in zip(
                feature.upperlevelindicies, feature.lowerlevelindicies, strict=False
            ):
                unaccounted_shellvol = 0.0  # account for the volume of empty shells
                unaccounted_shells = []
                for modelgridindex in modeldata.index:
                    try:
                        levelpop = dfnltepops.filter(
                            (pl.col("modelgridindex") == modelgridindex)
                            & (pl.col("timestep") == timestep)
                            & (pl.col("Z") == feature.atomic_number)
                            & (pl.col("ion_stage") == feature.ion_stage)
                            & (pl.col("level") == upperlevelindex)
                        )["n_NLTE"].item(0)

                        A_val = (
                            ion.transitions.query("upper == @upperlevelindex and lower == @lowerlevelindex").iloc[0].A
                        )

                        delta_ergs = (
                            ion.levels.iloc[upperlevelindex].energy_ev - ion.levels.iloc[lowerlevelindex].energy_ev
                        ) * ev_to_erg

                        # l = delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                        # print(f'  {modelgridindex} outer_velocity {modeldata.vel_r_max_kmps.to_numpy()[modelgridindex]}'
                        #       f' km/s shell_vol: {shell_volumes[modelgridindex] + unaccounted_shellvol} cm3'
                        #       f' n_level {levelpop} cm-3 shell_Lum {l} erg/s')

                        fluxdata[timeindex] += (
                            delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                        )

                        unaccounted_shellvol = 0.0

                    except IndexError:
                        unaccounted_shellvol += shell_volumes[modelgridindex]
                        unaccounted_shells.append(modelgridindex)
                if unaccounted_shells:
                    print(f"No data for cells {unaccounted_shells} (expected for empty cells)")
                assert len(unaccounted_shells) < len(modeldata.index)  # must be data for at least one shell

        dictlcdata[feature.colname] = list(fluxdata)

    return pl.DataFrame(dictlcdata)


def get_closelines(
    modelpath: Path | str,
    atomic_number: int,
    ion_stage: int,
    approxlambdalabel: str | int,
    lambdamin: float | None = None,
    lambdamax: float | None = None,
    lowerlevelindex: int | None = None,
    upperlevelindex: int | None = None,
) -> FeatureTuple:
    lzdflinelistclosematches = (
        at.misc
        .get_linelist_pldf(modelpath)
        .with_columns(upper_level=pl.col("upperlevelindex") + 1, lower_level=pl.col("lowerlevelindex") + 1)
        .filter(pl.col("atomic_number") == atomic_number, pl.col("ion_stage") == ion_stage)
    )

    if lambdamin is not None and lambdamin > 0:
        lzdflinelistclosematches = lzdflinelistclosematches.filter(lambdamin < pl.col("lambda_angstroms"))
    if lambdamax is not None and lambdamax > 0:
        lzdflinelistclosematches = lzdflinelistclosematches.filter(lambdamax > pl.col("lambda_angstroms"))
    if lowerlevelindex is not None and lowerlevelindex >= 0:
        lzdflinelistclosematches = lzdflinelistclosematches.filter(pl.col("lowerlevelindex") == lowerlevelindex)
    if upperlevelindex is not None and upperlevelindex >= 0:
        lzdflinelistclosematches = lzdflinelistclosematches.filter(pl.col("upperlevelindex") == upperlevelindex)

    dflinelistclosematches = lzdflinelistclosematches.collect()

    colname = f"flux_{at.get_ionstring(atomic_number, ion_stage, sep='')}_{approxlambdalabel}"
    featurelabel = f"{at.get_ionstring(atomic_number, ion_stage)} {approxlambdalabel} Å"
    lowestlambda = dflinelistclosematches["lambda_angstroms"].min()
    assert isinstance(lowestlambda, float | np.floating)
    highestlambda = dflinelistclosematches["lambda_angstroms"].max()
    assert isinstance(highestlambda, float | np.floating)

    return FeatureTuple(
        colname=colname,
        featurelabel=featurelabel,
        approxlambda=approxlambdalabel,
        linelistindices=tuple(dflinelistclosematches["lineindex"].to_list()),
        lowestlambda=lowestlambda,
        highestlambda=highestlambda,
        atomic_number=atomic_number,
        ion_stage=ion_stage,
        upperlevelindicies=tuple(dflinelistclosematches["upperlevelindex"].to_list()),
        lowerlevelindicies=tuple(dflinelistclosematches["lowerlevelindex"].to_list()),
    )


def get_labelandlineindices(modelpath: Path | str, emfeaturesearch: Sequence[t.Any]) -> list[FeatureTuple]:
    labelandlineindices = []
    for params in emfeaturesearch:
        feature = get_closelines(modelpath, params[0], params[1], params[2], *params[3:])
        print(
            f"{feature.featurelabel} includes {len(feature.linelistindices)} lines "
            f"[{feature.lowestlambda:.1f} Å, {feature.highestlambda:.1f} Å]"
        )
        labelandlineindices.append(feature)
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 26, 2, 7155, 7150, 7160)))
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 26, 2, 12570, 12470, 12670)))
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 28, 2, 7378, 7373, 7383)))

    return labelandlineindices


def make_flux_ratio_plot(args: argparse.Namespace) -> None:
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharey=False,
        figsize=(
            args.figscale * at.get_config()["figwidth"],
            args.figscale * at.get_config()["figwidth"] * (0.25 + nrows * 0.4),
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if nrows == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    axis = axes[0]
    axis.set_yscale("log")
    # axis.set_ylabel(r'log$_1$$_0$ F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\mathrm{{\AA}}$]')

    # axis.set_xlim(left=supxmin, right=supxmax)
    tmin = math.inf
    tmax = -math.inf

    for modelpath, modellabel, modelcolor in zip(args.modelpath, args.label, args.color, strict=False):
        print(f"====> {modellabel}")

        emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

        dflcdata = (
            get_line_fluxes_from_pops(
                emfeatures, modelpath, arr_tstart=args.timebins_tstart, arr_tend=args.timebins_tend
            )
            if args.frompops
            else get_line_fluxes_from_packets(
                args.emtypecolumn,
                emfeatures,
                modelpath,
                maxpacketfiles=args.maxpacketfiles,
                arr_tstart=args.timebins_tstart,
                arr_tend=args.timebins_tend,
            )
        )

        dflcdata = dflcdata.with_columns(fratio=pl.col(emfeatures[1].colname) / pl.col(emfeatures[0].colname))
        axis.set_ylabel(
            r"F$_{\mathrm{" + emfeatures[1].featurelabel + r"}}$ / F$_{\mathrm{" + emfeatures[0].featurelabel + r"}}$"
        )

        # \mathrm{\AA}
        print(dflcdata)

        axis.plot(
            dflcdata["time"],
            dflcdata["fratio"],
            label=modellabel,
            marker="x",
            lw=0,
            markersize=10,
            markeredgewidth=2,
            color=modelcolor,
            alpha=0.8,
            fillstyle="none",
        )

        tmin = min(tmin, dflcdata.select(pl.col("time").min()).item())
        tmax = max(tmax, dflcdata.select(pl.col("time").max()).item())

    if args.emfeaturesearch[0][:3] == (26, 2, 7155) and args.emfeaturesearch[1][:3] == (26, 2, 12570):
        axis.set_ylim(ymin=0.05)
        axis.set_ylim(ymax=7)
        arr_tdays = np.linspace(tmin, tmax, 3)
        arr_floersfit = [10 ** (0.0043 * timedays - 1.65) for timedays in arr_tdays]
        for ax in axes:
            ax.plot(arr_tdays, arr_floersfit, color="black", label="Flörs+2020 fit", lw=2.0)

        femis = pd.read_csv(
            "/Users/luke/iCloud/Papers (first-author)/2022 Artis ionisation/"
            "generateplots/floers_model_NIR_VIS_ratio_20201126.csv"
        )

        amodels: dict[str, tuple[list[int], list[float]]] = {}
        for _index, row in femis.iterrows():
            modelname = row.file.replace("fig-nne_Te_allcells-", "").replace(f"-{row.epoch}d.txt", "")
            if modelname not in amodels:
                amodels[modelname] = ([], [])
            if int(row.epoch) != 263:
                amodels[modelname][0].append(row.epoch)
                amodels[modelname][1].append(row.NIR_VIS_ratio)

        # for amodelname, (xlist, ylist) in amodels.items():
        for aindex, (amodelname, alabel) in enumerate([
            ("w7", "W7")
            # ("subch", "S0"),
            # ('subch_shen2018', r'1M$_\odot$'),
            # ('subch_shen2018_electronlossboost4x', '1M$_\odot$ (Shen+18) 4x e- loss'),
            # ('subch_shen2018_electronlossboost8x', r'1M$_\odot$ heatboost8'),
            # ('subch_shen2018_electronlossboost12x', '1M$_\odot$ (Shen+18) 12x e- loss'),
        ]):
            xlist, ylist = amodels[amodelname]
            color = args.color[aindex] if aindex < len(args.color) else None
            print(amodelname, xlist, ylist)
            axis.plot(
                xlist,
                ylist,
                color=color,
                label="Flörs " + alabel,
                marker="+",
                markersize=10,
                markeredgewidth=2,
                lw=0,
                alpha=0.8,
            )
    m18_tdays = np.array([206, 229, 303, 339])
    m18_pew = {}
    # m18_pew[(26, 2, 12570)] = np.array([2383, 1941, 2798, 6770])
    m18_pew[26, 2, 7155] = np.array([618, 417, 406, 474])
    m18_pew[28, 2, 7378] = np.array([157, 256, 236, 309])
    if args.emfeaturesearch[1][:3] in m18_pew and args.emfeaturesearch[0][:3] in m18_pew:
        axis.set_ylim(ymax=12)
        arr_fratio = m18_pew[args.emfeaturesearch[1][:3]] / m18_pew[args.emfeaturesearch[0][:3]]
        for ax in axes:
            ax.plot(m18_tdays, arr_fratio, color="black", label="Maguire et al. (2018)", lw=2.0, marker="s")

    for ax in axes:
        ax.set_xlabel(r"Time [days]")
        if not args.nolegend:
            ax.legend(loc="upper right", frameon=False, handlelength=1, ncol=2, numpoints=1, prop={"size": 9})

    defaultoutputfile = "linefluxes.pdf"
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile /= defaultoutputfile

    fig.savefig(args.outputfile, format="pdf")
    # plt.show()
    print(f"open {args.outputfile}")
    plt.close()


def plot_nne_te_points(
    axis: mplax.Axes,
    serieslabel: str,
    em_log10nne: Sequence[float] | npt.NDArray[np.floating],
    em_Te: Sequence[float] | npt.NDArray[np.floating],
    normtotalpackets: float,
    color: float | str | None,
    marker: MarkerType,
) -> None:
    color_adj = [(c + 0.1) / 1.1 for c in mpl.colors.to_rgb(color)]  # type: ignore[arg-type] # pyright: ignore[reportAttributeAccessIssue]
    hitcount: dict[tuple[float, float], int] = {}
    for log10nne, Te in zip(em_log10nne, em_Te, strict=True):
        assert isinstance(log10nne, float | np.floating)
        assert isinstance(Te, float | np.floating)
        dictkey = (float(log10nne), float(Te))
        hitcount[dictkey] = hitcount.get(dictkey, 0) + 1

    arr_log10nne, arr_te = zip(*hitcount.keys(), strict=False) if hitcount else ([], [])
    arr_weight = np.array([hitcount[x, y] for x, y in zip(arr_log10nne, arr_te, strict=False)])
    arr_weight = (arr_weight / normtotalpackets) * 500
    arr_size = np.sqrt(arr_weight) * 10

    # arr_weight = arr_weight / float(max(arr_weight))
    # arr_color = np.zeros((len(arr_x), 4))
    # arr_color[:, :3] = np.array([[c for c in mpl.colors.to_rgb(color)] for x in arr_weight])
    # arr_color[:, 3] = (arr_weight + 0.2) / 1.2
    # np.array([[c * z for c in mpl.colors.to_rgb(color)] for z in arr_z])

    # axis.scatter(arr_log10nne, arr_te, s=arr_weight * 20, marker=marker, color=color_adj, lw=0, alpha=1.0,
    #              edgecolors='none')
    alpha = 0.8
    axis.scatter(arr_log10nne, arr_te, s=arr_size, marker=marker, color=color_adj, lw=0, alpha=alpha)

    # make an invisible plot series to appear in the legend with a fixed marker size
    axis.plot([0], [0], marker=marker, markersize=3, color=color_adj, linestyle="None", label=serieslabel, alpha=alpha)

    # axis.plot(em_log10nne, em_Te, label=serieslabel, linestyle='None',
    #           marker='o', markersize=2.5, markeredgewidth=0, alpha=0.05,
    #           fillstyle='full', color=color_b)


def plot_nne_te_bars(
    axis: mplax.Axes,
    em_log10nne: Sequence[float] | npt.NDArray[np.floating],
    em_Te: Sequence[float] | npt.NDArray[np.floating],
    color: t.Any,
) -> None:
    if len(em_log10nne) == 0:
        return
    # black larger one for an outline
    axis.errorbar(
        np.mean(em_log10nne),
        np.mean(em_Te),
        xerr=np.std(em_log10nne),
        yerr=np.std(em_Te),
        color="black",
        markersize=12.0,
        fillstyle="full",
        capthick=4,
        capsize=15,
        linewidth=4.0,
        alpha=1.0,
    )
    axis.errorbar(
        np.mean(em_log10nne),
        np.mean(em_Te),
        xerr=np.std(em_log10nne),
        yerr=np.std(em_Te),
        color=color,
        markersize=8.0,
        fillstyle="full",
        capthick=2,
        capsize=14,
        linewidth=2.0,
        alpha=1.0,
    )


def make_emitting_regions_plot(args: argparse.Namespace) -> None:
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    # 'floers_te_nne.json',
    refdatafilenames = ["floers_te_nne.json"]  # , 'floers_te_nne_CMFGEN.json', 'floers_te_nne_Smyth.json']
    refdatalabels = ["Flörs+2020"]  # , 'Floers CMFGEN', 'Floers Smyth']
    refdatacolors = ["0.0", "C1", "C2", "C4"]
    refdatakeys: list[list[str]] = [[] for _ in refdatafilenames]
    refdatatimes = [np.array([], dtype=np.float64) for _ in refdatafilenames]
    refdatapoints: list[list[float]] = [[] for _ in refdatafilenames]
    for refdataindex, refdatafilename in enumerate(refdatafilenames):
        floers_te_nne = json.loads(Path(refdatafilename).read_text(encoding="utf-8"))

        # give an ordering and index to dict items
        refdatakeys_thisseries = [str(x) for x in sorted(floers_te_nne.keys(), key=float)]  # strings, not floats
        assert refdatakeys_thisseries is not None
        refdatakeys[refdataindex] = refdatakeys_thisseries
        refdatatimes[refdataindex] = np.array([float(t) for t in refdatakeys_thisseries])
        refdatapoints[refdataindex] = [floers_te_nne[t] for t in refdatakeys_thisseries]
        print(f"{refdatafilename} data available for times: {refdatakeys_thisseries}")

    times_days = ((np.array(args.timebins_tstart) + np.array(args.timebins_tend)) / 2.0).tolist()

    print(f"Chosen times: {times_days}")

    emdata_all: dict[int, dict[tuple[float, str], dict[str, npt.NDArray[np.floating]]]] = {}
    log10nnedata_all: dict[int, dict[int, list[float]]] = {}
    Tedata_all: dict[int, dict[int, list[float]]] = {}

    # data is collected, now make plots
    defaultoutputfile = "emittingregions.pdf"
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile /= defaultoutputfile

    args.modelpath.append(None)
    args.label.append(f"All models: {args.label}")
    args.modeltag.append("all")
    for modelindex, (modelpath, modellabel, modeltag) in enumerate(
        zip(args.modelpath, args.label, args.modeltag, strict=False)
    ):
        print(f"ARTIS model: '{modellabel}'")

        if modelpath is not None:
            print(f"Getting packets/nne/Te data for ARTIS model: '{modellabel}'")

            emdata_all[modelindex] = {}

            emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

            linelistindices_allfeatures = tuple(
                lineindex for feature in emfeatures for lineindex in feature.linelistindices
            )

            em_mgicolumn = "em_modelgridindex" if args.emtypecolumn == "emissiontype" else "emtrue_modelgridindex"

            dfpackets, _ = get_packets_with_emtype(
                modelpath, args.emtypecolumn, linelistindices_allfeatures, maxpacketfiles=args.maxpacketfiles
            )

            dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelpath=modelpath)

            dfestimators = (
                at.estimators
                .scan_estimators(modelpath=modelpath)
                .select(["timestep", "modelgridindex", "Te", "nne"])
                .drop_nulls()
                .rename({"timestep": "em_timestep", "modelgridindex": em_mgicolumn, "Te": "em_Te", "nne": "em_nne"})
            ).with_columns(em_log10nne=pl.col("em_nne").log10())

            dfpackets = dfpackets.join(dfestimators, on=["em_timestep", em_mgicolumn], how="inner")

            for tmid, tstart, tend in zip(times_days, args.timebins_tstart, args.timebins_tend, strict=False):
                for feature in emfeatures:
                    dfpackets_selected = (
                        dfpackets
                        .filter(pl.col("t_arrive_d").is_between(tstart, tend, closed="both"))
                        .filter(pl.col(args.emtypecolumn).is_in(feature.linelistindices))
                        .select("em_log10nne", "em_Te")
                        .collect()
                    )
                    if dfpackets_selected.is_empty():
                        emdata_all[modelindex][tmid, feature.colname] = {
                            "em_log10nne": np.array([]),
                            "em_Te": np.array([]),
                        }
                    else:
                        emdata_all[modelindex][tmid, feature.colname] = {
                            "em_log10nne": dfpackets_selected["em_log10nne"].to_numpy(),
                            "em_Te": dfpackets_selected["em_Te"].to_numpy(),
                        }

            estimators = at.estimators.read_estimators(modelpath)
            modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect().to_pandas(use_pyarrow_extension_array=True)
            Tedata_all[modelindex] = {}
            log10nnedata_all[modelindex] = {}
            for tmid, tstart, tend in zip(times_days, args.timebins_tstart, args.timebins_tend, strict=False):
                Tedata_all[modelindex][tmid] = []
                log10nnedata_all[modelindex][tmid] = []
                tstartlist = at.get_timestep_times(modelpath, loc="start")
                tendlist = at.get_timestep_times(modelpath, loc="end")
                tslist = [ts for ts in range(len(tstartlist)) if tendlist[ts] >= tstart and tstartlist[ts] <= tend]
                for timestep in tslist:
                    for modelgridindex in modeldata.index:
                        Te, log10nne = None, None
                        with contextlib.suppress(KeyError):
                            Te = estimators[timestep, modelgridindex]["Te"]
                            log10nne = math.log10(estimators[timestep, modelgridindex]["nne"])

                        if Te is not None and log10nne is not None:
                            Tedata_all[modelindex][tmid].append(Te)
                            log10nnedata_all[modelindex][tmid].append(log10nne)

        if modeltag != "all":
            continue

        nrows = 1
        for tmid in times_days:
            print(f"  Plot at {tmid} days")

            fig, axis = plt.subplots(
                nrows=nrows,
                ncols=1,
                sharey=False,
                sharex=False,
                figsize=(
                    args.figscale * at.get_config()["figwidth"],
                    args.figscale * at.get_config()["figwidth"] * (0.25 + nrows * 0.7),
                ),
                tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.2},
            )
            assert isinstance(axis, mplax.Axes)

            for refdataindex in range(len(refdatafilenames)):
                timeindex = np.abs(refdatatimes[refdataindex] - tmid).argmin()
                axis.plot(
                    refdatapoints[refdataindex][timeindex]["ne"],
                    refdatapoints[refdataindex][timeindex]["temp"],
                    color=refdatacolors[refdataindex],
                    lw=2,
                    label=f"{refdatalabels[refdataindex]} +{refdatakeys[refdataindex][timeindex]}d",
                )

                timeindexb = np.abs(refdatatimes[refdataindex] - tmid - 50).argmin()
                if timeindexb < len(refdatakeys[refdataindex]):
                    axis.plot(
                        refdatapoints[refdataindex][timeindexb]["ne"],
                        refdatapoints[refdataindex][timeindexb]["temp"],
                        color="0.4",
                        lw=2,
                        label=f"{refdatalabels[refdataindex]} +{refdatakeys[refdataindex][timeindexb]}d",
                    )

            if modeltag == "all":
                for bars in (False,):  # (False, True)
                    for truemodelindex in range(modelindex):
                        emfeatures = get_labelandlineindices(args.modelpath[truemodelindex], args.emfeaturesearch)

                        em_log10nne = np.concatenate([
                            emdata_all[truemodelindex][tmid, feature.colname]["em_log10nne"] for feature in emfeatures
                        ])

                        em_Te = np.concatenate([
                            emdata_all[truemodelindex][tmid, feature.colname]["em_Te"] for feature in emfeatures
                        ])

                        normtotalpackets = len(em_log10nne) * 8.0  # circles have more area than triangles, so decrease
                        modelcolor = args.color[truemodelindex]
                        label = args.label[truemodelindex].format(timeavg=tmid, modeltag=modeltag)
                        if not bars:
                            plot_nne_te_points(
                                axis, label, em_log10nne, em_Te, normtotalpackets, modelcolor, marker="s"
                            )
                        else:
                            plot_nne_te_bars(axis, em_log10nne, em_Te, modelcolor)
            else:
                emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

                featurecolours = ["blue", "red"]
                markers: list[MarkerType] = [
                    mplmarkers.MarkerStyle(mplmarkers.CARETUPBASE),
                    mplmarkers.MarkerStyle(mplmarkers.CARETDOWNBASE),
                ]
                # featurecolours = ['C0', 'C3']
                # featurebarcolours = ['blue', 'red']

                normtotalpackets = np.sum([
                    len(emdata_all[modelindex][tmid, feature.colname]["em_log10nne"]) for feature in emfeatures
                ])

                axis.scatter(
                    log10nnedata_all[modelindex][tmid],
                    Tedata_all[modelindex][tmid],
                    s=1.0,
                    marker="o",
                    color="0.4",
                    lw=0,
                    edgecolors="none",
                    label="All cells",
                )

                for bars in (False,):  # (False, True)
                    for featureindex, feature in enumerate(emfeatures):
                        emdata = emdata_all[modelindex][tmid, feature.colname]

                        if not bars:
                            print(f"   {len(emdata['em_log10nne'])} points plotted for {feature.featurelabel}")

                        serieslabel = (
                            (modellabel + " " + feature.featurelabel)
                            .format(timeavg=tmid, modeltag=modeltag)
                            .replace("Å", r" $\mathrm{\AA}$")
                        )

                        if not bars:
                            plot_nne_te_points(
                                axis,
                                serieslabel,
                                emdata["em_log10nne"],
                                emdata["em_Te"],
                                normtotalpackets,
                                featurecolours[featureindex],
                                marker=markers[featureindex],
                            )
                        else:
                            plot_nne_te_bars(axis, emdata["em_log10nne"], emdata["em_Te"], featurecolours[featureindex])

            if tmid == times_days[-1] and not args.nolegend:
                axis.legend(
                    loc="best",
                    frameon=False,
                    handlelength=1,
                    ncol=1,
                    borderpad=0,
                    numpoints=1,
                    fontsize=11,
                    markerscale=2.5,
                )

            axis.set_ylim(ymin=3000)
            axis.set_ylim(ymax=10000)
            axis.set_xlim(xmin=4.5, xmax=7.15)

            axis.set_xlabel(r"log$_{10}$(n$_{\mathrm{e}}$ [cm$^{-3}$])")
            axis.set_ylabel(r"Electron Temperature [K]")

            # axis.annotate(f'{tmid:.0f}d', xy=(0.98, 0.5), xycoords='axes fraction',
            #               horizontalalignment='right', verticalalignment='center', fontsize=16)

            outputfile = str(args.outputfile).format(timeavg=tmid, modeltag=modeltag)
            fig.savefig(outputfile, format="pdf")
            print(f"    Saved {outputfile}")
            plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath", default=[], nargs="*", type=Path, help="Paths to ARTIS folders with spec.out or packets files"
    )

    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument("-modeltag", default=[], nargs="*", help="List of model tags for file names")

    parser.add_argument("-color", default=[f"C{i}" for i in range(10)], nargs="*", help="List of line colors")

    parser.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")

    parser.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")

    parser.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")

    parser.add_argument(
        "-maxpacketfiles", "-maxpacketsfiles", type=int, default=None, help="Limit the number of packet files read"
    )

    parser.add_argument("-emfeaturesearch", default=[], nargs="*", help="List of tuples (TODO explain)")

    parser.add_argument(
        "--frompops", action="store_true", help="Sum up internal emissivity instead of outgoing packets"
    )

    parser.add_argument(
        "--use_lastemissiontype",
        action="store_true",
        help="Tag packets by their last scattering rather than thermal emission type",
    )

    # parser.add_argument('-timemin', type=float,
    #                     help='Lower time in days to integrate spectrum')
    #
    # parser.add_argument('-timemax', type=float,
    #                     help='Upper time in days to integrate spectrum')
    #
    parser.add_argument("-xmin", type=int, default=50, help="Plot range: minimum wavelength in Angstroms")

    parser.add_argument("-xmax", type=int, default=450, help="Plot range: maximum wavelength in Angstroms")

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument(
        "-timebins_tstart", default=[], nargs="*", action="append", help="Time bin start values in days"
    )

    parser.add_argument("-timebins_tend", default=[], nargs="*", action="append", help="Time bin end values in days")

    parser.add_argument(
        "-figscale", type=float, default=1.8, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    parser.add_argument("--plotemittingregions", action="store_true", help="Plot conditions where flux line is emitted")

    parser.add_argument(
        "-outputfile", "-o", action="store", dest="outputfile", type=Path, help="path/filename for PDF file"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot line flux ratios for comparisons to Floers."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=(__doc__))
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    if not args.modelpath:
        args.modelpath = [Path()]
    elif isinstance(args.modelpath, str | Path):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)

    args.label, args.modeltag, args.color = at.trim_or_pad(len(args.modelpath), args.label, args.modeltag, args.color)

    args.emtypecolumn = "emissiontype" if args.use_lastemissiontype else "trueemissiontype"

    assert isinstance(args.label, list)
    for i in range(len(args.label)):
        if args.label[i] is None:
            assert hasattr(args.label, "__setitem__")
            args.label[i] = at.get_model_name(args.modelpath[i])

    at.plottools.set_mpl_style()

    if args.plotemittingregions:
        make_emitting_regions_plot(args)
    else:
        make_flux_ratio_plot(args)


if __name__ == "__main__":
    main()
