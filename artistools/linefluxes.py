"""Plotting of emission line fluxes and flux ratios."""

import argparse
import contextlib
import json
import math
import typing as t
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import matplotlib.typing as mplt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import markers as mplmarkers
from matplotlib.typing import MarkerType

import artistools as at
from artistools.constants import day_to_s
from artistools.constants import EV_to_erg
from artistools.constants import km_to_cm
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_maxpacketfiles
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_outputfile
from artistools.misc import addarg_seriesstyle
from artistools.plottools import save_figure

# the Fe II 7155 Å / 12570 Å pair used for the Flörs et al. (2020) ratio comparison
DEFAULT_EMFEATURESEARCH: tuple[tuple[int, ...], ...] = ((26, 2, 7155, 7150, 7160), (26, 2, 12570, 12470, 12670))


def parse_emfeaturesearch(strfeature: str) -> tuple[int | float, ...]:
    """Parse an emission feature given on the command line, e.g. '(26, 2, 7155, 7150, 7160)'."""
    import ast

    try:
        feature = ast.literal_eval(strfeature)
    except (ValueError, SyntaxError) as exc:
        msg = f"Could not parse emission feature {strfeature!r}. Expected e.g. '(26, 2, 7155, 7150, 7160)'"
        raise argparse.ArgumentTypeError(msg) from exc

    # the atomic number, ion stage, and level indices must be integers, but the wavelengths (entries 2 to 4) may be
    # fractional. Exact type checks, because bool is a subclass of int and would otherwise be accepted
    if (
        not isinstance(feature, (tuple, list))
        or not (3 <= len(feature) <= 7)
        or not all(type(x) in {int, float} for x in feature)
        or not all(type(x) is int for x in (*feature[:2], *feature[5:]))
    ):
        msg = (
            f"Emission feature {strfeature!r} must be 3 to 7 numbers:"
            " (atomic_number, ion_stage, feature_wavelength[, lambdamin, lambdamax, lowerlevelindex,"
            " upperlevelindex]), with integer atomic number, ion stage, and level indices"
        )
        raise argparse.ArgumentTypeError(msg)

    return tuple(feature)


class FeatureTuple(t.NamedTuple):
    """A spectral feature: its label and wavelength range, and the line list entries that make it up."""

    colname: str
    featurelabel: str
    approxlambda: float | str
    linelistindices: Sequence[int]
    lowestlambda: float
    highestlambda: float
    atomic_number: int
    ion_stage: int
    upperlevelindices: Sequence[int]
    lowerlevelindices: Sequence[int]


def get_line_luminosities_from_packets(
    emtypecolumn: str,
    emfeatures: Sequence[FeatureTuple],
    modelpath: Path | str,
    maxpacketfiles: int | None = None,
    arr_tstart: Sequence[float] | None = None,
    arr_tend: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Get the emission line luminosities of the requested features vs time.

    The returned values are luminosities in [erg/s]: they are not divided by 4 pi d^2 for any observer distance.
    """
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times(modelpath, loc="start")
    if arr_tend is None:
        arr_tend = at.get_timestep_times(modelpath, loc="end")

    arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = (np.array(arr_tstart) + np.array(arr_tend)) / 2.0
    timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]]).tolist()

    linelistindices_allfeatures = tuple(lineindex for feature in emfeatures for lineindex in feature.linelistindices)

    nprocs_read, dfpackets = at.packets.get_packets(
        modelpath=modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    dfpackets = dfpackets.filter(pl.col(emtypecolumn).is_in(linelistindices_allfeatures))

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
                .select(pl.col("e_rf_sum") / nprocs_read / (day_to_s * pl.col("timedelta_days")))
                .collect()
                .to_series()
                .to_numpy()
            )
            for feature in emfeatures
        },
    }

    return pl.DataFrame(dictlcdata)


def get_line_luminosities_from_pops(
    emfeatures: Sequence[FeatureTuple],
    modelpath: Path | str,
    arr_tstart: Sequence[float] | None = None,
    arr_tend: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Return each feature's luminosity against time, computed from the NLTE level populations."""
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times(modelpath, loc="start")
    if arr_tend is None:
        arr_tend = at.get_timestep_times(modelpath, loc="end")

    # arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = (np.array(arr_tstart) + np.array(arr_tend)) / 2.0

    modeldata = at.inputmodel.get_modeldata(modelpath, derived_cols=["vel_r_min_kmps", "vel_r_max_kmps"])[0].collect()

    ionlist = [(feature.atomic_number, feature.ion_stage) for feature in emfeatures]
    adata = at.atomic.get_levels(modelpath, ionlist=tuple(ionlist), get_transitions=True)

    # timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]])

    # read_files is uncached, so read every rank's nlte output once rather than once per feature
    dfnltepops_allions = at.nltepops.read_files(modelpath)

    dictlcdata = {"time": arr_tmid}
    for feature in emfeatures:
        lumdata = np.zeros_like(arr_tmid, dtype=float)

        dfnltepops = dfnltepops_allions.filter(
            (pl.col("Z") == feature.atomic_number)
            & (pl.col("ion_stage") == feature.ion_stage)
            & pl.col("level").is_in(feature.upperlevelindices)
        )

        ion = adata.filter((pl.col("Z") == feature.atomic_number) & (pl.col("ion_stage") == feature.ion_stage)).row(
            0, named=True
        )

        # one pass over the populations instead of re-filtering the whole frame for every cell below.
        # setdefault keeps the first row for a duplicated key, matching the .item(0) this replaces
        levelpop_of_ts_level_mgi: dict[tuple[int, int, int], float] = {}
        for ts, level, mgi, n_nlte in zip(
            dfnltepops["timestep"], dfnltepops["level"], dfnltepops["modelgridindex"], dfnltepops["n_NLTE"], strict=True
        ):
            levelpop_of_ts_level_mgi.setdefault((ts, level, mgi), n_nlte)

        # the shell velocities do not change with time, so take them out of the loop and scale the volume by t^3
        v_inner = modeldata["vel_r_min_kmps"].cast(pl.Float64).to_numpy() * km_to_cm
        v_outer = modeldata["vel_r_max_kmps"].cast(pl.Float64).to_numpy() * km_to_cm
        shell_volumes_at_1s = (4 * math.pi / 3) * (v_outer**3 - v_inner**3)

        # the transition data is the same for every cell and timestep, so look it up once per line. An IndexError
        # here is a missing transition rather than an empty cell, and must not be silently absorbed below
        dftransitions_ion = ion["transitions"].collect()
        linedata = []
        for upperlevelindex, lowerlevelindex in zip(feature.upperlevelindices, feature.lowerlevelindices, strict=False):
            A_val = dftransitions_ion.filter(
                (pl.col("upper") == upperlevelindex) & (pl.col("lower") == lowerlevelindex)
            )["A"].item(0)

            delta_ergs = (
                ion["levels"]["energy_ev"].item(upperlevelindex) - ion["levels"]["energy_ev"].item(lowerlevelindex)
            ) * EV_to_erg
            linedata.append((upperlevelindex, A_val, delta_ergs))

        for timeindex, timedays in enumerate(arr_tmid):
            t_sec = timedays * day_to_s
            shell_volumes = shell_volumes_at_1s * t_sec**3

            timestep = at.get_timestep_of_timedays(modelpath, float(timedays))
            print(f"{feature.approxlambda}A {timedays}d (ts {timestep})")

            for upperlevelindex, A_val, delta_ergs in linedata:
                unaccounted_shellvol = 0.0  # account for the volume of empty shells
                unaccounted_shells: list[int] = []

                for modelgridindex in range(modeldata.height):
                    levelpop = levelpop_of_ts_level_mgi.get((timestep, upperlevelindex, modelgridindex))
                    if levelpop is None:
                        # no population data for this cell, so roll its volume into the next one that has data
                        unaccounted_shellvol += shell_volumes[modelgridindex]
                        unaccounted_shells.append(modelgridindex)
                        continue

                    # l = delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                    # print(f'  {modelgridindex} outer_velocity {modeldata.vel_r_max_kmps.to_numpy()[modelgridindex]}'
                    #       f' km/s shell_vol: {shell_volumes[modelgridindex] + unaccounted_shellvol} cm3'
                    #       f' n_level {levelpop} cm-3 shell_Lum {l} erg/s')

                    lumdata[timeindex] += (
                        delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                    )

                    unaccounted_shellvol = 0.0
                if unaccounted_shells:
                    print(f"No data for cells {unaccounted_shells} (expected for empty cells)")
                assert len(unaccounted_shells) < modeldata.height  # must be data for at least one shell

        dictlcdata[feature.colname] = lumdata

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
    """Return the feature made up of one ion's lines matching the given wavelength range and level indices."""
    lzdflinelistclosematches = (
        at.atomic
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

    colname = f"lum_{at.get_ionstring(atomic_number, ion_stage, sep='')}_{approxlambdalabel}"
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
        lowestlambda=float(lowestlambda),
        highestlambda=float(highestlambda),
        atomic_number=atomic_number,
        ion_stage=ion_stage,
        upperlevelindices=tuple(dflinelistclosematches["upperlevelindex"].to_list()),
        lowerlevelindices=tuple(dflinelistclosematches["lowerlevelindex"].to_list()),
    )


def get_labelandlineindices(modelpath: Path | str, emfeaturesearch: Sequence[t.Any]) -> list[FeatureTuple]:
    """Return one feature per search specification in emfeaturesearch."""
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


def plot_floers_model_ratios(axis: mplax.Axes, floersmodelratiopath: Path, args: argparse.Namespace) -> None:
    """Overplot the NIR/VIS ratios of the Flörs models from a CSV of columns file, epoch, NIR_VIS_ratio."""
    if not floersmodelratiopath.is_file():
        msg = f"{floersmodelratiopath} not found"
        raise FileNotFoundError(msg)

    # the 263 d epoch is excluded because those models were not run to that epoch with the same setup
    dffloers = (
        pl
        .read_csv(floersmodelratiopath)
        .filter(pl.col("epoch").cast(pl.Int32) != 263)
        .with_columns(
            modelname=pl
            .col("file")
            .str.replace("fig-nne_Te_allcells-", "", literal=True)
            .str.replace(r"-\d+d\.txt$", "")
        )
    )

    # the modelname column also holds the sub-Chandrasekhar variants (subch, subch_shen2018,
    # subch_shen2018_electronlossboost{4,8,12}x), but only W7 is overplotted
    dfmodel = dffloers.filter(pl.col("modelname") == "w7").sort("epoch")
    if dfmodel.is_empty():
        print(f"WARNING: no rows for Flörs model w7 in {floersmodelratiopath}")
        return

    axis.plot(
        dfmodel["epoch"].to_list(),
        dfmodel["NIR_VIS_ratio"].to_list(),
        color=args.color[0] if args.color else None,
        label="Flörs W7",
        marker="+",
        markersize=10,
        markeredgewidth=2,
        lw=0,
        alpha=0.8,
    )


def make_luminosity_ratio_plot(args: argparse.Namespace) -> None:
    """Plot the luminosity ratio of pairs of spectral features against time, and save the figure."""
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharey=False,
        figsize=(args.figscale * 5.0, args.figscale * 5.0 * (0.25 + nrows * 0.4)),
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
            get_line_luminosities_from_pops(
                emfeatures, modelpath, arr_tstart=args.timebins_tstart, arr_tend=args.timebins_tend
            )
            if args.frompops
            else get_line_luminosities_from_packets(
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

        if args.floersmodelratiofile:
            plot_floers_model_ratios(axis, Path(args.floersmodelratiofile), args)
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

    args.outputfile = at.resolve_outputfile(args.outputfile, "linefluxes.pdf")

    save_figure(fig, args.outputfile, format="pdf")


def plot_nne_te_points(
    axis: mplax.Axes,
    serieslabel: str,
    em_log10nne: Sequence[float] | npt.NDArray[np.floating],
    em_Te: Sequence[float] | npt.NDArray[np.floating],
    normtotalpackets: float,
    color: mplt.ColorType,
    marker: MarkerType,
) -> None:
    """Scatter plot the electron density and temperature of the emitting cells, sized by how many packets each emitted."""
    color_adj = [(c + 0.1) / 1.1 for c in mplcolors.to_rgb(color)]
    hitcount: Counter[tuple[float, float]] = Counter(
        zip(np.asarray(em_log10nne, dtype=float).tolist(), np.asarray(em_Te, dtype=float).tolist(), strict=True)
    )

    arr_log10nne: list[float] = []
    arr_te: list[float] = []
    if hitcount:
        log10nne_te_pairs = list(hitcount.keys())
        arr_log10nne = [x[0] for x in log10nne_te_pairs]
        arr_te = [x[1] for x in log10nne_te_pairs]
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
    """Draw error bars at the mean electron density and temperature of the emitting cells, sized by their spread."""
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
    """Plot the electron density and temperature of the cells emitting each feature, and save the figure."""
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    # 'floers_te_nne.json',
    refdatafilenames = ["floers_te_nne.json"]  # , 'floers_te_nne_CMFGEN.json', 'floers_te_nne_Smyth.json']
    refdatalabels = ["Flörs+2020"]  # , 'Floers CMFGEN', 'Floers Smyth']
    refdatacolors = ["0.0", "C1", "C2", "C4"]
    refdatakeys: list[list[str]] = [[] for _ in refdatafilenames]
    refdatatimes = [np.array([], dtype=np.float64) for _ in refdatafilenames]
    refdatapoints: list[list[dict[str, list[float]]]] = [[] for _ in refdatafilenames]
    for refdataindex, refdatafilename in enumerate(refdatafilenames):
        floers_te_nne: dict[str, dict[str, list[float]]] = json.loads(Path(refdatafilename).read_text(encoding="utf-8"))

        # give an ordering and index to dict items
        refdatakeys_thisseries = sorted(floers_te_nne.keys(), key=float)  # strings, not floats
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
    args.outputfile = at.resolve_outputfile(args.outputfile, "emittingregions.pdf")

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

            _nprocs_read, dfpackets = at.packets.get_packets(
                modelpath=modelpath,
                maxpacketfiles=args.maxpacketfiles,
                packet_type="TYPE_ESCAPE",
                escape_type="TYPE_RPKT",
            )

            dfpackets = at.packets.add_derived_columns_lazy(
                dfpackets.filter(pl.col(args.emtypecolumn).is_in(linelistindices_allfeatures)), modelpath=modelpath
            )

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
            modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
            Tedata_all[modelindex] = {}
            log10nnedata_all[modelindex] = {}
            for tmid, tstart, tend in zip(times_days, args.timebins_tstart, args.timebins_tend, strict=False):
                Tedata_all[modelindex][tmid] = []
                log10nnedata_all[modelindex][tmid] = []
                tstartlist = at.get_timestep_times(modelpath, loc="start")
                tendlist = at.get_timestep_times(modelpath, loc="end")
                tslist = [ts for ts in range(len(tstartlist)) if tendlist[ts] >= tstart and tstartlist[ts] <= tend]
                for timestep in tslist:
                    for modelgridindex in range(modeldata.height):
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
                figsize=(args.figscale * 5.0, args.figscale * 5.0 * (0.25 + nrows * 0.7)),
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
                assert isinstance(modelpath, Path | str)
                emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

                featurecolours = ["blue", "red"]
                markers: list[MarkerType] = [
                    mplmarkers.MarkerStyle(mplmarkers.CARETUPBASE),
                    mplmarkers.MarkerStyle(mplmarkers.CARETDOWNBASE),
                ]
                # featurecolours = ['C0', 'C3']
                # featurebarcolours = ['blue', 'red']

                normtotalpackets = float(
                    np.sum([
                        len(emdata_all[modelindex][tmid, feature.colname]["em_log10nne"]) for feature in emfeatures
                    ])
                )

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
            save_figure(fig, outputfile, format="pdf")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser, multiplepaths=True, default=[], helptext="Paths to ARTIS folders with spec.out or packets files"
    )

    addarg_seriesstyle(parser, colordefault=[f"C{i}" for i in range(10)])

    addarg_nolegend(parser)

    parser.add_argument("-modeltag", default=[], nargs="*", help="List of model tags for file names")

    addarg_maxpacketfiles(parser)

    parser.add_argument(
        "-emfeaturesearch",
        default=list(DEFAULT_EMFEATURESEARCH),
        nargs="*",
        type=parse_emfeaturesearch,
        help=(
            "Emission features as (atomic_number, ion_stage, feature_wavelength, lower_wavelength, upper_wavelength)"
            " tuples, e.g. '(26, 2, 7155, 7150, 7160)' for the Fe II 7155 Å feature. At least two are needed for the"
            " flux ratio plot"
        ),
    )

    parser.add_argument(
        "--frompops", action="store_true", help="Sum up internal emissivity instead of outgoing packets"
    )

    parser.add_argument(
        "-floersmodelratiofile",
        type=Path,
        default=None,
        help=(
            "Path to a CSV of Flörs model NIR/VIS ratios (columns: file, epoch, NIR_VIS_ratio) to overplot on the"
            " Fe II 7155/12570 ratio plot"
        ),
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
    # the x axis of this command is a time in days, thus it takes no wavelength aliases
    addarg_axislimits(
        parser,
        xmindefault=50,
        xmaxdefault=450,
        xminhelp="Plot range: minimum time in days",
        xmaxhelp="Plot range: maximum time in days",
    )

    parser.add_argument(
        "-timebins_tstart",
        default=None,
        nargs="*",
        type=float,
        help="Time bin start values in days. Defaults to the model timestep starts",
    )

    parser.add_argument(
        "-timebins_tend",
        default=None,
        nargs="*",
        type=float,
        help="Time bin end values in days. Defaults to the model timestep ends",
    )

    addarg_figscale(parser, figscaledefault=1.8)

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    parser.add_argument("--plotemittingregions", action="store_true", help="Plot conditions where flux line is emitted")

    addarg_outputfile(parser, helptext="path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot line flux ratios for comparisons to Floers."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.modelpath = at.normalize_path_list(args.modelpath)

    args.label, args.modeltag, args.color = at.trim_or_pad(len(args.modelpath), args.label, args.modeltag, args.color)

    args.emtypecolumn = "emissiontype" if args.use_lastemissiontype else "trueemissiontype"

    args.emfeaturesearch = [parse_emfeaturesearch(f) if isinstance(f, str) else tuple(f) for f in args.emfeaturesearch]
    if len(args.emfeaturesearch) < 2:
        msg = f"At least two emission features are needed for a flux ratio, but got {len(args.emfeaturesearch)}"
        raise ValueError(msg)

    if (args.timebins_tstart is None) != (args.timebins_tend is None):
        msg = "timebins_tstart and timebins_tend must be given together"
        raise ValueError(msg)

    if args.timebins_tstart is not None and len(args.timebins_tstart) != len(args.timebins_tend):
        msg = (
            f"timebins_tstart has {len(args.timebins_tstart)} values but timebins_tend has"
            f" {len(args.timebins_tend)}. They must match"
        )
        raise ValueError(msg)

    if args.plotemittingregions and any(color is None for color in args.color):
        # the emitting region markers shade their colour, so every series needs a real one. The default palette
        # runs out past 10 models, and trim_or_pad fills the rest with None
        ncolors = sum(color is not None for color in args.color)
        msg = (
            f"-plotemittingregions needs a colour for each of the {len(args.modelpath)} models,"
            f" but only {ncolors} are set. Pass -color with one value per model"
        )
        raise ValueError(msg)

    if args.plotemittingregions and args.timebins_tstart is None:
        # this plot needs concrete time bins, so fall back to the first model's timesteps. The flux ratio plot
        # leaves them as None, which makes each model use its own timesteps
        # copy the lists, because get_timestep_times() is lru_cached
        args.timebins_tstart = list(at.get_timestep_times(args.modelpath[0], loc="start"))
        args.timebins_tend = list(at.get_timestep_times(args.modelpath[0], loc="end"))

    args.label = [
        at.get_series_label(args.label, index, at.get_model_name(modelpath))
        for index, modelpath in enumerate(args.modelpath)
    ]

    at.plottools.set_mpl_style()

    if args.plotemittingregions:
        make_emitting_regions_plot(args)
    else:
        make_luminosity_ratio_plot(args)


if __name__ == "__main__":
    main()
