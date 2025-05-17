# PYTHON_ARGCOMPLETE_OK


import argparse
import math
import sys
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib as mpl
import matplotlib.axes as mplax
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.misc import print_theta_phi_definitions

color_list = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))


def plot_deposition_thermalisation(
    axis: mplax.Axes,
    axistherm: mplax.Axes | None,
    modelpath: str | Path,
    modelname: str,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    # if args.logscalex:
    #     axistherm.set_xscale("log")

    # if args.logscaley:
    #     axistherm.set_yscale("log")
    #     axistherm.set_ylim(bottom=0.1, top=1.0)

    if args.plotthermalisation:
        dfmodel, _ = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g", "vel_r_mid", "kinetic_en_erg"])

        model_mass_grams = dfmodel.select("mass_g").sum().collect().item()
        print(f"  model mass: {model_mass_grams / 1.989e33:.3f} Msun")

    depdata = at.get_deposition(modelpath)

    # color_total = axis._get_lines.get_next_color()

    # axis.plot(depdata['tmid_days'], depdata['eps_erg/s/g'] * model_mass_grams, **dict(
    #     plotkwargs, **{
    #         'label': plotkwargs['label'] + r' $\dot{\epsilon}_{\alpha\beta^\pm\gamma}$',
    #         'linestyle': 'dashed',
    #         'color': color_total,
    #     }))

    # axis.plot(depdata['tmid_days'], depdata['total_dep_Lsun'] * 3.826e33, **dict(
    #     plotkwargs, **{
    #         'label': plotkwargs['label'] + r' $\dot{E}_{dep,\alpha\beta^\pm\gamma}$',
    #         'linestyle': 'dotted',
    #         'color': color_total,
    #     }))
    # if args.plotthermalisation:
    #     # f = depdata['eps_erg/s/g'] / depdata['Qdot_ana_erg/s/g']
    #     f = depdata['total_dep_Lsun'] * 3.826e33 / (depdata['eps_erg/s/g'] * model_mass_grams)
    #     axistherm.plot(depdata['tmid_days'], f, **dict(
    #         plotkwargs, **{
    #             'label': plotkwargs['label'] + r' $\dot{E}_{dep}/\dot{E}_{rad}$',
    #             'linestyle': 'solid',
    #             'color': color_total,
    #         }))

    color_gamma = axis._get_lines.get_next_color()  # type: ignore[attr-defined] # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]
    color_gamma = axis._get_lines.get_next_color()  # type: ignore[attr-defined] # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]

    # axis.plot(depdata['tmid_days'], depdata['eps_gamma_Lsun'] * 3.826e33, **dict(
    #     plotkwargs, **{
    #         'label': plotkwargs['label'] + r' $\dot{E}_{rad,\gamma}$',
    #         'linestyle': 'dashed',
    #         'color': color_gamma,
    #     }))

    gammadep_lsun = depdata["gammadep_Lsun"]

    axis.plot(
        depdata["tmid_days"],
        gammadep_lsun * 3.826e33,
        **(
            plotkwargs
            | {"label": plotkwargs["label"] + r" $\dot{E}_{dep,\gamma}$", "linestyle": "dashed", "color": color_gamma}
        ),
    )

    color_beta = axis._get_lines.get_next_color()  # type: ignore[attr-defined] # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]

    if "eps_elec_Lsun" in depdata:
        axis.plot(
            depdata["tmid_days"],
            depdata["eps_elec_Lsun"] * 3.826e33,
            **(
                plotkwargs
                | {
                    "label": plotkwargs["label"] + r" $\dot{E}_{rad,\beta^-}$",
                    "linestyle": "dotted",
                    "color": color_beta,
                }
            ),
        )

    if "elecdep_Lsun" in depdata:
        axis.plot(
            depdata["tmid_days"],
            depdata["elecdep_Lsun"] * 3.826e33,
            **(
                plotkwargs
                | {
                    "label": plotkwargs["label"] + r" $\dot{E}_{dep,\beta^-}$",
                    "linestyle": "dashed",
                    "color": color_beta,
                }
            ),
        )

    # c23modelpath = Path(
    #     Path.home(), "Google Drive/Shared drives/ARTIS/artis_runs_published/Collinsetal2023/sfho_long_1-35-135Msun"
    # )

    # c23energyrate = at.inputmodel.energyinputfiles.get_energy_rate_fromfile(c23modelpath)
    # c23etot, c23energydistribution_data = at.inputmodel.energyinputfiles.get_etot_fromfile(c23modelpath)

    # dE = np.diff(c23energyrate["rate"] * c23etot)
    # dt = np.diff(c23energyrate["times"] * 24 * 60 * 60)

    # axis.plot(
    #     c23energyrate["times"][1:],
    #     dE / dt * 0.308,
    #     color="grey",
    #     linestyle="--",
    #     zorder=20,
    #     label=r"Collins+23 $\dot{E}_{rad,\beta^-}$",
    # )

    # color_alpha = axis._get_lines.get_next_color()
    color_alpha = "C1"

    plotalphadep = False
    if plotalphadep:
        if "eps_alpha_ana_Lsun" in depdata:
            axis.plot(
                depdata["tmid_days"],
                depdata["eps_alpha_ana_Lsun"] * 3.826e33,
                **(
                    plotkwargs
                    | {
                        "label": plotkwargs["label"] + r" $\dot{E}_{rad,\alpha}$ analytical",
                        "linestyle": "solid",
                        "color": color_alpha,
                    }
                ),
            )

        if "eps_alpha_Lsun" in depdata:
            axis.plot(
                depdata["tmid_days"],
                depdata["eps_alpha_Lsun"] * 3.826e33,
                **(
                    plotkwargs
                    | {
                        "label": plotkwargs["label"] + r" $\dot{E}_{rad,\alpha}$",
                        "linestyle": "dashed",
                        "color": color_alpha,
                    }
                ),
            )

        axis.plot(
            depdata["tmid_days"],
            depdata["alphadep_Lsun"] * 3.826e33,
            **(
                plotkwargs
                | {
                    "label": plotkwargs["label"] + r" $\dot{E}_{dep,\alpha}$",
                    "linestyle": "dotted",
                    "color": color_alpha,
                }
            ),
        )

    if args.plotthermalisation:
        assert axistherm is not None
        f_gamma = depdata["gammadep_Lsun"] / depdata["eps_gamma_Lsun"]
        axistherm.plot(
            depdata["tmid_days"],
            f_gamma,
            **(
                plotkwargs
                | {
                    "label": modelname + r" $\left(\dot{E}_{dep,\gamma} \middle/ \dot{E}_{rad,\gamma}\right)$",
                    "linestyle": "solid",
                    "color": color_gamma,
                }
            ),
        )

        f_beta = depdata["elecdep_Lsun"] / depdata["eps_elec_Lsun"]
        axistherm.plot(
            depdata["tmid_days"],
            f_beta,
            **(
                plotkwargs
                | {
                    "label": modelname + r" $\left(\dot{E}_{dep,\beta^-} \middle/ \dot{E}_{rad,\beta^-}\right)$",
                    "linestyle": "solid",
                    "color": color_beta,
                }
            ),
        )

        f_alpha = depdata["alphadep_Lsun"] / depdata["eps_alpha_Lsun"]

        axistherm.plot(
            depdata["tmid_days"],
            f_alpha,
            **(
                plotkwargs
                | {
                    "label": modelname + r" $\left(\dot{E}_{dep,\alpha} \middle/ \dot{E}_{rad,\alpha}\right)$",
                    "linestyle": "solid",
                    "color": color_alpha,
                }
            ),
        )

        ejecta_ke_erg: float = dfmodel.select("kinetic_en_erg").sum().collect().item()

        print(f"  ejecta kinetic energy: {ejecta_ke_erg / 1e7:.2e} [J] = {ejecta_ke_erg:.2e} [erg]")

        # velocity derived from ejecta kinetic energy to match Barnes et al. (2016) Section 2.1
        ejecta_v = np.sqrt(2 * ejecta_ke_erg / model_mass_grams)
        print(f"  Barnes average ejecta velocity: {ejecta_v / 29979245800:.2f}c")
        m5 = model_mass_grams / (5e-3 * 1.989e33)  # M / (5e-3 Msun)
        v2 = ejecta_v / (0.2 * 29979245800)  # ejecta_v / (0.2c)

        # Barnes et al (2016) scaling form from equation 17, with fiducial t_ineff_gamma of 1.4 days
        t_ineff_gamma = 1.4 * np.sqrt(m5) / v2
        # Barnes et al (2016) equation 33
        barnes_f_gamma = [1 - math.exp(-((t / t_ineff_gamma) ** -2)) for t in depdata["tmid_days"]]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_gamma,
            **(plotkwargs | {"label": r"Barnes+2016 $f_\gamma$", "linestyle": "dashed", "color": color_gamma}),
        )

        e0_beta_mev = 0.5
        # Barnes et al (2016) equation 20
        t_ineff_beta = 7.4 * (e0_beta_mev / 0.5) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))
        # Barnes et al (2016) equation 32
        barnes_f_beta = [
            math.log(1 + 2 * (t / t_ineff_beta) ** 2) / (2 * (t / t_ineff_beta) ** 2) for t in depdata["tmid_days"]
        ]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_beta,
            **(plotkwargs | {"label": r"Barnes+2016 $f_\beta$", "linestyle": "dashed", "color": color_beta}),
        )

        e0_alpha_mev = 6.0
        # Barnes et al (2016) equation 25 times equation 16 for t_peak
        t_ineff_alpha = 4.3 * 1.8 * (e0_alpha_mev / 6.0) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))
        # Barnes et al (2016) equation 32
        barnes_f_alpha = [
            math.log(1 + 2 * (t / t_ineff_alpha) ** 2) / (2 * (t / t_ineff_alpha) ** 2) for t in depdata["tmid_days"]
        ]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_alpha,
            **(plotkwargs | {"label": r"Barnes+2016 $f_\alpha$", "linestyle": "dashed", "color": color_alpha}),
        )


def plot_artis_lightcurve(
    modelpath: str | Path,
    axis: mplax.Axes,
    lcindex: int = 0,
    linelabel: str | None = None,
    escape_type: str = "TYPE_RPKT",
    frompackets: bool = False,
    maxpacketfiles: int | None = None,
    axistherm: mplax.Axes | None = None,
    directionbins: Sequence[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
    args: argparse.Namespace | None = None,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    **plotkwargs: t.Any,
) -> dict[int, pl.DataFrame] | None:
    if args is None:
        args = argparse.Namespace()
    if plotkwargs is None:
        plotkwargs = {}
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)

    lcfilename = None
    modelpath = Path(modelpath)
    if Path(modelpath).is_file():  # handle e.g. modelpath = 'modelpath/light_curve.out'
        lcfilename = Path(modelpath).parts[-1]
        modelpath = Path(modelpath).parent

    if not modelpath.is_dir():
        print(f"\nWARNING: Skipping because {modelpath} does not exist\n")
        return None

    linelabel_is_custom = linelabel is not None
    assert "label" not in plotkwargs, "label is already set in plotkwargs"
    linelabel = linelabel or at.get_model_name(modelpath)
    assert linelabel is not None
    if escape_type == "TYPE_GAMMA":
        linelabel += r" $\gamma$"
    if pellet_nucname is not None:
        linelabel += f" {pellet_nucname}"

    print(f"====> {linelabel}")
    print(f" modelpath: {modelpath.resolve().parts[-1]}")

    if hasattr(args, "title") and args.title:
        axis.set_title(linelabel)

    if directionbins is None:
        directionbins = [-1]

    if frompackets:
        lcdataframes = at.lightcurve.get_from_packets(
            modelpath,
            escape_type=escape_type,
            maxpacketfiles=maxpacketfiles,
            directionbins=directionbins,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            get_cmf_column=args.plotcmf,
            directionbins_are_vpkt_observers=args.plotvspecpol is not None,
            pellet_nucname=pellet_nucname,
            use_pellet_decay_time=use_pellet_decay_time,
        )
    else:
        assert pellet_nucname is None, "pellet_nucname is only valid with frompackets=True"
        assert not use_pellet_decay_time, "use_pellet_decay_time is only valid with frompackets=True"
        if lcfilename is None:
            lcfilename = (
                "light_curve_res.out"
                if directionbins != [-1]
                else "gamma_light_curve.out"
                if escape_type == "TYPE_GAMMA"
                else "light_curve.out"
            )

        try:
            lcpath = at.firstexisting(lcfilename, folder=modelpath, tryzipped=True)
        except FileNotFoundError:
            print(f"WARNING: Skipping because {lcfilename} does not exist")
            return None

        lcdataframes = at.lightcurve.readfile(lcpath)

        if average_over_phi:
            lcdataframes = at.average_direction_bins(lcdataframes, overangle="phi")

        if average_over_theta:
            lcdataframes = at.average_direction_bins(lcdataframes, overangle="theta")

    if args.dashes[lcindex]:
        plotkwargs["dashes"] = args.dashes[lcindex]
    if args.linewidth[lcindex]:
        plotkwargs["linewidth"] = args.linewidth[lcindex]

    # check if doing viewing angle stuff, and if so define which data to use
    dirbins, angle_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

    if args.colorbarcostheta or args.colorbarphi:
        costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=usedegrees)
        scaledmap = make_colorbar_viewingangles_colormap()

    lctimemin = lcdataframes[dirbins[0]]["time"].min()
    lctimemax = lcdataframes[dirbins[0]]["time"].max()
    assert isinstance(lctimemin, float)
    assert isinstance(lctimemax, float)

    print(f" range of light curve: {lctimemin:.2f} to {lctimemax:.2f} days")
    try:
        nts_last, validrange_start_days, validrange_end_days = at.get_escaped_arrivalrange(modelpath)
        if validrange_start_days is not None and validrange_end_days is not None:
            str_valid_range = f"{validrange_start_days:.2f} to {validrange_end_days:.2f} days"
        else:
            str_valid_range = f"{validrange_start_days} to {validrange_end_days} days"
        print(f" range of validity (last timestep {nts_last}): {str_valid_range}")
    except FileNotFoundError:
        print(
            " range of validity: could not determine due to missing files "
            "(requires deposition.out, input.txt, model.txt)"
        )
        nts_last, validrange_start_days, validrange_end_days = None, -math.inf, math.inf

    if any(dirbin != -1 for dirbin in dirbins):
        print_theta_phi_definitions()

    colorindex: t.Any = None
    for dirbin in dirbins:
        lcdata = lcdataframes[dirbin]

        label_with_tags: str | None = linelabel
        if dirbin != -1:
            print(f" directionbin {dirbin:4d}  {angle_definition[dirbin]}")

            if args.colorbarcostheta or args.colorbarphi:
                plotkwargs["alpha"] = 0.75
                if not linelabel_is_custom:
                    label_with_tags = None
                # Update plotkwargs with viewing angle colour
                plotkwargs, colorindex = get_viewinganglecolor_for_colorbar(
                    dirbin, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
                )
                if args.average_over_phi_angle:
                    plotkwargs["color"] = "lightgrey"
            else:
                # the first dirbin should use the color argument (which has been removed from the color cycle)
                if dirbin != dirbins[0]:
                    plotkwargs["color"] = None
                if len(dirbins) > 1 or not linelabel_is_custom:
                    assert label_with_tags is not None
                    label_with_tags += f" {angle_definition[dirbin]}"

        if pellet_nucname is not None:
            plotkwargs["color"] = None

        filterfunc = at.get_filterfunc(args)
        if filterfunc is not None:
            lcdata = lcdata.with_columns(
                pl.from_numpy(filterfunc(lcdata["lum"].to_numpy()), schema=["lum"]).get_column("lum")
            )

        if not args.Lsun or args.magnitude:
            # convert luminosity from Lsun to erg/s
            lcdata = lcdata.with_columns(pl.col("lum") * 3.826e33)
            if "lum_cmf" in lcdata.columns:
                lcdata = lcdata.with_columns(pl.col("lum_cmf") * 3.826e33)

        if args.magnitude:
            # convert to bol magnitude
            Lsun_to_erg_s = 3.828e33
            lcdata["mag"] = 4.74 - (2.5 * np.log10(lcdata["lum"] / Lsun_to_erg_s))
            ycolumn = "mag"
        else:
            ycolumn = "lum"

        if (
            args.average_over_phi_angle
            and dirbin % at.get_viewingdirection_costhetabincount() == 0
            and (args.colorbarcostheta or args.colorbarphi)
        ):
            plotkwargs["color"] = scaledmap.to_rgba(colorindex)  # Update colours for light curves averaged over phi
            plotkwargs["zorder"] = 10

        # show the parts of the light curve that are outside the valid arrival range as partially transparent
        if validrange_start_days is None or validrange_end_days is None:
            # entire range is invalid
            lcdata_before_valid = lcdata
            lcdata_after_valid = pl.DataFrame(schema=lcdata.schema)
            lcdata_valid = pl.DataFrame(schema=lcdata.schema)
        else:
            lcdata_valid = lcdata.filter(pl.col("time").is_between(validrange_start_days, validrange_end_days))
            if lcdata_valid.is_empty():
                # valid range doesn't contain any data points
                lcdata_before_valid = lcdata
                lcdata_after_valid = pl.DataFrame(schema=lcdata.schema)
            else:
                lcdata_before_valid = lcdata.filter(pl.col("time") <= lcdata_valid["time"].min())
                lcdata_after_valid = lcdata.filter(pl.col("time") >= lcdata_valid["time"].max())

        if args.plotinvalidpart:
            plotkwargs_invalidrange = plotkwargs.copy()
            plotkwargs_invalidrange.update({"label": None, "alpha": 0.5})
            axis.plot(lcdata_before_valid["time"], lcdata_before_valid[ycolumn], **plotkwargs_invalidrange)
            axis.plot(lcdata_after_valid["time"], lcdata_after_valid[ycolumn], **plotkwargs_invalidrange)
        elif lcdata_valid.is_empty():
            print("  WARNING: No data points in valid range")

        axis.plot(lcdata_valid["time"], lcdata_valid[ycolumn], label=label_with_tags, **plotkwargs)
        if args.print_data:
            print(lcdata[["time", ycolumn, "lum_cmf"]])

        if args.plotcmf:
            plotkwargs["linewidth"] = 1
            if not linelabel_is_custom:
                assert label_with_tags is not None
                label_with_tags += " (cmf)"
            plotkwargs["linestyle"] = "dashed"
            # plotkwargs['color'] = 'tab:orange'
            axis.plot(lcdata["time"], lcdata["lum_cmf"], label=label_with_tags, **plotkwargs)

    if args.plotdeposition or args.plotthermalisation:
        plot_deposition_thermalisation(
            axis, axistherm, modelpath, label=linelabel, args=args, modelname=plotkwargs["modelname"], **plotkwargs
        )

    return lcdataframes


def make_lightcurve_plot(
    modelpaths: Sequence[str | Path],
    filenameout: str | Path,
    frompackets: bool = False,
    showuvoir: bool = True,
    showgamma: bool = False,
    maxpacketfiles: int | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Plot light curves from light_curve.out, gamma_light_curve.out or light_curve_res.out or packets files."""
    if args is None:
        args = argparse.Namespace()

    conffigwidth = float(at.get_config()["figwidth"])
    fig, axis = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        figsize=(args.figscale * conffigwidth, args.figscale * conffigwidth / 1.6),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if args.plotthermalisation:
        figtherm, axistherm = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            figsize=(args.figscale * conffigwidth, args.figscale * conffigwidth / 1.6),
            tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
        )
    else:
        axistherm = None

    # take any specified colours our of the cycle
    colors = [
        color for i, color in enumerate(plt.rcParams["axes.prop_cycle"].by_key()["color"]) if f"C{i}" not in args.color
    ]
    axis.set_prop_cycle(color=colors)
    reflightcurveindex = 0

    plottedsomething = False
    lcindex = 0
    for modelpath in modelpaths:
        if not Path(modelpath).is_dir() and not Path(modelpath).exists() and "." in str(modelpath):
            bolreflightcurve = Path(modelpath)

            dflightcurve, metadata = at.lightcurve.read_bol_reflightcurve_data(bolreflightcurve)
            lightcurvelabel = args.label[lcindex] or metadata.get("label", bolreflightcurve)
            color = ["0.0", "0.5", "0.7"][reflightcurveindex]
            plotkwargs = {"label": lightcurvelabel, "color": color, "zorder": 0}
            if (
                "luminosity_errminus_erg/s" in dflightcurve.columns
                and "luminosity_errplus_erg/s" in dflightcurve.columns
            ):
                axis.errorbar(
                    dflightcurve["time_days"],
                    dflightcurve["luminosity_erg/s"],
                    yerr=[dflightcurve["luminosity_errminus_erg/s"], dflightcurve["luminosity_errplus_erg/s"]],
                    fmt="o",
                    capsize=3,
                    **plotkwargs,
                )
            else:
                axis.scatter(dflightcurve["time_days"], dflightcurve["luminosity_erg/s"], **plotkwargs)
            print(f"====> {lightcurvelabel}")
            reflightcurveindex += 1
            plottedsomething = True

            lcindex += 1
        else:
            dirbin = args.plotviewingangle or (args.plotvspecpol or [-1])
            escape_types = ["TYPE_RPKT"] if showuvoir else []
            if showgamma:
                escape_types.append("TYPE_GAMMA")

            topnucs = args.topnucs
            for escape_type in escape_types:
                pellet_nucnames: list[str | None] = [None]
                if topnucs > 0:
                    try:
                        dfnuclides = at.get_nuclides(modelpath=modelpath)
                        _, dfpackets = at.packets.get_packets_pl(
                            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
                        )
                        top_nuclides = (
                            dfpackets.group_by("pellet_nucindex")
                            .agg(pl.sum("e_rf").alias("e_rf_sum"))
                            .top_k(by="e_rf_sum", k=topnucs)
                            .join(dfnuclides, on="pellet_nucindex", how="left")
                            .select(["e_rf_sum", "nucname", "pellet_nucindex"])
                            .collect()
                        )
                        print(top_nuclides)
                        pellet_nucnames.extend(top_nuclides["nucname"])
                    except FileNotFoundError:
                        print("WARNING: no nuclides.out file found, skipping top nuclides")

                for pellet_nucname in pellet_nucnames:
                    lcdataframes = plot_artis_lightcurve(
                        modelpath=modelpath,
                        lcindex=lcindex,
                        axis=axis,
                        escape_type=escape_type,
                        frompackets=frompackets,
                        maxpacketfiles=maxpacketfiles,
                        axistherm=axistherm,
                        directionbins=dirbin,
                        average_over_phi=args.average_over_phi_angle,
                        average_over_theta=args.average_over_theta_angle,
                        usedegrees=args.usedegrees,
                        args=args,
                        pellet_nucname=pellet_nucname,
                        use_pellet_decay_time=args.use_pellet_decay_time,
                        linestyle=args.linestyle[lcindex] if escape_type == "TYPE_RPKT" else ":",
                        color=args.color[lcindex],
                        linelabel=args.label[lcindex],
                    )
                    plottedsomething = plottedsomething or (lcdataframes is not None)

        print()

        if plottedsomething:
            lcindex += 1

    if args.reflightcurves:
        for bolreflightcurve in args.reflightcurves:
            if args.Lsun:
                print("Check units - trying to plot ref light curve in erg/s")
                sys.exit(1)
            bollightcurve_data, metadata = at.lightcurve.read_bol_reflightcurve_data(bolreflightcurve)
            axis.scatter(
                bollightcurve_data["time_days"],
                bollightcurve_data["luminosity_erg/s"],
                label=metadata.get("label", bolreflightcurve),
                color="k",
            )
            plottedsomething = True

    assert plottedsomething

    if args.magnitude:
        axis.invert_yaxis()

    if args.xmin is not None:
        axis.set_xlim(left=args.xmin)
    if args.xmax is not None:
        axis.set_xlim(right=args.xmax)
    if args.ymin is not None:
        axis.set_ylim(bottom=args.ymin)
    if args.ymax is not None:
        axis.set_ylim(top=args.ymax)
    # axis.set_ylim(bottom=-0.1, top=1.3)

    if not args.nolegend:
        axis.legend(loc="best", handlelength=2, frameon=args.legendframeon, numpoints=1, prop={"size": 9})
        if args.plotthermalisation:
            assert axistherm is not None
            axistherm.legend(
                loc="upper right", handlelength=2, frameon=args.legendframeon, numpoints=1, prop={"size": 9}
            )

    axis.set_xlabel(r"Time [days]")

    if args.magnitude:
        axis.set_ylabel("Absolute Bolometric Magnitude")
    else:
        str_units = r" [{}$\mathrm{{L}}_\odot$]" if args.Lsun else " [{}erg/s]"
        if args.logscaley:
            str_units = str_units.replace("{}", "")
        if args.plotdeposition:
            yvarname = r"$L$ or $\dot{{E}}$"
        elif showgamma and not showuvoir:
            yvarname = r"$\mathrm{{L}}_\gamma$"
        elif showuvoir and not showgamma:
            yvarname = r"$\mathrm{{L}}_{{\mathrm{{UVOIR}}}}$"
        else:
            yvarname = r"$\mathrm{{L}}$"

        axis.set_ylabel(yvarname + str_units)

        if "{" in axis.get_ylabel() and not args.logscaley:
            axis.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(axis.get_ylabel(), decimalplaces=1))

    if args.colorbarcostheta or args.colorbarphi:
        _, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
        scaledmap = make_colorbar_viewingangles_colormap()
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, ax=axis)

    if args.logscalex:
        axis.set_xscale("log")

    if args.logscaley:
        axis.set_yscale("log")

    if args.show:
        plt.show()

    fig.savefig(str(filenameout), format="pdf")
    print(f"Saved {filenameout}")

    if args.plotthermalisation:
        assert axistherm is not None
        # axistherm.set_xscale('log')
        axistherm.set_ylabel("Thermalisation ratio")
        axistherm.set_xlabel(r"Time [days]")
        # axistherm.set_xlim(left=0., args.xmax)
        if args.xmin is not None:
            axistherm.set_xlim(left=args.xmin)
        if args.xmax is not None:
            axistherm.set_xlim(right=args.xmax)
        axistherm.set_ylim(bottom=0.0)
        # axistherm.set_ylim(top=1.05)

        # filenameout2 = "plotthermalisation.pdf"
        filenameout2 = str(filenameout).replace(".pdf", "_thermalisation.pdf")
        figtherm.savefig(filenameout2, format="pdf")
        print(f"Saved {filenameout2}")

    plt.close()


def create_axes(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[t.Any] | mplax.Axes]:
    if "labelfontsize" in args:
        font = {"size": args.labelfontsize}
        mpl.rc("font", **font)

    args.subplots = False  # TODO: set as command line arg

    if (args.filter and len(args.filter) > 1) or args.subplots:
        args.subplots = True
        rows = 2
        cols = 3
    elif args.colour_evolution and len(args.colour_evolution) > 1:
        args.subplots = True
        rows = 1
        cols = 3
    else:
        args.subplots = False
        rows = 1
        cols = 1

    if "figwidth" not in args:
        args.figwidth = at.get_config()["figwidth"] * 1.6 * cols
    if "figheight" not in args:
        args.figheight = at.get_config()["figwidth"] * 1.1 * rows * 1.5

    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True,
        figsize=(args.figwidth, args.figheight),
        tight_layout={"pad": 3.0, "w_pad": 0.6, "h_pad": 0.6},
    )  # (6.2 * 3, 9.4 * 3)

    if args.subplots:
        assert isinstance(ax, np.ndarray)
        ax = ax.flatten()

    return fig, ax


def get_linelabel(
    modelname: str,
    modelnumber: int,
    angle: int | None,
    angle_definition: dict[int, str] | None,
    args: argparse.Namespace,
) -> str:
    if angle is not None and angle != -1:
        assert angle_definition is not None
        linelabel = angle_definition[angle] if args.nomodelname else f"{modelname} {angle_definition[angle]}"
        # linelabel = None
        # linelabel = fr"{modelname} $\theta$ = {angle_names[index]}$^\circ$"
        # plt.plot(time, magnitude, label=linelabel, linewidth=3)
    elif args.label:
        linelabel = str(args.label[modelnumber])
    else:
        linelabel = modelname
        # linelabel = 'Angle averaged'

    if linelabel == "None" or linelabel is None:
        linelabel = modelname

    return linelabel


def set_lightcurveplot_legend(ax: mplax.Axes | npt.NDArray[t.Any], args: argparse.Namespace) -> None:
    if args.nolegend:
        return

    if args.subplots:
        assert isinstance(ax, np.ndarray)
        axis = ax[args.legendsubplotnumber]
        assert isinstance(axis, mplax.Axes)
        axis.legend(loc=args.legendposition, frameon=args.legendframeon, fontsize="x-small", ncol=args.ncolslegend)
    else:
        assert isinstance(ax, mplax.Axes)
        ax.legend(
            loc=args.legendposition,
            frameon=args.legendframeon,
            fontsize="small",
            ncol=args.ncolslegend,
            handlelength=0.7,
        )


def set_lightcurve_plot_labels(
    fig: mplfig.Figure,
    ax: mplax.Axes | npt.NDArray[t.Any],
    filternames_conversion_dict: dict[str, str],
    args: argparse.Namespace,
    band_name: str | None = None,
) -> tuple[mplfig.Figure, mplax.Axes | npt.NDArray[t.Any]]:
    ylabel = None
    if args.subplots:
        if args.filter:
            ylabel = "Absolute Magnitude"
        elif args.colour_evolution:
            ylabel = r"$\Delta$m"
        else:
            msg = "No filter or colour evolution specified"
            raise AssertionError(msg)
        fig.text(0.5, 0.025, "Time Since Explosion [days]", ha="center", va="center")
        fig.text(0.02, 0.5, ylabel, ha="center", va="center", rotation="vertical")
    else:
        assert isinstance(ax, mplax.Axes)
        if args.filter and band_name in filternames_conversion_dict:
            assert band_name is not None
            ylabel = f"{filternames_conversion_dict[band_name]} Magnitude"
        elif args.filter:
            ylabel = f"{band_name} Magnitude"
        elif args.colour_evolution:
            ylabel = r"$\Delta$m"
        else:
            msg = "No filter or colour evolution specified"
            raise AssertionError(msg)

        ax.set_ylabel(ylabel, fontsize=args.labelfontsize)  # r'M$_{\mathrm{bol}}$'
        ax.set_xlabel("Time Since Explosion [days]", fontsize=args.labelfontsize)

    return fig, ax


def make_colorbar_viewingangles_colormap() -> t.Any:
    norm = mplcolors.Normalize(vmin=0, vmax=9)
    scaledmap = mplcm.ScalarMappable(cmap="tab10", norm=norm)
    scaledmap.set_array([])
    return scaledmap


def get_viewinganglecolor_for_colorbar(
    angle: int,
    costheta_viewing_angle_bins: list[str],  # noqa: ARG001
    phi_viewing_angle_bins: list[str],  # noqa: ARG001
    scaledmap: t.Any,
    plotkwargs: dict[str, t.Any],
    args: argparse.Namespace,
) -> tuple[dict[str, t.Any], int]:
    nphibins = at.get_viewingdirection_phibincount()
    if args.colorbarcostheta:
        costheta_index = angle // nphibins
        colorindex = costheta_index
        plotkwargs["color"] = scaledmap.to_rgba(colorindex)
    if args.colorbarphi:
        phi_index = angle % nphibins
        assert nphibins == 10
        reorderphibins = {5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
        print("Reordering phi bins")
        colorindex = reorderphibins.get(colorindex, phi_index)
        plotkwargs["color"] = scaledmap.to_rgba(colorindex)

    return plotkwargs, colorindex


def make_colorbar_viewingangles(
    phi_viewing_angle_bins: list[str],  # noqa: ARG001
    scaledmap: t.Any,
    args: argparse.Namespace,
    fig: mplfig.Figure | None = None,
    ax: mplax.Axes | Iterable[mplax.Axes] | None = None,
) -> None:
    if args.colorbarcostheta:
        # ticklabels = costheta_viewing_angle_bins
        ticklabels = [" -1", " -0.8", " -0.6", " -0.4", " -0.2", " 0", " 0.2", " 0.4", " 0.6", " 0.8", " 1"]
        ticklocs = list(np.linspace(0, 9, num=11, dtype=float))
        label = "cos θ"
    if args.colorbarphi:
        print("reordered phi bins")
        phi_viewing_angle_bins_reordered = [
            "0",
            "π/5",
            "2π/5",
            "3π/5",
            "4π/5",
            "π",
            "6π/5",
            "7π/5",
            "8π/5",
            "9π/5",
            "2π",
        ]
        ticklabels = phi_viewing_angle_bins_reordered
        # ticklabels = phi_viewing_angle_bins
        ticklocs = list(np.linspace(0, 9, num=11, dtype=float))
        label = "ϕ bin"

    hidecolorbar = False
    if not hidecolorbar:
        if fig:
            # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # cax = plt.axes([0.3, 0.97, 0.45, 0.02])  #2nd and 4th move up and down. 1st left and right. 3rd bar width
            cax = plt.axes((0.2, 0.98, 0.65, 0.04))
            cbar = fig.colorbar(scaledmap, cax=cax, orientation="horizontal")
        else:
            cbar = plt.colorbar(scaledmap, ax=ax)
        if label:
            cbar.set_label(label, rotation=0)
        cbar.locator = mplticker.FixedLocator(ticklocs)
        cbar.formatter = mplticker.FixedFormatter(ticklabels)
        cbar.update_ticks()


def make_band_lightcurves_plot(
    modelpaths: Sequence[str | Path],
    filternames_conversion_dict: dict[str, str],
    outputfolder: Path | str,
    args: argparse.Namespace,
) -> None:
    # angle_names = [0, 45, 90, 180]
    # plt.style.use('dark_background')

    args.labelfontsize = 22  # TODO: make command line arg
    fig, ax = create_axes(args)

    plotkwargs: dict[str, t.Any] = {}

    if args.colorbarcostheta or args.colorbarphi:
        costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(
            usedegrees=args.usedegrees
        )
        scaledmap = make_colorbar_viewingangles_colormap()

    first_band_name = None
    for modelnumber, modelpath in enumerate(Path(m) for m in modelpaths):
        # check if doing viewing angle stuff, and if so define which data to use
        angles, angle_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

        for index, angle in enumerate(angles):
            modelname = at.get_model_name(modelpath)
            print(f"Reading spectra: {modelname} (angle {angle})")
            band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
                modelpath, args, angle, modelnumber=modelnumber
            )

            if modelnumber == 0 and args.plot_hesma_model:  # TODO: does this work?
                hesma_model = at.lightcurve.read_hesma_lightcurve(args)
                plotkwargs["label"] = str(args.plot_hesma_model).split("_")[:3]

            for plotnumber, band_name in enumerate(band_lightcurve_data):
                axis = ax[plotnumber] if isinstance(ax, np.ndarray) else ax
                assert isinstance(axis, mplax.Axes)
                if first_band_name is None:
                    first_band_name = band_name
                time, brightness_in_mag = at.lightcurve.get_band_lightcurve(band_lightcurve_data, band_name, args)

                if args.print_data or args.write_data:
                    txtlinesout = [f"# band: {band_name}", f"# model: {modelname}", "# time_days magnitude"]
                    txtlinesout.extend(f"{t_d} {m}" for t_d, m in zip(time, brightness_in_mag, strict=False))
                    txtout = "\n".join(txtlinesout)
                if args.write_data:
                    bandoutfile = (
                        Path(f"band_{band_name}_angle_{angle}.txt") if angle != -1 else Path(f"band_{band_name}.txt")
                    )
                    with bandoutfile.open("w", encoding="utf-8") as f:
                        f.write(txtout)
                    print(f"Saved {bandoutfile}")
                if args.print_data:
                    print(txtout)

                plotkwargs["label"] = get_linelabel(modelname, modelnumber, angle, angle_definition, args)
                # plotkwargs['label'] = '\n'.join(wrap(linelabel, 40))  # TODO: could be arg? wraps text in label

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    brightness_in_mag = filterfunc(brightness_in_mag)

                # This does the same thing as below -- leaving code in case I'm wrong (CC)
                # if args.plotviewingangle and args.plotviewingangles_lightcurves:
                #     global define_colours_list
                #     plt.plot(time, brightness_in_mag, label=modelname, color=define_colours_list[angle], linewidth=3)

                if modelnumber == 0 and args.plot_hesma_model and band_name in hesma_model:  # TODO: see if this works
                    assert isinstance(ax, mplax.Axes)
                    ax.plot(hesma_model.t, hesma_model[band_name], color="black")

                # axarr[plotnumber].axis([0, 60, -16, -19.5])
                text_key = filternames_conversion_dict.get(band_name, band_name)

                if args.subplots:
                    assert isinstance(text_key, str)
                    axis.annotate(
                        text_key,
                        xy=(1.0, 1.0),
                        xycoords="axes fraction",
                        textcoords="offset points",
                        xytext=(-30, -30),
                        horizontalalignment="right",
                        verticalalignment="top",
                    )
                # else:
                #     ax.text(args.xmax * 0.75, args.ymax * 0.95, text_key)

                # if not args.calculate_peak_time_mag_deltam15_bool:

                if args.reflightcurves and modelnumber == 0:
                    if len(angles) > 1 and index > 0:
                        print("already plotted reflightcurve")
                    else:
                        assert isinstance(ax, mplax.Axes)
                        define_colours_list = args.refspeccolors
                        markers = args.refspecmarkers
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_lightcurve_from_refdata(
                                list(band_lightcurve_data.keys()),
                                reflightcurve,
                                define_colours_list[i],
                                markers[i],
                                filternames_conversion_dict,
                                ax,
                                plotnumber,
                            )

                if len(angles) == 1:
                    if args.color:
                        plotkwargs["color"] = args.color[modelnumber]
                    else:
                        plotkwargs["color"] = define_colours_list[modelnumber]

                if args.colorbarcostheta or args.colorbarphi:
                    # Update plotkwargs with viewing angle colour
                    plotkwargs["label"] = None
                    plotkwargs, _ = get_viewinganglecolor_for_colorbar(
                        angle, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
                    )

                if args.linestyle:
                    plotkwargs["linestyle"] = args.linestyle[modelnumber]

                # if not (args.test_viewing_angle_fit or args.calculate_peak_time_mag_deltam15_bool):
                axis.plot(time, brightness_in_mag, linewidth=4 if args.subplots else 3.5, **plotkwargs)

    at.set_mpl_style()

    ax = at.plottools.set_axis_properties(ax, args)
    fig, ax = set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args, band_name=first_band_name)
    set_lightcurveplot_legend(ax, args)

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, fig=fig, ax=ax)

    if args.filter and len(band_lightcurve_data) == 1:
        args.outputfile = Path(outputfolder, f"plot{first_band_name}lightcurves.pdf")
    if args.show:
        plt.show()

    firstaxis = ax[0] if isinstance(ax, np.ndarray) else ax
    assert isinstance(firstaxis, mplax.Axes)
    ymin, ymax = firstaxis.get_ylim()
    if ymin < ymax:
        firstaxis.invert_yaxis()

    plt.savefig(args.outputfile, format="pdf")
    print(f"Saved figure: {args.outputfile}")


# In case this code is needed again...

# if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
#     # print('time before', time)
#     # print('z', args.redshifttoz[modelnumber])
#     time = np.array(time) * (1 + args.redshifttoz[modelnumber])
#     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
#     # print('time after', time)
#     linestyle = '--'
#     color = 'darkmagenta'
#     linelabel=args.label[1]
# else:
#     linestyle = '-'
#     color='k'
# plt.plot(time, magnitude, label=linelabel, linewidth=3)

# if (args.magnitude or args.plotviewingangles_lightcurves) and not (
#         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
#         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
#         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
#         args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
#     if args.reflightcurves:
#         colours = args.refspeccolors
#         markers = args.refspecmarkers
#         for i, reflightcurve in enumerate(args.reflightcurves):
#             plot_lightcurve_from_refdata(filters_dict.keys(), reflightcurve, colours[i], markers[i],
#                                       filternames_conversion_dict)


def colour_evolution_plot(
    modelpaths: Sequence[str | Path],
    filternames_conversion_dict: dict[str, str],
    outputfolder: str | Path,
    args: argparse.Namespace,
) -> None:
    args.labelfontsize = 24  # TODO: make command line arg
    angle_counter = 0

    fig, ax = create_axes(args)

    plotkwargs: dict[str, t.Any] = {}

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)
        modelname = at.get_model_name(modelpath)
        print(f"Reading spectra: {modelname}")

        angles, angle_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

        for index, angle in enumerate(angles):
            for plotnumber, filters in enumerate(args.colour_evolution):
                filter_names = filters.split("-")
                args.filter = filter_names
                band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
                    modelpath, args, angle=angle, modelnumber=modelnumber
                )

                plot_times, colour_delta_mag = at.lightcurve.get_colour_delta_mag(band_lightcurve_data, filter_names)

                plotkwargs["label"] = get_linelabel(modelname, modelnumber, angle, angle_definition, args)

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    colour_delta_mag = filterfunc(colour_delta_mag)

                if args.color and args.plotviewingangle:
                    print(
                        "WARNING: -color argument will not work with viewing angles for colour evolution plots,"
                        "colours are taken from color_list array instead"
                    )
                    # plotkwargs["color"] = color_list[angle_counter]  # index instead of angle_counter??
                    angle_counter += 1
                elif args.plotviewingangle:
                    plotkwargs["color"] = color_list[angle_counter]
                    angle_counter += 1
                elif args.color:
                    plotkwargs["color"] = args.color[modelnumber]
                if args.linestyle:
                    plotkwargs["linestyle"] = args.linestyle[modelnumber]

                if args.reflightcurves and modelnumber == 0:
                    if len(angles) > 1 and index > 0:
                        print("already plotted reflightcurve")
                    else:
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_color_evolution_from_data(
                                filter_names,
                                reflightcurve,
                                args.refspeccolors[i],
                                args.refspecmarkers[i],
                                filternames_conversion_dict,
                                ax,
                                plotnumber,
                                args,
                            )

                curax = ax if isinstance(ax, mplax.Axes) else ax[plotnumber]
                curax.plot(plot_times, colour_delta_mag, linewidth=4 if args.subplots else 3, **plotkwargs)

                assert isinstance(curax, mplax.Axes)
                curax.invert_yaxis()
                curax.annotate(
                    f"{filter_names[0]}-{filter_names[1]}",
                    xy=(1.0, 1.0),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    xytext=(-30, -30),
                    horizontalalignment="right",
                    verticalalignment="top",
                    fontsize="x-large",
                )

            # UNCOMMENT TO ESTIMATE COLOUR AT TIME B MAX
            # def match_closest_time(reftime):
            #     return ("{}".format(min([float(x) for x in plot_times], key=lambda x: abs(x - reftime))))
            #
            # tmax_B = 17.0  # CHANGE TO TIME OF B MAX
            # tmax_B = float(match_closest_time(tmax_B))
            # print(f'{filter_names[0]} - {filter_names[1]} at t_Bmax ({tmax_B}) = '
            #       f'{diff[plot_times.index(tmax_B)]}')

    fig, ax = set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args)
    ax = at.plottools.set_axis_properties(ax, args)
    set_lightcurveplot_legend(ax, args)

    args.outputfile = Path(outputfolder, f"plotcolorevolution{filter_names[0]}-{filter_names[1]}.pdf")
    for i in range(2):
        if filter_names[i] in filternames_conversion_dict:
            filter_names[i] = filternames_conversion_dict[filter_names[i]]
    # plt.text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')

    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format="pdf")


# Just in case it's needed...

# if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
#     plot_times = np.array(plot_times) * (1 + args.redshifttoz[modelnumber])
#     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
#     linestyle = '--'
#     color='darkmagenta'
#     linelabel = args.label[1]
# else:
#     linestyle = '-'
#     color='k'
#     color='k'


def plot_lightcurve_from_refdata(
    filter_names: Sequence[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    filternames_conversion_dict: dict[str, str],
    ax: npt.NDArray[t.Any] | mplax.Axes,
    plotnumber: int,
) -> str | None:
    from extinction import apply
    from extinction import ccm89

    lightcurve_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    linename = metadata["label"] if plotnumber == 0 else None
    assert linename is None or isinstance(linename, str)
    filterdir = Path(at.get_config()["path_artistools_dir"], "data/filters/")

    filter_data = {}
    for axnumber, filter_name in enumerate(filter_names):
        axis = ax[axnumber] if isinstance(ax, np.ndarray) else ax
        assert isinstance(axis, mplax.Axes)
        if filter_name == "bol":
            continue
        with Path(filterdir / f"{filter_name}.txt").open(encoding="utf-8") as f:
            lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name == "bol":
            continue
        if filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data[filter_name] = lightcurve_data.loc[lightcurve_data["band"] == filter_name]
        # plt.plot(limits_x, limits_y, 'v', label=None, color=color)
        # else:

        if "a_v" in metadata or "e_bminusv" in metadata:
            print("Correcting for reddening")

            clightinangstroms = 3e18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * len(filter_data[filter_name]["magnitude"]), dtype=float)

            filter_data[filter_name]["flux"] = (
                clightinangstroms / (lambda0**2) * 10 ** -((filter_data[filter_name]["magnitude"] + 48.6) / 2.5)
            )  # gs

            filter_data[filter_name]["dered"] = apply(
                ccm89(filters[:], a_v=-metadata["a_v"], r_v=metadata["r_v"]), filter_data[filter_name]["flux"]
            )

            filter_data[filter_name]["magnitude"] = (
                2.5 * np.log10(clightinangstroms / (filter_data[filter_name]["dered"] * lambda0**2)) - 48.6
            )
        else:
            print("WARNING: did not correct for reddening")

        axis.plot(
            filter_data[filter_name]["time"],
            filter_data[filter_name]["magnitude"],
            marker,
            label=linename,
            color=color,
            linewidth=4 if len(filter_names) == 1 else None,
        )

        # if linename == 'SN 2018byg':
        #     x_values = []
        #     y_values = []
        #     limits_x = []
        #     limits_y = []
        #     for index, row in filter_data[filter_name].iterrows():
        #         if row['date'] == 58252:
        #             plt.plot(row['time'], row['magnitude'], '*', label=linename, color=color)
        #         elif row['e_magnitude'] != -1:
        #             x_values.append(row['time'])
        #             y_values.append(row['magnitude'])
        #         else:
        #             limits_x.append(row['time'])
        #             limits_y.append(row['magnitude'])
        #     print(x_values, y_values)
        #     plt.plot(x_values, y_values, 'o', label=linename, color=color)
        #     plt.plot(limits_x, limits_y, 's', label=linename, color=color)
    return linename


def plot_color_evolution_from_data(
    filter_names: Iterable[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    filternames_conversion_dict: dict[str, str],
    ax: npt.NDArray[t.Any] | mplax.Axes,
    plotnumber: int,
    args: argparse.Namespace,
) -> None:
    from extinction import apply
    from extinction import ccm89

    lightcurve_from_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    filterdir = Path(at.get_config()["path_artistools_dir"], "data/filters/")

    filter_data = []
    for i, filter_name in enumerate(filter_names):
        with (filterdir / Path(f"{filter_name}.txt")).open(encoding="utf-8") as f:
            lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data.append(lightcurve_from_data.loc[lightcurve_from_data["band"] == filter_name])

        if "a_v" in metadata or "e_bminusv" in metadata:
            print("Correcting for reddening")
            if "r_v" not in metadata:
                metadata["r_v"] = metadata["a_v"] / metadata["e_bminusv"]
            elif "a_v" not in metadata:
                metadata["a_v"] = metadata["e_bminusv"] * metadata["r_v"]

            clightinangstroms = 3e18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * filter_data[i].shape[0], dtype=float)

            filter_data[i]["flux"] = (
                clightinangstroms / (lambda0**2) * 10 ** -((filter_data[i]["magnitude"] + 48.6) / 2.5)
            )

            filter_data[i]["dered"] = apply(
                ccm89(filters[:], a_v=-metadata["a_v"], r_v=metadata["r_v"]), filter_data[i]["flux"]
            )

            filter_data[i]["magnitude"] = (
                2.5 * np.log10(clightinangstroms / (filter_data[i]["dered"] * lambda0**2)) - 48.6
            )

    # for i in range(2):
    #     # if metadata['label'] == 'SN 2018byg':
    #     #     filter_data[i] = filter_data[i][filter_data[i].e_magnitude != -99.00]
    #     if metadata['label'] in ['SN 2016jhr', 'SN 2018byg']:
    #         filter_data[i]['time'] = filter_data[i]['time'].apply(lambda x: round(float(x)))  # round to nearest day

    merge_dataframes = filter_data[0].merge(filter_data[1], how="inner", on=["time"])
    axis = ax[plotnumber] if isinstance(ax, np.ndarray) else ax
    assert isinstance(axis, mplax.Axes)
    axis.plot(
        merge_dataframes["time"],
        merge_dataframes["magnitude_x"] - merge_dataframes["magnitude_y"],
        marker,
        label=metadata["label"],
        color=color,
        linewidth=4 if args.subplots else None,
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "modelpath",
        default=[],
        nargs="*",
        action=at.AppendPath,
        help="Path(s) to ARTIS folders with light_curve.out or packets files (may include wildcards such as * and **)",
    )

    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument("--title", action="store_true", help="Show title of plot")

    parser.add_argument("-color", default=[f"C{i}" for i in range(10)], nargs="*", help="List of line colors")

    parser.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")

    parser.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")

    parser.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")

    parser.add_argument(
        "-figscale", type=float, default=1.6, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("--frompackets", action="store_true", help="Read packets files instead of light_curve.out")

    parser.add_argument("-maxpacketfiles", type=int, default=None, help="Limit the number of packet files read")

    parser.add_argument("--gamma", action="store_true", help="Make light curve from gamma rays")

    parser.add_argument(
        "--rpkt",
        "--uvoir",
        dest="rpkt",
        action="store_true",
        help="Make light curve from R-packets (default unless --gamma is passed)",
    )

    parser.add_argument("-escape_type", default="TYPE_RPKT", help="Type of escaping packets")

    parser.add_argument("-o", "-outputfile", action="store", dest="outputfile", type=Path, help="Filename for PDF file")

    parser.add_argument(
        "--plotcmf",
        "--plot_cmf",
        "--showcmf",
        "--show_cmf",
        action="store_true",
        help="Plot comoving frame light curve",
    )

    parser.add_argument(
        "--plotinvalidpart",
        action="store_true",
        help="Plot the entire light curve including partially accumulated parts (light travel time effects)",
    )

    parser.add_argument("--plotdeposition", action="store_true", help="Plot model deposition rates")

    parser.add_argument(
        "--plotthermalisation", action="store_true", help="Plot thermalisation rates (in separate plot)"
    )

    parser.add_argument(
        "-topnucs", type=int, default=0, help="Show light curves from top n nuclides energy contributions."
    )

    parser.add_argument(
        "--use_pellet_decay_time", action="store_true", help="Use pellet decay time instead of observer arrival time"
    )

    parser.add_argument("--magnitude", action="store_true", help="Plot light curves in magnitudes")

    parser.add_argument("--Lsun", action="store_true", help="Plot light curves in units of Lsun")

    parser.add_argument(
        "-filter",
        "-band",
        dest="filter",
        type=str,
        nargs="+",
        help=(
            "Choose filter eg. bol U B V R I. Default B. "
            "WARNING: filter names are not case sensitive eg. sloan-r is not r, it is rs"
        ),
    )

    parser.add_argument("-colour_evolution", nargs="*", help="Plot of colour evolution. Give two filters eg. B-V")

    parser.add_argument("--print_data", action="store_true", help="Print plotted data")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a text file")

    parser.add_argument(
        "-plot_hesma_model",
        action="store",
        type=Path,
        default=False,
        help="Plot hesma model on top of lightcurve plot. Enter model name saved in data/hesma directory",
    )

    parser.add_argument(
        "-plotvspecpol", type=int, nargs="+", help="Plot vspecpol. Expects int for spec number in vspecpol files"
    )

    parser.add_argument(
        "-plotviewingangle",
        type=int,
        nargs="+",
        help=(
            "Plot viewing angles. Expects int for angle number in specpol_res.out"
            "use args = -2 to select all the viewing angles"
        ),
    )
    parser.add_argument(
        "--usedegrees",
        action="store_true",
        help="Use degrees instead of radians for viewing angles. Only works with -plotviewingangle",
    )

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-xmax", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-xmin", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-timemax", type=float, default=None, help="Time max to plot")

    parser.add_argument("-timemin", type=float, default=None, help="Time min to plot")

    parser.add_argument("--logscalex", action="store_true", help="Use log scale for horizontal axis")

    parser.add_argument("--logscaley", action="store_true", help="Use log scale for vertical axis")

    parser.add_argument(
        "-reflightcurves",
        type=str,
        nargs="+",
        dest="reflightcurves",
        help="Also plot reference lightcurves from these files",
    )

    parser.add_argument(
        "-refspeccolors", default=["0.0", "0.3", "0.5"], nargs="*", help="Set a list of color for reference spectra"
    )

    parser.add_argument(
        "-refspecmarkers", default=["o", "s", "h"], nargs="*", help="Set a list of markers for reference spectra"
    )

    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order.e.g. -filtersavgol 5 3",
    )

    parser.add_argument(
        "-redshifttoz",
        type=float,
        nargs="+",
        help="Redshift to z = x. Expects array length of number modelpaths.If not to be redshifted then = 0.",
    )

    parser.add_argument("--show", action="store_true", default=False, help="Show plot before saving")

    # parser.add_argument('--calculate_peakmag_risetime_delta_m15', action='store_true',
    #                     help='Calculate band risetime, peak mag and delta m15 values for '
    #                     'the models specified using a polynomial fitting method and '
    #                     'print to screen')

    parser.add_argument(
        "--save_angle_averaged_peakmag_risetime_delta_m15_to_file",
        action="store_true",
        help="Save the band risetime, peak mag and delta m15 values for the angle averaged model lightcurves to file",
    )

    parser.add_argument(
        "--save_viewing_angle_peakmag_risetime_delta_m15_to_file",
        action="store_true",
        help=(
            "Save the band risetime, peak mag and delta m15 values for "
            "all viewing angles specified for plotting at a later time "
            "as these values take a long time to calculate for all "
            "viewing angles. Need to run this command first alongside "
            "-plotviewingangle in order to save the data for the "
            "viewing angles you want to use before making the scatter"
            "plots"
        ),
    )

    parser.add_argument(
        "--test_viewing_angle_fit",
        action="store_true",
        help=(
            "Plots the lightcurves for each  viewing angle along with"
            "the polynomial fit for each viewing angle specified"
            "to check the fit is working properly: use alongside"
            "-plotviewingangle "
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_risetime_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak mag with risetime with the "
            "angle averaged values being the solid dot and the errors bars"
            "representing the standard deviation of the viewing angle"
            "distribution"
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_delta_m15_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak with delta m15 with the angle"
            "averaged values being the solid dot and the error bars representing "
            "the standard deviation of the viewing angle distribution"
        ),
    )

    parser.add_argument(
        "--include_delta_m40",
        action="store_true",
        help="When calculating delta_m15, calculate delta_m40 as well.Only affects the saved viewing angle data.",
    )

    parser.add_argument(
        "--noerrorbars", action="store_true", help="Don't plot error bars on viewing angle scatter plots"
    )

    parser.add_argument(
        "--noangleaveraged", action="store_true", help="Don't plot angle averaged values on viewing angle scatter plots"
    )

    parser.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # for backwards compatibility with above option
    parser.add_argument("--average_every_tenth_viewing_angle", action="store_true")

    parser.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
    )

    parser.add_argument(
        "-calculate_costheta_phi_from_viewing_angle_numbers",
        type=int,
        nargs="+",
        help=(
            "calculate costheta and phi for each viewing angle given the number of the viewing angle"
            "Expects ints for angle number supplied from the argument of plot viewing angle"
            "use args = -1 to select all viewing angles"
            "Note: this method will only work if the number of angle bins (MABINS) = 100"
            "if this is not the case an error will be printed"
        ),
    )

    parser.add_argument(
        "--colorbarcostheta", action="store_true", help="Colour viewing angles by cos theta and show color bar"
    )

    parser.add_argument("--colorbarphi", action="store_true", help="Colour viewing angles by phi and show color bar")

    parser.add_argument(
        "--colouratpeak", action="store_true", help="Make scatter plot of colour at peak for viewing angles"
    )

    parser.add_argument(
        "--brightnessattime",
        action="store_true",
        help="Make scatter plot of light curve brightness at a given time (requires timedays)",
    )

    parser.add_argument("-timedays", "-time", "-t", type=float, help="Time in days to plot")

    parser.add_argument("--nomodelname", action="store_true", help="Model name not added to linename in legend")

    parser.add_argument(
        "-legendsubplotnumber", type=int, default=1, help="Subplot number to place legend in. Default is subplot[1]"
    )

    parser.add_argument("-legendposition", type=str, default="best", help="Position of legend in plot. Default is best")

    parser.add_argument("-ncolslegend", type=int, default=1, help="Number of columns in legend")

    parser.add_argument("--legendframeon", action="store_true", help="Frame on in legend")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS light curve."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)
        if args.average_every_tenth_viewing_angle:
            print("WARNING: --average_every_tenth_viewing_angle is deprecated. use --average_over_phi_angle instead")
            args.average_over_phi_angle = True

    at.set_mpl_style()

    if not args.modelpath:
        args.modelpath = ["."]

    if not isinstance(args.modelpath, Iterable):
        args.modelpath = [args.modelpath]

    assert isinstance(args.modelpath, list)
    args.modelpath = at.flatten_list(args.modelpath)

    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    args.color, args.label, args.linestyle, args.dashes, args.linewidth = at.trim_or_pad(
        len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth
    )

    if args.rpkt is False and not args.gamma:
        # if we're not plotting gamma, then we want to plot the r-packets by default
        args.rpkt = True
    if args.topnucs > 0:
        print("Enabling --frompackets because topnucs > 0")
        args.frompackets = True

    if args.filter:
        defaultoutputfile = "plotlightcurves.pdf"
    elif args.colour_evolution:
        defaultoutputfile = "plot_colour_evolution.pdf"
    else:
        defaultoutputfile = "plotlightcurve.pdf"

    if not args.outputfile:
        outputfolder = Path()
        args.outputfile = defaultoutputfile
    elif args.outputfile.is_dir():
        outputfolder = Path(args.outputfile)
        args.outputfile = outputfolder / defaultoutputfile
    else:
        outputfolder = Path()

    filternames_conversion_dict = {"rs": "r", "gs": "g", "is": "i", "zs": "z"}

    # determine if this will be a scatter plot or not
    args.calculate_peak_time_mag_deltam15_bool = False
    if (  # args.calculate_peakmag_risetime_delta_m15 or
        args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
        or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        or args.make_viewing_angle_peakmag_risetime_scatter_plot
        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot
    ):
        args.calculate_peak_time_mag_deltam15_bool = True
        at.lightcurve.peakmag_risetime_declinerate_init(modelpaths, filternames_conversion_dict, args)
        return

    if args.colouratpeak:  # make scatter plot of colour at peak, eg. B-V at Bmax
        at.lightcurve.make_peak_colour_viewing_angle_plot(args)
        return

    if args.brightnessattime:
        if args.timedays is None:
            print("Specify timedays")
            sys.exit(1)
        if not args.plotviewingangle:
            args.plotviewingangle = [-1]
        if not args.colorbarcostheta and not args.colorbarphi:
            args.colorbarphi = True
        at.lightcurve.plot_viewanglebrightness_at_fixed_time(Path(modelpaths[0]), args)
        return

    if args.filter:
        make_band_lightcurves_plot(modelpaths, filternames_conversion_dict, outputfolder, args)

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args)
        print(f"Saved figure: {args.outputfile}")
    else:
        make_lightcurve_plot(
            modelpaths=args.modelpath,
            filenameout=args.outputfile,
            frompackets=args.frompackets,
            showuvoir=args.rpkt,
            showgamma=args.gamma,
            maxpacketfiles=args.maxpacketfiles,
            args=args,
        )


if __name__ == "__main__":
    main()
