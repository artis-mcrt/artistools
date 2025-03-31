#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import typing as t
from pathlib import Path

import argcomplete
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at
from artistools.misc import print_theta_phi_definitions


def plot_spherical(
    modelpath: str | Path,
    dfpackets: pl.LazyFrame,
    nprocs_read: int,
    timemindays: float | None,
    timemaxdays: float | None,
    nphibins: int,
    ncosthetabins: int,
    dfestimators: pl.LazyFrame | None = None,
    atomic_number: int | None = None,
    ion_stage: int | None = None,
    gaussian_sigma: int | None = None,
    plotvars: list[str] | None = None,
    figscale: float = 1.0,
    cmap: str | None = None,
) -> tuple[mplfig.Figure, t.Any, float, float, str]:
    condition = ""
    if plotvars is None:
        plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc"]

    _, tmin_d_valid, tmax_d_valid = at.get_escaped_arrivalrange(modelpath)
    if tmin_d_valid is None or tmax_d_valid is None:
        print("WARNING! The observer never gets light from the entire ejecta. Plotting all packets anyway")
        timemindays, timemaxdays = (
            dfpackets.select(tmin=pl.col("t_arrive_d").min(), tmax=pl.col("t_arrive_d").max()).collect().to_numpy()[0]
        )
    else:
        if timemindays is None:
            print(f"setting timemindays to start of valid observable range {tmin_d_valid:.2f} d")
            timemindays = tmin_d_valid
        elif timemindays < tmin_d_valid:
            print(
                f"WARNING! timemindays {timemindays} is too early for light to travel from the entire ejecta "
                f" ({tmin_d_valid:.2f} d)"
            )

        if timemaxdays is None:
            print(f"setting timemaxdays to end of valid observable range {tmax_d_valid:.2f} d")
            timemaxdays = tmax_d_valid
        elif timemaxdays > tmax_d_valid:
            print(
                f"WARNING! timemaxdays {timemaxdays} is too late to receive light from the entire ejecta "
                f" ({tmax_d_valid:.2f} d)"
            )
        dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(timemindays, timemaxdays))

    assert timemindays is not None
    assert timemaxdays is not None

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    dfpackets = at.packets.bin_packet_directions_lazypolars(
        dfpackets=dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins, phibintype="phibinmonotonicasc"
    )

    # for figuring out where the axes are on the plot, make a cut
    # dfpackets = dfpackets.filter(pl.col("dirz") > 0.9)

    aggs = []
    if nnelement_vars := [var for var in plotvars if var.startswith("nnelement_")]:
        aggs += [
            ((pl.col(var) * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800).alias(var)
            for var in nnelement_vars
        ]

    if "emvelocityoverc" in plotvars:
        aggs.append(
            ((pl.col("emission_velocity") * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800).alias(
                "emvelocityoverc"
            )
        )

    if "emvelocityoverc_sigma" in plotvars:
        aggs.append(((pl.col("emission_velocity") / 29979245800).std()).alias("emvelocityoverc_sigma"))

    if "emlosvelocityoverc" in plotvars:
        aggs.append(
            (
                (pl.col("emission_velocity_lineofsight") * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800
            ).alias("emlosvelocityoverc")
        )

    if "luminosity" in plotvars:
        solidanglefactor = nphibins * ncosthetabins
        aggs.append(
            (pl.col("e_rf").sum() / nprocs_read * solidanglefactor / (timemaxdays - timemindays) / 86400).alias(
                "luminosity"
            )
        )

    if "temperature" in plotvars or "temperature_sigma" in plotvars or nnelement_vars:
        timebins = [tstart * 86400.0 for tstart in at.get_timestep_times(modelpath, loc="start")] + [
            at.get_timestep_times(modelpath, loc="end")[-1] * 86400.0
        ]
        dfpackets = dfpackets.with_columns(
            (
                pl.col("em_time")
                .cut(breaks=timebins, labels=[str(x) for x in range(-1, len(timebins))])
                .cast(str)
                .cast(pl.Int32)
            ).alias("em_timestep")
        )

        assert dfestimators is not None
        dfestimators = (
            dfestimators.select(["timestep", "modelgridindex", "TR", *nnelement_vars])
            .drop_nulls()
            .rename({"timestep": "em_timestep", "modelgridindex": "em_modelgridindex"})
        )
        dfpackets = dfpackets.join(dfestimators, on=["em_timestep", "em_modelgridindex"], how="left")

    if "temperature" in plotvars:
        aggs.append(((pl.col("TR") * pl.col("e_rf")).mean() / pl.col("e_rf").mean()).alias("temperature"))

    if "temperature_sigma" in plotvars:
        aggs.append((pl.col("TR").std()).alias("temperature_sigma"))

    if atomic_number is not None or ion_stage is not None:
        dflinelist = at.get_linelist_pldf(modelpath)
        elem_cond = f"Z={atomic_number} {at.get_elsymbol(atomic_number)}" if atomic_number is not None else ""
        ion_stage_cond = f"ion stage {ion_stage}" if ion_stage is not None else ""
        condition = f"last emitted/absorbed by {elem_cond} {ion_stage_cond}"
        print(f"Including only packets {condition}")
        if atomic_number is not None:
            dflinelist = dflinelist.filter(pl.col("atomic_number") == atomic_number)
        if ion_stage is not None:
            dflinelist = dflinelist.filter(pl.col("ion_stage") == ion_stage)

        selected_emtypes = dflinelist.select("lineindex").collect().get_column("lineindex")
        dfpackets = dfpackets.filter(
            pl.col("emissiontype").is_in(selected_emtypes) | pl.col("absorption_type").is_in(selected_emtypes)
        )

    aggs.append(pl.len().alias("count"))
    dfpackets = (
        dfpackets.group_by(["costhetabin", "phibinmonotonicasc"])
        .agg(aggs)
        .select(["costhetabin", "phibinmonotonicasc", "count", *plotvars])
    )

    ndirbins = nphibins * ncosthetabins
    alldirbins = (
        pl.DataFrame(
            {
                "phibinmonotonicasc": (d % nphibins for d in range(ndirbins)),
                "costhetabin": (d // nphibins for d in range(ndirbins)),
            },
            schema={"phibinmonotonicasc": pl.Int32, "costhetabin": pl.Int32},
        )
        .lazy()
        .join(dfpackets, how="left", on=["costhetabin", "phibinmonotonicasc"], coalesce=True)
        .fill_null(0)
        .sort(["costhetabin", "phibinmonotonicasc"])
    ).collect()

    print(f"packets plotted: {alldirbins.select('count').sum().item(0, 0):.1e}")

    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphibins + 1, endpoint=True, dtype=np.float64)

    # costhetabin zero is (0,0,-1) so theta angle
    costhetagrid = np.linspace(-1, 1, ncosthetabins + 1, endpoint=True, dtype=np.float64)
    # for Molleweide projection, theta range is [-pi/2, +pi/2]
    thetagrid = np.pi / 2 - np.arccos(costhetagrid)

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)

    xwidth = float(figscale * at.get_config()["figwidth"])
    fig, axes = plt.subplots(
        len(plotvars),
        1,
        figsize=(xwidth, xwidth * 0.715 * len(plotvars)),
        subplot_kw={"projection": "mollweide"},
        layout="constrained",
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    if len(plotvars) == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    # for ax, axcbar, plotvar in zip(axes[::2], axes[1::2], plotvars):
    for ax, plotvar in zip(axes, plotvars, strict=False):
        data = alldirbins.get_column(plotvar).to_numpy().reshape((ncosthetabins, nphibins))

        if gaussian_sigma is not None and gaussian_sigma > 0:
            import scipy.ndimage

            sigma_bins = gaussian_sigma / 360 * nphibins
            data = scipy.ndimage.gaussian_filter(data, sigma=sigma_bins, mode="wrap")

        colormesh = ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True, cmap=cmap)

        match plotvar:
            case "emlosvelocityoverc":
                colorbartitle = r"Mean line of sight velocity [c]"
            case "emvelocityoverc":
                colorbartitle = r"Last interaction ejecta velocity [c]"
            case "emvelocityoverc_sigma":
                colorbartitle = r"Last interaction ejecta velocity standard deviation [c]"
            case "luminosity":
                colorbartitle = r"Radiant intensity $\cdot\,4π$ [{}erg/s]"
            case "temperature":
                colorbartitle = r"Temperature [{}K]"
            case "temperature_sigma":
                colorbartitle = r"Temperature standard deviation [{}K]"
            case s if s.startswith("nnelement_"):
                elemsymbol = s.split("_", maxsplit=1)[1]
                colorbartitle = f"Number density of {elemsymbol} " + r"[{}/cm³]"
            case _:
                msg = f"Unknown plotvar {plotvar}"
                raise AssertionError(msg)

        cbar = fig.colorbar(colormesh, location="bottom", pad=0.03, ax=ax, shrink=0.95)
        cbar.outline.set_linewidth(0)  # type: ignore[operator] # pyright: ignore[reportCallIssue]
        cbar.ax.tick_params(axis="both", direction="out")
        cbar.ax.xaxis.set_ticks_position("top")
        # cbar.ax.set_title(colorbartitle)
        cbar.ax.set_xlabel(colorbartitle)
        cbar.ax.xaxis.set_label_position("top")
        if r"{}" in colorbartitle:
            cbar.ax.xaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(colorbartitle))

        # ax.set_xlabel("Azimuthal angle")
        # ax.set_ylabel("Polar angle")
        # ax.set_xlabel(r"$\phi$")
        # ax.set_ylabel(r"$\theta$")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.grid(visible=True, color="black")
        ax.axis("off")
        # xticks_deg = np.arange(0, 360, 90)[1:]
        # ax.set_xticks(ticks=xticks_deg / 180 * np.pi - np.pi, labels=[rf"${deg:.0f}\degree$" for deg in xticks_deg])

        # yticks_deg = np.linspace(0, 180, 7)
        # ax.set_yticks(
        #     ticks=-yticks_deg / 180 * np.pi + np.pi / 2.0, labels=[rf"${deg:.0f}\degree$" for deg in yticks_deg]
        # )

    return fig, axes, timemindays, timemaxdays, condition


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path to ARTIS folder")
    parser.add_argument("-timestep", "-ts", action="store", type=str, default=None, help="Timestep index")
    parser.add_argument("-timemin", "-tmin", action="store", type=float, default=None, help="Time minimum [d]")
    parser.add_argument("-timemax", "-tmax", action="store", type=float, default=None, help="Time maximum [d]")
    parser.add_argument("-nphibins", action="store", type=int, default=64, help="Number of azimuthal bins")
    parser.add_argument("-ncosthetabins", action="store", type=int, default=32, help="Number of polar angle bins")
    parser.add_argument("-maxpacketfiles", type=int, default=None, help="Limit the number of packet files read")
    parser.add_argument("-gaussian_sigma", type=int, default=None, help="Apply Gaussian filter")
    parser.add_argument(
        "-plotvars",
        default=["luminosity", "emvelocityoverc", "emlosvelocityoverc"],
        choices=["luminosity", "emvelocityoverc", "emlosvelocityoverc", "temperature"],
        nargs="+",
        help="Variable to plot: luminosity, emvelocityoverc, emlosvelocityoverc, temperature",
    )
    parser.add_argument("-elem", type=str, default=None, help="Filter emitted packets by element of last emission")
    parser.add_argument(
        "-atomic_number", type=int, default=None, help="Filter emitted packets by element of last emission"
    )
    parser.add_argument(
        "-ion_stage", type=int, default=None, help="Filter emitted packets by ionistion stage of last emission"
    )
    parser.add_argument("-cmap", default=None, type=str, help="Matplotlib color map name")

    parser.add_argument(
        "-figscale", type=float, default=1.0, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("--makegif", action="store_true", help="Make a gif with time evolution")

    parser.add_argument(
        "-o", action="store", dest="outputfile", type=str, default="", help="Filename for plot output file"
    )

    parser.add_argument("-format", "-f", default="", choices=["pdf", "png"], help="Set format of output plot files")


def main(args: argparse.Namespace | None = None, argsraw: list[str] | None = None, **kwargs: t.Any) -> None:
    """Plot direction maps based on escaped packets."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    if not args.modelpath:
        args.modelpath = Path()

    if args.elem is not None:
        assert args.atomic_number is None
        args.atomic_number = at.get_atomic_number(args.elem)

    dfmodel, modelmeta = at.get_modeldata(modelpath=args.modelpath, getheadersonly=True, printwarningsonly=True)
    dfestimators = at.estimators.scan_estimators(modelpath=args.modelpath) if "temperature" in args.plotvars else None

    nprocs_read, dfpackets = at.packets.get_packets_pl(
        args.modelpath, args.maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelmeta=modelmeta, dfmodel=dfmodel)

    tstarts = at.get_timestep_times(args.modelpath, loc="start")
    tends = at.get_timestep_times(args.modelpath, loc="end")
    if args.makegif:
        time_ranges = [
            (tstart, tend, f"timestep {ts}")
            for ts, (tstart, tend) in enumerate(zip(tstarts, tends, strict=False))
            if ((args.timemin is None or tstart >= args.timemin) and (args.timemax is None or tend <= args.timemax))
        ]
        outformat = "png"
    elif args.timestep is not None:
        time_ranges = [
            (tstarts[int(ts)], tends[int(ts)], f"timestep {ts}") for ts in at.parse_range_list(args.timestep)
        ]
        outformat = args.format or "pdf"
    else:
        time_ranges = [(args.timemin, args.timemax, "")]
        outformat = args.format or "pdf"

    print_theta_phi_definitions()
    outputfilenames = []
    for tstart, tend, label in time_ranges:
        if tend is not None:
            print(f"Plotting spherical map for {tstart:.2f}-{tend:.2f} days {label}")
        # tstart and tend are requested, but the actual plotted time range may be different
        fig, axes, timemindays, timemaxdays, condition = plot_spherical(
            modelpath=args.modelpath,
            dfpackets=dfpackets,
            dfestimators=dfestimators,
            nprocs_read=nprocs_read,
            timemindays=tstart,
            timemaxdays=tend,
            nphibins=args.nphibins,
            ncosthetabins=args.ncosthetabins,
            gaussian_sigma=args.gaussian_sigma,
            atomic_number=args.atomic_number,
            ion_stage=args.ion_stage,
            plotvars=args.plotvars,
            cmap=args.cmap,
            figscale=args.figscale,
        )

        axes[0].set_title(
            f"{timemindays:.2f}-{timemaxdays:.2f} days{f' ({condition})' if condition else ''}", loc="left", pad=0
        )

        defaultfilename = "plotspherical_{timemindays:.2f}-{timemaxdays:.2f}d.{outformat}"  # noqa: RUF027
        outfilename = str(
            args.outputfile
            if (args.outputfile and not Path(args.outputfile).is_dir() and not args.makegif)
            else Path(args.outputfile) / defaultfilename
        ).format(timemindays=timemindays, timemaxdays=timemaxdays, outformat=outformat)

        fig.savefig(outfilename, format=outformat, dpi=300, pad_inches=0.0)
        print(f"Saved {outfilename}")
        plt.close()
        plt.clf()

        outputfilenames.append(outfilename)

    if args.makegif:
        import imageio.v2 as iio

        gifname = (
            Path(args.outputfile) / "sphericalplot.gif"
            if Path(args.outputfile).is_dir()
            else args.outputfile.format(outformat=outformat)
        )
        with iio.get_writer(gifname, mode="I", duration=(1000 * 1 / 1.5)) as writer:
            for filename in outputfilenames:
                image = iio.imread(filename)
                writer.append_data(image)  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
        print(f"Created gif: {gifname}")


if __name__ == "__main__":
    main()
