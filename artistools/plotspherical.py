#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# mypy: warn-unused-configs, disallow-any-generics, disallow-subclassing-any, disallow-untyped-calls,
# mypy: disallow-untyped-defs, disallow-incomplete-defs, check-untyped-defs, disallow-untyped-decorators,
# mypy: warn-redundant-casts, warn-unused-ignores, warn-return-any, no-implicit-reexport, strict-equality, strict-concatenate
from __future__ import annotations

import argparse
import typing as t
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typeguard import typechecked

import artistools as at


@typechecked
def plot_spherical(
    modelpath: str | Path,
    timemindays: float | None,
    timemaxdays: float | None,
    nphibins: int,
    ncosthetabins: int,
    maxpacketfiles: int | None = None,
    atomic_number: int | None = None,
    ion_stage: int | None = None,
    gaussian_sigma: int | None = None,
    plotvars: list[str] | None = None,
    figscale: float = 1.0,
    cmap: str | None = None,
) -> tuple[plt.Figure, np.ndarray[t.Any, np.dtype[plt.Axes]]]:
    if plotvars is None:
        plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc"]

    _, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=True, printwarningsonly=True)

    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    _, tmin_d_valid, tmax_d_valid = at.get_escaped_arrivalrange(modelpath)
    if tmin_d_valid is None or tmax_d_valid is None:
        print("WARNING! The observer never gets light from the entire ejecta. Plotting all packets anyway")
        timemindays, timemaxdays = (
            dfpackets.select(pl.col("t_arrive_d").min().alias("tmin"), pl.col("t_arrive_d").max().alias("tmax"))
            .collect()
            .to_numpy()[0]
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
                f"WARNING! timemaxdays {timemaxdays} is too late to recieve light from the entire ejecta "
                f" ({tmax_d_valid:.2f} d)"
            )
        dfpackets = dfpackets.filter((pl.col("t_arrive_d") >= timemindays) & (pl.col("t_arrive_d") <= timemaxdays))

    assert timemindays is not None
    assert timemaxdays is not None

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    dfpackets = at.packets.bin_packet_directions_lazypolars(
        dfpackets=dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins, phibintype="monotonic"
    )

    # for figuring out where the axes are on the plot, make a cut
    # dfpackets = dfpackets.filter(pl.col("dirz") > 0.9)

    aggs = []
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelmeta=modelmeta)

    if "emvelocityoverc" in plotvars:
        aggs.append(
            ((pl.col("emission_velocity") * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800).alias(
                "emvelocityoverc"
            )
        )

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

    if "temperature" in plotvars:
        timebins = [
            *at.get_timestep_times(modelpath, loc="start") * 86400.0,
            at.get_timestep_times(modelpath, loc="end")[-1] * 86400.0,
        ]
        dfpackets = dfpackets.with_columns(
            (
                pl.col("em_time")
                .cut(breaks=list(timebins), labels=[str(x) for x in range(-1, len(timebins))])
                .cast(str)
                .cast(pl.Int32)
            ).alias("em_timestep")
        )

        dfest_parquetfile = Path(modelpath, "temperatures.parquet.tmp")

        if not dfest_parquetfile.is_file():
            estimators = at.estimators.read_estimators(
                modelpath,
                get_ion_values=False,
                get_heatingcooling=False,
                skip_emptycells=True,
            )
            assert len(estimators) > 0
            pl.DataFrame(
                {
                    "timestep": (tsmgi[0] for tsmgi in estimators),
                    "modelgridindex": (tsmgi[1] for tsmgi in estimators),
                    "TR": (estimators[tsmgi].get("TR", -1) for tsmgi in estimators),
                },
            ).filter(pl.col("TR") >= 0).with_columns(pl.col(pl.Int64).cast(pl.Int32)).write_parquet(
                dfest_parquetfile, compression="zstd"
            )

        df_estimators = pl.scan_parquet(dfest_parquetfile).rename(
            {"timestep": "em_timestep", "modelgridindex": "em_modelgridindex", "TR": "em_TR"}
        )
        dfpackets = dfpackets.join(df_estimators, on=["em_timestep", "em_modelgridindex"], how="left")
        aggs.append(((pl.col("em_TR") * pl.col("e_rf")).mean() / pl.col("e_rf").mean()).alias("temperature"))

    if atomic_number is not None or ion_stage is not None:
        dflinelist = at.get_linelist_pldf(modelpath)
        if atomic_number is not None:
            print(f"Including only packets emitted by Z={atomic_number} {at.get_elsymbol(atomic_number)}")
            dflinelist = dflinelist.filter(pl.col("atomic_number") == atomic_number)
        if ion_stage is not None:
            print(f"Including only packets emitted by ionisation stage {ion_stage}")
            dflinelist = dflinelist.filter(pl.col("ion_stage") == ion_stage)

        selected_emtypes = dflinelist.select("lineindex").collect().get_column("lineindex")
        dfpackets = dfpackets.filter(pl.col("emissiontype").is_in(selected_emtypes))

    aggs.append(pl.count())

    dfpackets = dfpackets.groupby(["costhetabin", "phibin"]).agg(aggs)
    dfpackets = dfpackets.select(["costhetabin", "phibin", "count", *plotvars])

    ndirbins = nphibins * ncosthetabins
    alldirbins = pl.DataFrame(
        {"phibin": (d % nphibins for d in range(ndirbins)), "costhetabin": (d // nphibins for d in range(ndirbins))}
    ).with_columns(pl.all().cast(pl.Int32))
    alldirbins = (
        alldirbins.join(
            dfpackets.collect(),
            how="left",
            on=["costhetabin", "phibin"],
        )
        .fill_null(0)
        .sort(["costhetabin", "phibin"])
    )

    print(f'total packets contributed: {alldirbins.select("count").sum().to_numpy()[0][0]:.1e}')

    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphibins + 1, endpoint=True, dtype=np.float64)

    # costhetabin zero is (0,0,-1) so theta angle
    costhetagrid = np.linspace(-1, 1, ncosthetabins + 1, endpoint=True, dtype=np.float64)
    # for Molleweide projection, theta range is [-pi/2, +pi/2]
    thetagrid = np.pi / 2 - np.arccos(costhetagrid)

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)

    fig, axes = plt.subplots(
        len(plotvars),
        1,
        figsize=(figscale * at.get_config()["figwidth"], 3.7 * len(plotvars)),
        subplot_kw={"projection": "mollweide"},
        tight_layout={"pad": 0.1, "w_pad": 0.0, "h_pad": 0.0},
    )

    if len(plotvars) == 1:
        axes = np.array([axes])

    for ax, plotvar in zip(axes, plotvars):
        data = alldirbins.get_column(plotvar).to_numpy().reshape((ncosthetabins, nphibins))

        if gaussian_sigma is not None and gaussian_sigma > 0:
            import scipy.ndimage

            sigma_bins = gaussian_sigma / 360 * nphibins
            data = scipy.ndimage.gaussian_filter(data, sigma=sigma_bins, mode="wrap")

        colormesh = ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True, cmap=cmap)

        if plotvar == "emlosvelocityoverc":
            colorbartitle = r"Mean line of sight velocity [c]"
        elif plotvar == "emvelocityoverc":
            colorbartitle = r"Last interaction ejecta velocity [c]"
        elif plotvar == "luminosity":
            colorbartitle = r"Radiant intensity $\cdot\,4π$ [{}erg/s]"
        elif plotvar == "temperature":
            colorbartitle = r"Temperature [{}K]"
        else:
            raise AssertionError

        cbar = fig.colorbar(colormesh, ax=ax, location="bottom")
        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(axis="both", direction="out")
        cbar.ax.xaxis.set_ticks_position("top")
        # cbar.ax.set_title(colorbartitle)
        cbar.ax.set_xlabel(colorbartitle)
        cbar.ax.xaxis.set_label_position("top")
        if r"{}" in colorbartitle:
            cbar.ax.xaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(colorbartitle, useMathText=True))

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

    return fig, axes


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath",
        type=Path,
        default=Path(),
        help="Path to ARTIS folder",
    )
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

    parser.add_argument(
        "-o",
        action="store",
        dest="outputfile",
        type=Path,
        default=Path("plotspherical.pdf"),
        help="Filename for PDF file",
    )


def main(args: argparse.Namespace | None = None, argsraw: list[str] | None = None, **kwargs: t.Any) -> None:
    """Plot direction maps based on escaped packets."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ["."]

    if args.elem is not None:
        assert args.atomic_number is None
        args.atomic_number = at.get_atomic_number(args.elem)

    fig, axes = plot_spherical(
        modelpath=args.modelpath,
        timemindays=args.timemin,
        timemaxdays=args.timemax,
        nphibins=args.nphibins,
        ncosthetabins=args.ncosthetabins,
        maxpacketfiles=args.maxpacketfiles,
        gaussian_sigma=args.gaussian_sigma,
        atomic_number=args.atomic_number,
        ion_stage=args.ion_stage,
        plotvars=args.plotvars,
        cmap=args.cmap,
        figscale=args.figscale,
    )

    fig.savefig(args.outputfile)
    print(f"Saved {args.outputfile}")


if __name__ == "__main__":
    main()
