"""Plot 2D histograms of where in the ejecta packets were last emitted or scattered."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.atomic import decode_roman_numeral
from artistools.atomic import get_atomic_number
from artistools.atomic import get_lineindices
from artistools.constants import c_ang_per_s
from artistools.constants import C_cm_per_s as CLIGHT
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.linecontributions import add_velocity_bin_args
from artistools.linecontributions import convert_velocity_columns
from artistools.linecontributions import format_velocity
from artistools.linecontributions import get_velocity_bin_edges_kmps
from artistools.linecontributions import get_velocity_bin_index_expr
from artistools.linecontributions import get_velocity_unit
from artistools.linecontributions import INVERSE_VELOCITYUNIT_AXISLABELS
from artistools.linecontributions import KMPS_PER_VELOCITYUNIT
from artistools.linecontributions import VELOCITYUNIT_AXISLABELS
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_unsupported
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import resolve_outputfile
from artistools.misc.dirbins import get_dirbin_labels
from artistools.packets.core import filter_packets_dirbin
from artistools.packets.core import get_emission_time_expr
from artistools.packets.core import get_emission_velocity_expr
from artistools.packets.core import get_packets
from artistools.plottools import get_next_color
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend
from artistools.plottools import set_mpl_style
from artistools.rustext import get_bin_indices
from artistools.rustext import sum_weights_in_bins

if t.TYPE_CHECKING:
    import matplotlib.axes as mplax


def get_required_packets(
    modelpath: Path,
    Z_list: Sequence[int] | None,
    ion_stage_list: Sequence[int] | None,
    srII_triplet: bool = False,
    *,
    onlylineabsorptions: bool = True,
) -> tuple[int, pl.LazyFrame]:
    """Return the escaped packets that a line of the given elements and ion stages last absorbed.

    A None list selects every element or every ion stage. The Sr II triplet takes the place of both lists. With
    onlylineabsorptions=False and no list, the function returns every escaped r-packet.
    """
    # careful: ion_stage is counted from 1 here, i.e. 1 <-> neutral, 2 <-> singly ionized
    nprocs_read, dfpackets = get_packets(
        modelpath=modelpath, maxpacketfiles=None, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if not srII_triplet and Z_list is ion_stage_list is None:
        if not onlylineabsorptions:
            return nprocs_read, dfpackets
        # the plot keeps every line, thus the filter needs no list of line indices. A packet
        # that no line absorbed has a negative absorption_type
        return nprocs_read, dfpackets.filter(pl.col("absorption_type") >= 0)

    if srII_triplet:
        lineindices = get_lineindices(
            modelpath,
            [38],
            [2],
            linefilter=((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 3))
            | ((pl.col("lowerlevelindex") == 2) & (pl.col("upperlevelindex") == 4))
            | ((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 4)),
        )
    else:
        lineindices = get_lineindices(modelpath, Z_list, ion_stage_list)

    return nprocs_read, dfpackets.filter(pl.col("absorption_type").is_in(lineindices.implode()))


def get_reduced_packet_set(
    modelpath: Path,
    dirbin: int,
    Z: Sequence[int] | None,
    ion_stage: Sequence[int] | None,
    wavelen: float | None = None,
    binwidth: float | None = None,
    srII_triplet: bool = False,
    *,
    absorption: bool = False,
    onlylineabsorptions: bool = True,
) -> tuple[int, pl.LazyFrame, float]:
    """Get packets in specific escape angle bins for observer direction.

    The function returns the number of MPI ranks, the packets, and the solid-angle factor
    (4 pi / solidangle) of the direction bin. `get_required_packets()` selects the packets for the
    given element and ion filters. If `wavelen` and `binwidth` both have a value, the function keeps
    only the packets in that wavelength slice. The slice applies to the rest-frame wavelength at the escape, or with
    absorption, at the last absorption. The function then keeps only the packets of the direction bin.
    """
    nprocs_read, dfpackets_selected = get_required_packets(
        modelpath, Z, ion_stage, srII_triplet=srII_triplet, onlylineabsorptions=onlylineabsorptions
    )
    frequencycolumn = "absorption_freq" if absorption else "nu_rf"
    dfpackets_selected = dfpackets_selected.with_columns((c_ang_per_s / pl.col(frequencycolumn)).alias("lambda_rf"))

    if wavelen is not None and binwidth is not None:
        lam_min = wavelen - binwidth / 2
        lam_max = wavelen + binwidth / 2

        dfpackets_selected = dfpackets_selected.filter(
            (pl.col("lambda_rf") > lam_min) & (pl.col("lambda_rf") < lam_max)
        )
    dfpackets_selected, solidangle_factor = filter_packets_dirbin(dfpackets_selected, dirbin, average_over_phi=True)

    return nprocs_read, dfpackets_selected, solidangle_factor


def packets_2d_hist_bin_and_ejecta_vel(
    modelpath: Path,
    tdays: float,
    srIItriplet: bool,
    colorlogscale: bool,
    dirbin: int,
    trueem: bool,
    Z: int | None = None,
    ion_stage_str: str | None = None,
    wavelen: float | None = None,
    binwidth: float | None = None,
    outputfile: Path | None = None,
    args: argparse.Namespace | None = None,
    *,
    absorption: bool = False,
) -> None:
    """Plot a 2D histogram of packet emission position against ejecta velocity, and save the figure.

    An outputfile of None or of a folder gives the file a name from the selection of the packets. With absorption, the
    wavelength slice applies to the wavelength of the last absorption.
    """
    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if wavelen is not None:
        start_of_filename = f"{start_of_filename}{wavelen:.0f}A_"
    start_of_filename = f"{start_of_filename}Z={Z}_" if Z else f"{start_of_filename}allelements_"
    start_of_filename = f"{start_of_filename}I={ion_stage_str}_" if ion_stage_str else f"{start_of_filename}allions_"

    # Step 1) collect packets IDs and select according to arrival time. None selects every element or ion stage
    Z_list = [Z] if Z else None
    ion_stage_list = [decode_roman_numeral(ion_stage_str)] if ion_stage_str else None
    if ion_stage_list == [-1]:
        exit_with_error(f"-ionstage {ion_stage_str} is no Roman numeral", "Give the ion stage as e.g. II")

    nprocs_read, dfpackets, inverse_solidangle_fraction = get_reduced_packet_set(
        modelpath,
        dirbin,
        Z_list,
        ion_stage_list,
        wavelen=wavelen,
        binwidth=binwidth,
        srII_triplet=srIItriplet,
        absorption=absorption,
    )

    start_of_filename += f"t_arrive_d_{tdays}_"
    timestep, t_min, t_max = get_arrival_timestep(modelpath, tdays)
    Delta_t_secs = (t_max - t_min) * day_to_s
    Delta_beta = 0.5 / 25

    position: t.Literal["em", "trueem"] = "em"
    if trueem:
        required_cols = {"trueem_posx", "trueem_posy", "trueem_posz", "trueem_time"}
        missing_cols = required_cols - set(dfpackets.collect_schema().names())
        if missing_cols:
            message = (
                "--use_thermalemissiontype requires packets with columns "
                f"{sorted(required_cols)} (missing {sorted(missing_cols)})"
            )
            raise ValueError(message)
        position = "trueem"
    print(f"t_min selected: {t_min} t_max_selected: {t_max}, is {Delta_t_secs} seconds")
    # a timestep holds the times from its start up to its end, and the end belongs to the next timestep
    dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="left"))
    # a packet with no record of the emission has a time of NaN, thus it is outside each bin of the histogram
    emtime = get_emission_time_expr(position)
    dfpackets = dfpackets.with_columns(
        ((pl.col(f"{position}_posx") ** 2 + pl.col(f"{position}_posy") ** 2).sqrt() / emtime / CLIGHT).alias(
            "beta_r_cyl_em"
        )
    ).with_columns((pl.col(f"{position}_posz") / emtime / CLIGHT).alias("beta_z_em"))

    dfpackets = dfpackets.with_columns(
        ((pl.col("beta_r_cyl_em") / Delta_beta).floor() * Delta_beta * CLIGHT * emtime).alias("R_cyl_inner_em")
    ).with_columns((pl.col("R_cyl_inner_em") + Delta_beta * CLIGHT * emtime).alias("R_cyl_outer_em"))
    dfpackets_selected = dfpackets.with_columns(
        (
            np.pi
            * (pl.col("R_cyl_outer_em").cast(pl.Float64) ** 2 - pl.col("R_cyl_inner_em").cast(pl.Float64) ** 2)
            * CLIGHT
            * Delta_beta
            * emtime
        ).alias("hollow_cyl_vol_em")
    ).collect()
    energy_sum = float(dfpackets_selected["e_rf"].sum())
    print(
        "Directional 4pi-equivalent bol. luminosity of"
        f" {energy_sum / nprocs_read / Delta_t_secs * inverse_solidangle_fraction}"
    )

    # Step 2) create the heatmap. Normalise packet energy to modelgrid cell volume at packet emission time (lab frame)
    # the kernel gives the bins of np.histogram2d. np.histogram2d took 0.13 s for 2.5 million packets at 2 days of
    # a 3D kilonova model. The kernel made the command 0.11 s faster
    xedges = np.linspace(0, 0.5, num=26)
    yedges = np.linspace(-0.5, 0.5, num=51)
    xbinindex = get_bin_indices(dfpackets_selected.select("beta_r_cyl_em"), "beta_r_cyl_em", xedges.tolist())[
        "binindex"
    ]
    dfinxrange = dfpackets_selected.select(
        pl.col("beta_z_em").cast(pl.Float64),
        weight=(pl.col("e_rf") / pl.col("hollow_cyl_vol_em")).cast(pl.Float64),
        xbinindex=xbinindex,
    ).filter(pl.col("xbinindex") >= 0)
    hist2D = (
        sum_weights_in_bins(dfinxrange, "beta_z_em", "weight", yedges.tolist(), "xbinindex", len(xedges) - 1)["sum"]
        .to_numpy()
        .reshape(len(xedges) - 1, len(yedges) - 1)
    )
    heatmap = hist2D / Delta_t_secs / nprocs_read * inverse_solidangle_fraction
    heatmap = np.ma.masked_less_equal(heatmap, 0.0)
    if colorlogscale:
        heatmap = np.ma.log10(heatmap)

    # an image with a colorbar keeps plt.subplots: fig.colorbar takes space that the
    # Divider of make_frame_figure gives back at draw time, and the two then overlap
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(3.5, 4.5), layout="constrained")
    z = heatmap.T

    im = ax.imshow(z, origin="lower", cmap="viridis", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$v_r$ [$c$]")
    ax.set_ylabel(r"$v_z$ [$c$]")
    cbar = fig.colorbar(im, ax=ax)
    if colorlogscale:
        cbar.set_label(r"log$_{10}$ volumetric emissivity [erg/(s cm$^3$)]")
    else:
        cbar.set_label(r"volumetric emissivity [erg/(s cm$^3$)]")

    ax.set_xticks(np.linspace(xedges[0], xedges[-1], 6))
    ax.set_yticks(np.linspace(yedges[0], yedges[-1], 6))

    outfilename = resolve_outputfile(outputfile, start_of_filename + f"ts{timestep}_into_dirbin{dirbin}.pdf")
    if getattr(args, "print_data", False) or getattr(args, "write_data", False):
        assert args is not None
        valuecolumn = "log10_emissivity_erg_per_s_per_cm3" if colorlogscale else "emissivity_erg_per_s_per_cm3"
        output_data(args, {"": get_heatmap_table(z, xedges, yedges, valuecolumn)}, outfilename)
    save_figure(fig, outfilename, dpi=300, args=args)


def get_arrival_timestep(modelpath: Path, tdays: float) -> tuple[int, float, float]:
    """Return the timestep that holds tdays, and the start and the end of that timestep [days]."""
    timestep = get_timestep_of_timedays(modelpath, tdays)
    t_min = get_timestep_times(modelpath=modelpath, loc="start")[timestep]
    t_max = get_timestep_times(modelpath=modelpath, loc="end")[timestep]
    return timestep, t_min, t_max


def get_group_velocity_profiles(
    dfpackets: pl.DataFrame,
    typecolumn: str,
    dflabels: pl.DataFrame,
    velocityedges_kmps: Sequence[float],
    maxseriescount: int | None,
) -> tuple[pl.DataFrame, list[str]]:
    """Return the packet energy and the number of packets of each group in each velocity bin, and the drawn groups.

    dfpackets has the columns typecolumn, e_rf, and velocity_kmps. dflabels gives the label of each type code. A
    code with no label and each group after the first maxseriescount groups join the group "Other". A packet
    outside the velocity bins is in no bin.
    """
    from artistools.spectra.core import rank_flux_series_names

    dfgrouped = (
        dfpackets
        .lazy()
        .join(dflabels.lazy(), on=typecolumn, how="left", maintain_order="left")
        .with_columns(
            label=pl.col("label").fill_null("Other"),
            velocitybin=get_velocity_bin_index_expr(pl.col("velocity_kmps"), velocityedges_kmps),
        )
        .group_by("label", "velocitybin")
        .agg(energy=pl.col("e_rf").cast(pl.Float64).sum(), packetcount=pl.len())
        .collect()
    )
    dftotals = dfgrouped.group_by("label").agg(pl.col("energy").sum()).sort("label")
    drawnnames, _ = rank_flux_series_names(
        dict(zip(dftotals["label"].to_list(), dftotals["energy"].to_list(), strict=True)), maxseriescount
    )
    dfprofiles = (
        dfgrouped
        .with_columns(label=pl.when(pl.col("label").is_in(drawnnames)).then("label").otherwise(pl.lit("Other")))
        .group_by("label", "velocitybin")
        .agg(pl.col("energy", "packetcount").sum())
        .sort("label", "velocitybin", nulls_last=True)
    )
    return dfprofiles, drawnnames


def plot_velocity_profiles_by_group(
    modelpath: Path,
    tdays: float,
    dirbin: int,
    groupby: str,
    velocityedges_kmps: Sequence[float],
    *,
    trueem: bool = False,
    absorption: bool = False,
    wavelen: float | None = None,
    binwidth: float | None = None,
    maxseriescount: int | None = 8,
    unit: t.Literal["kmps", "c"] = "kmps",
    outputfile: Path | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Plot the luminosity per unit velocity against the radial velocity of the last interaction, for each group.

    A group is an element, an ion, or a line of the last emission, or with absorption, of the last absorption. The
    last absorption of a packet happens at the position of its last emission. unit is the unit of the velocity axis.
    An outputfile of None or of a folder gives the file a name from the selection of the packets.
    """
    from artistools.spectra.core import get_absorption_type_labels
    from artistools.spectra.core import get_emission_type_labels
    from artistools.spectra.core import get_linelist_label_columns

    position: t.Literal["em", "trueem"] = "trueem" if trueem else "em"
    typecolumn = "absorption_type" if absorption else ("trueemissiontype" if trueem else "emissiontype")
    nprocs_read, lzdfpackets, inverse_solidangle_fraction = get_reduced_packet_set(
        modelpath,
        dirbin,
        None,
        None,
        wavelen=wavelen,
        binwidth=binwidth,
        absorption=absorption,
        onlylineabsorptions=False,
    )
    timestep, t_min, t_max = get_arrival_timestep(modelpath, tdays)
    delta_t_s = (t_max - t_min) * day_to_s
    dfpackets = (
        lzdfpackets
        .filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="left"))
        .select(typecolumn, "e_rf", velocity_kmps=get_emission_velocity_expr(position) / km_to_cm)
        .collect()
    )
    if dfpackets.is_empty():
        exit_with_error(
            f"no escaped r-packet of the selection arrives from {t_min:.2f} to {t_max:.2f} d",
            "Give a different time, a wider wavelength range, or -dirbin -1",
        )

    dflines = get_linelist_label_columns(modelpath, groupby)
    typecodes = dfpackets[typecolumn].unique()
    dflabels = (
        get_absorption_type_labels(typecodes, dflines, groupby)
        if absorption
        else get_emission_type_labels(modelpath, typecodes, dflines, groupby, typecolumn)
    ).select(typecolumn, label=pl.col("absorptiontype_str" if absorption else "emissiontype_str"))
    del dflines

    dfprofiles, drawnnames = get_group_velocity_profiles(
        dfpackets, typecolumn, dflabels, velocityedges_kmps, maxseriescount
    )
    luminosity_factor = inverse_solidangle_fraction / (delta_t_s * nprocs_read)
    print_group_profile_method(
        dfprofiles, dfpackets.height, position, typecolumn, t_min, t_max, velocityedges_kmps, luminosity_factor, unit
    )

    dftable = get_group_profile_table(dfprofiles, velocityedges_kmps, luminosity_factor, [*drawnnames, "Other"])
    fig, axesgrid = make_frame_figure(args, aspect=0.6)
    ax = axesgrid[0, 0]
    draw_group_profiles(ax, dftable, velocityedges_kmps, unit)
    interaction = "absorption" if absorption else ("thermal emission" if trueem else "interaction")
    ax.set_xlabel(f"Radial velocity of the last {interaction} [{VELOCITYUNIT_AXISLABELS[unit]}]")
    ax.set_ylabel(rf"$\Delta L / \Delta v$ [erg s$^{{-1}}$ {INVERSE_VELOCITYUNIT_AXISLABELS[unit]}]")
    titleparts = [f"{get_model_name(modelpath)} {(t_min + t_max) / 2:.2f} d"]
    if wavelen is not None and binwidth is not None:
        titleparts.append(f"{wavelen - binwidth / 2:.0f}-{wavelen + binwidth / 2:.0f} Å")
    if dirbin != -1:
        titleparts.append(get_dirbin_labels([dirbin], average_over_phi=True, usedegrees=True)[dirbin])
    ax.set_title(", ".join(titleparts))
    set_legend(ax, args)

    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if wavelen is not None:
        start_of_filename = f"{start_of_filename}{wavelen:.0f}A_"
    kind = "absorption" if absorption else "emission"
    outfilename = resolve_outputfile(
        outputfile, f"{start_of_filename}{kind}_by{groupby}_ts{timestep}_into_dirbin{dirbin}.pdf"
    )
    if args is not None:
        output_data(args, {"": convert_velocity_columns(dftable, unit)}, outfilename)
    save_figure(fig, outfilename, args=args)


def print_group_profile_method(
    dfprofiles: pl.DataFrame,
    packetcount: int,
    position: t.Literal["em", "trueem"],
    typecolumn: str,
    t_min: float,
    t_max: float,
    velocityedges_kmps: Sequence[float],
    luminosity_factor: float,
    unit: t.Literal["kmps", "c"],
) -> None:
    """Print the method of the profiles, and the luminosity, the mean velocity, and the number of packets of each group.

    The mean velocity takes the energy of each bin as the weight and the centre of each bin as the velocity.
    """
    print_detail(
        f"{packetcount} escaped r-packets arrive from {t_min:.2f} to {t_max:.2f} d. The group of a packet is the"
        f" {typecolumn} of the packet"
    )
    if position == "em":
        print_detail(
            "The velocity is |em_pos| / em_time, the radial velocity of the last interaction. An electron scattering"
            " also sets em_pos, thus a packet that scatters after its last emission gives the place of the scattering"
        )
    else:
        print_detail("The velocity is |trueem_pos| / trueem_time, the radial velocity of the last thermal emission")
    nbins = len(velocityedges_kmps) - 1
    binwidth_kmps = velocityedges_kmps[1] - velocityedges_kmps[0]
    print_detail(
        f"{nbins} velocity bins of {format_velocity(binwidth_kmps, unit)} from 0 to"
        f" {format_velocity(velocityedges_kmps[-1], unit)}. The luminosity"
        " is the packet energy divided by the width of the timestep and by the number of MPI ranks. A factor of 4 pi"
        " divided by the solid angle of the direction bin then applies"
    )
    totalenergy = float(dfprofiles["energy"].sum())
    outside = dfprofiles.filter(pl.col("velocitybin").is_null())
    if not outside.is_empty():
        print_detail(
            f"{int(outside['packetcount'].sum())} packets with {float(outside['energy'].sum()) / totalenergy:.1%} of"
            " the energy have a velocity outside the bins, or no record of the position"
        )

    bincentres = pl.col("velocitybin").cast(pl.Float64).add(0.5) * binwidth_kmps
    dfgroups = (
        dfprofiles
        .group_by("label")
        .agg(
            energy=pl.col("energy").sum(),
            packetcount=pl.col("packetcount").sum(),
            meanvelocity=(pl.col("energy") * bincentres).filter(bincentres.is_not_null()).sum()
            / pl.col("energy").filter(bincentres.is_not_null()).sum(),
        )
        .sort("energy", descending=True)
    )
    print_detail(
        "These are the groups, with the mean velocity of their bins. The mean takes the energy of each bin as the"
        " weight:"
    )
    for label, energy, groupcount, meanvelocity in dfgroups.select(
        "label", "energy", "packetcount", "meanvelocity"
    ).iter_rows():
        # a group with all its packets outside the bins has no mean velocity
        velocitytext = (
            f"mean velocity {format_velocity(meanvelocity, unit)}"
            if math.isfinite(meanvelocity)
            else "no packet in the bins"
        )
        print_detail(
            f"  {label}: {energy * luminosity_factor:.3e} erg/s ({energy / totalenergy:.1%}), {velocitytext},"
            f" {groupcount} packets"
        )


def get_group_profile_table(
    dfprofiles: pl.DataFrame, velocityedges_kmps: Sequence[float], luminosity_factor: float, labels: Sequence[str]
) -> pl.DataFrame:
    """Return one row for each group and velocity bin, with the luminosity and the number of packets of the bin.

    A bin with no packet of a group has a luminosity of zero.
    """
    nbins = len(velocityedges_kmps) - 1
    binwidth_kmps = velocityedges_kmps[1] - velocityedges_kmps[0]
    presentlabels = [label for label in labels if label in set(dfprofiles["label"].to_list())]
    dfgrid = pl.DataFrame({"label": presentlabels}).join(
        pl.DataFrame({"velocitybin": range(nbins)}, schema={"velocitybin": pl.Int64}), how="cross"
    )
    return dfgrid.join(dfprofiles, on=["label", "velocitybin"], how="left", maintain_order="left").select(
        "label",
        x_lower_kmps=pl.lit(pl.Series(velocityedges_kmps[:-1])).gather(pl.col("velocitybin")),
        x_upper_kmps=pl.lit(pl.Series(velocityedges_kmps[1:])).gather(pl.col("velocitybin")),
        luminosity_erg_per_s=pl.col("energy").fill_null(0.0) * luminosity_factor,
        luminosity_per_velocity_erg_per_s_per_kmps=pl.col("energy").fill_null(0.0) * luminosity_factor / binwidth_kmps,
        packetcount=pl.col("packetcount").fill_null(0),
    )


def draw_group_profiles(
    ax: "mplax.Axes", dftable: pl.DataFrame, velocityedges_kmps: Sequence[float], unit: t.Literal["kmps", "c"]
) -> None:
    """Draw the sum of all the groups, and one step series for each group, with its fraction of the luminosity.

    dftable is in km/s, and the axis takes the velocity unit unit.
    """
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    edges = np.array(velocityedges_kmps) / kmpsperunit
    yvalues = "luminosity_per_velocity_erg_per_s_per_kmps"
    dftotal = dftable.group_by("x_lower_kmps").agg(pl.col(yvalues).sum()).sort("x_lower_kmps")
    totalluminosity = float(dftable["luminosity_erg_per_s"].sum())
    ax.stairs(
        dftotal[yvalues].to_numpy() * kmpsperunit, edges, label="All packets in the bins", color="black", linewidth=1.5
    )
    for (label,), dfgroup in dftable.group_by("label", maintain_order=True):
        fraction = float(dfgroup["luminosity_erg_per_s"].sum()) / totalluminosity if totalluminosity > 0 else 0.0
        # stairs with `color=None` draws black and not the next colour of the cycle
        color = "grey" if label == "Other" else get_next_color(ax)
        ax.stairs(dfgroup[yvalues].to_numpy() * kmpsperunit, edges, label=f"{label} ({fraction:.1%})", color=color)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)


def get_heatmap_table(
    image: np.ma.MaskedArray, xedges: npt.NDArray[np.floating], yedges: npt.NDArray[np.floating], valuecolumn: str
) -> pl.DataFrame:
    """Return one row for each bin of the image, with the velocity edges of the bin in units of c.

    The image has one row for each v_z bin and one column for each v_r bin. A bin that the image leaves blank has no
    value.
    """
    nzbins, nrbins = image.shape
    # the Fortran order keeps the v_z bins of one v_r bin together
    values = np.ma.filled(image.astype(np.float64), np.nan).ravel(order="F")
    return pl.DataFrame({
        "v_r_min_c": np.repeat(xedges[:-1], nzbins),
        "v_r_max_c": np.repeat(xedges[1:], nzbins),
        "v_z_min_c": np.tile(yedges[:-1], nrbins),
        "v_z_max_c": np.tile(yedges[1:], nrbins),
        valuecolumn: pl.Series(values, nan_to_null=True),
    })


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, required=True, helptext="Path to ARTIS simulation")
    addarg_darkmode(parser)

    addarg_timedays(
        parser,
        kind="float",
        helptext="Time in days. The plot takes the packets that arrive in the timestep of this time",
    )
    parser.add_argument("-tdays", dest="timedays", type=float, help=argparse.SUPPRESS)
    addarg_unsupported(parser, "-timestep", "-ts", instead="-timedays")

    parser.add_argument("-wavelen", type=float, default=None, help="Central wavelength in Angstrom")
    parser.add_argument("-binwidth", type=float, default=None, help="Wavelength bin width in Angstrom")

    parser.add_argument("-element", type=str, default=None, help="Element symbol")
    parser.add_argument("-ionstage", type=str, default=None, help="Ionisation stage (spectroscopic notation)")

    parser.add_argument(
        "-dirbin",
        type=int,
        default=-1,
        help=(
            "The first direction bin of a costheta bin, e.g. 0, 10, or 90. The plot takes all the phi bins of that"
            " costheta bin. The value -1 takes all directions"
        ),
    )
    parser.add_argument(
        "-groupby",
        choices=["element", "ion", "line"],
        help=(
            "Plot the luminosity per unit velocity against the radial velocity of the last interaction. Draw one"
            " series for each element, ion, or line of the last emission. Without this option, the plot is a 2D image"
        ),
    )
    parser.add_argument(
        "--absorption",
        action="store_true",
        help=(
            "Select the packets by the wavelength of their last absorption. With -groupby, group them by the element,"
            " the ion, or the line of that absorption. The last absorption happens at the position of the last"
            " emission"
        ),
    )
    parser.add_argument(
        "-maxseriescount",
        type=int,
        default=8,
        help='Number of groups that -groupby draws. The other groups join the group "Other"',
    )
    add_velocity_bin_args(parser, " of -groupby")
    parser.add_argument("--srIItriplet", action="store_true", help="Plot packets from SrII triplet only")
    parser.add_argument("--colorlogscale", action="store_true", help="Show the log10 of the emissivity in the 2D plot")

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Plot true thermal emission rather than last interaction location",
    )

    addarg_output(parser, kind="file", helptext="Path/filename for the PDF file")
    addarg_data_output(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot last packet interaction properties versus ejecta velocity for selected packets."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if (args.wavelen is None) != (args.binwidth is None):
        message = "Wavelength mode requires both -wavelen and -binwidth to be provided."
        raise ValueError(message)

    if args.timedays is None:
        exit_with_error("the time is missing", "Give the time in days, e.g. -timedays 300")

    nphibins = get_viewingdirection_phibincount()
    ndirbins = get_viewingdirectionbincount()
    if args.dirbin != -1 and not (0 <= args.dirbin < ndirbins and args.dirbin % nphibins == 0):
        exit_with_error(
            f"-dirbin {args.dirbin} is not the first direction bin of a costheta bin",
            f"Give -1 for all directions, or a multiple of {nphibins} from 0 to {ndirbins - nphibins}",
        )

    # get_atomic_number gives -1 for a text that is no element symbol, and 0 for the free neutron, which has no lines
    atomic_number = get_atomic_number(args.element) if args.element else None
    if atomic_number is not None and atomic_number < 1:
        exit_with_error(f"-element {args.element} is no element symbol", "Give the element as e.g. Sr")

    if args.groupby is not None:
        plot_velocity_profiles_from_args(args)
        return

    packets_2d_hist_bin_and_ejecta_vel(
        Path(args.modelpath),
        args.timedays,
        args.srIItriplet,
        args.colorlogscale,
        dirbin=args.dirbin,
        Z=atomic_number,
        trueem=args.use_thermalemissiontype,
        ion_stage_str=args.ionstage,
        wavelen=args.wavelen,
        binwidth=args.binwidth,
        outputfile=args.outputfile,
        args=args,
        absorption=args.absorption,
    )


def plot_velocity_profiles_from_args(args: argparse.Namespace) -> None:
    """Check the options of -groupby, then plot the velocity profile of each group."""
    for option, isgiven in (
        ("-element", args.element is not None),
        ("-ionstage", args.ionstage is not None),
        ("--srIItriplet", args.srIItriplet),
        ("--colorlogscale", args.colorlogscale),
    ):
        if isgiven:
            exit_with_error(
                f"{option} applies to the 2D image, and -groupby gives one series for each group",
                f"Remove {option}, or remove -groupby",
            )
    if args.absorption and args.use_thermalemissiontype:
        exit_with_error(
            "--absorption takes the last absorption, and --use_thermalemissiontype takes the last thermal emission",
            "Give one of the two options",
        )
    modelpath = Path(args.modelpath)
    unit = get_velocity_unit(modelpath, args.velocityunit)
    plot_velocity_profiles_by_group(
        modelpath,
        args.timedays,
        args.dirbin,
        args.groupby,
        get_velocity_bin_edges_kmps(modelpath, args.nvelocitybins, args.velocitymax, unit),
        trueem=args.use_thermalemissiontype,
        absorption=args.absorption,
        wavelen=args.wavelen,
        binwidth=args.binwidth,
        maxseriescount=args.maxseriescount,
        unit=unit,
        outputfile=args.outputfile,
        args=args,
    )
