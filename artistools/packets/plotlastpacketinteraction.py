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
from artistools.atomic import get_elsymbol
from artistools.atomic import get_ion_tuple
from artistools.atomic import get_ionstring
from artistools.atomic import get_lineindices
from artistools.atomic import roman_numerals
from artistools.constants import c_ang_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_maxpacketfiles
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_plotinvalidpart
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import check_time_range_is_valid
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import get_selected_timestep
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_weighted_quantile_expr
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc.cliutils import CommaJoinAction
from artistools.misc.cliutils import positive_int_arg
from artistools.misc.dirbins import check_costheta_dirbin
from artistools.misc.dirbins import get_costheta_bins
from artistools.misc.dirbins import get_dirbin_labels
from artistools.packets.core import addarg_velocitykind
from artistools.packets.core import filter_packets_dirbin
from artistools.packets.core import get_emission_time_expr
from artistools.packets.core import get_last_interaction_velocity_expr
from artistools.packets.core import get_packets
from artistools.packets.core import get_relative_error_expr
from artistools.packets.core import NOISY_RELATIVE_ERROR
from artistools.packets.core import print_arrival_time_method
from artistools.packets.core import print_last_interaction_convention
from artistools.packets.core import print_velocity_method
from artistools.plottools import addarg_velocitybins
from artistools.plottools import convert_velocity_columns
from artistools.plottools import format_velocity
from artistools.plottools import get_next_color
from artistools.plottools import get_signed_velocity_bin_edges_kmps
from artistools.plottools import get_velocity_bin_edges_kmps
from artistools.plottools import get_velocity_bin_index_expr
from artistools.plottools import get_velocity_unit
from artistools.plottools import INVERSE_VELOCITYUNIT_AXISLABELS
from artistools.plottools import KMPS_PER_VELOCITYUNIT
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend
from artistools.plottools import set_mpl_style
from artistools.plottools import VELOCITYUNIT_AXISLABELS
from artistools.rustext import get_bin_indices
from artistools.rustext import sum_weights_in_bins

if t.TYPE_CHECKING:
    from collections.abc import Mapping

    import matplotlib.axes as mplax


def get_required_packets(
    modelpath: Path,
    ions: Sequence[tuple[int | None, int | None]] | None,
    srII_triplet: bool = False,
    maxpacketfiles: int | None = None,
) -> tuple[int, pl.LazyFrame]:
    """Return the escaped r-packets, or with ions, the escaped r-packets that a line of the ions last absorbed.

    Each item of ions is (Z, ion_stage), and a None in it selects each element or each ion stage. The Sr II triplet
    takes the place of ions.
    """
    nprocs_read, dfpackets = get_packets(
        modelpath=modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if srII_triplet:
        lineindices = get_lineindices(
            modelpath,
            [38],
            [2],
            linefilter=((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 3))
            | ((pl.col("lowerlevelindex") == 2) & (pl.col("upperlevelindex") == 4))
            | ((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 4)),
        )
    elif ions is None:
        return nprocs_read, dfpackets
    else:
        lineindices = get_lineindices(
            modelpath,
            None,
            None,
            linefilter=pl.any_horizontal(
                pl.all_horizontal(
                    pl.lit(Z is None) | (pl.col("atomic_number") == Z),
                    pl.lit(ion_stage is None) | (pl.col("ion_stage") == ion_stage),
                )
                for Z, ion_stage in ions
            ),
        )

    return nprocs_read, dfpackets.filter(pl.col("absorption_type").is_in(lineindices.implode()))


def get_reduced_packet_set(
    modelpath: Path,
    dirbin: int,
    ions: Sequence[tuple[int | None, int | None]] | None,
    lambdarange: tuple[float, float] | None = None,
    srII_triplet: bool = False,
    *,
    absorption: bool = False,
    maxpacketfiles: int | None = None,
) -> tuple[int, pl.LazyFrame, float]:
    """Get packets in specific escape angle bins for observer direction.

    The function returns the number of MPI ranks, the packets, and the solid-angle factor
    (4 pi / solidangle) of the direction bin. `get_required_packets()` selects the packets for the
    ions. A lambdarange keeps only the packets in that range of wavelength [Å], with
    both ends. The range applies to the rest-frame wavelength at the escape, or with absorption, at the last
    absorption. ARTIS records no wavelength for a free-free or a bound-free absorption, thus
    the slice leaves out its packet, as plotspectra does. The function then keeps only the packets of the direction bin.
    """
    from artistools.spectra.core import get_boundbound_absorption_freq_expr

    nprocs_read, dfpackets_selected = get_required_packets(
        modelpath, ions, srII_triplet=srII_triplet, maxpacketfiles=maxpacketfiles
    )
    frequency = get_boundbound_absorption_freq_expr() if absorption else pl.col("nu_rf")
    dfpackets_selected = dfpackets_selected.with_columns((c_ang_per_s / frequency).alias("lambda_rf"))

    if lambdarange is not None:
        dfpackets_selected = dfpackets_selected.filter(pl.col("lambda_rf").is_between(*lambdarange))
    dfpackets_selected, solidangle_factor = filter_packets_dirbin(dfpackets_selected, dirbin, average_over_phi=True)

    return nprocs_read, dfpackets_selected, solidangle_factor


def get_ion_label(ion: tuple[int | None, int | None]) -> str:
    """Return the name of an ion, an element, or an ion stage of the selection of -ions, e.g. "Sr II" or "Sr"."""
    Z, ion_stage = ion
    if Z is None:
        assert ion_stage is not None
        return f"ion stage {roman_numerals[ion_stage]}"
    return get_elsymbol(Z) if ion_stage is None else get_ionstring(Z, ion_stage)


def packets_2d_hist_bin_and_ejecta_vel(
    modelpath: Path,
    tdays: float,
    dirbin: int,
    nbins: int,
    velocitymax_kmps: float | None,
    unit: t.Literal["kmps", "c"],
    *,
    ions: Sequence[tuple[int | None, int | None]] | None,
    srIItriplet: bool,
    logcolorscale: bool,
    trueem: bool,
    lambdarange: tuple[float, float] | None,
    outputfile: Path | None,
    args: argparse.Namespace | None,
    absorption: bool,
) -> None:
    """Plot the observed luminosity per unit volume at the last interaction of the packets, and save the figure.

    The image has nbins pixels of v_cyl = sqrt(v_x^2 + v_y^2) from 0 to velocitymax_kmps, and 2 nbins pixels of v_z
    from minus to plus velocitymax_kmps. A velocitymax_kmps of None takes the largest v_cyl or |v_z| of the last
    interactions. unit is the unit of the axes. An outputfile of None or of a folder gives the file a name from the
    selection of the packets. With absorption, the wavelength range applies to the wavelength of the last absorption.
    """
    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if lambdarange is not None:
        # the slice of absorption selects different packets, thus its files must not replace the files of the escape
        start_of_filename = (
            f"{start_of_filename}{lambdarange[0]:.0f}-{lambdarange[1]:.0f}A_{'absorption_' if absorption else ''}"
        )
    if srIItriplet:
        start_of_filename += "SrIItriplet_"
    elif ions is not None:
        start_of_filename += "-".join(get_ion_label(ion).replace(" ", "") for ion in ions) + "_"
    else:
        start_of_filename += "allions_"

    nprocs_read, lzpackets, inverse_solidangle_fraction = get_reduced_packet_set(
        modelpath,
        dirbin,
        ions,
        lambdarange=lambdarange,
        srII_triplet=srIItriplet,
        absorption=absorption,
        maxpacketfiles=getattr(args, "maxpacketfiles", None),
    )

    start_of_filename += f"t_arrive_d_{tdays}_"
    timestep, t_min, t_max = get_arrival_timestep(modelpath, tdays)
    delta_t_s = (t_max - t_min) * day_to_s

    position: t.Literal["em", "trueem"] = "em"
    if trueem:
        required_cols = {"trueem_posx", "trueem_posy", "trueem_posz", "trueem_time"}
        missing_cols = required_cols - set(lzpackets.collect_schema().names())
        if missing_cols:
            exit_with_error(
                f"the 2D image of --use_thermalemissiontype needs the packet columns {', '.join(sorted(missing_cols))}",
                "Give -groupby, which can take the column true_emission_velocity of an old packets file",
            )
        position = "trueem"
    # a packet with no record of the emission has a time of NaN, thus it is outside each bin of the image
    emtime = get_emission_time_expr(position)
    dfpackets = (
        lzpackets
        # a timestep holds the times from its start up to its end, and the end belongs to the next timestep
        .filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="left"))
        .select(
            pl.col("e_rf").cast(pl.Float64),
            emtime_s=emtime.cast(pl.Float64),
            v_cyl_kmps=(pl.col(f"{position}_posx") ** 2 + pl.col(f"{position}_posy") ** 2).sqrt() / emtime / km_to_cm,
            v_z_kmps=(pl.col(f"{position}_posz") / emtime / km_to_cm).cast(pl.Float64),
        )
        .collect()
    )

    if velocitymax_kmps is None:
        fastest_kmps = dfpackets.select(pl.max_horizontal("v_cyl_kmps", pl.col("v_z_kmps").abs()).fill_nan(None).max())
        if fastest_kmps.item() is None:
            exit_with_error("no selected packet has a record of the position", "Give a different selection")
        # the factor keeps the fastest packet inside the last pixel
        velocitymax_kmps = float(fastest_kmps.item()) * (1.0 + 1e-9)
        print_detail(
            f"The axes end at {format_velocity(velocitymax_kmps, unit)}, the largest v_cyl or |v_z| of the last"
            " interactions. -velocitymax gives a different end"
        )
    xedges = np.linspace(0.0, velocitymax_kmps, nbins + 1)
    pixelwidth_kmps = float(xedges[1])
    yedges = np.array(get_signed_velocity_bin_edges_kmps(xedges.tolist()))
    # the ring of bin i has a volume of pi ((i + 1)^2 - i^2) w^3 in velocity space, and t^3 makes it a real volume
    dfinimage = (
        dfpackets
        .with_columns(
            xbinindex=get_bin_indices(dfpackets.select("v_cyl_kmps"), "v_cyl_kmps", xedges.tolist())["binindex"]
        )
        .filter((pl.col("xbinindex") >= 0) & pl.col("v_z_kmps").is_between(yedges[0], yedges[-1]))
        .with_columns(
            weight=pl.col("e_rf")
            / (math.pi * (2 * pl.col("xbinindex") + 1) * (pixelwidth_kmps * km_to_cm * pl.col("emtime_s")) ** 3)
        )
    )
    dfinimage = dfinimage.with_columns(weightsquared=pl.col("weight") ** 2, one=pl.lit(1.0))
    hist2D, histsquared, histcount = (
        sum_weights_in_bins(dfinimage, "v_z_kmps", column, yedges.tolist(), "xbinindex", len(xedges) - 1)["sum"]
        .to_numpy()
        .reshape(len(xedges) - 1, len(yedges) - 1)
        for column in ("weight", "weightsquared", "one")
    )
    luminosityfactor = inverse_solidangle_fraction / (delta_t_s * nprocs_read)
    print_arrival_time_method(modelpath, [timestep], "the timestep that holds -timedays")
    print_last_interaction_convention(modelpath)
    print_heatmap_method(
        dfpackets,
        dfinimage,
        position,
        format_velocity(pixelwidth_kmps, unit),
        luminosityfactor,
        # a pixel with no packet has no error
        np.sqrt(histsquared[hist2D > 0.0]) / hist2D[hist2D > 0.0],
        selectiontext=(
            "the Sr II triplet"
            if srIItriplet
            else None
            if ions is None
            else ", ".join(get_ion_label(ion) for ion in ions)
        ),
    )

    heatmap = np.ma.masked_less_equal(hist2D * luminosityfactor, 0.0)
    if logcolorscale:
        heatmap = np.ma.log10(heatmap)

    # an image with a colorbar keeps plt.subplots: fig.colorbar takes space that the
    # Divider of make_frame_figure gives back at draw time, and the two then overlap
    set_mpl_style()
    figscale = getattr(args, "figscale", 1.0)
    fig, ax = plt.subplots(figsize=(3.5 * figscale, 4.5 * figscale), layout="constrained")
    z = heatmap.T

    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    im = ax.imshow(
        z,
        origin="lower",
        cmap="viridis",
        extent=(xedges[0] / kmpsperunit, xedges[-1] / kmpsperunit, yedges[0] / kmpsperunit, yedges[-1] / kmpsperunit),
    )
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$v_{{\rm cyl}} = \sqrt{{v_x^2 + v_y^2}}$ [{VELOCITYUNIT_AXISLABELS[unit]}]")
    ax.set_ylabel(rf"$v_z$ [{VELOCITYUNIT_AXISLABELS[unit]}]")
    cbar = fig.colorbar(im, ax=ax)
    luminositylabel = r"observed luminosity per volume [erg/(s cm$^3$)]"
    cbar.set_label(rf"log$_{{10}}$ {luminositylabel}" if logcolorscale else f"O{luminositylabel[1:]}")
    if dirbin != -1:
        draw_observer_direction(ax, dirbin, 0.8 * velocitymax_kmps / kmpsperunit)

    outfilename = resolve_outputfile(outputfile, start_of_filename + f"ts{timestep}_into_dirbin{dirbin}.pdf")
    if getattr(args, "print_data", False) or getattr(args, "write_data", False):
        assert args is not None
        valuecolumn = (
            "log10_escaped_luminosity_erg_per_s_per_cm3" if logcolorscale else "escaped_luminosity_erg_per_s_per_cm3"
        )
        # the empty table replaces a _summary file of the same name from an earlier command with -groupby
        output_data(
            args,
            {
                "": convert_velocity_columns(
                    get_heatmap_table(z, xedges, yedges, valuecolumn).with_columns(
                        packetcount=pl.Series(histcount.T.ravel(order="F")).cast(pl.Int64),
                        luminosity_relative_error=pl.Series(
                            np.sqrt(histsquared.T.ravel(order="F")) / hist2D.T.ravel(order="F"), nan_to_null=True
                        ),
                    ),
                    unit,
                ),
                "_summary": pl.DataFrame(),
            },
            outfilename,
        )
    save_figure(fig, outfilename, dpi=300, args=args)


def draw_observer_direction(ax: "mplax.Axes", dirbin: int, length: float) -> None:
    """Draw an arrow from the origin towards the observer, at the mean cos θ of the direction bin."""
    costhetalower, costhetaupper, _ = get_costheta_bins(usedegrees=False)
    costhetaindex = dirbin // get_viewingdirection_phibincount()
    costheta = (costhetalower[costhetaindex] + costhetaupper[costhetaindex]) / 2.0
    sintheta = math.sqrt(1.0 - costheta**2)
    head = (length * sintheta, length * costheta)
    ax.annotate("", xy=head, xytext=(0.0, 0.0), arrowprops={"arrowstyle": "->", "color": "red"})
    ax.text(*head, " observer", color="red", ha="left", va="center")


def print_heatmap_method(
    dfpackets: pl.DataFrame,
    dfinimage: pl.DataFrame,
    position: t.Literal["em", "trueem"],
    pixelwidthtext: str,
    luminosityfactor: float,
    pixelrelativeerrors: "npt.NDArray[np.floating]",
    *,
    selectiontext: str | None,
) -> None:
    """Print the selection of the packets of the 2D image, the meaning of its colour, and the luminosity in it.

    dfpackets holds the selected packets of the timestep, and dfinimage the packets inside the image.
    pixelrelativeerrors gives sqrt(sum(w^2)) / sum(w) of each pixel with packets. selectiontext names the lines of
    the last absorption that select the packets, or None for all the packets.
    """
    if selectiontext is None:
        print_detail(f"{dfpackets.height} escaped r-packets of the direction bin arrive in the timestep")
    else:
        print_detail(
            f"{dfpackets.height} escaped r-packets of the direction bin arrive in the timestep and have a last"
            f" absorption by a line of {selectiontext}. The image takes only such packets"
        )
    if position == "em":
        print_detail(
            "The position is em_pos at em_time, the place of the last interaction. An electron scattering also sets"
            " em_pos"
        )
    else:
        print_detail("The position is trueem_pos at trueem_time, the place of the last thermal emission")
    print_detail(
        f"Each pixel is a ring around the z axis, {pixelwidthtext} wide in v_cyl and in v_z, with v_cyl ="
        " sqrt(v_x^2 + v_y^2). The command divides the energy of each packet by the volume of its ring at the time of"
        " its last interaction. The colour is the sum of these values in the ring, divided by the width of the"
        " timestep and by the number of packet files (one for each MPI rank). The command then multiplies the colour"
        " by 4 pi divided by the solid angle of the direction bin"
    )
    print_detail(
        "The data table gives the number of packets and the relative error sqrt(sum(w^2)) / sum(w) of each pixel, with"
        " the weight w of each packet"
    )
    # the packets can have different energies, thus the error comes from the weights and not from the counts
    if pixelrelativeerrors.size > 0 and float(np.median(pixelrelativeerrors)) > NOISY_RELATIVE_ERROR:
        print_warning(
            "The pixels that hold packets have a median relative error of"
            f" {float(np.median(pixelrelativeerrors)):.0%}, thus the colour of a pixel has a large packet noise. Give"
            " a smaller -nvelocitybins, a wider wavelength range, or -dirbin -1"
        )
    totalluminosity = float(dfpackets["e_rf"].sum()) * luminosityfactor
    imageluminosity = float(dfinimage["e_rf"].sum()) * luminosityfactor
    print_detail(
        "Thus the colour is the observed luminosity per unit volume at the place of the last interaction. It is not"
        f" the emissivity of the ejecta. The packets give an observed luminosity of {totalluminosity:.3e} erg/s (4 pi"
        f" times the luminosity per unit solid angle), and {imageluminosity / totalluminosity:.1%} of it is inside"
        " the image. A packet outside the image has a velocity above the axes or no record of the position"
        if totalluminosity > 0.0
        else "Thus the colour is the observed luminosity per unit volume at the place of the last interaction. No"
        " packet is in the image"
    )


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
    outside the velocity bins is in no bin. A group with no packet in the bins has no series to draw, thus it also
    joins "Other". An example is the group NOT SET of the thermal emission, because its packets have no position.
    """
    from artistools.spectra.core import rank_flux_series_names

    dfgrouped = (
        dfpackets
        .lazy()
        .join(dflabels.lazy(), on=typecolumn, how="left", maintain_order="left")
        .with_columns(
            label=pl.col("label").fill_null(pl.lit("Other")),
            velocitybin=get_velocity_bin_index_expr(pl.col("velocity_kmps"), velocityedges_kmps),
        )
        .group_by("label", "velocitybin")
        .agg(
            energy=pl.col("e_rf").cast(pl.Float64).sum(),
            energysquared=(pl.col("e_rf").cast(pl.Float64) ** 2).sum(),
            packetcount=pl.len(),
        )
        .collect()
    )
    dftotals = (
        dfgrouped
        .group_by("label")
        .agg(pl.col("energy").sum(), binnedenergy=pl.col("energy").filter(pl.col("velocitybin").is_not_null()).sum())
        .filter(pl.col("binnedenergy") > 0.0)
        .sort("label")
    )
    drawnnames, _ = rank_flux_series_names(
        dict(zip(dftotals["label"].to_list(), dftotals["energy"].to_list(), strict=True)), maxseriescount
    )
    dfprofiles = (
        dfgrouped
        .with_columns(label=pl.when(pl.col("label").is_in(drawnnames)).then(pl.col("label")).otherwise(pl.lit("Other")))
        .group_by("label", "velocitybin")
        .agg(pl.col("energy", "energysquared", "packetcount").sum())
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
    velocitykind: t.Literal["radial", "lineofsight"] = "radial",
    lambdarange: tuple[float, float] | None = None,
    maxseriescount: int | None = 8,
    unit: t.Literal["kmps", "c"] = "kmps",
    outputfile: Path | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Plot the luminosity per unit velocity against the velocity of the last interaction, for each group.

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
        lambdarange=lambdarange,
        absorption=absorption,
        maxpacketfiles=getattr(args, "maxpacketfiles", None),
    )
    timestep, t_min, t_max = get_arrival_timestep(modelpath, tdays)
    delta_t_s = (t_max - t_min) * day_to_s
    try:
        velocity = get_last_interaction_velocity_expr(position, velocitykind, lzdfpackets.collect_schema().names())
    except ValueError as exc:
        exit_with_error(str(exc))
    dfpackets = (
        lzdfpackets
        .filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="left"))
        .select(typecolumn, "e_rf", velocity_kmps=velocity / km_to_cm)
        .collect()
    )
    if dfpackets.is_empty():
        exit_with_error(
            f"no escaped r-packet of the selection arrives from {t_min:.2f} to {t_max:.2f} d",
            "Give a different time, a wider wavelength range, or -dirbin -1",
        )
    print_group_selection(lambdarange, groupby, absorption=absorption)

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
    dfgroupsummary = get_group_summary(dfpackets, typecolumn, dflabels, drawnnames, luminosity_factor)
    print_arrival_time_method(modelpath, [timestep], "the timestep that holds -timedays")
    print_last_interaction_convention(modelpath)
    print_group_profile_method(
        dfprofiles, dfpackets.height, position, velocitykind, typecolumn, velocityedges_kmps, dfgroupsummary, unit
    )

    dftable = get_group_profile_table(dfprofiles, velocityedges_kmps, luminosity_factor, [*drawnnames, "Other"])
    fig, axesgrid = make_frame_figure(args, aspect=0.6)
    ax = axesgrid[0, 0]
    fractionoflabel = dict(zip(dfgroupsummary["label"], dfgroupsummary["luminosity_fraction"], strict=True))
    draw_group_profiles(ax, dftable, velocityedges_kmps, unit, fractionoflabel)
    # with --absorption the velocity is still the one of em_pos, which an electron scattering after the absorption moves
    interaction = "thermal emission" if trueem else "interaction"
    velocityname = "Radial velocity" if velocitykind == "radial" else "Line-of-sight velocity (towards the observer)"
    ax.set_xlabel(f"{velocityname} of the last {interaction} [{VELOCITYUNIT_AXISLABELS[unit]}]")
    ax.set_ylabel(rf"$\Delta L / \Delta v$ [erg s$^{{-1}}$ {INVERSE_VELOCITYUNIT_AXISLABELS[unit]}]")
    titleparts = [f"{get_model_name(modelpath)} {get_timestep_time(modelpath, timestep):.2f} d"]
    if lambdarange is not None:
        titleparts.append(f"{lambdarange[0]:.0f}-{lambdarange[1]:.0f} Å")
    if dirbin != -1:
        titleparts.append(get_dirbin_labels([dirbin], average_over_phi=True, usedegrees=True)[dirbin])
    ax.set_title(", ".join(titleparts))
    set_legend(ax, args)

    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if lambdarange is not None:
        start_of_filename = f"{start_of_filename}{lambdarange[0]:.0f}-{lambdarange[1]:.0f}A_"
    # the last absorption, the last thermal emission, and the last emission select different groups, thus the files
    # must not replace each other
    kind = "absorption" if absorption else ("thermalemission" if trueem else "emission")
    if velocitykind == "lineofsight":
        kind += "_lineofsight"
    outfilename = resolve_outputfile(
        outputfile, f"{start_of_filename}{kind}_by{groupby}_ts{timestep}_into_dirbin{dirbin}.pdf"
    )
    if args is not None:
        output_data(
            args,
            {"": convert_velocity_columns(dftable, unit), "_summary": convert_velocity_columns(dfgroupsummary, unit)},
            outfilename,
        )
    save_figure(fig, outfilename, args=args)


def print_group_selection(lambdarange: tuple[float, float] | None, groupby: str, *, absorption: bool) -> None:
    """Print the wavelength selection of the packets of -groupby, the position of an absorption, and the line labels."""
    from artistools.spectra.core import print_line_label_convention

    if lambdarange is not None:
        event = "last absorption" if absorption else "escape"
        print_detail(
            f"The packets have a rest-frame wavelength from {lambdarange[0]:g} to {lambdarange[1]:g} Å at the {event}"
        )
    if absorption:
        print_detail(
            f"The group of a packet is the {groupby} of its last absorption. That absorption happens at the position"
            " of the last emission, thus em_pos gives its velocity. An electron scattering after the last emission also"
            " sets em_pos, thus the axis gives the velocity of the last interaction. ARTIS records no wavelength for a"
            " free-free or a bound-free absorption"
        )
    if groupby == "line":
        print_line_label_convention()


def print_group_profile_method(
    dfprofiles: pl.DataFrame,
    packetcount: int,
    position: t.Literal["em", "trueem"],
    velocitykind: t.Literal["radial", "lineofsight"],
    typecolumn: str,
    velocityedges_kmps: Sequence[float],
    dfgroupsummary: pl.DataFrame,
    unit: t.Literal["kmps", "c"],
) -> None:
    """Print the method of the profiles, and the luminosity, velocity quartiles, and packet count of each group."""
    grouptext = {
        "emissiontype": "the line or the continuum process of its last emission",
        "trueemissiontype": "the line or the continuum process of its last thermal emission",
        "absorption_type": "the line of its last absorption",
    }[typecolumn]
    print_detail(
        f"{packetcount} escaped r-packets of the direction bin arrive in the timestep. The group of a packet is"
        f" {grouptext} ({typecolumn})"
    )
    print_detail(
        "Each packet reaches the observer, thus the luminosity of a group is its part of the observed luminosity. It"
        " is not the emission of the ejecta. A line that emits much in an optically thick region has only a small part"
        " of the observed luminosity, because other lines absorb its photons again. plotlinecontributions gives the"
        " emission of each line in each cell"
    )
    print_velocity_method(position, velocitykind)
    nbins = len(velocityedges_kmps) - 1
    binwidth_kmps = velocityedges_kmps[1] - velocityedges_kmps[0]
    print_detail(
        f"{nbins} velocity bins of {format_velocity(binwidth_kmps, unit)} from"
        f" {format_velocity(velocityedges_kmps[0], unit)} to {format_velocity(velocityedges_kmps[-1], unit)}. The"
        " luminosity is the packet energy divided by the width of the timestep and by the number of packet files (one"
        " for each MPI rank). The command then multiplies it by 4 pi divided by the solid angle of the direction bin"
    )
    print_detail(
        "The relative error of a luminosity is sqrt(sum(e^2)) / sum(e) of the packet energies e, which is"
        " 1 / sqrt(n) for n packets of one energy. The table gives it in the column luminosity_relative_error"
    )
    totalenergy = float(dfprofiles["energy"].sum())
    outside = dfprofiles.filter(pl.col("velocitybin").is_null())
    if not outside.is_empty():
        print_detail(
            f"{int(outside['packetcount'].sum())} packets with {float(outside['energy'].sum()) / totalenergy:.1%} of"
            " the energy have a velocity outside the bins, or no record of the position"
        )

    print_detail(
        "These are the groups. The quartiles of the velocity take the energy of each packet as the weight, and they"
        " include the packets outside the bins:"
    )
    for row in dfgroupsummary.iter_rows(named=True):
        quartiles = [row[f"velocity_luminosity_q{quantile}_kmps"] for quantile in (25, 50, 75)]
        velocitytext = (
            "velocity quartiles " + " ".join(format_velocity(value, unit) for value in quartiles)
            if None not in quartiles
            else "no record of the position"
        )
        print_detail(
            f"  {row['label']}: {row['luminosity_erg_per_s']:.3e} erg/s ({row['luminosity_fraction']:.1%}) with a"
            f" relative error of {row['luminosity_relative_error']:.0%}, {velocitytext}, {row['packetcount']} packets"
        )
    noisylabels = dfgroupsummary.filter(pl.col("luminosity_relative_error") > NOISY_RELATIVE_ERROR)["label"].to_list()
    if noisylabels:
        print_warning(
            f"These groups have a luminosity with a relative error above {NOISY_RELATIVE_ERROR:.0%}:"
            f" {', '.join(noisylabels)}. Thus the packet noise of their fractions and velocity profiles is large. Give"
            " a wider wavelength range or -dirbin -1. A smaller -maxseriescount puts the weak groups in Other"
        )


def get_group_summary(
    dfpackets: pl.DataFrame,
    typecolumn: str,
    dflabels: pl.DataFrame,
    drawnnames: Sequence[str],
    luminosity_factor: float,
) -> pl.DataFrame:
    """Return one row for each drawn group and for "Other", with the luminosity, the packet count, and the quartiles.

    The quartiles take the energy of each packet as the weight. A packet with no record of the position has no
    velocity, thus it is in the luminosity but not in the quartiles.
    """
    luminosity = pl.col("luminosity_erg_per_s")
    return (
        dfpackets
        .join(dflabels, on=typecolumn, how="left", maintain_order="left")
        .with_columns(
            label=pl.when(pl.col("label").is_in(list(drawnnames))).then(pl.col("label")).otherwise(pl.lit("Other"))
        )
        .group_by("label")
        .agg(
            luminosity_erg_per_s=pl.col("e_rf").cast(pl.Float64).sum() * luminosity_factor,
            packetcount=pl.len(),
            luminosity_relative_error=get_relative_error_expr(pl.col("e_rf")),
            **{
                f"velocity_luminosity_q{round(quantile * 100)}_kmps": get_weighted_quantile_expr(
                    "velocity_kmps", "e_rf", quantile
                )
                for quantile in (0.25, 0.5, 0.75)
            },
        )
        .with_columns(luminosity_fraction=luminosity / luminosity.sum())
        .sort("luminosity_erg_per_s", "label", descending=[True, False])
        .select(
            "label",
            "luminosity_erg_per_s",
            "luminosity_fraction",
            "packetcount",
            "luminosity_relative_error",
            pl.exclude(
                "label", "luminosity_erg_per_s", "luminosity_fraction", "packetcount", "luminosity_relative_error"
            ),
        )
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
        luminosity_relative_error=pl.col("energysquared").sqrt() / pl.col("energy"),
    )


def draw_group_profiles(
    ax: "mplax.Axes",
    dftable: pl.DataFrame,
    velocityedges_kmps: Sequence[float],
    unit: t.Literal["kmps", "c"],
    fractionoflabel: "Mapping[str, float]",
) -> None:
    """Draw the sum of all the groups, and one step series for each group, with its fraction of the luminosity.

    dftable is in km/s, and the axis takes the velocity unit unit. fractionoflabel gives the fraction of the luminosity
    of all the selected packets, as the log and the _summary table give it.
    """
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    edges = np.array(velocityedges_kmps) / kmpsperunit
    yvalues = "luminosity_per_velocity_erg_per_s_per_kmps"
    dftotal = dftable.group_by("x_lower_kmps").agg(pl.col(yvalues).sum()).sort("x_lower_kmps")
    ax.stairs(
        dftotal[yvalues].to_numpy() * kmpsperunit, edges, label="All packets in the bins", color="black", linewidth=1.5
    )
    for (label,), dfgroup in dftable.group_by("label", maintain_order=True):
        # stairs with `color=None` draws black and not the next colour of the cycle
        color = "grey" if label == "Other" else get_next_color(ax)
        ax.stairs(
            dfgroup.sort("x_lower_kmps")[yvalues].to_numpy() * kmpsperunit,
            edges,
            label=f"{label} ({fractionoflabel[str(label)]:.1%})",
            color=color,
        )
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)


def get_heatmap_table(
    image: np.ma.MaskedArray, xedges: npt.NDArray[np.floating], yedges: npt.NDArray[np.floating], valuecolumn: str
) -> pl.DataFrame:
    """Return one row for each bin of the image, with the velocity edges [km/s] of the bin.

    The image has one row for each v_z bin and one column for each v_cyl bin. A bin that the image leaves blank has no
    value.
    """
    nzbins, nrbins = image.shape
    # the Fortran order keeps the v_z bins of one v_cyl bin together
    values = np.ma.filled(image.astype(np.float64), np.nan).ravel(order="F")
    return pl.DataFrame({
        "v_cyl_min_kmps": np.repeat(xedges[:-1], nzbins),
        "v_cyl_max_kmps": np.repeat(xedges[1:], nzbins),
        "v_z_min_kmps": np.tile(yedges[:-1], nrbins),
        "v_z_max_kmps": np.tile(yedges[1:], nrbins),
        valuecolumn: pl.Series(values, nan_to_null=True),
    })


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of ARTIS model")
    addarg_darkmode(parser)

    addarg_timedays(
        parser,
        kind="float",
        helptext="Time in days. The plot takes the packets that arrive in the timestep of this time",
    )
    parser.add_argument("-tdays", dest="timedays", type=float, help=argparse.SUPPRESS)
    addarg_timestep(parser, helptext="Timestep of the arrival time of the packets, e.g. 40, or last")

    parser.add_argument(
        "-xmin",
        "-lambdamin",
        type=float,
        help=(
            "Minimum rest-frame wavelength of the packets in Angstroms, at the escape or with --absorption, at the last"
            " absorption. Default: no limit"
        ),
    )
    parser.add_argument(
        "-xmax", "-lambdamax", type=float, help="Maximum rest-frame wavelength of the packets in Angstroms"
    )
    # the earlier form of the wavelength range: a centre and a width
    parser.add_argument("-wavelen", type=float, help=argparse.SUPPRESS)
    parser.add_argument("-binwidth", type=float, help=argparse.SUPPRESS)

    parser.add_argument(
        "-ions",
        action=CommaJoinAction,
        help=(
            'Ions or elements of the 2D image, e.g. "Sr II,Fe". The image takes the packets with a last absorption by'
            " a line of these ions. Default: all the escaped packets"
        ),
    )
    # the earlier form of -ions: one element, one ion stage, or both
    parser.add_argument("-element", type=str, help=argparse.SUPPRESS)
    parser.add_argument("-ionstage", type=str, help=argparse.SUPPRESS)

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
            "Plot the luminosity per unit velocity against the velocity (-velocity) of the last interaction. Draw one"
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
        type=positive_int_arg,
        default=8,
        help='Number of groups that -groupby draws. The other groups join the group "Other"',
    )
    addarg_velocitybins(parser)
    addarg_velocitykind(parser)
    parser.add_argument("--srIItriplet", action="store_true", help="Plot packets from SrII triplet only")
    parser.add_argument(
        "--logcolorscale",
        action="store_true",
        help="Show the log10 of the observed luminosity per unit volume in the 2D image",
    )
    # the earlier name of --logcolorscale
    parser.add_argument("--colorlogscale", dest="logcolorscale", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Take the last thermal emission and not the last interaction",
    )

    addarg_output(parser, kind="file", helptext="Path/filename for the PDF file")
    addarg_figscale(parser)
    addarg_show(parser)
    addarg_data_output(parser)
    addarg_plotinvalidpart(parser, "the packets")
    addarg_maxpacketfiles(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot last packet interaction properties versus ejecta velocity for selected packets."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if (args.wavelen is None) != (args.binwidth is None):
        exit_with_error("-wavelen and -binwidth give a range together", "Give -lambdamin and -lambdamax")
    if args.wavelen is not None:
        if args.xmin is not None or args.xmax is not None:
            exit_with_error("give the wavelength range one time", "Remove -wavelen and -binwidth")
        args.xmin, args.xmax = args.wavelen - args.binwidth / 2, args.wavelen + args.binwidth / 2
    if (args.xmin is None) != (args.xmax is None):
        exit_with_error("the wavelength range needs two ends", "Give -lambdamin and -lambdamax together")
    if args.absorption and args.use_thermalemissiontype:
        exit_with_error(
            "--absorption takes the last absorption, and --use_thermalemissiontype takes the last thermal emission",
            "Give one of the two options",
        )

    # the arrival time bins are the timesteps, thus -timestep and -timedays select the same bin
    selectedtimestep = get_selected_timestep(Path(args.modelpath), args.timestep, args.timedays)
    args.timedays = get_timestep_time(Path(args.modelpath), selectedtimestep)
    try:
        check_time_range_is_valid(
            Path(args.modelpath),
            get_timestep_times(Path(args.modelpath), loc="start")[selectedtimestep],
            get_timestep_times(Path(args.modelpath), loc="end")[selectedtimestep],
            args.plotinvalidpart,
        )
    except ValueError as exc:
        exit_with_error(str(exc))

    check_costheta_dirbin(args.dirbin, "-dirbin", allowall=True)

    ions = get_selected_ions(args)

    if args.groupby is None and args.velocitykind != "radial":
        exit_with_error(
            "-velocity applies only with -groupby. The 2D image always gives v_cyl and v_z",
            "Remove -velocity, or give -groupby",
        )
    if args.groupby is not None:
        plot_velocity_profiles_from_args(args, ions)
        return

    modelpath = Path(args.modelpath)
    unit = get_velocity_unit(modelpath, args.velocityunit)
    packets_2d_hist_bin_and_ejecta_vel(
        modelpath,
        args.timedays,
        args.dirbin,
        args.nvelocitybins,
        None
        if args.velocitymax_kmps is None
        else get_velocity_bin_edges_kmps(modelpath, args.nvelocitybins, args.velocitymax_kmps, unit)[-1],
        unit,
        ions=ions,
        srIItriplet=args.srIItriplet,
        logcolorscale=args.logcolorscale,
        trueem=args.use_thermalemissiontype,
        lambdarange=get_lambdarange(args),
        outputfile=args.outputfile,
        args=args,
        absorption=args.absorption,
    )


def get_selected_ions(args: argparse.Namespace) -> list[tuple[int | None, int | None]] | None:
    """Return the (Z, ion_stage) of each item of -ions, or of -element and -ionstage, or None for no selection.

    A None in an item selects each element or each ion stage.
    """
    if args.element is None and args.ionstage is None:
        if not args.ions:
            return None
        ions: list[tuple[int | None, int | None]] = []
        for iontext in str(args.ions).split(","):
            try:
                iontuple = get_ion_tuple(iontext.strip())
            except ValueError:
                exit_with_error(
                    f"-ions takes ions or elements, and '{iontext}' is neither", 'Give e.g. -ions "Sr II,Fe"'
                )
            Z = iontuple[0] if isinstance(iontuple, tuple) else iontuple
            # get_atomic_number gives -1 for a text that is no element symbol, and 0 for the free neutron
            if Z < 1:
                exit_with_error(
                    f"-ions takes ions or elements, and '{iontext}' is neither", 'Give e.g. -ions "Sr II,Fe"'
                )
            ions.append(iontuple if isinstance(iontuple, tuple) else (iontuple, None))
        return ions
    if args.ions:
        exit_with_error("give the ions one time", "Remove -element and -ionstage")
    Z = get_atomic_number(args.element) if args.element else None
    if Z is not None and Z < 1:
        exit_with_error(f"-element {args.element} is no element symbol", "Give the element as e.g. Sr")
    ion_stage = decode_roman_numeral(args.ionstage) if args.ionstage else None
    if ion_stage == -1:
        exit_with_error(f"-ionstage {args.ionstage} is no Roman numeral", "Give the ion stage as e.g. II")
    return [(Z, ion_stage)]


def get_lambdarange(args: argparse.Namespace) -> tuple[float, float] | None:
    """Return the range of rest-frame wavelength [Å] of -lambdamin and -lambdamax, or None for no limit."""
    return None if args.xmin is None else (args.xmin, args.xmax)


def plot_velocity_profiles_from_args(
    args: argparse.Namespace, ions: Sequence[tuple[int | None, int | None]] | None
) -> None:
    """Check the options of -groupby, then plot the velocity profile of each group."""
    for option, isgiven in (
        ("-ions", ions is not None),
        ("--srIItriplet", args.srIItriplet),
        ("--logcolorscale", args.logcolorscale),
    ):
        if isgiven:
            exit_with_error(
                f"{option} applies to the 2D image, and -groupby gives one series for each group",
                f"Remove {option}, or remove -groupby",
            )
    modelpath = Path(args.modelpath)
    unit = get_velocity_unit(modelpath, args.velocityunit)
    velocityedges_kmps = get_velocity_bin_edges_kmps(modelpath, args.nvelocitybins, args.velocitymax_kmps, unit)
    if args.velocitykind == "lineofsight":
        # a line-of-sight velocity can be negative, thus each side of zero gets -nvelocitybins bins
        velocityedges_kmps = get_signed_velocity_bin_edges_kmps(velocityedges_kmps)
    plot_velocity_profiles_by_group(
        modelpath,
        args.timedays,
        args.dirbin,
        args.groupby,
        velocityedges_kmps,
        trueem=args.use_thermalemissiontype,
        absorption=args.absorption,
        velocitykind=args.velocitykind,
        lambdarange=get_lambdarange(args),
        maxseriescount=args.maxseriescount,
        unit=unit,
        outputfile=args.outputfile,
        args=args,
    )
