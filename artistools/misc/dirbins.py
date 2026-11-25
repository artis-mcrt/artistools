"""Viewing-direction (costheta/phi) bin definitions, labels, selection, and averaging."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

from artistools.misc.argparser import get_standard_flag
from artistools.misc.console import exit_with_error
from artistools.misc.console import print_detail
from artistools.misc.console import print_warning
from artistools.misc.modelinfo import get_vpkt_config


def split_multitable_dataframe(
    res_df: pl.DataFrame | pl.LazyFrame, source: Path | str = "the data"
) -> dict[int, pl.LazyFrame]:
    """Return each table of a file that repeats its first column, e.g. one table for each direction bin.

    A table starts where the first column stops increasing. An old file can round the values differently in each table,
    thus a count of the different values is not correct. Tables of different lengths show a cut file. source names the
    file in the error.
    """
    res_df = res_df.lazy()
    tablelengths = (
        res_df
        .select(tableindex=(cs.by_index(0).cast(pl.Float64).diff() <= 0).fill_null(value=False).cum_sum())
        .group_by("tableindex", maintain_order=True)
        .len()
        .collect()["len"]
    )
    if tablelengths.n_unique() != 1:
        msg = f"The tables have different lengths in {source}, or it holds no table. A cut file gives this"
        raise ValueError(msg)
    nu_points, tablecount = tablelengths[0], tablelengths.len()

    # polars 2.0.0rc2 gave incorrect rows for pl.collect_all of several filtered slices of one scan.
    # A filter on the row index read the whole file for each table
    return {tableindex: res_df.head((tableindex + 1) * nu_points).tail(nu_points) for tableindex in range(tablecount)}


def get_common_first_column(
    dirbindataframes: dict[int, pl.DataFrame] | dict[int, pl.LazyFrame], columnname: str
) -> pl.Series:
    """Return the first column of direction bin 0, after a check that each direction bin agrees with it.

    ARTIS writes a value with six significant digits, thus the same time can differ by one unit of the last digit in
    two tables. A larger difference shows a different grid, and an average of such tables has no meaning.
    """
    maxrounding_rtol = 2e-5
    commonvalues = dirbindataframes[0].lazy().select(columnname).collect().to_series()
    for dirbin, table in dirbindataframes.items():
        values = table.lazy().select(columnname).collect().to_series()
        if values.len() != commonvalues.len() or not np.allclose(
            values.to_numpy(), commonvalues.to_numpy(), rtol=maxrounding_rtol, atol=0.0
        ):
            msg = (
                f"Cannot average the direction bins: the column {columnname} of direction bin {dirbin} does not agree"
                " with the column of direction bin 0"
            )
            raise ValueError(msg)
    return commonvalues


def average_direction_bins(
    dirbindataframes: dict[int, pl.DataFrame] | dict[int, pl.LazyFrame], overangle: t.Literal["phi", "theta"]
) -> dict[int, pl.LazyFrame]:
    """Average dict of direction-binned polars DataFrames according to the phi or theta angle.

    Every column is averaged, which is only valid for a column that is linear in the bin contributions. A
    column derived non-linearly from the others, such as a magnitude, is not the average of its bins: a
    single dark bin sends the mean to infinity. Derive such a column after averaging, not before.
    """
    dirbincount = get_viewingdirectionbincount()

    if overangle not in {"phi", "theta"}:
        msg = "overangle must be 'phi' or 'theta'"
        raise ValueError(msg)
    start_bin_range = get_dirbins(average_over_phi=overangle == "phi", average_over_theta=overangle == "theta")

    if missingbins := sorted(set(range(dirbincount)) - set(dirbindataframes)):
        # averaging twice (e.g. over theta and then over phi) leaves too few bins to average again
        msg = (
            f"Cannot average over {overangle}: expected all {dirbincount} direction bins, but"
            f" {len(missingbins)} are missing (first missing bin is {missingbins[0]})"
        )
        raise ValueError(msg)

    # a different build of ARTIS can write more bins, and the geometry of the default bins would ignore the others
    if extrabins := sorted(set(dirbindataframes) - set(range(dirbincount))):
        msg = (
            f"Cannot average over {overangle}: artistools takes the geometry of {dirbincount} direction bins, but the"
            f" data holds {len(extrabins)} more bins (first extra bin is {extrabins[0]})"
        )
        raise ValueError(msg)

    # we will make a copy to ensure that we don't cause side effects from altering the original DataFrames
    # that might be returned again later by an lru_cached function
    dirbindataframesout: dict[int, pl.LazyFrame] = {}
    commonfirstcolumn = get_common_first_column(
        dirbindataframes, dirbindataframes[0].lazy().collect_schema().names()[0]
    )

    for start_bin in start_bin_range:
        contribbins = get_average_group_dirbins(
            start_bin, average_over_phi=overangle == "phi", average_over_theta=overangle == "theta"
        )

        colnames = dirbindataframes[start_bin].collect_schema().names()
        firstcolname = colnames[0]

        dirbindataframesout[start_bin] = (
            pl
            .concat(
                [
                    # the group_by compares exact values, thus each table needs the same values
                    dirbindataframes[dirbin].lazy().select(colnames).with_columns(pl.lit(commonfirstcolumn))
                    for dirbin in contribbins
                ],
                how="vertical",
            )
            .group_by(firstcolname, maintain_order=True)
            .agg([(pl.col(col).sum() / len(contribbins)).alias(col) for col in colnames[1:]])
        )

        print(f"Bin number {start_bin:2d} = the average of bins {contribbins}")

    return dirbindataframesout


def get_viewingdirectionbincount() -> int:
    """Return the total number of viewing direction bins."""
    return get_viewingdirection_phibincount() * get_viewingdirection_costhetabincount()


def get_viewingdirection_phibincount() -> int:
    """Return the number of phi bins that the viewing directions are divided into."""
    return 10


def get_viewingdirection_costhetabincount() -> int:
    """Return the number of cos(theta) bins that the viewing directions are divided into."""
    return 10


def check_averaging_angles(average_over_phi: bool, average_over_theta: bool) -> None:
    """Reject averaging over both angles at once, which leaves too few direction bins to average again.

    The command-line flags are already mutually exclusive (see addarg_viewingangle), so this covers the callers
    that pass the values directly or build an argparse.Namespace from keyword arguments.
    """
    if average_over_phi and average_over_theta:
        msg = "Cannot average over both the phi and theta viewing angles"
        raise ValueError(msg)


def get_dirbins(average_over_phi: bool = False, average_over_theta: bool = False) -> list[int]:
    """Return the indices of the viewing direction bins.

    An average over the phi angle or the theta angle gives only the first bin of each group of the average.
    """
    check_averaging_angles(average_over_phi, average_over_theta)
    if average_over_phi:
        return list(range(0, get_viewingdirectionbincount(), get_viewingdirection_phibincount()))
    if average_over_theta:
        return list(range(get_viewingdirection_phibincount()))
    return list(range(get_viewingdirectionbincount()))


def get_average_group_dirbins(dirbin: int, *, average_over_phi: bool, average_over_theta: bool) -> list[int]:
    """Return the direction bins that the average of a bin holds, or the bin alone with no average.

    The bin of an average is the first bin of its group, and a different bin raises ValueError. The function reads
    no file of the model.
    """
    get_dirbin_labels([dirbin], average_over_phi=average_over_phi, average_over_theta=average_over_theta)
    # dirbin == costheta_index * nphibins + phi_index, thus the bins of one costheta index are contiguous, and the bins
    # of one phi index are nphibins apart
    nphibins = get_viewingdirection_phibincount()
    if average_over_phi:
        return list(range(dirbin, dirbin + nphibins))
    if average_over_theta:
        return list(range(dirbin, get_viewingdirectionbincount(), nphibins))
    return [dirbin]


def get_dirbin_definitions(
    modelpath: Path | str,
    dirbins: Sequence[int] | None = None,
    *,
    vpkt_observers: bool = False,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
) -> dict[int, str]:
    """Return a label for each direction bin, or for each virtual packet observer when vpkt_observers is set.

    The function raises ValueError for averages over both angles, and for a bin that an average leaves out.
    """
    # the check comes before the virtual packet observers, because both averages are also an error for an observer
    check_averaging_angles(average_over_phi, average_over_theta)
    if vpkt_observers:
        return get_vspec_dir_labels(modelpath=modelpath, usedegrees=usedegrees)

    return get_dirbin_labels(
        dirbins=dirbins,
        modelpath=modelpath,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
        usedegrees=usedegrees,
    )


def print_theta_phi_definitions() -> None:
    """Print the spherical polar convention that the theta and phi direction bins follow."""
    print(
        "Spherical polar: x = r sinθ cosϕ, y = r sinθ sinϕ, z = r cosθ -> θ=0 is +Z and θ=π is -Z."
        " At Z=0, ϕ=0 is +X and ϕ=π/2 is +Y"
    )


def get_phibin_rank_ascending(phibin: int) -> int:
    """Return the rank of an ARTIS phi bin in the order that phi increases.

    The ARTIS phi bins do not ascend with phi. The second half covers phi from 0 upward, and the
    first half holds the phi above pi in descending order, thus bin 0 holds the highest phi. A colour
    scale or an axis that ascends with phi needs this rank and not the bin index. See get_phi_bins
    for the range of each bin.
    """
    nphibins = get_viewingdirection_phibincount()
    halfbins = nphibins // 2
    return phibin - halfbins if phibin >= halfbins else nphibins - 1 - phibin


def get_phi_bin_steps() -> list[int]:
    """Return the phi step of each direction bin.

    Direction bin b covers the phi range that step phisteps[b] names, i.e. 2 pi (1 - (step + 1) / n)
    to 2 pi (1 - step / n). For historical reasons the phi bins descend and reverse at the middle
    bin, thus the steps are [0, 1, 2, 3, 4, 9, 8, 7, 6, 5] for ten bins.
    """
    nphibins = get_viewingdirection_phibincount()
    # pi/2 must be an exact boundary because of the change in behaviour there
    assert nphibins % 2 == 0

    return list(range(nphibins // 2)) + list(reversed(range(nphibins // 2, nphibins)))


def get_phi_bin_edges_ascending(nphibins: int) -> npt.NDArray[np.float64]:
    """Return the nphibins + 1 edges of the ascending phi bins, from 0 to 2 pi.

    Ascending bin j holds 2 pi j / nphibins <= phi < 2 pi (j + 1) / nphibins. The column phibinmonotonicasc
    of bin_packet_directions_polars in packets/core.py gives this bin.
    """
    return np.arange(nphibins + 1, dtype=np.float64) * 2 * np.pi / nphibins


def get_phi_bins(usedegrees: bool) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], list[str]]:
    """Return the lower and upper phi boundaries of each direction bin, and a label for each."""
    nphibins = get_viewingdirection_phibincount()
    phisteps = get_phi_bin_steps()

    # set up monotonic descending phi bin boundaries
    phi_lower = np.array([2 * math.pi * (1 - (step + 1) / nphibins) for step in phisteps])
    phi_upper = np.array([2 * math.pi * (1 - step / nphibins) for step in phisteps])

    binlabels = []
    for phibin in range(nphibins):
        if usedegrees:
            str_phi_lower = f"{phi_lower[phibin] / math.pi * 180:3.0f}°"
            str_phi_upper = f"{phi_upper[phibin] / math.pi * 180:3.0f}°"
        else:
            coeff_lower = phi_lower[phibin] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_lower, round(coeff_lower), rtol=0.01), coeff_lower
            str_phi_lower = f"{round(coeff_lower)}π/{nphibins // 2}" if phi_lower[phibin] > 0.0 else "0"
            coeff_upper = phi_upper[phibin] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_upper, round(coeff_upper), rtol=0.01)
            str_phi_upper = f"{round(coeff_upper)}π/{nphibins // 2}" if phi_upper[phibin] < 2 * math.pi else "2π"

        # ARTIS puts a packet of the first half in the bin int(acos(cosphi) / 2 pi * n), and
        # acos descends as phi ascends. Thus such a bin is open at the lower boundary and closed
        # at the upper one. The second half takes int((acos(cosphi) + pi) / 2 pi * n), thus it is
        # closed at the lower boundary and open at the upper one. The last bin also holds
        # phi = pi, because ARTIS clamps that index. See bin_packet_directions_polars in packets/core.py
        firsthalf = phibin < nphibins // 2
        lower_compare = "<" if firsthalf else "≤"
        upper_compare = "≤" if firsthalf or phibin == nphibins - 1 else "<"
        binlabels.append(f"{str_phi_lower} {lower_compare} ϕ {upper_compare} {str_phi_upper}")

    # if nphibins == 10, then binlabels = [
    #     "9π/5 < ϕ ≤ 2π",
    #     "8π/5 < ϕ ≤ 9π/5",
    #     "7π/5 < ϕ ≤ 8π/5",
    #     "6π/5 < ϕ ≤ 7π/5",
    #     "5π/5 < ϕ ≤ 6π/5",
    #     "0 ≤ ϕ < 1π/5",
    #     "1π/5 ≤ ϕ < 2π/5",
    #     "2π/5 ≤ ϕ < 3π/5",
    #     "3π/5 ≤ ϕ < 4π/5",
    #     "4π/5 ≤ ϕ ≤ 5π/5",
    # ]

    return phi_lower, phi_upper, binlabels


def get_costheta_bins(usedegrees: bool) -> tuple[tuple[float, ...], tuple[float, ...], list[str]]:
    """Return the lower and upper cos(theta) boundaries of each direction bin, and a label for each.

    The boundaries are always cos(theta); usedegrees only changes how the labels are written.
    """
    ncosthetabins = get_viewingdirection_costhetabincount()
    # the costheta bins are ordered by ascending cos θ from -1. to 1.,
    # which means that they are in descending order of theta from π to 0
    # i.e. costhetabins[0] is the θ=π or -Z axis direction
    # np.arange gives -1.1e-16 in place of 0 at the middle edge, and the label then read "-0.0 ≤ cos θ". The rounded
    # edge is -0.0, and the addition of 0.0 gives 0.0
    costhetabins_lower = np.round(np.arange(-1.0, 1.0, 2.0 / ncosthetabins), 12) + 0.0
    costhetabins_upper = costhetabins_lower + 2.0 / ncosthetabins
    if usedegrees:
        thetabins_upper = np.arccos(costhetabins_lower) / np.pi * 180
        thetabins_lower = np.arccos(costhetabins_upper) / np.pi * 180

        # the edges of θ come from the edges of cos θ, thus a closed edge of cos θ gives a closed edge of θ
        binlabels = [
            f"{lower:.0f}° {'≤' if binindex == ncosthetabins - 1 else '<'} θ ≤ {upper:.0f}°"
            for binindex, (lower, upper) in enumerate(zip(thetabins_lower, thetabins_upper, strict=True))
        ]
    else:
        # ARTIS puts a packet with cos θ = 1 in the last bin, thus that bin includes its upper edge
        binlabels = [
            f"{lower:.1f} ≤ cos θ {'≤' if binindex == ncosthetabins - 1 else '<'} {upper:.1f}"
            for binindex, (lower, upper) in enumerate(zip(costhetabins_lower, costhetabins_upper, strict=True))
        ]
    return tuple(float(x) for x in costhetabins_lower), tuple(costhetabins_upper), binlabels


def get_dirbin_centre_direction(dirbin: int) -> npt.NDArray[np.float64]:
    """Return the unit vector at the centre cos θ and the centre phi of a direction bin. The polar axis is +z.

    The phi bins of ARTIS descend and reverse at the middle bin, thus the phi of a bin comes from get_phi_bins.
    """
    costhetabin, phibin = divmod(dirbin, get_viewingdirection_phibincount())
    costhetalower, costhetaupper, _ = get_costheta_bins(usedegrees=False)
    philower, phiupper, _ = get_phi_bins(usedegrees=False)
    costheta = (costhetalower[costhetabin] + costhetaupper[costhetabin]) / 2.0
    phi = (philower[phibin] + phiupper[phibin]) / 2.0
    sintheta = math.sqrt(1.0 - costheta**2)
    return np.array([sintheta * math.cos(phi), sintheta * math.sin(phi), costheta])


def get_costhetabin_phibin_labels(usedegrees: bool) -> tuple[list[str], list[str]]:
    """Return the cos(theta) and phi bin labels."""
    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)
    return costhetabinlabels, phibinlabels


def get_opacity_condition_label(z_exclude: int) -> str:
    """Return the label for a virtual packet opacity exclusion code, such as 'no-bb' or 'no-Fe'."""
    # artistools.atomic imports artistools.misc, thus this import runs at call time
    from artistools.atomic import get_elsymbol

    # codes match the opacityexclusions handling in read_vpktparameterfile() and trace_vpkt_direction()
    labels = {0: "", -1: "no-bb", -2: "no-bf", -3: "no-ff", -4: "no-es"}

    return labels[z_exclude] if z_exclude in labels else f"no-{get_elsymbol(z_exclude)}"


def get_vspec_dir_labels(modelpath: str | Path, usedegrees: bool = False) -> dict[int, str]:
    """Return a label for each virtual packet observer direction and opacity choice combination."""
    vpkt_config = get_vpkt_config(Path(modelpath))
    dirlabels = {}
    for dirindex in range(vpkt_config["nobsdirections"]):
        phi_angle = round(vpkt_config["phi"][dirindex])
        for opacchoiceindex in range(vpkt_config["nspectraperobs"]):
            opacity_condition_label = get_opacity_condition_label(int(vpkt_config["z_excludelist"][opacchoiceindex]))
            ind_comb = vpkt_config["nspectraperobs"] * dirindex + opacchoiceindex
            cos_theta = vpkt_config["cos_theta"][dirindex]
            if usedegrees:
                theta_degrees = round(math.degrees(math.acos(cos_theta)))
                dirlabels[ind_comb] = rf"θ = {theta_degrees}°, ϕ = {phi_angle}° {opacity_condition_label}"
            else:
                dirlabels[ind_comb] = rf"cos θ = {cos_theta}, ϕ = {phi_angle}° {opacity_condition_label}"

    return dirlabels


def get_dirbin_labels(
    dirbins: npt.NDArray[np.int32] | Sequence[int] | None = None,
    modelpath: Path | str | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
) -> dict[int, str]:
    """Return a dict of text labels for viewing direction bins."""
    check_averaging_angles(average_over_phi, average_over_theta)
    if modelpath:
        modelpath = Path(modelpath)
        MABINS = get_viewingdirectionbincount()
        # ARTIS sets MABINS when it compiles, thus a different build can write a different count of direction bins
        if list(modelpath.glob("*_res_00.out*")) and (
            not list(modelpath.glob(f"*_res_{MABINS - 1:02d}.out*")) or list(modelpath.glob(f"*_res_{MABINS:02d}.out*"))
        ):
            print_warning(
                f"The *_res_ files in {modelpath} do not give {MABINS} direction bins, and artistools gives the angles"
                f" of {MABINS} direction bins. Thus the angle of a direction bin can be incorrect"
            )

    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)

    nphibins = get_viewingdirection_phibincount()
    ndirbins = get_viewingdirectionbincount()

    if dirbins is None:
        dirbins = get_dirbins(average_over_phi=average_over_phi, average_over_theta=average_over_theta)

    angle_definitions: dict[int, str] = {}
    for dirbin in dirbins:
        dirbin_int = int(dirbin)
        if dirbin_int == -1:
            angle_definitions[dirbin_int] = "all directions"
            continue

        if not 0 <= dirbin_int < ndirbins:
            msg = f"The direction bin {dirbin_int} is not one of the {ndirbins} bins 0 to {ndirbins - 1}, or -1 for all"
            raise ValueError(msg)

        costheta_index = dirbin_int // nphibins
        phi_index = dirbin_int % nphibins

        # an averaged bin is the first bin of its group, thus a different bin has no label
        if (average_over_phi and phi_index != 0) or (average_over_theta and costheta_index != 0):
            validbins = get_dirbins(average_over_phi=average_over_phi, average_over_theta=average_over_theta)
            msg = f"Direction bin {dirbin_int} is not the first bin of an average group. Valid bins: {validbins}"
            raise ValueError(msg)

        if average_over_phi:
            angle_definitions[dirbin_int] = costhetabinlabels[costheta_index]
        elif average_over_theta:
            angle_definitions[dirbin_int] = phibinlabels[phi_index]
        else:
            angle_definitions[dirbin_int] = f"{costhetabinlabels[costheta_index]}, {phibinlabels[phi_index]}"

    return angle_definitions


def check_dirbin(
    dirbin: int, optionname: str, *, averageoverphi: bool, allowall: bool, averageovertheta: bool = False
) -> None:
    """Exit with an error if dirbin is not a direction bin. With allowall, -1 for all the directions is also valid.

    With averageoverphi, as with --average-over-phi-angle of plotspectra, dirbin must be the first bin of its costheta
    bin. With averageovertheta, dirbin must be the bin of its ϕ range in the first costheta bin.
    """
    if allowall and dirbin == -1:
        return
    # the caller can give the declared spelling, and the message gives the spelling of the help
    optionname = get_standard_flag(optionname)
    nphibins = get_viewingdirection_phibincount()
    ndirbins = get_viewingdirectionbincount()
    isdirbin = 0 <= dirbin < ndirbins
    alltext = "-1 for all the directions, or " if allowall else ""
    if averageoverphi:
        if isdirbin and dirbin % nphibins == 0:
            return
        exit_with_error(
            f"{optionname} {dirbin} is not the first direction bin of a costheta bin",
            f"Give {alltext}a multiple of {nphibins} from 0 to {ndirbins - nphibins}, e.g. 90 for the pole at +z",
        )
    if averageovertheta:
        if 0 <= dirbin < nphibins:
            return
        exampletext = f", e.g. {dirbin % nphibins} for the ϕ range of {dirbin}" if isdirbin else ""
        exit_with_error(
            f"{optionname} {dirbin} is not the first direction bin of a phi bin",
            f"Give {alltext}a bin from 0 to {nphibins - 1}{exampletext}",
        )
    if not isdirbin:
        exit_with_error(f"{optionname} {dirbin} is no direction bin", f"Give {alltext}a bin from 0 to {ndirbins - 1}")


def get_phi_expr(x: pl.Expr, y: pl.Expr, z: pl.Expr) -> pl.Expr:
    """Return ϕ = atan2(y, x) of each direction, from 0 to 2π, with ϕ from +x towards +y, as in ARTIS.

    ARTIS takes cos ϕ = 1 for a direction with sin θ < 1e-12 (get_escapedirectionbin in vectors.h), where ϕ has no
    definition. Thus such a direction gets ϕ = 0, or ϕ = 2π for a negative y, and the bin of ARTIS.
    """
    isonzaxis = x**2 + y**2 <= 1e-24 * (x**2 + y**2 + z**2)
    return pl.when(~isonzaxis).then(pl.arctan2(y, x) % (2 * math.pi)).when(y < 0).then(2 * math.pi).otherwise(0.0)


def get_costhetabin_expr(costheta: pl.Expr, ncosthetabins: int | None = None) -> pl.Expr:
    """Return the costheta bin of ARTIS for each cos θ. The last bin includes cos θ = 1, and NaN gives bin 0."""
    if ncosthetabins is None:
        ncosthetabins = get_viewingdirection_costhetabincount()
    return pl.min_horizontal(((costheta + 1) / 2.0 * ncosthetabins).fill_nan(0).cast(pl.Int32), ncosthetabins - 1)


def get_phibin_expr(phi: pl.Expr, diry: pl.Expr, nphibins: int | None = None) -> pl.Expr:
    """Return the historical phi bin of ARTIS for each direction, from phi and the sign of the y component.

    ARTIS bins the angle acos(cos ϕ), which is 2π - ϕ for a negative y component, or acos(cos ϕ) + π, which is ϕ + π
    for the other directions. Thus the bin index decreases as phi increases from π to 2π.
    """
    if nphibins is None:
        nphibins = get_viewingdirection_phibincount()
    # the angle ϕ + π reaches exactly 2π at ϕ = π, which would otherwise land in the first phi bin of the next costheta
    # ring, so clamp to the last bin
    return pl.min_horizontal(
        (pl.when(diry < 0).then(2 * math.pi - phi).otherwise(phi + math.pi) / (2 * math.pi) * nphibins)
        .fill_nan(0)
        .cast(pl.Int32),
        nphibins - 1,
    )


class CellDirectionRange(t.NamedTuple):
    """The range of the directions of the centres of the cells that a command selects.

    cos θ = v_z / |v|, and ϕ is the azimuth from the +x axis toward the +y axis, from 0 to 2π, as in ARTIS.
    costhetabin and phibin are the bins of ARTIS that --dirbin selects, and None selects each cos θ or each ϕ. Thus
    each cell is in one bin. With no bin, costhetamin and costhetamax give the range of --costhetamin and
    --costhetamax, and that range includes both edges.
    """

    costhetamin: float
    costhetamax: float
    costhetabin: int | None
    phibin: int | None


def addarg_celldirectionrange(parser: argparse.ArgumentParser, *, addaverageoverphi: bool = True) -> None:
    """Add the options that select cells by their direction, e.g. --dirbin and --costhetamin.

    The options are --dirbin, --average-over-phi-angle, --average-over-theta-angle, --costhetamin, and --costhetamax. A
    command that applies --average-over-phi-angle also to a different option gives addaverageoverphi=False, and then
    adds the flag with addarg_averageoverphi.
    """
    parser.add_argument(
        "-dirbin",
        type=int,
        help=(
            "Select the cells whose centre is in this direction bin. The cos θ = v_z / |v| and the azimuth ϕ of the"
            " centre must be in the ranges of the bin. These are the direction bins of the spectra and the light"
            " curves. A 2D model is symmetric around the z axis, thus the ϕ range has no effect for it"
        ),
    )
    if addaverageoverphi:
        parser.add_argument(
            "--average_over_phi_angle",
            action="store_true",
            help=(
                "Select the cells in the cos θ range of --dirbin at each ϕ. Then --dirbin must be the first bin of"
                " its costheta bin, e.g. 90 or 50"
            ),
        )
    parser.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help=(
            "Select the cells in the ϕ range of --dirbin at each cos θ. Then --dirbin must be from 0 to 9, the bin of"
            " its ϕ range in the first costheta bin. A 2D model has no ϕ, thus it refuses this option"
        ),
    )
    parser.add_argument(
        "-costhetamin",
        type=float,
        help=(
            "Minimum cos θ = v_z / |v| of the centre of a cell, e.g. 0.8 for the cells near the pole at +z, at each ϕ."
            " Default: -1"
        ),
    )
    parser.add_argument(
        "-costhetamax",
        type=float,
        help="Maximum cos θ = v_z / |v| of the centre of a cell, e.g. 0.2 with --costhetamin 0, at each ϕ. Default: 1",
    )


def get_cell_direction_range(args: argparse.Namespace, quiet: bool = False) -> CellDirectionRange | None:
    """Return the directions of --dirbin or of --costhetamin and --costhetamax, or None to keep each cell.

    --dirbin uses the bins of ARTIS, thus each cell is in one bin. The range of --costhetamin and --costhetamax
    includes both edges. With quiet, the function does not log the range of --dirbin.
    """
    averageoverphi = getattr(args, "average_over_phi_angle", False)
    averageovertheta = getattr(args, "average_over_theta_angle", False)
    if args.dirbin is not None:
        if args.costhetamin is not None or args.costhetamax is not None:
            exit_with_error(
                "--dirbin gives a range of directions, thus --costhetamin and --costhetamax do not apply",
                "Remove --dirbin, or remove --costhetamin and --costhetamax",
            )
        if averageoverphi and averageovertheta:
            exit_with_error(
                "--average-over-phi-angle and --average-over-theta-angle together leave no direction bin",
                "Remove one of them",
            )
        check_dirbin(
            args.dirbin, "-dirbin", averageoverphi=averageoverphi, averageovertheta=averageovertheta, allowall=False
        )
        costhetalowers, costhetauppers, _ = get_costheta_bins(usedegrees=False)
        costhetabin, phibin = divmod(args.dirbin, get_viewingdirection_phibincount())
        directionrange = (
            CellDirectionRange(-1.0, 1.0, None, phibin)
            if averageovertheta
            else CellDirectionRange(
                float(costhetalowers[costhetabin]),
                float(costhetauppers[costhetabin]),
                costhetabin,
                None if averageoverphi else phibin,
            )
        )
        if not quiet:
            print_detail(
                f"--dirbin {args.dirbin} selects the cells whose centre has"
                f" {format_cell_direction_range(directionrange)}. The z axis is the polar axis of the direction bins,"
                " and ϕ = 0 is the +x axis"
            )
        return directionrange

    # plotlinecontributions applies the flag also to --troughdirbin
    if averageoverphi and getattr(args, "troughdirbin", None) is None:
        exit_with_error(
            "--average-over-phi-angle needs a direction bin", "Give --dirbin, or remove --average-over-phi-angle"
        )
    if averageovertheta:
        exit_with_error(
            "--average-over-theta-angle needs a direction bin", "Give --dirbin, or remove --average-over-theta-angle"
        )
    if args.costhetamin is None and args.costhetamax is None:
        return None
    costhetarange = (
        -1.0 if args.costhetamin is None else args.costhetamin,
        1.0 if args.costhetamax is None else args.costhetamax,
    )
    if not -1.0 <= costhetarange[0] < costhetarange[1] <= 1.0:
        exit_with_error(
            f"--costhetamin {costhetarange[0]:g} and --costhetamax {costhetarange[1]:g} give no range of cos θ",
            "Give -1 ≤ --costhetamin < --costhetamax ≤ 1",
        )
    return CellDirectionRange(*costhetarange, costhetabin=None, phibin=None)


def format_cell_direction_range(directionrange: CellDirectionRange) -> str:
    """Return the text of a range of the cell directions, e.g. "0.0 ≤ cos θ < 0.2, 9π/5 < ϕ ≤ 2π".

    The text of --dirbin gives the edges of the bin of ARTIS, as get_dirbin_labels does. The range of --costhetamin and
    --costhetamax includes both edges.
    """
    nphibins = get_viewingdirection_phibincount()
    costhetabin, phibin = directionrange.costhetabin, directionrange.phibin
    if costhetabin is not None and phibin is not None:
        dirbin = costhetabin * nphibins + phibin
        return get_dirbin_labels([dirbin])[dirbin]
    if costhetabin is not None:
        # get_dirbin_labels gives the label of a costheta bin to its first direction bin
        return get_dirbin_labels([costhetabin * nphibins], average_over_phi=True)[costhetabin * nphibins]
    if phibin is not None:
        return get_dirbin_labels([phibin], average_over_theta=True)[phibin]
    return f"{directionrange.costhetamin:g} ≤ cos θ ≤ {directionrange.costhetamax:g}"


def warn_direction_bin_noise(source: str, *, average_over_phi: bool, average_over_theta: bool) -> None:
    """Print a warning that a direction bin holds a small part of the packets, and thus has a larger noise.

    A file of the direction bins, e.g. spec_res.out, gives no packet count. If the packets escape in all directions
    alike, a bin of the n bins holds 1/n of the packets, and its relative noise is sqrt(n) times that of the average.
    """
    bincount = get_viewingdirection_costhetabincount() * get_viewingdirection_phibincount()
    if average_over_phi:
        bincount = get_viewingdirection_costhetabincount()
    elif average_over_theta:
        bincount = get_viewingdirection_phibincount()
    print_warning(
        f"Each direction bin of {source} holds about 1/{bincount} of the escaped packets, if they escape in all"
        f" directions alike. Thus its Monte Carlo noise is about {math.sqrt(bincount):.3g} times the noise of the"
        f" angle average. {source} gives no packet count, and --frompackets gives the noise of each bin"
    )
