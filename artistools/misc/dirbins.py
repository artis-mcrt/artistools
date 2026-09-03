"""Viewing-direction (costheta/phi) bin definitions, labels, selection, and averaging."""

import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

from artistools.misc.modelinfo import get_vpkt_config


def split_multitable_dataframe(res_df: pl.DataFrame | pl.LazyFrame) -> dict[int, pl.LazyFrame]:
    """Res (angle-resolved) files include a table for each direction bin."""
    res_df = res_df.lazy()
    rowcount = res_df.select(pl.len()).collect().item()
    nu_points = res_df.select(cs.by_index(0).n_unique()).collect().item()
    assert rowcount % nu_points == 0
    tablecount = rowcount // nu_points

    return {
        tableindex: (res_df.select(pl.all().slice(tableindex * nu_points, nu_points)))
        for tableindex in range(tablecount)
    }


def average_direction_bins(
    dirbindataframes: dict[int, pl.DataFrame] | dict[int, pl.LazyFrame], overangle: t.Literal["phi", "theta"]
) -> dict[int, pl.LazyFrame]:
    """Average dict of direction-binned polars DataFrames according to the phi or theta angle.

    Every column is averaged, which is only valid for a column that is linear in the bin contributions. A
    column derived non-linearly from the others, such as a magnitude, is not the average of its bins: a
    single dark bin sends the mean to infinity. Derive such a column after averaging, not before.
    """
    dirbincount = get_viewingdirectionbincount()
    nphibins = get_viewingdirection_phibincount()

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

    # we will make a copy to ensure that we don't cause side effects from altering the original DataFrames
    # that might be returned again later by an lru_cached function
    dirbindataframesout: dict[int, pl.LazyFrame] = {}

    for start_bin in start_bin_range:
        # dirbin == costheta_index * nphibins + phi_index, so the bins sharing a costheta index are contiguous, while
        # the bins sharing a phi index are nphibins apart
        contribbins = list(
            range(start_bin, start_bin + nphibins) if overangle == "phi" else range(start_bin, dirbincount, nphibins)
        )

        dirbindataframesout[start_bin] = dirbindataframes[start_bin].lazy()
        firstcolname = dirbindataframes[start_bin].collect_schema().names()[0]
        for dirbin in contribbins[1:]:
            dirbindataframesout[start_bin] = dirbindataframesout[start_bin].join(
                dirbindataframes[dirbin].lazy(),
                on=firstcolname,
                how="left",
                suffix=f"_dirbin{dirbin}",
                maintain_order="left",
            )

        dirbindataframesout[start_bin] = dirbindataframesout[start_bin].select(
            cs.by_index(0),
            *[
                (
                    pl.sum_horizontal([pl.col(col), *[pl.col(f"{col}_dirbin{dirbin}") for dirbin in contribbins[1:]]])
                    / len(contribbins)
                ).alias(col)
                for col in dirbindataframes[start_bin].collect_schema().names()[1:]
            ],
        )

        print(f"bin number {start_bin:2d} = the average of bins {contribbins}")

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
    """Return the viewing direction bin indices, reduced to the first bin of each averaging group when averaging over phi or theta angle."""
    check_averaging_angles(average_over_phi, average_over_theta)
    if average_over_phi:
        return list(range(0, get_viewingdirectionbincount(), get_viewingdirection_phibincount()))
    if average_over_theta:
        return list(range(get_viewingdirection_phibincount()))
    return list(range(get_viewingdirectionbincount()))


def get_dirbin_definitions(
    modelpath: Path | str,
    dirbins: Sequence[int] | None = None,
    *,
    vpkt_observers: bool = False,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
) -> dict[int, str]:
    """Return a label for each direction bin, or for each virtual packet observer when vpkt_observers is set."""
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
        "Spherical polar: x = r sinθ cosϕ, y = r sinθ sinϕ, z = r cosθ -> θ=0 is +Z and θ=π is -Z. At Z=0, ϕ=0 is +X and ϕ=π/2 is +Y"
    )


def get_phi_bins(usedegrees: bool) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], list[str]]:
    """Return the lower and upper phi boundaries of each direction bin, and a label for each."""
    nphibins = get_viewingdirection_phibincount()
    # pi/2 must be an exact boundary because of the change in behaviour there
    assert nphibins % 2 == 0

    # for historical reasons, phi bins are descending and include a flip at half way
    # phisteps = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5] for nphibins == 10
    phisteps = list(range(nphibins // 2)) + list(reversed(range(nphibins // 2, nphibins)))

    # set up monotonic descending phi bin boundaries
    phi_lower = np.array([2 * math.pi * (1 - (step + 1) / nphibins) for step in phisteps])
    phi_upper = np.array([2 * math.pi * (1 - step / nphibins) for step in phisteps])

    binlabels = ["" for _ in range(nphibins)]
    for phibin, phibinmonotonicdesc in enumerate(phisteps):
        if usedegrees:
            str_phi_lower = f"{phi_lower[phibinmonotonicdesc] / math.pi * 180:3.0f}°"
            str_phi_upper = f"{phi_upper[phibinmonotonicdesc] / math.pi * 180:3.0f}°"
        else:
            coeff_lower = phi_lower[phibinmonotonicdesc] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_lower, round(coeff_lower), rtol=0.01), coeff_lower
            str_phi_lower = f"{round(coeff_lower)}π/{nphibins // 2}" if phi_lower[phibinmonotonicdesc] > 0.0 else "0"
            coeff_upper = phi_upper[phibinmonotonicdesc] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_upper, round(coeff_upper), rtol=0.01)
            str_phi_upper = (
                f"{round(coeff_upper)}π/{nphibins // 2}" if phi_upper[phibinmonotonicdesc] < 2 * math.pi else "2π"
            )

        lower_compare = "≤" if phibin < (nphibins // 2) else "<"
        upper_compare = "≤" if phibin > (nphibins // 2) else "<"
        binlabels[phibinmonotonicdesc] = f"{str_phi_lower} {lower_compare} ϕ {upper_compare} {str_phi_upper}"

    # if nphibins == 10, then binlabels = [
    #     "9π/5 ≤ ϕ < 2π",
    #     "8π/5 ≤ ϕ < 9π/5",
    #     "7π/5 ≤ ϕ < 8π/5",
    #     "6π/5 ≤ ϕ < 7π/5",
    #     "5π/5 ≤ ϕ < 6π/5",
    #     "0 < ϕ ≤ 1π/5",
    #     "1π/5 < ϕ ≤ 2π/5",
    #     "2π/5 < ϕ ≤ 3π/5",
    #     "3π/5 < ϕ ≤ 4π/5",
    #     "4π/5 < ϕ < 5π/5",
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
    costhetabins_lower = np.arange(-1.0, 1.0, 2.0 / ncosthetabins)
    costhetabins_upper = costhetabins_lower + 2.0 / ncosthetabins
    if usedegrees:
        thetabins_upper = np.arccos(costhetabins_lower) / np.pi * 180
        thetabins_lower = np.arccos(costhetabins_upper) / np.pi * 180

        binlabels = [
            f"{lower:.0f}° < θ < {upper:.0f}°" for lower, upper in zip(thetabins_lower, thetabins_upper, strict=False)
        ]
    else:
        binlabels = [
            f"{lower:.1f} ≤ cos θ < {upper:.1f}"
            for lower, upper in zip(costhetabins_lower, costhetabins_upper, strict=False)
        ]
    return tuple(float(x) for x in costhetabins_lower), tuple(costhetabins_upper), binlabels


def get_costhetabin_phibin_labels(usedegrees: bool) -> tuple[list[str], list[str]]:
    """Return the cos(theta) and phi bin labels."""
    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)
    return costhetabinlabels, phibinlabels


def get_opacity_condition_label(z_exclude: int) -> str:
    """Return the label for a virtual packet opacity exclusion code, such as 'no-bb' or 'no-Fe'."""
    from artistools.atomic._atomic_core import get_elsymbol

    # codes match the opacityexclusions handling in read_vpktparameterfile() and trace_vpkt_direction()
    labels = {0: "", -1: "no-bb", -2: "no-bf", -3: "no-ff", -4: "no-es"}

    return labels[z_exclude] if z_exclude in labels else f"no-{get_elsymbol(z_exclude)}"


def get_vspec_dir_labels(modelpath: str | Path, usedegrees: bool = False) -> dict[int, str]:
    """Return a label for each virtual packet observer direction and opacity choice combination."""
    vpkt_config = get_vpkt_config(modelpath)
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
    if modelpath:
        modelpath = Path(modelpath)
        MABINS = get_viewingdirectionbincount()
        if list(modelpath.glob("*_res_00.out*")):
            # if the first direction bin file exists, check:
            # check last bin exists
            assert list(modelpath.glob(f"*_res_{MABINS - 1:02d}.out*"))
            # check one beyond does not exist
            assert not list(modelpath.glob(f"*_res_{MABINS:02d}.out*"))

    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)

    nphibins = get_viewingdirection_phibincount()

    if dirbins is None:
        dirbins = get_dirbins(average_over_phi=average_over_phi, average_over_theta=average_over_theta)

    angle_definitions: dict[int, str] = {}
    for dirbin in dirbins:
        dirbin_int = int(dirbin)
        if dirbin_int == -1:
            angle_definitions[dirbin_int] = "all directions"
            continue

        costheta_index = dirbin_int // nphibins
        phi_index = dirbin_int % nphibins

        if average_over_phi:
            angle_definitions[dirbin_int] = costhetabinlabels[costheta_index]
            assert phi_index == 0
            assert not average_over_theta
        elif average_over_theta:
            angle_definitions[dirbin_int] = phibinlabels[phi_index]
            assert costheta_index == 0
        else:
            angle_definitions[dirbin_int] = f"{costhetabinlabels[costheta_index]}, {phibinlabels[phi_index]}"

    return angle_definitions
