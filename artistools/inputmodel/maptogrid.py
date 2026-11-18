# adapted from Fortran maptogrid.f90 and kernelmodule.f90
# original Fortran code by Andreas Bauswein
"""Map SPH ejecta particles to a Cartesian grid or a spherical grid with a cubic spline kernel."""

import argparse
import math
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import geometric_time_to_s
from artistools.inputmodel.core import savetologfile
from artistools.inputmodel.modelfromhydro import read_ejectasnapshot
from artistools.inputmodel.modelfromhydro import SPHERICAL_GRIDDAT_EDGE_KEYS
from artistools.inputmodel.rprocess_from_trajectory import save_gridparticlecontributions
from artistools.misc import addarg_force
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import parse_cli_args

itable = 40000  # wie fein Kernelfkt interpoliert wird
itab = itable + 5

#
# --maximum interaction length and step size
#
v2max = 4.0
dvtable = v2max / itable
i1 = int(1.0 // dvtable)


def get_wij() -> npt.NDArray[np.float64]:
    """Return a lookup table of the normalised cubic spline kernel sampled on a grid of v^2."""
    #
    # --normalisation constant
    #
    cnormk = 1.0 / math.pi
    # build the tables. The entries above itable stay zero
    wij = np.zeros(itab + 1)
    i = np.arange(1, itable + 1)
    v2 = i * dvtable
    v = np.sqrt(v2)
    dif2 = 2.0 - v
    # v less than 1 for the entries up to i1, and v greater than 1 above
    vsum = np.where(i <= i1, 1.0 - 1.5 * v2 + 0.75 * (v * v2), 0.25 * dif2 * dif2 * dif2)
    wij[1 : itable + 1] = cnormk * vsum
    # the kernel at v = 0 takes the v < 1 branch, which gives 1 before the normalisation. Entry 0 held
    # zero, thus kernelvals2 gave too small a weight to each pair inside one step of the table
    wij[0] = cnormk

    return wij


def kernelvals2(
    rij2: npt.NDArray[np.float64], hmean: float, wij: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:  # the calculation is fast, but it gives no gradients
    """Return the kernel values for the pair separations rij2 and a mean smoothing length, interpolated from wij."""
    hmean21 = 1.0 / hmean**2
    hmean31 = hmean21 / hmean
    v2 = rij2 * hmean21
    indexfloat = np.floor(v2 / dvtable)
    dxx = v2 - indexfloat * dvtable
    index = indexfloat.astype(np.int64)
    wijlow = np.take(wij, index)
    dwdx = (np.take(wij, index + 1) - wijlow) / dvtable
    return (wijlow + dwdx * dxx) * hmean31


# the default cell counts n_r, n_theta, and n_phi of a spherical grid
SPHERICAL_GRID_SHAPE_DEFAULT = (25, 16, 32)


def get_modified_smoothing_lengths(
    h: npt.NDArray[np.float64], dis: npt.NDArray[np.float64], rmean: float, hmean: float, modifysmoothinglength: str
) -> npt.NDArray[np.float64]:
    """Return the smoothing length of each particle for a spherical map, after the option of --modifysmoothinglength.

    The options give the lengths that the Cartesian map gives. option1, option5, and option6 set the length from the
    width of a Cartesian cell, thus a spherical map stops for them.
    """
    if modifysmoothinglength == "False":
        return h
    if modifysmoothinglength == "option2":
        return np.maximum(h, 0.25 * dis)
    if modifysmoothinglength == "option3":
        return np.where(dis > 1.5 * rmean, np.maximum(h, 0.4 * dis), h)
    if modifysmoothinglength == "option4":
        return np.where(dis > rmean, np.maximum(h, 1.5 * hmean), h)
    msg = (
        f"--modifysmoothinglength {modifysmoothinglength} sets the smoothing length from the width of a Cartesian"
        " cell. For --coordinates spherical, give False, option2, option3, or option4"
    )
    raise ValueError(msg)


def get_spherical_grid_edges(
    r_outer: float, ngridr: int, ngridtheta: int, ngridphi: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the edges of the cells of a spherical grid in the radius, the polar angle, and the azimuthal angle.

    The radial edges and the azimuthal edges are uniform. The polar edges are uniform in cos(theta), thus the
    cells of one shell have equal volumes.
    """
    theta_edges = np.arccos(np.linspace(1.0, -1.0, ngridtheta + 1))
    phi_edges = np.linspace(0.0, 2.0 * math.pi, ngridphi + 1)
    # arccos(-1) and the last value of linspace can differ from pi and 2 pi in the last bit
    theta_edges[-1] = math.pi
    phi_edges[-1] = 2.0 * math.pi
    return np.linspace(0.0, r_outer, ngridr + 1), theta_edges, phi_edges


def get_spherical_cell_volumes(
    r_edges: npt.NDArray[np.float64], theta_edges: npt.NDArray[np.float64], phi_edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the volume of each cell of a spherical grid, with the shape (n_r, n_theta, n_phi)."""
    radialpart = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
    polarpart = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    return radialpart[:, np.newaxis, np.newaxis] * polarpart[np.newaxis, :, np.newaxis] * np.diff(phi_edges)


def get_spherical_deposit(
    position: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
    edges: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    wij: npt.NDArray[np.float64],
    normalisekernelweights: bool = False,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the mass that each particle gives to each cell of a spherical grid.

    position has one row of x, y, and z for each particle, and edges holds the cell edges in r, theta, and phi. The
    result holds the particle index, the cell index, and the mass of each pair, and the sum of the weights of each
    particle. The cell index is i_r + n_r * (i_theta + n_theta * i_phi).

    A particle gives mass to the cells whose mid-points are in its kernel, which has the radius 2h. The weight of a
    cell is the kernel value at its mid-point times its volume.

    The default method is the method of the Cartesian map. The mass of a pair is the mass of the particle times the
    weight, thus the density of a cell is the sum of mass times kernel value over the particles. The sum of the
    weights of a particle is 1 only if the cells resolve its kernel, thus the grid does not hold exactly the mass of
    the particles. A particle outside the grid gives mass to each cell mid-point in its kernel.

    With normalisekernelweights, the function normalises the weights of a particle to a sum of 1. A particle with no
    cell mid-point in its kernel gives all its mass to the cell that holds it, and its sum of weights is zero. A
    particle outside the outer radius gives no mass, and its sum of weights is NaN. Thus the grid holds exactly the
    mass of each particle inside the outer radius. The particle with the largest radius is at the outer radius for a
    grid to that radius, and it is inside the grid.
    """
    r_edges, theta_edges, phi_edges = edges
    ngridr, ngridtheta, ngridphi = (len(axisedges) - 1 for axisedges in edges)
    r_mid, theta_mid, phi_mid = ((axisedges[:-1] + axisedges[1:]) / 2.0 for axisedges in edges)
    sintheta_mid, costheta_mid = np.sin(theta_mid), np.cos(theta_mid)
    # a search of three copies of the azimuthal mid-points finds a range that goes through phi = 0
    phi_mid_periodic = np.concatenate((phi_mid - 2.0 * math.pi, phi_mid, phi_mid + 2.0 * math.pi))
    volumes = get_spherical_cell_volumes(r_edges, theta_edges, phi_edges)
    alltheta, allphi = np.arange(ngridtheta), np.arange(ngridphi)

    r_particle = np.sqrt(np.sum(position**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        costheta_particle = np.where(r_particle > 0.0, position[:, 2] / r_particle, 1.0)
    theta_particle = np.arccos(np.clip(costheta_particle, -1.0, 1.0))
    phi_particle = np.arctan2(position[:, 1], position[:, 0]) % (2.0 * math.pi)
    # the remainder of a small negative angle rounds to 2 pi, which is the angle 0
    phi_particle[phi_particle >= 2.0 * math.pi] = 0.0

    weightsums = np.full(len(h), math.nan)
    pairparticles: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    paircells: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    pairmasses: list[npt.NDArray[np.float64]] = [np.empty(0)]

    particleindices = np.flatnonzero(r_particle <= r_edges[-1]) if normalisekernelweights else np.arange(len(h))
    for n in particleindices.tolist():
        kernelradius = 2.0 * float(h[n])
        r_p, theta_p, phi_p = float(r_particle[n]), float(theta_particle[n]), float(phi_particle[n])

        # the kernel is inside a cone around the direction of the particle with the half angle asin(2h / r). A kernel
        # that holds the centre of the grid reaches each direction, and a kernel that holds the z axis reaches each phi
        thetaindices, phiindices = alltheta, allphi
        if kernelradius < r_p:
            halfangle = math.asin(kernelradius / r_p)
            thetaindices = np.arange(
                np.searchsorted(theta_mid, theta_p - halfangle, side="left"),
                np.searchsorted(theta_mid, theta_p + halfangle, side="right"),
            )
            if theta_p - halfangle > 0.0 and theta_p + halfangle < math.pi:
                phihalfrange = math.asin(min(1.0, math.sin(halfangle) / math.sin(theta_p)))
                phistart = int(np.searchsorted(phi_mid_periodic, phi_p - phihalfrange, side="left"))
                phiend = int(np.searchsorted(phi_mid_periodic, phi_p + phihalfrange, side="right"))
                if phiend - phistart < ngridphi:
                    phiindices = np.arange(phistart, phiend) % ngridphi
        rstart = int(np.searchsorted(r_mid, r_p - kernelradius, side="left"))
        rend = int(np.searchsorted(r_mid, r_p + kernelradius, side="right"))

        # the distance to a cell mid-point is a Cartesian distance, from the law of cosines
        cosangle = sintheta_mid[thetaindices, np.newaxis] * math.sin(theta_p) * np.cos(
            phi_mid[phiindices] - phi_p
        ) + costheta_mid[thetaindices, np.newaxis] * math.cos(theta_p)
        r_cells = r_mid[rstart:rend, np.newaxis, np.newaxis]
        dis2 = np.maximum(r_cells**2 + r_p**2 - 2.0 * r_cells * r_p * cosangle, 0.0)
        box_r, box_theta, box_phi = np.nonzero(dis2 <= kernelradius**2)
        cell_r, cell_theta, cell_phi = box_r + rstart, thetaindices[box_theta], phiindices[box_phi]
        weights = kernelvals2(dis2[box_r, box_theta, box_phi], float(h[n]), wij) * volumes[cell_r, cell_theta, cell_phi]
        weightsum = float(weights.sum())
        weightsums[n] = weightsum

        if not normalisekernelweights:
            # as in the Cartesian map, each cell mid-point in the kernel makes a pair, also with a weight of zero
            paircells.append(cell_r + ngridr * (cell_theta + ngridtheta * cell_phi))
            pairmasses.append(mass[n] * weights)
        elif weightsum > 0.0:
            hasweight = weights > 0.0
            paircells.append((cell_r + ngridr * (cell_theta + ngridtheta * cell_phi))[hasweight])
            pairmasses.append(mass[n] * weights[hasweight] / weightsum)
        else:
            # a cell holds its lower edge, and the last cell of an axis also holds the upper edge of the grid
            index_r, index_theta, index_phi = (
                min(int(np.searchsorted(axisedges, value, side="right")) - 1, len(axisedges) - 2)
                for axisedges, value in zip(edges, (r_p, theta_p, phi_p), strict=True)
            )
            paircells.append(np.array([index_r + ngridr * (index_theta + ngridtheta * index_phi)], dtype=np.int64))
            pairmasses.append(np.array([mass[n]]))
        pairparticles.append(np.full(paircells[-1].size, n, dtype=np.int64))

        # one small array for each particle costs much memory for millions of particles
        if len(paircells) > 65536:
            pairparticles = [np.concatenate(pairparticles)]
            paircells = [np.concatenate(paircells)]
            pairmasses = [np.concatenate(pairmasses)]

    return np.concatenate(pairparticles), np.concatenate(paircells), np.concatenate(pairmasses), weightsums


def map_particles_to_spherical_grid(
    outputfolderpath: Path,
    dfparticles: pl.DataFrame,
    gridshape: tuple[int, int, int],
    r_outer: float,
    dtextra: float,
    logprint: Callable[..., None],
    normalisekernelweights: bool = False,
) -> None:
    """Map SPH particles to a spherical grid of r, theta, and phi, and write grid.dat and gridcontributions.txt.zst.

    dfparticles holds the columns id, x, y, z, h, pmass, mass, and ye of the particles, with the smoothing length
    after the option of --modifysmoothinglength. mass is pmass * rho / rho_rst, which is the mass that the particle
    gives to the density field. gridshape gives the cell counts n_r, n_theta, and n_phi. get_spherical_deposit
    describes the default method and the method of normalisekernelweights. The cell ids of both files start at 0, as
    format version 2 of model.txt needs. The files of a Cartesian map have cell ids from 1.
    """
    edges = get_spherical_grid_edges(r_outer, *gridshape)
    volumes = get_spherical_cell_volumes(*edges)
    ncells = volumes.size
    logprint(
        "Coordinates: spherical. theta is the polar angle from the +z axis, and phi = atan2(y, x) is the azimuthal"
        " angle. The index of a cell is i_r + n_r * (i_theta + n_theta * i_phi)"
    )
    logprint(
        f"Grid: n_r x n_theta x n_phi = {' x '.join(str(count) for count in gridshape)} = {ncells} cells. The radial"
        f" edges are uniform from 0 to {r_outer}. The polar edges are uniform in cos(theta), thus the cells of one"
        " shell have equal volumes. The azimuthal edges are uniform"
    )
    if normalisekernelweights:
        logprint(
            "Deposit method: normalised kernel weights (--normalise-kernel-weights). A particle gives its mass to the"
            " cells whose mid-points are in its kernel of radius 2h. The weight of a cell is the cubic spline kernel at"
            " its mid-point times its volume. The deposit normalises the weights of a particle to a sum of 1. A"
            " particle with no cell mid-point in its kernel gives all its mass to the cell that holds it. A particle"
            " outside the outer radius gives no mass. The Cartesian map does not have this method"
        )
    else:
        logprint(
            "Deposit method: kernel sample at the cell mid-point, which is the method of the Cartesian map. The density"
            " of a cell is the sum over the particles of pmass * rho / rho_rst * W(d, h). W is the cubic spline"
            " kernel, d is the distance from the cell mid-point to the particle, and h is the smoothing length. The"
            " mass on the grid is the mass of the particles only if the cells resolve each kernel"
        )
    logprint("The mid-point of a cell is the mean of its edges in r, theta, and phi")
    logprint(
        "Assumption: the mass of a particle on the grid is pmass * rho / rho_rst, the same factor as the Cartesian"
        " map (source: maptogrid.f90 of A. Bauswein)"
    )

    particlemass = dfparticles["mass"].to_numpy()
    particleye = dfparticles["ye"].to_numpy()
    position = dfparticles.select("x", "y", "z").to_numpy()
    pairparticle, paircell, pairmass, weightsums = get_spherical_deposit(
        position, dfparticles["h"].to_numpy(), particlemass, edges, get_wij(), normalisekernelweights
    )

    isinside = np.sqrt(np.sum(position**2, axis=1)) <= r_outer
    hasnocellinkernel = isinside & ~(weightsums > 0.0)
    logprint(
        f"Particles inside the outer radius: {np.count_nonzero(isinside)} of {len(weightsums)}, with the mass"
        f" pmass * rho / rho_rst = {particlemass[isinside].sum()} of {particlemass.sum()}"
    )
    logprint(
        "Particles inside the outer radius with no cell mid-point in the kernel:"
        f" {np.count_nonzero(hasnocellinkernel)}, with the mass {particlemass[hasnocellinkernel].sum()}. "
        + ("Each one gives its mass to the cell that holds it" if normalisekernelweights else "They give no mass")
    )
    if np.any(weightsums > 0.0):
        quantiles = np.quantile(weightsums[weightsums > 0.0], [0.05, 0.5, 0.95])
        logprint(
            "Sum of the weights of a particle before the normalisation:"
            f" 5% quantile {quantiles[0]:.3g}, median {quantiles[1]:.3g}, 95% quantile {quantiles[2]:.3g}."
            " The sum is 1 for a kernel that the cells resolve"
        )
    logprint(
        f"Pairs of a particle and a cell: {pairmass.size}, which is"
        f" {pairmass.size / max(1, np.count_nonzero(isinside)):.1f} for each particle inside the outer radius"
    )

    cellmass = np.bincount(paircell, weights=pairmass, minlength=ncells)
    cellyemass = np.bincount(paircell, weights=pairmass * particleye[pairparticle], minlength=ncells)
    celltracercount = np.bincount(paircell, minlength=ncells)
    # the order of the flat arrays has the radial index fastest, which is the Fortran order of [i_r, i_theta, i_phi]
    cellrho = cellmass / volumes.ravel(order="F")
    with np.errstate(divide="ignore", invalid="ignore"):
        cellye = np.where(cellmass > 0.0, cellyemass / cellmass, 0.0)

    gridcontribpath = save_gridparticlecontributions(
        pl.DataFrame({
            "particleid": dfparticles["id"].to_numpy()[pairparticle],
            # the cell ids of a spherical model start at 0, as format version 2 of model.txt needs
            "cellindex": paircell,
            "frac_of_cellmass": pairmass / cellmass[paircell],
        }),
        Path(outputfolderpath, "gridcontributions.txt"),
        float_precision=None,
    )
    logprint(f"Saved {gridcontribpath}")

    gridmass = float(np.sum(cellrho * volumes.ravel(order="F")))
    totmass = float(dfparticles["pmass"].sum())
    logprint(f"Fraction of total mass on grid {gridmass / totmass}")
    logprint(
        f"{'WARNING!' if gridmass / totmass < 0.9 else ''} mass on grid from rho*V: {gridmass} mass of particles:"
        f" {totmass}. Mass pmass * rho / rho_rst of the particles inside the outer radius:"
        f" {particlemass[isinside].sum()}"
    )
    nempty = int(np.count_nonzero(cellmass == 0.0))
    logprint(f"Number of cells with rho=0 {nempty}, total num of cells {ncells}, fraction {nempty / ncells}")

    with Path(outputfolderpath, "grid.dat").open("w", encoding="utf-8") as fgrid:
        fgrid.write("coordinates spherical3d\n")
        fgrid.write(f"dtextra_geomunits {dtextra} # extra time after explosion simulation ended (in geom units)\n")
        fgrid.writelines(
            f"{key} {' '.join(f'{edge:.17g}' for edge in axisedges.tolist())}\n"
            for key, axisedges in zip(SPHERICAL_GRIDDAT_EDGE_KEYS, edges, strict=True)
        )
        fgrid.write(" gridindex    rho    Ye    tracercount\n")
        fgrid.writelines(
            f"{gridindex:8d} {rho} {ye} {tracercount}\n"
            for gridindex, rho, ye, tracercount in zip(
                range(ncells), cellrho.tolist(), cellye.tolist(), celltracercount.tolist(), strict=True
            )
        )

    logprint(f"Saved {outputfolderpath / 'grid.dat'}")


def maptogrid(
    ejectasnapshotpath: Path,
    outputfolderpath: Path | str,
    ncoordgrid: int = 50,
    downsamplefactor: int = 1,
    dtextra_seconds: float = 0.5,
    setgrid_fractionrmax: float = 0.5,
    modifysmoothinglength: str = "option4",
    samplelowercorner: bool = False,
    coordinates: str = "cartesian",
    sphericalgridshape: tuple[int, int, int] = SPHERICAL_GRID_SHAPE_DEFAULT,
    normalisekernelweights: bool = False,
) -> None:
    """Map an SPH ejecta snapshot to a grid and write grid.dat and gridcontributions.txt.zst.

    The coordinates cartesian give an ncoordgrid^3 Cartesian grid. The coordinates spherical give a grid of r,
    theta, and phi with the cell counts of sphericalgridshape, and map_particles_to_spherical_grid makes it. Both
    maps use one deposit method, unless normalisekernelweights selects the normalised weights for a spherical map.

    samplelowercorner evaluates the kernel at the lower corner of each cell, as artistools did before. Use it only
    to make a grid.dat again that an older version of artistools wrote.
    """
    if not ejectasnapshotpath.is_file():
        msg = f"{ejectasnapshotpath} does not exist"
        raise FileNotFoundError(msg)

    outputfolderpath = Path(outputfolderpath)
    if not outputfolderpath.exists():
        outputfolderpath.mkdir(parents=True)

    # save the printed output to a log file
    logprint = savetologfile(outputfolderpath=outputfolderpath, logfilename="maptogridlog.txt")

    wij = get_wij()

    if coordinates == "cartesian":
        assert ncoordgrid % 2 == 0

    snapshot_columns_used = ["id", "h", "x", "y", "z", "vx", "vy", "vz", "pmass", "rho", "p", "rho_rst", "ye"]

    dfsnapshot = read_ejectasnapshot(
        ejectasnapshotpath, usecols=snapshot_columns_used, downsamplefactor=downsamplefactor
    )

    logprint(dfsnapshot)
    if coordinates == "cartesian":
        logprint(f"Ncoordgrid: {ncoordgrid}")

    assert len(dfsnapshot.columns) == len(snapshot_columns_used)

    npart = len(dfsnapshot)

    # Propagate particles to dtextra using velocities
    logprint(f"Propagating particles for dtextra_seconds={dtextra_seconds}")
    dtextra = dtextra_seconds / geometric_time_to_s
    dfsnapshot = (
        dfsnapshot
        .with_columns(
            dis_orig=(pl.col("x") ** 2 + pl.col("y") ** 2 + pl.col("z") ** 2).sqrt(),
            x_orig=pl.col("x"),
            y_orig=pl.col("y"),
            z_orig=pl.col("z"),
            x=pl.col("x") + pl.col("vx") * dtextra,
            y=pl.col("y") + pl.col("vy") * dtextra,
            z=pl.col("z") + pl.col("vz") * dtextra,
        )
        .with_columns(dis=(pl.col("x") ** 2 + pl.col("y") ** 2 + pl.col("z") ** 2).sqrt())
        .with_columns(
            h=pl.col("h") / pl.col("dis_orig") * pl.col("dis"),
            vrad=(pl.col("vx") * pl.col("x") + pl.col("vy") * pl.col("y") + pl.col("vz") * pl.col("z")) / pl.col("dis"),
            vtot=(pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).sqrt(),
        )
        .with_columns(
            vperp=pl
            .when(pl.col("vtot") > pl.col("vrad"))
            .then((pl.col("vtot") ** 2 - pl.col("vrad") ** 2).sqrt())
            .otherwise(0.0)
        )
    )

    particleid = dfsnapshot["id"].to_numpy()
    x = dfsnapshot["x"].to_numpy()
    y = dfsnapshot["y"].to_numpy()
    z = dfsnapshot["z"].to_numpy()
    h = dfsnapshot["h"].to_numpy().copy()
    pmass = dfsnapshot["pmass"].to_numpy()
    rho_rst = dfsnapshot["rho_rst"].to_numpy()
    rho = dfsnapshot["rho"].to_numpy()
    Ye = dfsnapshot["ye"].to_numpy()

    totmass = float(dfsnapshot["pmass"].sum())
    rmean = dfsnapshot["dis"].mean()
    assert isinstance(rmean, float)
    hmean = dfsnapshot["h"].mean()
    assert isinstance(hmean, float)
    hmin = dfsnapshot["h"].min()
    assert isinstance(hmin, float)
    rmax = dfsnapshot["dis"].max()
    assert isinstance(rmax, float)
    with Path(outputfolderpath, "ejectapartanalysis.dat").open(mode="w", encoding="utf-8") as fpartanalysis:
        fpartanalysis.writelines(
            f"{dis} {hpart} {h_on_dis} {vrad} {vperp} {vtot}\n"
            for dis, hpart, h_on_dis, vrad, vperp, vtot in dfsnapshot.select(
                "dis", "h", (pl.col("h") / pl.col("dis")).alias("h_on_dis"), "vrad", "vperp", "vtot"
            ).iter_rows()
        )

    logprint(f"Saved {outputfolderpath / 'ejectapartanalysis.dat'}")

    logprint(f"Total mass of sph particle {totmass} max dist {rmax} mean dist {rmean}")
    logprint(f"Smoothing length min {hmin} mean {hmean}")
    logprint("Mean ratio of vperp to vrad", dfsnapshot.select(pl.col("vperp") / pl.col("vrad")).mean().item(0, 0))

    # check maybe cm and correct by shifting

    # ...

    # set up grid
    logprint(
        f"setgrid_fractionrmax={setgrid_fractionrmax}: gridmax is set to {setgrid_fractionrmax}*rmax"
        " of the SPH particles"
    )
    if coordinates == "spherical":
        logprint(f"Modifysmoothinglength: {modifysmoothinglength}")
        map_particles_to_spherical_grid(
            outputfolderpath,
            dfsnapshot.select(
                "id",
                "x",
                "y",
                "z",
                h=pl.Series(
                    get_modified_smoothing_lengths(h, dfsnapshot["dis"].to_numpy(), rmean, hmean, modifysmoothinglength)
                ),
                pmass="pmass",
                mass=pl.col("pmass") * pl.col("rho") / pl.col("rho_rst"),
                ye="ye",
            ),
            sphericalgridshape,
            r_outer=setgrid_fractionrmax * rmax,
            dtextra=dtextra,
            logprint=logprint,
            normalisekernelweights=normalisekernelweights,
        )
        return

    # x0 is a fraction of the largest radius of the SPH particles
    x0 = -setgrid_fractionrmax * rmax

    dx = 2.0 * abs(x0) / (ncoordgrid)  # -1 to be symmetric, right?

    y0 = x0
    z0 = x0
    dy = dx
    dz = dx

    grho = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid))
    gye = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid))
    gparticlecounter = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid), dtype=int)
    # the particle index, the cell indices, and the density contribution of each particle-cell pair,
    # with one array for each particle. The empty first array lets np.concatenate operate on a grid with no pair.
    contrib_particle: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    contrib_celli: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    contrib_cellj: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    contrib_cellk: list[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    contrib_rho: list[npt.NDArray[np.float64]] = [np.empty(0)]

    logprint(f"Grid properties {x0=}, {dx=}, {x0 + dx * (ncoordgrid - 1)=}")

    # grid.dat gives the lower edge x0 + i * dx of each cell, thus the kernel samples each cell at its centre.
    # A sample at the lower corner moves the mapped ejecta by half a cell towards +x, +y, and +z
    logprint(f"Kernel sample point of each cell: {'lower corner' if samplelowercorner else 'centre'}")
    arrgx = x0 + dx * (np.arange(ncoordgrid) + (0.0 if samplelowercorner else 0.5))
    arrgy = arrgx
    arrgz = arrgx

    particlesused = set()
    particlesinsidegrid = set()

    logprint(f"Modifysmoothinglength: {modifysmoothinglength}")

    for n in range(npart):
        # the smoothing length changes first, because the search box below and the kernel must use
        # the same value. A box from the snapshot h drops every contribution that the larger h adds
        if modifysmoothinglength != "False":
            # -- change h by hand ---------

            # option 1 minimum that no particle is lost

            # option 2 increase smoothing everywhere, i.e. less holes but also less structure

            # option 3 increase smoothing beyond some distance

            # a combination of the options is possible. Option 1 alone fills the hole in the centre.
            # Later ejecta can also fill this hole
            if modifysmoothinglength == "option1":
                h[n] = max(h[n], 1.5 * dx)  # option 1

            dis = math.sqrt(x[n] * x[n] + y[n] * y[n] + z[n] * z[n])

            if modifysmoothinglength == "option2":
                h[n] = max(h[n], 0.25 * dis)  # option 2

            if modifysmoothinglength == "option3" and dis > 1.5 * rmean:
                h[n] = max(h[n], 0.4 * dis)  # option 3

            # option 4 (default) -- for particles with radius > mean particle radius choose the larger h
            # from the particle h and 150% of the mean h for all particles
            if modifysmoothinglength == "option4" and dis > rmean:
                h[n] = max(h[n], hmean * 1.5)
            # option 5 -- set a minimum smoothing length of 0.75 * dx and a maximum of 2500.
            # This applies to a particle with a radius above the mean, and it keeps the length
            # of an outer particle within a limit
            if modifysmoothinglength == "option5" and dis > rmean:
                h[n] = max(h[n], 0.75 * dx)
                h[n] = min(h[n], 2500)
            # option 6 -- as option 5, but with no maximum. Use this option to let the smoothing
            # length of an outer particle grow above 0.75 * dx
            if modifysmoothinglength == "option6" and dis > rmean:
                h[n] = max(h[n], 0.75 * dx)

            # -------------------------------

            # a further option could set the smoothing length from the neighbours. No code
            # does this yet

        maxdist = 2.0 * h[n]
        maxdist2 = maxdist**2

        ilow = max(math.floor((x[n] - maxdist - x0) / dx), 0)
        ihigh = min(math.ceil((x[n] + maxdist - x0) / dx), ncoordgrid - 1)
        jlow = max(math.floor((y[n] - maxdist - y0) / dy), 0)
        jhigh = min(math.ceil((y[n] + maxdist - y0) / dy), ncoordgrid - 1)
        klow = max(math.floor((z[n] - maxdist - z0) / dz), 0)
        khigh = min(math.ceil((z[n] + maxdist - z0) / dz), ncoordgrid - 1)

        if min(ihigh, jhigh, khigh) >= 0 and max(ilow, jlow, klow) <= ncoordgrid - 1:
            particlesinsidegrid.add(n)

        # a kernel that ends two cells or more before the grid gives a negative slice end. A slice reads that
        # as an offset from the end of the axis, thus skip the particle, because its box is empty
        if ilow > ihigh or jlow > jhigh or klow > khigh:
            continue

        distx2 = (arrgx[ilow : ihigh + 1] - x[n]) ** 2
        disty2 = (arrgy[jlow : jhigh + 1] - y[n]) ** 2
        distz2 = (arrgz[klow : khigh + 1] - z[n]) ** 2
        dis2box = distx2[:, np.newaxis, np.newaxis] + disty2[np.newaxis, :, np.newaxis] + distz2
        boxi, boxj, boxk = np.nonzero(dis2box <= maxdist2)
        if boxi.size == 0:
            continue

        wtij = kernelvals2(dis2box[boxi, boxj, boxk], float(h[n]), wij)

        # this particle's contribution to mass density (rho) in each cell
        grho_contrib = pmass[n] * rho[n] / rho_rst[n] * wtij

        # the cells of one particle are all different, thus an add with an index array is correct without np.add.at
        cells = (boxi + ilow, boxj + jlow, boxk + klow)
        grho[cells] += grho_contrib

        # mass-weighted electron fraction (needs to be normalised by cell density afterwards)
        gye[cells] += grho_contrib * Ye[n]

        # count number of particles contributing to each grid cell
        gparticlecounter[cells] += 1
        particlesused.add(n)

        contrib_particle.append(np.full(boxi.size, n, dtype=np.int64))
        contrib_celli.append(cells[0])
        contrib_cellj.append(cells[1])
        contrib_cellk.append(cells[2])
        contrib_rho.append(grho_contrib)

    logprint(
        f"Particles with any cell contribution: {len(particlesused)} of {len(particlesinsidegrid)} inside grid out of"
        f" {npart} total"
    )
    unusedparticles = [n for n in range(npart) if n not in particlesused]
    for n in unusedparticles:
        loc_i = math.floor((x[n] - x0) / dx)
        loc_j = math.floor((y[n] - y0) / dy)
        loc_k = math.floor((z[n] - z0) / dz)
        # ignore particles outside grid boundary
        if min(loc_i, loc_j, loc_k) < 0 or max(loc_i, loc_j, loc_k) > ncoordgrid - 1:
            continue
        logprint(f"Particle {n} is totally unused but located in cell {loc_i} {loc_j} {loc_k}")

    with np.errstate(divide="ignore", invalid="ignore"):
        gye = np.divide(gye, grho)

        contrib_i = np.concatenate(contrib_celli)
        contrib_j = np.concatenate(contrib_cellj)
        contrib_k = np.concatenate(contrib_cellk)
        contrib_gridindex = (contrib_k * ncoordgrid + contrib_j) * ncoordgrid + contrib_i + 1
        contrib_frac_of_cellmass = np.concatenate(contrib_rho) / grho[contrib_i, contrib_j, contrib_k]
        # full precision, because the reader keeps frac_of_cellmass as Float64
        gridcontribpath = save_gridparticlecontributions(
            pl.DataFrame({
                "particleid": particleid[np.concatenate(contrib_particle)],
                "cellindex": contrib_gridindex,
                "frac_of_cellmass": contrib_frac_of_cellmass,
            }),
            Path(outputfolderpath, "gridcontributions.txt"),
            float_precision=None,
        )
        logprint(f"Saved {gridcontribpath}")

    # check some stuff on the grid

    gmass = np.sum(grho) * dx * dy * dz

    isempty = grho < 1.0e-20
    # compare squared distances to avoid a sqrt over every cell
    dis2 = (
        arrgx[:, np.newaxis, np.newaxis] ** 2
        + arrgy[np.newaxis, :, np.newaxis] ** 2
        + arrgz[np.newaxis, np.newaxis, :] ** 2
    )

    nzero = int(np.count_nonzero(isempty))
    nzerocentral = int(np.count_nonzero(isempty & (dis2 < rmean**2)))

    logprint(f"Fraction of total mass on grid {gmass / totmass}")

    logprint(
        f"{'WARNING!' if gmass / totmass < 0.9 else ''} mass on grid from rho*V: {gmass} mass of particles: {totmass} "
    )

    logprint(
        f"Number of cells with rho=0 {nzero}, total num of cells {ncoordgrid**3}, fraction of cells with rho=0:"
        f" {(nzero) / (ncoordgrid**3)}"
    )

    logprint(
        f"Number of central cells (dis<rmean) with rho=0 {nzerocentral}, ratio"
        f" {(nzerocentral) / (4.0 * math.pi / 3.0 * rmean**3 / (dx * dy * dz))}"
    )

    logprint("Probably we want to choose grid size, i.e. x0, as compromise between mapped mass and rho=0 cells")

    # output grid - adapt as you need output

    with Path(outputfolderpath, "grid.dat").open("w", encoding="utf-8") as fgrid:
        fgrid.write(f"{ncoordgrid**3} # ncoordgrid\n")
        fgrid.write(f"{dtextra} # extra time after explosion simulation ended (in geom units)\n")
        fgrid.write(f"{x0} # xmax\n")
        fgrid.write(" gridindex    pos_x_min    pos_y_min    pos_z_min    rho    Ye    tracercount\n")
        # the cell order varies x fastest, which is the Fortran order of the [i, j, k] arrays
        ncells = ncoordgrid**3
        fgrid.writelines(
            f"{gridindex:8d} {gx} {gy} {gz} {cellrho} {cellye} {tracercount}\n"
            for gridindex, gx, gy, gz, cellrho, cellye, tracercount in zip(
                range(1, ncells + 1),
                np.tile(x0 + dx * np.arange(ncoordgrid), ncoordgrid**2).tolist(),
                np.tile(np.repeat(y0 + dy * np.arange(ncoordgrid), ncoordgrid), ncoordgrid).tolist(),
                np.repeat(z0 + dz * np.arange(ncoordgrid), ncoordgrid**2).tolist(),
                grho.ravel(order="F").tolist(),
                gye.ravel(order="F").tolist(),
                gparticlecounter.ravel(order="F").tolist(),
                strict=True,
            )
        )

    logprint(f"Saved {outputfolderpath / 'grid.dat'}")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputpath", "-i", default=".", help="The path to the ejectasnapshot")
    parser.add_argument(
        "--coordinates",
        default="cartesian",
        choices=["cartesian", "spherical"],
        help="The coordinates of the grid. spherical gives cells in the radius r, the polar angle theta from the +z"
        " axis, and the azimuthal angle phi. The cells are uniform in r, in cos(theta), and in phi. Both coordinates"
        " use one deposit method: the density of a cell is the kernel sum at its mid-point. (default: cartesian)",
    )
    parser.add_argument(
        "--normalise-kernel-weights",
        action="store_true",
        help="For --coordinates spherical: use a different deposit method. Each particle inside the outer radius gives"
        " all its mass to the grid. The weight of a cell is the kernel value at its mid-point times its volume. The"
        " method normalises the weights of a particle to a sum of 1. A particle with no cell mid-point in its kernel"
        " gives its mass to the cell that holds it. A particle outside the outer radius gives no mass. The default"
        " method is the method of the Cartesian map, which does not keep the mass exactly",
    )
    parser.add_argument(
        "-ncoordgrid",
        type=int,
        default=None,
        help="The number of cells on each axis of a Cartesian grid (the grid has ncoordgrid^3 cells, default: 50)",
    )
    for (axis, description), default in zip(
        (("r", "radial shells"), ("theta", "cells along the polar angle"), ("phi", "cells along the azimuthal angle")),
        SPHERICAL_GRID_SHAPE_DEFAULT,
        strict=True,
    ):
        parser.add_argument(
            f"--ngrid-{axis}",
            type=int,
            default=None,
            help=f"The number of {description} of a spherical grid (default: {default})",
        )
    parser.add_argument(
        "-dtextra_seconds",
        type=float,
        default=0.5,
        help="Time in seconds to propagate SPH particles ballistically after end of SPH simulation."
        " 0 for no extrapolation",
    )
    parser.add_argument(
        "-setgrid_fractionrmax",
        type=float,
        default=0.5,
        help="The size of the grid as a fraction of rmax, which is the largest radius of the particles. The size is the"
        " half width of a Cartesian grid, and the outer radius of a spherical grid",
    )
    parser.add_argument(
        "-downsamplefactor",
        type=int,
        default=1,
        help="Randomly sample particles, reducing the number by this factor (e.g. 2 will ignore half of the particles)",
    )
    parser.add_argument(
        "-modifysmoothinglength",
        default="option4",
        choices=[
            "option1",
            "option2",
            "option3",
            "option4",
            "option5",
            "option6",
            "False",
        ],  # We should choose if the default should be false and how we want to name these
        help="Option to modify smoothing length h. Choose from options. "
        "Default modifies h. Set to False for no modifications to h",
    )

    parser.add_argument(
        "--sample_cell_lower_corner",
        action="store_true",
        help="Evaluate the kernel at the lower corner of each cell, as artistools did before. This recreates an older"
        " grid.dat. The default samples the centre of each cell, because the lower corner moves the ejecta by half a"
        " cell towards +x, +y, and +z",
    )

    addarg_output(parser, kind="folder", default=Path())
    addarg_force(parser)


def get_spherical_grid_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return the cell counts n_r, n_theta, and n_phi of a spherical grid from the arguments.

    The function stops with an error for an argument that does not apply to the coordinates of the map, and for a
    cell count below 1. A count that the user does not give gets its default value.
    """
    gridcounts = (args.ngrid_r, args.ngrid_theta, args.ngrid_phi)
    if args.coordinates == "spherical":
        if args.ncoordgrid is not None or args.sample_cell_lower_corner:
            exit_with_error(
                "--ncoordgrid and --sample_cell_lower_corner apply to a Cartesian grid only",
                "For --coordinates spherical, give --ngrid-r, --ngrid-theta, and --ngrid-phi",
            )
        if args.modifysmoothinglength in {"option1", "option5", "option6"}:
            exit_with_error(
                f"--modifysmoothinglength {args.modifysmoothinglength} sets the smoothing length from the width of a"
                " Cartesian cell",
                "For --coordinates spherical, give False, option2, option3, or option4",
            )
    elif any(count is not None for count in gridcounts) or args.normalise_kernel_weights:
        exit_with_error(
            "--ngrid-r, --ngrid-theta, --ngrid-phi, and --normalise-kernel-weights apply to a spherical grid only",
            "Give --coordinates spherical, or give --ncoordgrid for a Cartesian grid",
        )
    # a count of 0 is a value of the user, thus only a count that the user did not give takes its default
    ngrid_r, ngrid_theta, ngrid_phi = (
        default if count is None else count
        for count, default in zip(gridcounts, SPHERICAL_GRID_SHAPE_DEFAULT, strict=True)
    )
    if min(ngrid_r, ngrid_theta, ngrid_phi) < 1:
        exit_with_error("--ngrid-r, --ngrid-theta, and --ngrid-phi must each be 1 or more")
    return ngrid_r, ngrid_theta, ngrid_phi


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Map tracer particle trajectories to a Cartesian grid or a spherical grid."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    sphericalgridshape = get_spherical_grid_shape(args)

    ejectasnapshotpath = Path(args.inputpath, "ejectasnapshot.dat")
    confirm_overwrite([args.outputfile], ("grid.dat", "gridcontributions.txt"), force=args.force)

    maptogrid(
        ejectasnapshotpath=ejectasnapshotpath,
        ncoordgrid=50 if args.ncoordgrid is None else args.ncoordgrid,
        outputfolderpath=args.outputfile,
        downsamplefactor=args.downsamplefactor,
        dtextra_seconds=args.dtextra_seconds,
        setgrid_fractionrmax=args.setgrid_fractionrmax,
        modifysmoothinglength=args.modifysmoothinglength,
        samplelowercorner=args.sample_cell_lower_corner,
        coordinates=args.coordinates,
        sphericalgridshape=sphericalgridshape,
        normalisekernelweights=args.normalise_kernel_weights,
    )
