# PYTHON_ARGCOMPLETE_OK
# adapted from Fortran maptogrid.f90 and kernelmodule.f90
# original Fortran code by Andreas Bauswein
"""Map SPH ejecta particles onto a Cartesian grid using a cubic spline kernel."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at

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
    # --build tables. Entry 0 and the entries above itable stay zero
    wij = np.zeros(itab + 1)
    i = np.arange(1, itable + 1)
    v2 = i * dvtable
    v = np.sqrt(v2)
    dif2 = 2.0 - v
    # v less than 1 for the entries up to i1, and v greater than 1 above
    vsum = np.where(i <= i1, 1.0 - 1.5 * v2 + 0.75 * (v * v2), 0.25 * dif2 * dif2 * dif2)
    wij[1 : itable + 1] = cnormk * vsum

    return wij


def kernelvals2(
    rij2: npt.NDArray[np.float64], hmean: float, wij: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:  # ist schnell berechnet aber keine Gradienten
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


def maptogrid(
    ejectasnapshotpath: Path,
    outputfolderpath: Path | str,
    ncoordgrid: int = 50,
    downsamplefactor: int = 1,
    dtextra_seconds: float = 0.5,
    setgrid_fractionrmax: float = 0.5,
    modifysmoothinglength: str = "option4",
) -> None:
    """Map an SPH ejecta snapshot onto an ncoordgrid^3 Cartesian grid and write grid.dat and gridcontributions.txt."""
    if not ejectasnapshotpath.is_file():
        msg = f"{ejectasnapshotpath} does not exist"
        raise FileNotFoundError(msg)

    outputfolderpath = Path(outputfolderpath)
    if not outputfolderpath.exists():
        outputfolderpath.mkdir(parents=True)

    # save the printed output to a log file
    logprint = at.inputmodel.inputmodel_misc.savetologfile(
        outputfolderpath=outputfolderpath, logfilename="maptogridlog.txt"
    )

    wij = get_wij()

    assert ncoordgrid % 2 == 0

    snapshot_columns_used = ["id", "h", "x", "y", "z", "vx", "vy", "vz", "pmass", "rho", "p", "rho_rst", "ye"]

    dfsnapshot = at.inputmodel.modelfromhydro.read_ejectasnapshot(
        ejectasnapshotpath, usecols=snapshot_columns_used, downsamplefactor=downsamplefactor
    )

    logprint(dfsnapshot)
    logprint(f"ncoordgrid: {ncoordgrid}")

    assert len(dfsnapshot.columns) == len(snapshot_columns_used)

    npart = len(dfsnapshot)

    # Propagate particles to dtextra using velocities
    logprint(f"Propagating particles for dtextra_seconds={dtextra_seconds}")
    dtextra = dtextra_seconds / 4.926e-6  # convert to geom units.
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

    logprint(f"saved {outputfolderpath / 'ejectapartanalysis.dat'}")

    logprint(f"total mass of sph particle {totmass} max dist {rmax} mean dist {rmean}")
    logprint(f"smoothing length min {hmin} mean {hmean}")
    logprint("ratio between vrad and vperp mean", dfsnapshot.select(pl.col("vperp") - pl.col("vrad")).mean().item(0, 0))

    # check maybe cm and correct by shifting

    # ...

    # set up grid
    logprint(
        f"setgrid_fractionrmax={setgrid_fractionrmax}: gridmax is set to {setgrid_fractionrmax}*rmax of the SPH particles"
    )
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

    logprint(f"grid properties {x0=}, {dx=}, {x0 + dx * (ncoordgrid - 1)=}")

    arrgx = x0 + dx * np.arange(ncoordgrid)
    arrgy = arrgx
    arrgz = arrgx

    particlesused = set()
    particlesinsidegrid = set()

    logprint(f"modifysmoothinglength: {modifysmoothinglength}")

    for n in range(npart):
        # the smoothing length changes first, because the search box below and the kernel must use
        # the same value. A box from the snapshot h drops every contribution that the larger h adds
        if modifysmoothinglength != "False":
            # -- change h by hand ---------

            # option 1 minimum that no particle is lost

            # option 2 increase smoothing everywhere, i.e. less holes but also less structure

            # option 3 increase smoothing beyond some distance

            # options can be combined, i.e. option 1 alone fills the hole in the center
            # (which we could also replace by later ejecta)
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

        if min(ihigh, jhigh, khigh) >= 1 and max(ilow, jlow, klow) <= ncoordgrid:
            particlesinsidegrid.add(n)

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
        f"particles with any cell contribution: {len(particlesused)} of {len(particlesinsidegrid)} inside grid out of"
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
        logprint(f"particle {n} is totally unused but located in cell {loc_i} {loc_j} {loc_k}")

    with np.errstate(divide="ignore", invalid="ignore"):
        gye = np.divide(gye, grho)

        contrib_i = np.concatenate(contrib_celli)
        contrib_j = np.concatenate(contrib_cellj)
        contrib_k = np.concatenate(contrib_cellk)
        contrib_gridindex = (contrib_k * ncoordgrid + contrib_j) * ncoordgrid + contrib_i + 1
        contrib_frac_of_cellmass = np.concatenate(contrib_rho) / grho[contrib_i, contrib_j, contrib_k]
        with Path(outputfolderpath, "gridcontributions.txt").open("w", encoding="utf-8") as fcontribs:
            fcontribs.write("particleid cellindex frac_of_cellmass\n")
            fcontribs.writelines(
                f"{pid} {gridindex} {frac}\n"
                for pid, gridindex, frac in zip(
                    particleid[np.concatenate(contrib_particle)].tolist(),
                    contrib_gridindex.tolist(),
                    contrib_frac_of_cellmass.tolist(),
                    strict=True,
                )
            )
        logprint(f"saved {outputfolderpath / 'gridcontributions.txt'}")

    # check some stuff on the grid

    gmass = np.sum(grho) * dx * dy * dz

    # k starts at 1 to match the original scan, which skipped the k=0 plane
    isempty = grho[:, :, 1:] < 1.0e-20
    # compare squared distances to avoid a sqrt over every cell
    dis2 = (
        arrgx[:, np.newaxis, np.newaxis] ** 2
        + arrgy[np.newaxis, :, np.newaxis] ** 2
        + arrgz[np.newaxis, np.newaxis, 1:] ** 2
    )

    nzero = int(np.count_nonzero(isempty))
    nzerocentral = int(np.count_nonzero(isempty & (dis2 < rmean**2)))

    logprint(f"fraction of total mass on grid {gmass / totmass}")

    logprint(
        f"{'WARNING!' if gmass / totmass < 0.9 else ''} mass on grid from rho*V: {gmass} mass of particles: {totmass} "
    )

    logprint(
        f"number of cells with rho=0 {nzero}, total num of cells {ncoordgrid**3}, fraction of cells with rho=0:"
        f" {(nzero) / (ncoordgrid**3)}"
    )

    logprint(
        f"number of central cells (dis<rmean) with rho=0 {nzerocentral}, ratio"
        f" {(nzerocentral) / (4.0 * math.pi / 3.0 * rmean**3 / (dx * dy * dz))}"
    )

    logprint("probably we want to choose grid size, i.e. x0, as compromise between mapped mass and rho=0 cells")

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

    logprint(f"saved {outputfolderpath / 'grid.dat'}")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputpath", "-i", default=".", help="Path to ejectasnapshot")
    parser.add_argument(
        "-ncoordgrid", type=int, default=50, help="Number of grid positions per axis (numcells = ncoordgrid^3)"
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
        help="Setup grid to have max equal to fraction of particle rmax",
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

    at.addarg_output(parser, kind="folder", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Map tracer particle trajectories to a Cartesian grid."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    ejectasnapshotpath = Path(args.inputpath, "ejectasnapshot.dat")

    maptogrid(
        ejectasnapshotpath=ejectasnapshotpath,
        ncoordgrid=args.ncoordgrid,
        outputfolderpath=args.outputfile,
        downsamplefactor=args.downsamplefactor,
        dtextra_seconds=args.dtextra_seconds,
        setgrid_fractionrmax=args.setgrid_fractionrmax,
        modifysmoothinglength=args.modifysmoothinglength,
    )


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
