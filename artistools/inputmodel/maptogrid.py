#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# adapted from Fortran maptogrid.f90 and kernelmodule.f90
# original Fortran code by Andreas Bauswein
from __future__ import annotations

import argparse
import math
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd

import artistools as at

itable = 40000  # wie fein Kernelfkt interpoliert wird
itab = itable + 5

#
# --maximum interaction length and step size
#
v2max = 4.0
dvtable = v2max / itable
i1 = int(1.0 // dvtable)


def get_wij() -> np.ndarray:
    igphi = 0
    #
    # --normalisation constant
    #
    cnormk = 1.0 / math.pi
    # --build tables
    #
    #  a) v less than 1
    #
    wij = np.zeros(itab + 1)
    if igphi == 1:
        for i in range(1, i1 + 1):
            v2 = i * dvtable
            v = math.sqrt(v2)
            v3 = v * v2
            v4 = v * v3
            vsum = 1.0 - 1.5 * v2 + 0.75 * v3
            wij[i] = cnormk * vsum
    else:
        for i in range(1, i1 + 1):
            v2 = i * dvtable
            v = math.sqrt(v2)
            v3 = v * v2
            vsum = 1.0 - 1.5 * v2 + 0.75 * v3
            wij[i] = cnormk * vsum

    #
    #  b) v greater than 1
    #
    if igphi == 1:
        for i in range(i1 + 1, itable + 1):
            v2 = i * dvtable
            v = math.sqrt(v2)
            dif2 = 2.0 - v
            vsum = 0.25 * dif2 * dif2 * dif2
            wij[i] = cnormk * vsum
    else:
        for i in range(i1 + 1, itable + 1):
            v2 = i * dvtable
            v = math.sqrt(v2)
            dif2 = 2.0 - v
            vsum = 0.25 * dif2 * dif2 * dif2
            wij[i] = cnormk * vsum

    return wij


def kernelvals2(rij2: float, hmean: float, wij: np.ndarray) -> float:  # ist schnell berechnet aber keine Gradienten
    hmean21 = 1.0 / hmean**2
    hmean31 = hmean21 / hmean
    v2 = rij2 * hmean21
    index = math.floor(v2 / dvtable)
    dxx = v2 - index * dvtable
    index1 = index + 1
    dwdx = (wij[index1] - wij[index]) / dvtable
    return (wij[index] + dwdx * dxx) * hmean31


def maptogrid(
    ejectasnapshotpath: Path, outputfolderpath: Path | str, ncoordgrid: int = 50, downsamplefactor: int = 1
) -> None:
    if not ejectasnapshotpath.is_file():
        print(f"{ejectasnapshotpath} not found")
        return

    # save the printed output to a log file
    logfilepath = Path("maptogridlog.txt")
    logfilepath.unlink(missing_ok=True)

    def logprint(*args, **kwargs):
        print(*args, **kwargs)
        with logfilepath.open("a", encoding="utf-8") as logfile:
            logfile.write(" ".join([str(x) for x in args]) + "\n")

    wij = get_wij()

    assert ncoordgrid % 2 == 0

    snapshot_columns = [
        "id",
        "h",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "vstx",
        "vsty",
        "vstz",
        "u",
        "psi",
        "alpha",
        "pmass",
        "rho",
        "p",
        "rho_rst",
        "tau",
        "av",
        "ye",
        "temp",
        "prev_rho(i)",
        "ynue(i)",
        "yanue(i)",
        "enuetrap(i)",
        "eanuetrap(i)",
        "enuxtrap(i)",
        "iwasequil(i, 1)",
        "iwasequil(i, 2)",
        "iwasequil(i, 3)",
    ]

    snapshot_columns_used = ["id", "h", "x", "y", "z", "vx", "vy", "vz", "pmass", "rho", "p", "rho_rst", "ye"]

    dfsnapshot = pd.read_csv(
        ejectasnapshotpath,
        names=snapshot_columns,
        delim_whitespace=True,
        usecols=snapshot_columns_used,
        dtype_backend="pyarrow",
    )

    if downsamplefactor > 1:
        dfsnapshot = dfsnapshot.sample(len(dfsnapshot) // downsamplefactor)

    logprint(dfsnapshot)

    assert len(dfsnapshot.columns) == len(snapshot_columns_used)

    npart = len(dfsnapshot)

    totmass = 0.0
    rmax = 0.0
    rmean = 0.0
    hmean = 0.0
    hmin = 100000.0
    vratiomean = 0.0

    # Propagate particles to dtextra using velocities
    dtextra_seconds = 0.5  # in seconds ---  dtextra = 0.0 # for no extrapolation

    dtextra = dtextra_seconds / 4.926e-6  # convert to geom units.

    particleid = dfsnapshot.id.to_numpy()
    x = dfsnapshot["x"].to_numpy().copy()
    y = dfsnapshot["y"].to_numpy().copy()
    z = dfsnapshot["z"].to_numpy().copy()
    h = dfsnapshot["h"].to_numpy().copy()
    vx = dfsnapshot["vx"].to_numpy()
    vy = dfsnapshot["vy"].to_numpy()
    vz = dfsnapshot["vz"].to_numpy()
    pmass = dfsnapshot["pmass"].to_numpy()
    rho_rst = dfsnapshot["rho_rst"].to_numpy()
    rho = dfsnapshot["rho"].to_numpy()
    Ye = dfsnapshot["ye"].to_numpy()

    with Path(outputfolderpath, "ejectapartanalysis.dat").open(mode="w", encoding="utf-8") as fpartanalysis:
        for n in range(npart):
            totmass = totmass + pmass[n]

            dis = math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)  # original dis

            x[n] += vx[n] * dtextra
            y[n] += vy[n] * dtextra
            z[n] += vz[n] * dtextra

            # actually we should also extrapolate smoothing length h unless we disrgard it below

            # extrapolate h such that ratio between dis and h remains constant
            h[n] = h[n] / dis * math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)

            dis = math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)  # possibly new distance

            rmean = rmean + dis

            rmax = max(rmax, dis)

            hmean = hmean + h[n]

            hmin = min(hmin, h[n])

            vtot = math.sqrt(vx[n] ** 2 + vy[n] ** 2 + vz[n] ** 2)

            vrad = (vx[n] * x[n] + vy[n] * y[n] + vz[n] * z[n]) / dis  # radial velocity

            # velocity perpendicular
            # if we extrapolate roundoff error can lead to Nan, ly?
            vperp = math.sqrt(vtot * vtot - vrad * vrad) if vtot > vrad else 0.0

            vratiomean = vratiomean + vperp / vrad

            # output some ejecta properties in file

            fpartanalysis.write(f"{dis} {h[n]} {h[n] / dis} {vrad} {vperp} {vtot}\n")

    logprint("saved ejectapartanalysis.dat")
    rmean = rmean / npart

    hmean = hmean / npart

    vratiomean = vratiomean / npart

    logprint(f"total mass of sph particle {totmass} max dist {rmax} mean dist {rmean}")
    logprint(f"smoothing length min {hmin} mean {hmean}")
    logprint("ratio between vrad and vperp mean", vratiomean)

    # check maybe cm and correct by shifting

    # ...

    # set up grid

    x0 = -0.5 * rmax  # 90% is hand waving - choose #

    # x0 = - rmean

    dx = 2.0 * abs(x0) / (ncoordgrid)  # -1 to be symmetric, right?

    y0 = x0
    z0 = x0
    dy = dx
    dz = dx

    grho = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid))
    gye = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid))
    gparticlecounter = np.zeros((ncoordgrid, ncoordgrid, ncoordgrid), dtype=int)
    particle_rho_contribs = {}

    logprint(f"grid properties {x0=}, {dx=}, {x0 + dx * (ncoordgrid - 1)=}")

    arrgx = x0 + dx * np.arange(ncoordgrid)
    arrgy = arrgx
    arrgz = arrgx

    particlesused = set()
    particlesinsidegrid = set()

    for n in range(npart):
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
        # check some min max

        # ... kernel reweighting ?

        searchcoords = [
            (i, j, k, (arrgx[i] - x[n]) ** 2 + (arrgy[j] - y[n]) ** 2 + (arrgz[k] - z[n]) ** 2)
            for i in range(ilow, ihigh + 1)
            for j in range(jlow, jhigh + 1)
            for k in range(klow, khigh + 1)
        ]

        for i, j, k, dis2 in searchcoords:
            # -- change h by hand --------- we could do these particle thinsg also further up

            # option 1 minimum that no particle is lost

            # option 2 increase smoothing everywhere, i.e. less holes but also less strcuture

            # option 3 increase smoothing beyond some distance

            # options can be combined, i.e. option 1 alone fills the hole in the center
            # (which we could also replace by later ejecta)

            # h[n] = max(h[n],1.5*dx) # option 1

            # h[n] = max(h[n],0.25*dis) #  option 2

            dis = math.sqrt(x[n] * x[n] + y[n] * y[n] + z[n] * z[n])

            # if (dis>1.5*rmean) h[n]=max(h[n],0.4*dis) # option 3
            if dis > rmean:
                h[n] = max(h[n], hmean * 1.5)

            # -------------------------------

            # or via neighbors  - not yet implemented

            if dis2 <= maxdist2:
                wtij = kernelvals2(dis2, h[n], wij)

                # USED PREVIOUSLY: less accurate?
                # grho_contrib = pmass[n] * wtij

                # this particle's contribution to mass density (rho) in the cell
                grho_contrib = pmass[n] * rho[n] / rho_rst[n] * wtij

                grho[i, j, k] += grho_contrib

                particle_rho_contribs[(n, i, j, k)] = grho_contrib

                # mass-weighted electron fraction (needs to be normalised by cell density afterwards)
                gye[i, j, k] += grho_contrib * Ye[n]

                # count number of particles contributing to each grid cell
                gparticlecounter[i, j, k] += 1
                particlesused.add(n)

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

        with Path(outputfolderpath, "gridcontributions.txt").open("w", encoding="utf-8") as fcontribs:
            fcontribs.write("particleid cellindex frac_of_cellmass\n")
            for (n, i, j, k), rho_contrib in particle_rho_contribs.items():
                gridindex = (k * ncoordgrid + j) * ncoordgrid + i + 1
                fcontribs.write(f"{particleid[n]} {gridindex} {rho_contrib / grho[i, j, k]}\n")
        logprint("saved gridcontributions.txt")

    # check some stuff on the grid

    nzero = 0
    nzerocentral = 0
    gmass = np.sum(grho) * dx * dy * dz
    # nzero = np.count_nonzero(grho[1:][1:][1:] < 1.e-20)

    for i in range(ncoordgrid):
        gx = x0 + dx * i
        for j in range(ncoordgrid):
            gy = y0 + dy * j
            for k in range(1, ncoordgrid):
                # how many cells with rho=0?

                if grho[i, j, k] < 1.0e-20:
                    nzero = nzero + 1

                gz = z0 + dz * k

                dis = math.sqrt(gx * gx + gy * gy + gz * gz)

                if grho[i, j, k] < 1.0e-20 and dis < rmean:
                    nzerocentral = nzerocentral + 1

    logprint(
        f"{'WARNING!' if gmass / totmass < 0.9 else ''} mass on grid from rho*V: {gmass} mass of particles: {totmass} "
    )

    logprint(
        f"number of cells with rho=0 {nzero}, total num of cells {ncoordgrid**3}, fraction of cells with rho=0:"
        f" {(nzero) / (ncoordgrid**3)}"
    )

    logprint(
        f"number of central cells (dis<rmean) with rho=0 {nzerocentral}, ratio"
        f" {(nzerocentral) / (4.0 * 3.14 / 3.0 * rmean**3 / (dx * dy * dz))}"
    )

    logprint("probably we want to choose grid size, i.e. x0, as compromise between mapped mass and rho=0 cells")

    # output grid - adapt as you need output

    with Path(outputfolderpath, "grid.dat").open("w", encoding="utf-8") as fgrid:
        fgrid.write(f"{ncoordgrid**3} # ncoordgrid\n")
        fgrid.write(f"{dtextra} # extra time after explosion simulation ended (in geom units)\n")
        fgrid.write(f"{x0} # xmax\n")
        fgrid.write(" gridindex    pos_x_min    pos_y_min    pos_z_min    rho    cellYe    tracercount\n")
        gridindex = 1
        for k in range(ncoordgrid):
            gz = z0 + dz * k
            for j in range(ncoordgrid):
                gy = y0 + dy * j
                for i in range(ncoordgrid):
                    fgrid.write(
                        f"{gridindex:8d} {x0+dx*i} {gy} {gz} {grho[i,j,k]} {gye[i,j,k]} {gparticlecounter[i,j,k]}\n"
                    )
                    # gridindex2 = ((k - 1) * ncoordgrid + (j - 1)) * ncoordgrid + (i - 1) + 1

                    gridindex = gridindex + 1

    logprint("saved grid.dat")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputpath", "-i", default=".", help="Path to ejectasnapshot")
    parser.add_argument(
        "-ncoordgrid", type=int, default=50, help="Number of grid positions per axis (numcells = ncoordgrid^3)"
    )
    parser.add_argument(
        "-downsamplefactor",
        type=int,
        default=1,
        help="Randomly sample particles, reducing the number by this factor (e.g. 2 will ignore half of the particles)",
    )
    parser.add_argument("-outputfolderpath", "-o", default=".", help="Path for output files")


def main(args=None, argsraw=None, **kwargs) -> None:
    """Map tracer particle trajectories to a Cartesian grid."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    ejectasnapshotpath = Path(args.inputpath, "ejectasnapshot.dat")

    maptogrid(
        ejectasnapshotpath=ejectasnapshotpath,
        ncoordgrid=args.ncoordgrid,
        outputfolderpath=args.outputfolderpath,
        downsamplefactor=args.downsamplefactor,
    )


if __name__ == "__main__":
    main()
