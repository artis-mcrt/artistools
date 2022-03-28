#!/usr/bin/env python3

# adapted from Fortran maptogrid.f90 and kernelmodule.f90
# original Fortran code by Andreas Bauswein

import argparse
import math
import numpy as np
import pandas as pd

from pathlib import Path
from artistools import CustomArgHelpFormatter

itable = 40000  # wie fein Kernelfkt interpoliert wird
itab = itable + 5


wij = np.zeros(itab + 1)

#
# --maximum interaction length and step size
#
v2max = 4.
dvtable = v2max / itable
i1 = int(1. // dvtable)


igphi = 0
#
# --normalisation constant
#
cnormk = 1. / math.pi
# --build tables
#
#  a) v less than 1
#
if (igphi == 1):
    for i in range(1, i1 + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        v3 = v * v2
        v4 = v * v3
        sum = 1. - 1.5 * v2 + 0.75 * v3
        wij[i] = cnormk * sum
else:
    for i in range(1, i1 + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        v3 = v * v2
        sum = 1. - 1.5 * v2 + 0.75 * v3
        wij[i] = cnormk * sum

#
#  b) v greater than 1
#
if (igphi == 1):
    for i in range(i1 + 1, itable + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        dif2 = 2. - v
        sum = 0.25 * dif2 * dif2 * dif2
        wij[i] = cnormk * sum
else:
    for i in range(i1 + 1, itable + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        dif2 = 2. - v
        sum = 0.25 * dif2 * dif2 * dif2
        wij[i] = cnormk * sum


def kernelvals2(rij2, hmean):  # ist schnell berechnet aber keine Gradienten
    hmean21 = 1. / (hmean * hmean)
    hmean31 = hmean21 / hmean
    v2 = rij2 * hmean21
    index = math.floor(v2 / dvtable)
    dxx = v2 - index * dvtable
    index1 = index + 1
    dwdx = (wij[index1] - wij[index]) / dvtable
    wtij = (wij[index] + dwdx * dxx) * hmean31
    return wtij


def maptogrid(ejectasnapshotpath, outputpath):
    if not ejectasnapshotpath.is_file():
        print(f'{ejectasnapshotpath} not found')
        return

    snapshot_columns = [
        'id', 'h', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vstx', 'vsty', 'vstz', 'u',
        'psi', 'alpha', 'pmass', 'rho', 'p', 'rst', 'tau', 'av', 'ye', 'temp',
        'prev_rho(i)', 'ynue(i)', 'yanue(i)', 'enuetrap(i)', 'eanuetrap(i)',
        'enuxtrap(i)', 'iwasequil(i, 1)', 'iwasequil(i, 2)', 'iwasequil(i, 3)']
    snapshot_columns_used = ['id', 'h', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'pmass', 'rho', 'p', 'rst', 'ye']
    dfsnapshot = pd.read_csv(
        ejectasnapshotpath, names=snapshot_columns,
        delim_whitespace=True, usecols=snapshot_columns_used)

    print(dfsnapshot)

    assert len(dfsnapshot.columns) == len(snapshot_columns_used)

    npart = len(dfsnapshot)
    print("number of particles", npart)

    fpartanalysis = open(Path(outputpath, 'ejectapartanalysis.dat'), mode='w')

    totmass = 0.0
    rmax = 0.0
    rmean = 0.0
    hmean = 0.0
    hmin = 100000.
    vratiomean = 0.

    # Propagate particles to dtextra using velocities
    dtextra = 0.5  # in seconds ---  dtextra = 0.0 # for no extrapolation

    dtextra = dtextra / 4.926e-6  # convert to geom units.

    particleid = dfsnapshot.id.values
    x = dfsnapshot.x.values
    y = dfsnapshot.y.values
    z = dfsnapshot.z.values
    h = dfsnapshot.h.values
    vx = dfsnapshot.vx.values
    vy = dfsnapshot.vy.values
    vz = dfsnapshot.vz.values
    pmass = dfsnapshot.pmass.values
    rst = dfsnapshot.rst.values
    Ye = dfsnapshot.ye.values

    for n in range(npart):
        totmass = totmass + pmass[n]

        dis = math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)  # original dis

        x[n] += vx[n] * dtextra
        y[n] += vy[n] * dtextra
        z[n] += vz[n] * dtextra

        # actually we should also extrapolate smoothing length h unless we disrgard it below

        # extrapolate h such that ratio betwen dis and h remains constant
        h[n] = h[n] / dis * math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)

        dis = math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)  # possibly new distance

        rmean = rmean + dis

        rmax = max(rmax, dis)

        hmean = hmean + h[n]

        hmin = min(hmin, h[n])

        vtot = math.sqrt(vx[n] ** 2 + vy[n] ** 2 + vz[n] ** 2)

        vrad = (vx[n] * x[n] + vy[n] * y[n] + vz[n] * z[n]) / dis  # radial velocity

        if (vtot > vrad):
            vperp = math.sqrt(vtot * vtot - vrad * vrad)  # velicty perpendicular
        else:
            vperp = 0.0  # if we rxtrapolate roundoff error can lead to Nan, ly?

        vratiomean = vratiomean + vperp / vrad

        # output some ejecta properties in file

        fpartanalysis.write(f'{dis} {h[n]} {h[n] / dis} {vrad} {vperp} {vtot}\n')

    rmean = rmean / npart

    hmean = hmean / npart

    vratiomean = vratiomean / npart

    print("total mass of sph particle, max, mean distance", totmass, rmax, rmean)
    print("smoothing length min, mean", hmin, hmean)
    print("ratio between vrad and vperp mean", vratiomean)

    # check maybe cm and correct by shifting

    # ...

    # set up grid

    ngrid = 50

    x0 = - 0.6 * rmax  # 90% is hand waving - choose #

    # x0 = - rmean

    dx = 2. * abs(x0) / (ngrid)  # -1 to be symmetric, right?

    y0 = x0
    z0 = x0
    dy = dx
    dz = dx

    grho = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    norm = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    gye = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    gparticlecounter = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1), dtype=int)
    particlecontribs = {}

    print('grid properties', x0, dx, x0 + dx * (ngrid - 1))

    arrgx = x0 + dx * (np.arange(ngrid + 1) - 1)
    arrgy = arrgx
    arrgz = arrgx

    for n in range(npart):

        maxdist = 2. * h[n]
        maxdist2 = maxdist ** 2

        ilow = max(math.floor((x[n] - maxdist - x0) / dx), 1)
        ihigh = min(math.ceil((x[n] + maxdist - x0) / dx), ngrid)
        jlow = max(math.floor((y[n] - maxdist - y0) / dy), 1)
        jhigh = min(math.ceil((y[n] + maxdist - y0) / dy), ngrid)
        klow = max(math.floor((z[n] - maxdist - z0) / dz), 1)
        khigh = min(math.ceil((z[n] + maxdist - z0) / dz), ngrid)

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

            # dis = sqrt(x[n]*x[n]+y[n]*y[n]+z[n]*z[n])

            # if (dis>1.5*rmean) h[n]=max(h[n],0.4*dis) # option 3

            # -------------------------------

            # or via neighbors  - not yet implemented

            if dis2 <= maxdist2:

                wtij = kernelvals2(dis2, h[n])

                grho[i, j, k] += pmass[n] * wtij
                # grho[i, j, k] = grho[i, j, k] + pmass[n] / particle.rst * wtij

                particle_contrib = pmass[n] / rst[n] * wtij
                particlecontribs[(n, i, j, k)] = particle_contrib

                # gye[i, j, k] += pmass[n] * particle.Ye * wtij
                gye[i, j, k] += particle_contrib * Ye[n]

                # norm[i, j, k] += pmass[n] * wtij
                norm[i, j, k] += particle_contrib

                # Counts particles contributing to grid cell
                gparticlecounter[i, j, k] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        gye = np.divide(gye, norm)
        # for i in range(1, ngrid + 1):
        #     for j in range(1, ngrid + 1):
        #         for k in range(1, ngrid + 1):
        #             gye[i, j, k] = gye[i, j, k] / norm[i, j, k]
        #             # if not norm[i, j, k] > 0:
        #             #     print(i, j, k, norm[i, j, k], gye[i, j, k] )

        with open(Path(outputpath, 'gridcontributions.txt'), 'w') as fcontribs:
            fcontribs.write('particleid cellindex frac_of_cellmass\n')
            for (n, i, j, k), old_value in particlecontribs.items():
                # particle_contribs[n, i, j, k] = old_value / norm[i, j, k]
                gridindex = ((k - 1) * ngrid + (j - 1)) * ngrid + (i - 1) + 1
                fcontribs.write(f'{particleid[n]} {gridindex} {old_value / norm[i, j, k]}\n')
    # check some stuff on the grid

    gmass = 0.0
    nzero = 0
    nzerocentral = 0
    gmass = np.sum(grho) * dx * dy * dz
    # nzero = np.count_nonzero(grho[1:][1:][1:] < 1.e-20)

    for i in range(1, ngrid + 1):
        gx = x0 + dx * (i - 1)
        for j in range(1, ngrid + 1):
            gy = y0 + dy * (j - 1)
            for k in range(1, ngrid + 1):

                # how many cells with rho=0?

                if grho[i, j, k] < 1.e-20:
                    nzero = nzero + 1

                gz = z0 + dz * (k - 1)

                dis = math.sqrt(gx * gx + gy * gy + gz * gz)

                if (grho[i, j, k] < 1.e-20 and dis < rmean):
                    nzerocentral = nzerocentral + 1

    print("mass on grid and particles", gmass, totmass)

    print("number of cells with zero rho, total num of cells, fraction of cells w rho=0",
          nzero, ngrid**3, (nzero) / (ngrid**3))

    print("number of central cells (dis<rmean) with zero rho, ratio",
          nzerocentral, (nzerocentral) / (4. * 3.14 / 3. * rmean**3 / (dx * dy * dz)))

    print("probably we want to choose grid size, i.e. x0, as compromise between mapped mass and rho=0 cells")

    # output grid - adapt as you need output

    with open(Path(outputpath, 'grid.dat'), 'w') as fgrid:

        fgrid.write(f'{ngrid**3} # ngrid\n')
        fgrid.write(f'{dtextra} # extra time after explosion simulation ended (in geom units)\n')
        fgrid.write(f'{x0} # xmax\n')
        fgrid.write(' gridindex    pos_x_min    pos_y_min    pos_z_min    rho    cellYe    tracercount\n')
        gridindex = 1
        for k in range(1, ngrid + 1):
            gz = z0 + dz * (k - 1)
            for j in range(1, ngrid + 1):
                gy = y0 + dy * (j - 1)
                for i in range(1, ngrid + 1):
                    fgrid.write(
                        f'{gridindex:8d} {x0+dx*(i-1)} {gy} {gz} '
                        f'{grho[i,j,k]} {gye[i,j,k]} {gparticlecounter[i,j,k]}\n')
                    # gridindex2 = ((k - 1) * ngrid + (j - 1)) * ngrid + (i - 1) + 1

                    gridindex = gridindex + 1


def addargs(parser):
    parser.add_argument('-inputpath', '-i',
                        default='.',
                        help='Path to ejectasnapshot')
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=CustomArgHelpFormatter,
            description='Map tracer particle trajectories to a Cartesian grid.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    ejectasnapshotpath = Path(args.inputpath, 'ejectasnapshot.dat')

    maptogrid(ejectasnapshotpath=ejectasnapshotpath, outputpath=args.outputpath)


if __name__ == "__main__":
    main()
