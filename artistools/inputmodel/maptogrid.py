#!/usr/bin/env python3

# adapted from Fortran maptogrid.f90 and kernelmodule.f90
# original Fortran code possibly by Andreas Bauswein?

import argparse
import math
import numpy as np
import pandas as pd

from pathlib import Path
from artistools import CustomArgHelpFormatter

idim = 700000     # ax anzahl der teilchgen urspruenglich400000
ineigh = 140  # max # nachbarn als oberes cut off, wenn ueberschritten wir h manuell
# minimiert, sonst wird h ueber dgl siehe unten bestimmt
ineigh2 = idim
nsymm = idim
tablelen = 1000
idgr = idim
idco = idim
idfl = idim
itable = 40000  # wie fein Kernelfkt interpoliert wird
ilist = 200
ilistgr = 400
lmx = 64  # lm... wird nicht mehr gebraucht
lmy = 64
lmz = 32
ipm = 5
itab = itable + 5
nel = 4
ncell = 36
ncell2 = (ncell - 1) / 2
ncell22 = ncell2 + 2
itableg = itable
emean = 1.
lma = 4 * lmx * (lmx + 1)
epsb = 1.e-4


wij = np.zeros(itab + 1)
grwij = np.zeros(itab + 1)
fmass = np.zeros(itab + 1)
fpoten = np.zeros(itab + 1)
dphidh = np.zeros(itab + 1)
fmass2 = np.zeros(itab + 1)
fmass3 = np.zeros(itab + 1)

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
        v5 = v * v4
        v6 = v * v5
        v7 = v * v6
        v8 = v * v7
        sum = 1. - 1.5 * v2 + 0.75 * v3
        wij[i] = cnormk * sum
        sum = -3. * v + 2.25 * v2
        grwij[i] = cnormk * sum
# I2
        sum = 1.3333333333 * v3 - 1.2 * v5 + 0.5 * v6
        fmass[i] = sum
# I4
        sum = 0.8 * v5 - 6. / 7. * v7 + 3. / 8. * v8
        fmass2[i] = sum
# -(1/d*I2+J1)
        sum = 0.66666666666 * v2 - 0.3 * v4 + 0.1 * v5 - 1.4
        fpoten[i] = sum
# -J1
        sum = -1.4 + 2. * v2 - 1.5 * v4 + 0.6 * v5
        dphidh[i] = sum
else:
    for i in range(1, i1 + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        v3 = v * v2
        sum = 1. - 1.5 * v2 + 0.75 * v3
        wij[i] = cnormk * sum
        sum = -3. * v + 2.25 * v2
        grwij[i] = cnormk * sum

#
#  b) v greater than 1
#
if (igphi == 1):
    for i in range(i1 + 1, itable + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        v3 = v * v2
        v4 = v * v3
        v5 = v * v4
        v6 = v * v5
        v7 = v * v6
        v8 = v * v7
        dif2 = 2. - v
        sum = 0.25 * dif2 * dif2 * dif2
        wij[i] = cnormk * sum
        sum = -0.75 * v2 + 3. * v - 3.
        grwij[i] = cnormk * sum
# I2
        sum = -0.16666666666 * v6 + 1.2 * v5 - 3. * v4 + 2.66666666666 * v3 - 0.0666666666666
        fmass[i] = sum
# I4
        sum = -2. * v6 + 1.6 * v5 + 6. / 7. * v7 - 1. / 8. * v8 - 1. / 70.
        fmass2[i] = sum
# -(1/v*I2+J1) ohne 1/d-Term
        sum = -0.033333333333 * v5 + 0.3 * v4 - v3 + 1.3333333333 * v2 - 1.6
        fpoten[i] = sum
# -J1
        sum = -1.6 + 4. * v2 - 4. * v3 + 1.5 * v4 - 0.2 * v5
        dphidh[i] = sum
else:
    for i in range(i1 + 1, itable + 1):
        v2 = i * dvtable
        v = math.sqrt(v2)
        dif2 = 2. - v
        sum = 0.25 * dif2 * dif2 * dif2
        wij[i] = cnormk * sum
        sum = -0.75 * v2 + 3. * v - 3.
        grwij[i] = cnormk * sum


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


def addargs(parser):
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
    snapshot_columns = [
        'id', 'h', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vstx', 'vsty',
        'vstz', 'u', 'psi', 'alpha', 'pmass', 'rho', 'p', 'rst',
        'tau', 'av', 'Ye', 'temp', '?1', '?2', '?3', '?4', '?5', '?6', '?7', '?8', '?9']

    dfsnapshot = pd.read_csv('ejectasnapshot.dat', names=snapshot_columns, delim_whitespace=True)
    assert len(dfsnapshot.columns) == len(snapshot_columns)

    # print(dfsnapshot)

    npart = len(dfsnapshot)
    print("number of particles", npart)

    fpartanalysis = open(Path(args.outputpath, 'ejectapartanalysis.dat'), mode='w')

    totmass = 0.0
    rmax = 0.0
    rmean = 0.0
    hmean = 0.0
    hmin = 100000.
    vratiomean = 0.

    # Propagate particles to dtextra using velocities
    dtextra = 0.5  # in seconds ---  dtextra = 0.0 # for no extrapolation

    dtextra = dtextra / 4.926e-6  # convert to geom units.

    x = dfsnapshot.x.values
    y = dfsnapshot.y.values
    z = dfsnapshot.z.values
    h = dfsnapshot.h.values
    vx = dfsnapshot.vx.values
    vy = dfsnapshot.vy.values
    vz = dfsnapshot.vz.values
    pmass = dfsnapshot.pmass.values
    rst = dfsnapshot.rst.values
    Ye = dfsnapshot.Ye.values

    for n in range(npart):
        totmass = totmass + pmass[n]

        dis = math.sqrt(x[n] ** 2 + y[n] ** 2 + z[n] ** 2)  # original dis

        x[n] += vx[n] * dtextra
        y[n] += vy[n] * dtextra
        z[n] += vz[n] * dtextra

        # actually we should also extra polate smoothing length h## unless we disrgard it below

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
            vperp = math.sqrt(vtot * vtot - vrad * vrad)  # velicty perp# endicular
        else:
            vperp = 0.0  # if we xtrapolate roundoff error can lead to Nan, ly?

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

    x0 = - 0.7 * rmax  # 90% is hand waving - choose #

    # x0 = - rmean

    dx = 2. * abs(x0) / (ngrid)  # -1 to be symmetric, right?

    y0 = x0
    z0 = x0
    dy = dx
    dz = dx

    grho = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    norm = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    gye = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))
    gparticlecounter = np.zeros((ngrid + 1, ngrid + 1, ngrid + 1))

    print('grid properties', x0, dx, x0 + dx * (ngrid - 1))

    print('bigloop start')
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

            if (dis2 <= maxdist2):

                wtij = kernelvals2(dis2, h[n])

                grho[i, j, k] += pmass[n] * wtij
                # grho[i, j, k] = grho[i, j, k] + pmass[n] / particle.rst * wtij

                # gye[i, j, k] += pmass[n] * particle.Ye * wtij
                gye[i, j, k] += (pmass[n] / rst[n] * Ye[n] * wtij)

                # norm[i, j, k] += pmass[n] * wtij
                norm[i, j, k] += pmass[n] / rst[n] * wtij

                # Counts particles contributing to grid cell
                gparticlecounter[i, j, k] += 1

    print('bigloop end')
    for i in range(1, ngrid + 1):
        for j in range(1, ngrid + 1):
            for k in range(1, ngrid + 1):
                gye[i, j, k] = gye[i, j, k] / norm[i, j, k]
                # if not norm[i, j, k] > 0:
                #     print(i, j, k, norm[i, j, k], gye[i, j, k] )

    # check some stuff on the grid

    gmass = 0.0
    nzero = 0
    nzerocentral = 0

    for i in range(1, ngrid + 1):
        for j in range(1, ngrid + 1):
            for k in range(1, ngrid + 1):

                gmass = gmass + grho[i, j, k] * dx * dy * dz

                # how many cells with rho=0?

                if (grho[i, j, k] < 1.e-20):
                    nzero = nzero + 1

                gx = x0 + dx * (i - 1)
                gy = y0 + dy * (j - 1)
                gz = z0 + dz * (k - 1)

                dis = math.sqrt(gx * gx + gy * gy + gz * gz)

                if (grho[i, j, k] < 1.e-20 and dis < rmean):
                    nzerocentral = nzerocentral + 1

    print("mass on grid and particles", gmass, totmass)

    print("number of cells with zero rho, total num of cels, fraction of cells w rho=0",
          nzero, ngrid**3, (nzero) / (ngrid**3))

    print("number of central cells (dis<rmean) with zero rho, ratio",
          nzerocentral, (nzerocentral) / (4. * 3.14 / 3. * rmean**3 / (dx * dy * dz)))

    print("probably we want to choose grid size, i.e. x0, as compromise between mapped mass and rho=0 cells")

    # output grid - adapt as you need output

    with open(Path(args.outputpath, 'grid.dat'), 'w') as fgrid:

        fgrid.write(f'{ngrid**3} # ngrid\n')
        fgrid.write(f'{dtextra} # extra time after explosion simulation # ended (in geom units)\n')
        fgrid.write(f'{x0} # xmax\n')
        fgrid.write(' gridindex    posx    posy    posz    rho    cellYe    tracercount\n')
        gridindex = 1
        for k in range(1, ngrid + 1):
            for j in range(1, ngrid + 1):
                for i in range(1, ngrid + 1):
                    fgrid.write(
                        f'{gridindex:8d} {x0+dx*(i-1)} {y0+dy*(j-1)} {z0+dz*(k-1)} '
                        f'{grho[i,j,k]} {gye[i,j,k]} {gparticlecounter[i,j,k]}\n')
                    gridindex = gridindex + 1


if __name__ == "__main__":
    main()
