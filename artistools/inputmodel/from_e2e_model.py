#
# prepare data for ARTIS KN calculation from end-to-end hydro models
#

from __future__ import annotations

from pathlib import Path
from numpy import *

import artistools as at
import numpy as np
import pandas as pd
import pdb
# import h5py,gzip,pickle,os,itertools

cl = 29979245800.0
day = 86400.0
msol = 1.988e33
tsnap = 0.1 * day

# 1) Script parameters: TODO make them all script parameters via arg parsing

# choose model / files
base = "/lustre/theory/gleck/e2emodels/"
dat_file = "rhine/kn_input_1216n1a6nse.npz"
iso_file = "iso_table.npy"

# time of this snapshot
vmax = 0.5  # maximum velocity in units of c

# dimension the final model.txt file shall have
numb_cells_ARTIS_radial = 25
numb_cells_ARTIS_z = 50 # has to be even for models with equatorial symmetry

# ejecta types to include
dyn = True
HMNS = True
torus = True

# 2) Main code

# *******************************************************************


def sphkernel(dist, hsph, nu):
    # smoothing kernel for SPH-like interpolation of particle
    # data

    q = dist / hsph
    w = where(
        q < 1.0,
        1.0 - 1.5 * q**2 + 0.75 * q**3,
        where(q < 2.0, 0.25 * (2.0 - q) ** 3, 0.0),
    )

    if nu == 3:
        sigma = 1.0 / pi
    elif nu == 2:
        sigma = 10.0 / (7.0 * pi)

    w = w * sigma / hsph**nu

    return w


# *******************************************************************


def f1corr(rcyl, hsph, nu):
    # correction factor to improve behavior near the axis
    # see Garcia-Senz et al Mon. Not. R. Astron. Soc. 392, 346–360 (2009)

    xi = abs(rcyl) / hsph
    f1 = where(
        xi < 1.0,
        1.0
        / (7.0 / 15.0 / xi + 2.0 / 3.0 * xi - 1.0 / 6.0 * xi**3 + 1.0 / 20.0 * xi**4),
        where(
            xi < 2.0,
            1.0
            / (
                8.0 / 15.0 / xi
                - 1.0 / 3.0
                + 4.0 / 3.0 * xi
                - 2.0 / 3.0 * xi**2
                + 1.0 / 6.0 * xi**3
                - 1.0 / 60.0 * xi**4
            ),
            1.0,
        ),
    )

    return f1


# *******************************************************************
# *** main routine *************************************************
# *******************************************************************


def main() -> None:
    # dat      = np.load(base+'rhine/kn_input_HR_14n1a6nse.npz')
    dat = np.load(base + dat_file)
    iso = np.load(base + iso_file)
    # dattem = np.load(base + dattem_file)
    """
    colnames0 = [
        "Id",
        "Mass",
        "time",
        "t9",
        "Ye",
        "entropy",
        "n/seed",
        "tau",
        "radius",
        "velocity",
        "angle",
    ]
    traj_summ_data = pd.read_csv(
        Path(".", "summary-all.dat"),
        delimiter=r"\s+",
        skiprows=1,
        names=colnames0,
        dtype_backend="pyarrow",
    )
    ye_summ_file = traj_summ_data["Ye"].to_numpy()
    print(
        f"Average Y_e before interpolation: {((traj_summ_data['Mass'] * traj_summ_data['Ye']).sum()) / traj_summ_data['Mass'].sum()}"
    )
    """

    # assume equatorial symmetry? (1=no, 2=yes)
    eqsymfac = 2 if amax(dat.f.pos[:, 1]) < pi / 2.0 else 1

    # first re-construct the original post-merger trajectories by merging the
    # splitted dynamical ejecta trajectories
    idx = array([round(i) for i in dat.f.idx])  # unique particle ID of all trajectories
    state = array(
        [round(i) for i in dat.f.state]
    )  # == -1,0,1 for dynamical, NS-torus, BH-torus
    dyncond = state == -1
    dynidall = array([i % 10000 for i in idx])
    dynid = list(
        set([i % 10000 for i in idx[dyncond]])
    )  # original IDs of dynamical ejecta
    ndyn = len(dynid)
    nodid = list(set([i % 10000 for i in idx[~dyncond]]))  # IDs of other ejecta trajs.
    nnod = len(nodid)
    ntraj = ndyn + nnod
    mtraj = zeros(ntraj)  # final trajectory mass
    isoA0 = iso[:, 0] + iso[:, 1]  # mass number = neutron number + proton number
    xiso0 = dat.f.nz[:, :] * isoA0[:]  # number fraction -> mass fraction
    ncomp = len(xiso0[0, :])  # number of isotopes
    xtraj = zeros((ntraj, ncomp))  # final mass fractions for each isotope at t = tsnap
    ttraj = zeros(ntraj)  # final temperature in Kelvin
    vtraj = zeros(ntraj)  # final radial velocity
    atraj = zeros(ntraj)  # final polar angle
    qtraj = zeros(ntraj)  # integrated energy release up to snapshot
    # yetraj = np.zeros(ntraj)  # initial electron fraction
    nsplit = 5
    # fill arrays depending on the type of ejecta
    i = -1
    # ... non-dynamical ejecta
    for i1 in nodid:  # index of my original list
        i = i + 1  # index in the new list accounting for unprocessed trajs.
        i2 = list(dynidall).index(i1)  # index in Zeweis extended list of trajs.
        mtraj[i] = dat.f.mass[i2] * msol
        xtraj[i, :] = xiso0[i2, :]
        # ttraj[i] = dattem.f.T9[i2] * 1e9
        qtraj[i] = sum(dat.f.qdot[i2]) * msol
        # yetraj[i] = ye_summ_file[int(i1)]
        vtraj[i] = dat.f.pos[i2, 0]
        atraj[i] = dat.f.pos[i2, 1]
    # ... dynamical ejecta
    for i1 in dynid:  # index of my original list
        i = i + 1  # index in the new list accounting for unprocessed trajs.
        i2 = where(dynidall == i1)[0]  # indices in Zeweis extended list of trajs.
        # if len(i2)<nsplit:
        #     print('missing dyn ejecta at i=',i,len(i2))
        mtraj[i] = sum(dat.f.mass[i2]) * msol
        weights = dat.f.mass[i2] * msol / mtraj[i]
        xtraj[i, :] = sum(weights * xiso0[i2, :].T, 1)
        # ttraj[i] = sum(weights * dattem.f.T9[i2] * 1e9)
        qtraj[i] = sum(dat.f.qdot[i2]) * msol
        # yetraj[i] = np.sum(weights * ye_summ_file[int(i1)])
        vtraj[i] = sum(weights * dat.f.pos[i2, 0])
        atraj[i] = sum(weights * dat.f.pos[i2, 1])

    # now do the mapping using an SPH like interpolation
    # (see e.g. Price 2007, http://adsabs.harvard.edu/abs/2007PASA...24..159P,
    #  Price & Monaghan 2007, https://ui.adsabs.harvard.edu/abs/2007MNRAS.374.1347P,
    #  and Garcia-Senz 2009, https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..346G)

    # ... smoothing length prefactor and number of dimensions (see Eq. 10 of P2007)
    hsmeta = 1.01
    nu = 2
    # ... cylindrical coordinates of the particle positions
    rcyltraj, zcyltraj = zeros(ntraj), zeros(ntraj)
    for i in arange(ntraj):
        rcyltraj[i] = vtraj[i] * sin(atraj[i]) * cl * tsnap
        zcyltraj[i] = vtraj[i] * cos(atraj[i]) * cl * tsnap

    # ... cylindrical coordinates of the grid onto which we want to map
    # maximum velocities, will be changed if equatorial symmetry is given
    if eqsymfac == 1: 
        # no equatorial symmetry -> mapping grid has to have negative z as well
        vminr, vmaxr = 0.0, vmax
        vminz, vmaxz = 0.0, vmax
        nvr = numb_cells_ARTIS_radial
        nvz = numb_cells_ARTIS_z
    else:
        # equatorial symmetry -> mapping grid will be in the z >= 0 domain
        vminr, vmaxr = 0.0, vmax
        vminz, vmaxz = 0.0, vmax
        nvr = numb_cells_ARTIS_radial
        nvz = int(numb_cells_ARTIS_z / 2)
    vrgridl = array([vminr + i * (vmaxr - vminr) / nvr for i in arange(nvr)])
    vrgridr = flip(array([vmaxr - i * (vmaxr - vminr) / nvr for i in arange(nvr)]))
    vrgridc = 0.5 * (vrgridl + vrgridr)
    vzgridl = array([vminz + i * (vmaxz - vminz) / nvz for i in arange(nvz)])
    vzgridr = flip(array([vmaxz - i * (vmaxz - vminz) / nvz for i in arange(nvz)]))
    vzgridc = 0.5 * (vzgridl + vzgridr)
    op = multiply.outer
    rgridc2d = op(vrgridc, ones(nvz)) * cl * tsnap
    zgridc2d = op(ones(nvr), vzgridc) * cl * tsnap
    volgrid2d = (
        2.0
        * pi
        * op(vrgridr**2 / 2.0 - vrgridl**2 / 2.0, vzgridr - vzgridl)
        * (cl * tsnap) ** 3
    )

    # compute mass density and smoothing length of each particle
    # by solving Eq. 10 of P2007 where rho is replaced by the
    # 2D density rho_2D = rho_3D/(2 \pi R) = \sum_i m_i W_2D
    # with particle masses m_i and 2D interpolation kernel W_2D
    print("computing particle densities...")
    rho2dtraj = zeros(ntraj)  # this is the 2D density!!!
    hsmooth = zeros(ntraj)
    for i in arange(ntraj):
        # print(i)
        cont = True
        hl, hr = 0.00001 * cl * tsnap, 1.0 * cl * tsnap
        dist = sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
        ic = 0
        while cont:
            ic = ic + 1
            h1 = 0.5 * (hl + hr)
            wsph = sphkernel(dist, h1, nu)
            rhos = sum(wsph * mtraj)
            fun = (mtraj[i] / ((h1 / hsmeta) ** nu) - rhos) / rhos
            if fun > 0.0:
                hl = h1
            else:
                hr = h1
            if abs(hr - hl) / hl < 1e-5:
                cont = False
                hsmooth[i] = 0.5 * (hl + hr)
                wsph = sphkernel(dist, 0.5 * (hl + hr), nu)
                rho2dtraj[i] = sum(wsph * mtraj)
            if ic > 50:
                print("Not good:", ic, hl, hr, fun)
                if ic > 60:
                    raise Exception
        # print(hsmooth[i])

    # f1 correction a la Garcia-Senz? (does not seem to make a significant difference)
    rho2dhat = rho2dtraj * f1corr(rcyltraj, hsmooth, nu)
    # r2dhat = rho2dtraj

    # cross check: count number of neighbors within smoothing length
    neinum = zeros(ntraj)
    for i in arange(ntraj):
        dist = sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
        neinum[i] = sum(where(dist / hsmooth < 2.0, 1.0, 0.0))
    neinumavg = sum(neinum * mtraj) / sum(mtraj)
    print("average number of neighbors:", neinumavg)

    # now interpolate all quantities onto the grid
    print("interpolating...")
    oa = add.outer
    distall = sqrt(oa(rgridc2d, -rcyltraj) ** 2 + oa(zgridc2d, -zcyltraj) ** 2)
    hall = op(ones((nvr, nvz)), hsmooth)
    wall = sphkernel(distall, hall, nu)
    weight = wall * (mtraj / rho2dhat)
    weinor = (weight.T / (sum(weight, axis=2) + 1.0e-100).T).T
    hint = sum(weinor * hsmooth, axis=2)
    # ... density
    rho2d = sum(wall * mtraj * rho2dtraj / rho2dhat, axis=2)
    rhoint = rho2d / (2.0 * pi * rgridc2d)
    # rhoint     = rho2d/(2.*pi*clip(rgridc2d,0.5*hint,None))  # limiting to 0.5*h seems to prevent artefacts near the axis
    # ... mass fractions
    xint = tensordot(xtraj.T, wall * mtraj, axes=(1, 2)) / (
        sum(wall * mtraj, axis=2) + 1e-100
    )
    xin2 = tensordot(xtraj.T, weinor, axes=(1, 2))  # for testing
    # ... temperature
    temint = np.sum(weinor * ttraj, axis=2)
    qinterpol = np.sum(weinor * qtraj, axis=2)
    # yeinterpol = np.sum(weinor * yetraj, axis=2)

    # renormalize so that interpolated mass = sum of particle masses
    dmgrid = rhoint * volgrid2d
    print("total mass after interpolation  :", sum(dmgrid) / msol * eqsymfac)
    rescfac = sum(mtraj) / sum(dmgrid)
    dmgrid = dmgrid * rescfac
    mtot = sum(dmgrid)

    # rescale each partial density (?)
    # ... not good -> sum X = 1 violated
    # xin2 = copy(xint)
    # for l in arange(ncomp):
    #     xin2[l,:,:] = xint[l,:,:]*sum(xiso0[:,l]*dat.f.mass*msol)/(sum(dmgrid*xint[l,:,:])+1e-100)

    print("===> mapped data")
    print("total mass                :", mtot / msol * eqsymfac)
    print(
        "total element mass He Z=2 :",
        sum(sum(xint[iso[:, 1] == 2, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Zr Z=40:",
        sum(sum(xint[iso[:, 1] == 40, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Sn Z=50:",
        sum(sum(xint[iso[:, 1] == 50, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Te Z=52:",
        sum(sum(xint[iso[:, 1] == 52, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Xe Z=54:",
        sum(sum(xint[iso[:, 1] == 54, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass W  Z=74:",
        sum(sum(xint[iso[:, 1] == 74, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Pt Z=78:",
        sum(sum(xint[iso[:, 1] == 78, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    """
    print("===> mapped data 2")
    print("total mass                :", mtot / msol * eqsymfac)
    print(
        "total element mass He Z=2 :",
        sum(sum(xin2[iso[:, 1] == 2, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Zr Z=40:",
        sum(sum(xin2[iso[:, 1] == 40, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Sn Z=50:",
        sum(sum(xin2[iso[:, 1] == 50, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Te Z=52:",
        sum(sum(xin2[iso[:, 1] == 52, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Xe Z=54:",
        sum(sum(xin2[iso[:, 1] == 54, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass W  Z=74:",
        sum(sum(xin2[iso[:, 1] == 74, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    print(
        "total element mass Pt Z=78:",
        sum(sum(xin2[iso[:, 1] == 78, :, :], axis=0) * dmgrid) * eqsymfac / msol,
    )
    """
    print("===> tracer data")
    print("total mass                :", sum(dat.f.mass) * eqsymfac)
    print(
        "total element mass He Z=2 :",
        sum(sum(xiso0[:, iso[:, 1] == 2], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass Zr Z=40:",
        sum(sum(xiso0[:, iso[:, 1] == 40], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass Sn Z=50:",
        sum(sum(xiso0[:, iso[:, 1] == 50], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass Te Z=52:",
        sum(sum(xiso0[:, iso[:, 1] == 52], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass Xe Z=54:",
        sum(sum(xiso0[:, iso[:, 1] == 54], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass W  Z=74:",
        sum(sum(xiso0[:, iso[:, 1] == 74], axis=1) * dat.f.mass) * eqsymfac,
    )
    print(
        "total element mass Pt Z=78:",
        sum(sum(xiso0[:, iso[:, 1] == 78], axis=1) * dat.f.mass) * eqsymfac,
    )

    test = sum(xint, axis=0) - 1.0
    test = where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid    :", amax(where(test > -1, abs(test), 0.0)))

    test = sum(xin2, axis=0) - 1.0
    test = where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid 2  :", amax(where(test > -1, abs(test), 0.0)))

    # # write file containing the contribution of each trajectory to each interpolated grid cell
    # with open(base+'gridcontributions_'+'138n1a6'+'.txt','w') as f:
    #     f.write('r-grid-index z-grid-index particle-ID frac_of_cellmass'+'\n')
    #     f.write('(only cells with non-vanishing matter content are listed here)'+'\n')
    #     for i in arange(nvr):
    #         for j in arange(nvz):
    #             if dmgrid[i,j]>(1e-100*mtot):
    #                 wloc = wall[i,j,:]*rho2dtraj/rho2dhat
    #                 wloc = wloc/sum(wloc)
    #                 cond = wloc>1.e-20
    #                 pids = where(cond)[0]
    #                 for pid in pids:
    #                     f.write('{:<8} {:<8} {:<10} {:25.15e} \n'.format(i,j,pid,wloc[pid]))

    CLIGHT = 2.99792458e10  # Speed of light [cm/s]
    STEBO = 5.670400e-5  # Stefan-Boltzmann constant [erg cm^−2 s^−1 K^−4.]
    # Luke: get the energy per gram in the cell from the temperature by working backwards from:
    # T_initial = pow(CLIGHT / 4 / STEBO * rho_tmin * q_ergperg, 1. / 4.);
    q_ergperg = temint**4 * 4 * STEBO / CLIGHT / rhoint

    # write file containing the contribution of each trajectory to each interpolated grid cell
    grid_contribs_path = base + "gridcontributions.txt"
    with open(grid_contribs_path, "w", encoding="utf-8") as fgridcontributions:
        fgridcontributions.write(
            "particleid cellindex frac_of_cellmass frac_of_cellmass_includemissing"
            + "\n"
        )
        for nz in np.arange(nvz):
            for nr in np.arange(nvr):
                cellid = nz * nvr + nr + 1
                if dmgrid[nr, nz] > (1e-100 * mtot):
                    # print(
                    # f"{nr} {nz} {temint[nr, nz]} {q_ergperg[nr, nz]} {rhoint[nr, nz]} {dmgrid[nr, nz]} {xint[nr, nz]}"
                    # )
                    wloc = wall[nr, nz, :] * rho2dtraj / rho2dhat
                    wloc = wloc / np.sum(wloc)
                    pids = np.where(wloc > 1.0e-20)[0]
                    for pid in pids:
                        fgridcontributions.write(
                            f"{pid:<10}  {cellid:<8} {wloc[pid]:25.15e} {wloc[pid]:25.15e}\n",
                        )
    # pdb.set_trace()
    # return nvr, nvz, rgridc2d, zgridc2d, rhoint, xint, iso, q_ergperg, yeinterpol
    return nvr, nvz, rgridc2d, zgridc2d, rhoint, xint, iso, q_ergperg, eqsymfac


def z_reflect(arr: np.ndarray, sign=1) -> np.ndarray:
    """Flatten an array and add a reflection in z. Add a sign if the reflected quantity needs to be negated"""
    ngridrcyl, ngridz = arr.shape
    arr_ref = np.concatenate([sign*np.flip(arr[:,  :], axis=1), arr[:,  :]], axis=1)
    return arr_ref.flatten(order="F")


# function added by Luke and Gerrit to create the ARTIS model.txt
def create_ARTIS_modelfile(
    ngridrcyl,
    ngridz,
    pos_t_s_grid_rad,
    pos_t_s_grid_z,
    rho_interpol,
    X_cells,
    isot_table,
    q_ergperg,
    eqsymfac
):
    assert pos_t_s_grid_rad.shape == (ngridrcyl, ngridz)
    assert pos_t_s_grid_z.shape == (ngridrcyl, ngridz)
    assert rho_interpol.shape == (ngridrcyl, ngridz)
    assert q_ergperg.shape == (ngridrcyl, ngridz)
    numb_cells = numb_cells_ARTIS_radial * numb_cells_ARTIS_z
    
    if eqsymfac == 1:
        dfmodel = pd.DataFrame(
            {
                "inputcellid": range(1, numb_cells + 1),
                "pos_rcyl_mid": (pos_t_s_grid_rad).flatten(order="F"),
                "pos_z_mid": (pos_t_s_grid_z).flatten(order="F"),
                "rho": (rho_interpol).flatten(order="F"),
                "q": (q_ergperg).flatten(order="F"),
                # "cellYe": z_reflect(ye).flatten(order="F"),
            }
        )
    else:
        # equatorial symmetry -> have to reflect
        dfmodel = pd.DataFrame(
            {
                "inputcellid": range(1, numb_cells + 1),
                "pos_rcyl_mid": z_reflect(pos_t_s_grid_rad).flatten(order="F"),
                "pos_z_mid": z_reflect(pos_t_s_grid_z,sign=-1).flatten(order="F"),
                "rho": z_reflect(rho_interpol).flatten(order="F"),
                "q": z_reflect(q_ergperg).flatten(order="F"),
                # "cellYe": z_reflect(ye).flatten(order="F"),
            }
        )

    # add mass fraction columns
    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = pd.concat(
            [dfmodel, pd.DataFrame({"X_Fegroup": np.ones(len(dfmodel))})], axis=1
        )
    # pdb.set_trace()

    dictabunds = {}
    dictelabunds = {"inputcellid": range(1, numb_cells + 1)}
    for tuple_idx, isot_tuple in enumerate(isot_table):
        if eqsymfac == 1:
            flat_isoabund = np.nan_to_num(
                (X_cells[tuple_idx]).flatten(order="F"), 0.0
            )
        else:
            flat_isoabund = np.nan_to_num(
                z_reflect(X_cells[tuple_idx]).flatten(order="F"), 0.0
            )
        if np.any(flat_isoabund):
            elem_str = f"X_{at.get_elsymbol(isot_tuple[1])}"
            isotope_str = f"{elem_str}{isot_tuple[0] + isot_tuple[1]}"
            dictabunds[isotope_str] = flat_isoabund
            dictelabunds[elem_str] = (
                dictelabunds[elem_str] + flat_isoabund
                if elem_str in dictelabunds
                else flat_isoabund
            )
            # if elem_str == "X_Ni":
                # print(
                    # f"cell 25 massfrac {isotope_str}: {flat_isoabund[26]} {elem_str} {dictelabunds[elem_str][26]}"
                # )
    print(f"Number of non-zero nuclides {len(dictabunds)}")
    dfmodel = pd.concat([dfmodel, pd.DataFrame(dictabunds)], axis=1)

    dfabundances = pd.DataFrame(dictelabunds)
    dfabundances = dfabundances.fillna(0.0)

    # create init abundance file
    at.inputmodel.save_initelemabundances(dfabundances, "abundances.txt")

    # create modelmeta dictionary
    modelmeta = {
        "dimensions": 2,
        "ncoordgridrcyl": numb_cells_ARTIS_radial,
        "ncoordgridz": numb_cells_ARTIS_z,
        "t_model_init_days": tsnap / day,
        "vmax_cmps": vmax * 29979245800.0,
    }
    # create model.txt
    at.inputmodel.save_modeldata(dfmodel=dfmodel, modelmeta=modelmeta, modelpath=".")


def test_ye_txt():
    # function to test the Ye.txt
    pass


if __name__ == "__main__":
    (
        ncoordrcyl,
        ncoordz,
        pos_t_s_grid_rad,
        pos_t_s_grid_z,
        rho_interpol,
        X_cells,
        isot_table,
        q_ergperg,
        eqsymfac
    ) = main()
    create_ARTIS_modelfile(
        ncoordrcyl,
        ncoordz,
        pos_t_s_grid_rad,
        pos_t_s_grid_z,
        rho_interpol,
        X_cells,
        isot_table,
        q_ergperg,
        eqsymfac
    )
