"""Prepare data for ARTIS KN calculation from end-to-end hydro models. Original script by Oliver Just with modifications by Gerrit Leck for abundance mapping."""

"""
Output:
- ARTIS model.txt, abundances.txt
- gridcontributions.txt
- Ye.txt for grey runs
"""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import numpy.typing as npt
import pandas as pd

import artistools as at

cl = 29979245800.0
day = 86400.0
msol = 1.989e33  # solar mass in g
tsnap = 0.1 * day # snapshot time is fixed by the npz files


def sphkernel(
    dist: npt.NDArray[np.floating], hsph: float | npt.NDArray[np.floating], nu: float
) -> npt.NDArray[np.floating]:
    # smoothing kernel for SPH-like interpolation of particle
    # data

    q = dist / hsph
    w = np.where(q < 1.0, 1.0 - 1.5 * q**2 + 0.75 * q**3, np.where(q < 2.0, 0.25 * (2.0 - q) ** 3, 0.0))

    if nu == 3:
        sigma = 1.0 / np.pi
    elif nu == 2:
        sigma = 10.0 / (7.0 * np.pi)

    return w * sigma / hsph**nu


# *******************************************************************


def f1corr(rcyl: npt.NDArray[np.floating], hsph: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    # correction factor to improve behavior near the axis
    # see Garcia-Senz et al Mon. Not. R. Astron. Soc. 392, 346-360 (2009)

    xi = abs(rcyl) / hsph
    return np.where(
        xi < 1.0,
        1.0 / (7.0 / 15.0 / xi + 2.0 / 3.0 * xi - 1.0 / 6.0 * xi**3 + 1.0 / 20.0 * xi**4),
        np.where(
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


def get_grid(dat_path, iso_path, vmax, numb_cells_ARTIS_radial, numb_cells_ARTIS_z, dynej, hmns, torus) -> tuple[
    int,
    int,
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    # base = Path('/the/ojust/lustredata/luke/hmnskn_2023/138n1a6/')
    dat = np.load(dat_path)
    iso = np.load(iso_path)

    # assume equatorial symmetry? (1=no, 2=yes)
    # determine whether the model assumes equatorial symmetry. Criterion: if the input npz files contains
    # tracer particles only in the upper half space (z > 0 or angle < pi/2), equatorial symmetry is assumed
    # and the reflection w.r.t. the z-axis for the final model.txt has to be done
    eqsymfac = 2 if np.amax(dat.f.pos[:, 1]) < np.pi / 2.0 else 1

    # first re-construct the original post-merger trajectories by merging the
    # splitted dynamical ejecta trajectories
    idx = np.array([round(i) for i in dat.f.idx])  # unique particle ID of all trajectories
    state = np.array(
        [round(i) for i in dat.f.state]
    )  # == -1,0,1 for dynamical, NS-torus, BH-torus
    dyncond = state == -1
    dynidall = np.array([i % 10000 for i in idx])
    dynid = list(
        set([i % 10000 for i in idx[dyncond]])
    )  # original IDs of dynamical ejecta
    ndyn = len(dynid)
    nodid = list(set([i % 10000 for i in idx[~dyncond]]))  # IDs of other ejecta trajs.
    nnod = len(nodid)
    ntraj = ndyn + nnod
    mtraj = np.zeros(ntraj)  # final trajectory mass
    isoA0 = iso[:, 0] + iso[:, 1]  # mass number = neutron number + proton number
    xiso0 = dat.f.nz[:, :] * isoA0[:]  # number fraction -> mass fraction
    ncomp = len(xiso0[0, :])  # number of isotopes
    xtraj = np.zeros((ntraj, ncomp))  # final mass fractions for each isotope at t = tsnap
    vtraj = np.zeros(ntraj)  # final radial velocity
    atraj = np.zeros(ntraj)  # final polar angle
    qtraj = np.zeros(ntraj)  # integrated energy release up to snapshot
    yetraj = np.zeros(ntraj)

    time_s = dat.f.time
    closest_idx = (np.abs(time_s - tsnap)).argmin()
    # yetraj = np.zeros(ntraj)  # initial electron fraction
    # fill arrays depending on the type of ejecta
    i = -1

    # first get masses and see if they have to be set to zero if the corresponding ejecta types shall be excluded
    # also set to integrated energy release up to snapshot time to zero

    # ... non-dynamical ejecta
    for i1 in nodid: # index of Oli Just's original list
        i = i+1                          # index in the new list accounting for unprocessed trajs.
        i2 = list(dynidall).index(i1)    # index in Zeweis extended list of trajs.
        mtraj[i] = dat.f.mass[i2]*msol
        qtraj[i] = sum(dat.f.qdot[i2][:closest_idx]) # no multiplication with mass to keep it a specific energy release
    if not hmns:
        # exclude HMNS ejecta
        mtraj[np.where(state == 0)] = 1e-15
        qtraj[np.where(state == 0)] = 1e-15
    if not torus:
        # exclude torus ejecta
        mtraj[np.where(state == 1)] = 1e-15
        qtraj[np.where(state == 1)] = 1e-15

    # ... dynamical ejecta
    for i1 in dynid:
        i  = i+1                         # index in the new list accounting for unprocessed trajs.
        i2 = np.where(dynidall==i1)[0]      # indices in Zeweis extended list of trajs.
        mtraj[i] = sum(dat.f.mass[i2]) * msol
        qtraj[i] = sum(dat.f.qdot[i2][:closest_idx])
    if not dynej:
        # exclude dynamical ejecta
        mtraj[np.where(state == -1)] = 1e-15
        qtraj[np.where(state == -1)] = 1e-15     

    # ... non-dynamical ejecta
    for i1 in nodid:  # index of my original list
        i = i + 1  # index in the new list accounting for unprocessed trajs.
        i2 = list(dynidall).index(i1)  # index in Zeweis extended list of trajs.
        xtraj[i, :] = xiso0[i2, :]
        # ttraj[i] = dattem.f.T9[i2] * 1e9
        qtraj[i] = sum(dat.f.qdot[i2]) * msol
        yetraj[i] = dat.f.t5out[i2, 4]
        vtraj[i] = dat.f.pos[i2, 0]
        atraj[i] = dat.f.pos[i2, 1]
    # ... dynamical ejecta
    for i1 in dynid:  # index of my original list
        i = i + 1  # index in the new list accounting for unprocessed trajs.
        i2 = np.where(dynidall == i1)[0]  # indices in Zeweis extended list of trajs.
        # if len(i2)<nsplit:
        #     print('missing dyn ejecta at i=',i,len(i2))
        weights = dat.f.mass[i2] / sum(dat.f.mass[i2])
        xtraj[i, :] = sum(weights * xiso0[i2, :].T, 1)
        # ttraj[i] = sum(weights * dattem.f.T9[i2] * 1e9)
        qtraj[i] = sum(dat.f.qdot[i2]) * msol
        # yetraj[i] = np.sum(weights * ye_summ_file[int(i1)])
        yetraj[i] = sum(weights * dat.f.t5out[i2, 4])
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
    rcyltraj, zcyltraj = np.zeros(ntraj), np.zeros(ntraj)
    for i in np.arange(ntraj):
        rcyltraj[i] = vtraj[i] * np.sin(atraj[i]) * cl * tsnap
        zcyltraj[i] = vtraj[i] * np.cos(atraj[i]) * cl * tsnap

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
    vrgridl = np.array([vminr + i * (vmaxr - vminr) / nvr for i in np.arange(nvr)])
    vrgridr = np.flip(np.array([vmaxr - i * (vmaxr - vminr) / nvr for i in np.arange(nvr)]))
    vrgridc = 0.5 * (vrgridl + vrgridr)
    vzgridl = np.array([vminz + i * (vmaxz - vminz) / nvz for i in np.arange(nvz)])
    vzgridr = np.flip(np.array([vmaxz - i * (vmaxz - vminz) / nvz for i in np.arange(nvz)]))
    vzgridc = 0.5 * (vzgridl + vzgridr)
    op = np.multiply.outer
    rgridc2d = op(vrgridc, np.ones(nvz)) * cl * tsnap
    zgridc2d = op(np.ones(nvr), vzgridc) * cl * tsnap
    volgrid2d = (
        2.0
        * np.pi
        * op(vrgridr**2 / 2.0 - vrgridl**2 / 2.0, vzgridr - vzgridl)
        * (cl * tsnap) ** 3
    )

    # compute mass density and smoothing length of each particle
    # by solving Eq. 10 of P2007 where rho is replaced by the
    # 2D density rho_2D = rho_3D/(2 \pi R) = \sum_i m_i W_2D
    # with particle masses m_i and 2D interpolation kernel W_2D
    print("computing particle densities...")
    rho2dtraj = np.zeros(ntraj)  # this is the 2D density!!!
    hsmooth = np.zeros(ntraj)
    for i in np.arange(ntraj):
        # print(i)
        cont = True
        hl, hr = 0.00001 * cl * tsnap, 1.0 * cl * tsnap
        dist = np.sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
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
    neinum = np.zeros(ntraj)
    for i in np.arange(ntraj):
        dist = np.sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
        neinum[i] = sum(np.where(dist / hsmooth < 2.0, 1.0, 0.0))
    neinumavg = sum(neinum * mtraj) / sum(mtraj)
    print("average number of neighbors:", neinumavg)

    # now interpolate all quantities onto the grid
    print("interpolating...")
    oa = np.add.outer
    distall = np.sqrt(oa(rgridc2d, -rcyltraj) ** 2 + oa(zgridc2d, -zcyltraj) ** 2)
    hall = op(np.ones((nvr, nvz)), hsmooth)
    wall = sphkernel(distall, hall, nu)
    weight = wall * (mtraj / rho2dhat)
    weinor = (weight.T / (sum(weight, axis=2) + 1.0e-100).T).T
    hint = sum(weinor * hsmooth, axis=2)
    # ... density
    rho2d = sum(wall * mtraj * rho2dtraj / rho2dhat, axis=2)
    rhoint = rho2d / (2.0 * np.pi * rgridc2d)
    # rhoint     = rho2d/(2.*pi*clip(rgridc2d,0.5*hint,None))  # limiting to 0.5*h seems to prevent artefacts near the axis
    # ... mass fractions
    xint = np.tensordot(xtraj.T, wall * mtraj, axes=(1, 2)) / (
        sum(wall * mtraj, axis=2) + 1e-100
    )
    xin2 = np.tensordot(xtraj.T, weinor, axes=(1, 2))  # for testing
    # ... temperature
    qinterpol = np.sum(weinor * qtraj, axis=2)
    yeinterpol = np.sum(weinor * yetraj, axis=2)

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

    # test outputs
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
    test = np.where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid    :", np.amax(np.where(test > -1, abs(test), 0.0)))

    test = sum(xin2, axis=0) - 1.0
    test = np.where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid 2  :", np.amax(np.where(test > -1, abs(test), 0.0)))

    # write file containing the contribution of each trajectory to each interpolated grid cell
    grid_contribs_path = "gridcontributions.txt"
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
    return rgridc2d, zgridc2d, rhoint, xint, iso, qinterpol, yeinterpol, eqsymfac

def z_reflect(arr: np.ndarray, sign=1) -> np.ndarray:
    """Flatten an array and add a reflection in z. Add a sign if the reflected quantity needs to be negated"""
    ngridrcyl, ngridz = arr.shape
    reflected = np.concatenate([sign*np.flip(arr[:,  :], axis=1), arr[:,  :]], axis=1)
    assert isinstance(reflected, np.ndarray)
    return reflected.flatten(order="F")

# function added by Luke and Gerrit to create the ARTIS model.txt
def create_ARTIS_modelfile(
    numb_cells_ARTIS_radial,
    numb_cells_ARTIS_z,
    vmax,
    pos_t_s_grid_rad,
    pos_t_s_grid_z,
    rho_interpol,
    X_cells,
    isot_table,
    q_ergperg,
    ye_traj,
    eqsymfac
):
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
                "cellYe": z_reflect(ye_traj).flatten(order="F"),
            }
        )
    
    assert pos_t_s_grid_rad.shape == (numb_cells_ARTIS_radial, numb_cells_ARTIS_z)
    assert pos_t_s_grid_z.shape == (numb_cells_ARTIS_radial, numb_cells_ARTIS_z)
    assert rho_interpol.shape == (numb_cells_ARTIS_radial, numb_cells_ARTIS_z)
    assert q_ergperg.shape == (numb_cells_ARTIS_radial, numb_cells_ARTIS_z)

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


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputpath", "-i", default=".", help="Path of snapshot files")
    parser.add_argument("-outputpath", "-o", default=".", help="Path of output ARTIS model file")

    parser.add_argument("-npz", required=True, help="Path to the model npz file")

    parser.add_argument("-iso", required=True, help="Path to the nuclide information npy (!) file")

    parser.add_argument("-vmax", required=True, help="Maximum one-direction velocity in units of c the ARTIS model shall have")

    parser.add_argument("-Ncell_r", required=True, help="Number of cells in radial direction the ARTIS model shall have")

    parser.add_argument("-Ncell_z", required=True, help="Number of cells in z direction the ARTIS model shall have")

    parser.add_argument("-dyn", default=True, help="Set to false if the model shall exclude dynamical ejecta")

    parser.add_argument("-hmns", default=True, help="Set to false if the model shall exclude neutrino wind ejecta")

    parser.add_argument("-torus", default=True, help="Set to false if the model shall exclude torus ejecta")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    numb_cells_ARTIS_radial = int(args.Ncell_r)
    numb_cells_ARTIS_z = int(args.Ncell_z)
    pos_t_s_grid_rad, pos_t_s_grid_z, rho_interpol, X_cells, isot_table, q_ergperg, ye_traj, eqsymfac = get_grid(
        args.npz,
        args.iso,
        float(args.vmax),
        numb_cells_ARTIS_radial,
        numb_cells_ARTIS_z,
        args.dyn,
        args.hmns,
        args.torus
    )

    create_ARTIS_modelfile(
        numb_cells_ARTIS_radial,
        numb_cells_ARTIS_z,
        float(args.vmax),
        pos_t_s_grid_rad,
        pos_t_s_grid_z,
        rho_interpol,
        X_cells,
        isot_table,
        q_ergperg,
        ye_traj,
        eqsymfac,
        outpath=args.outputpath,
    )


if __name__ == "__main__":
    main()
