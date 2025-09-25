"""Prepare data for ARTIS KN calculation from end-to-end hydro models. Original script by Oliver Just with modifications by Gerrit Leck for abundance mapping."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import itertools
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at

cl = 29979245800.0
day = 86400.0
msol = 1.989e33  # solar mass in g
tsnap = 0.1 * day  # snapshot time is fixed by the npz files


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


def get_grid(
    dat_path: Path,
    iso_path: Path,
    vmax: float,
    numb_cells_ARTIS_radial: int,
    numb_cells_ARTIS_z: int,
    nodynej: bool,
    nohmns: bool,
    notorus: bool,
    outputpath: Path,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    int,
]:
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
    state = np.array([round(i) for i in dat.f.state])  # == -1,0,1 for dynamical, NS-torus, BH-torus
    dyncond = state == -1
    dynidall = np.array([i % 10000 for i in idx])
    dynid = {i % 10000 for i in idx[dyncond]}  # original IDs of dynamical ejecta
    ndyn = len(dynid)
    nodid = {i % 10000 for i in idx[~dyncond]}  # IDs of other ejecta trajs.
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
    yetraj = np.zeros(ntraj)  # initial electron fraction

    time_s = dat.f.time
    starting_idx = (np.abs(time_s - 0.1)).argmin()
    closest_idx = (np.abs(time_s - tsnap)).argmin()
    i = -1
    tot_Q_rel = 0

    # first get masses and see if they have to be set to zero if the corresponding ejecta types shall be excluded
    # also set to integrated energy release up to snapshot time to zero

    mass_arr = dat.f.mass.copy()
    qdot_arr = dat.f.qdot.copy()

    # ... non-dynamical ejecta
    if nohmns:
        # exclude HMNS ejecta
        mass_arr[np.where(state == 0)] = 1e-30
        qdot_arr[np.where(state == 0)] = 1e-30
    if notorus:
        # exclude torus ejecta
        mass_arr[np.where(state == 1)] = 1e-30
        qdot_arr[np.where(state == 1)] = 1e-30
    for i1 in nodid:  # index of Oli's original list
        i += 1  # index in the new list accounting for unprocessed trajs.
        i2 = list(dynidall).index(i1)  # index in Zeweis extended list of trajs.
        mtraj[i] = mass_arr[i2] * msol
        # no multiplication with mass to keep it a specific energy release
        time_by_t_snap = time_s / tsnap
        qtraj[i] = np.trapezoid(
            time_by_t_snap[starting_idx:closest_idx] * qdot_arr[i2][starting_idx:closest_idx],
            time_s[starting_idx:closest_idx],
        )
        tot_Q_rel += mtraj[i] * qtraj[i]

    # ... dynamical ejecta

    if nodynej:
        # exclude dynamical ejecta
        mass_arr[np.where(state == -1)] = 1e-30
        qdot_arr[np.where(state == -1)] = 1e-30
    for i1 in dynid:
        i += 1  # index in the new list accounting for unprocessed trajs.
        i3 = np.where(dynidall == i1)[0]  # indices in Zeweis extended list of trajs.
        mtraj[i] = np.sum(mass_arr[i3]) * msol
        time_by_t_snap = time_s / tsnap
        qtraj[i] = np.trapezoid(
            time_by_t_snap[starting_idx:closest_idx] * np.sum(qdot_arr[i3], axis=0)[starting_idx:closest_idx],
            time_s[starting_idx:closest_idx],
        )
        tot_Q_rel += mtraj[i] * qtraj[i]
    print(f"tot_Q_rel: {tot_Q_rel}")
    i = -1
    # ... non-dynamical ejecta
    for i1 in nodid:  # index of Oli's original list
        i += 1  # index in the new list accounting for unprocessed trajs.
        i2 = list(dynidall).index(i1)  # index in Zeweis extended list of trajs.
        xtraj[i, :] = xiso0[i2, :]
        # ttraj[i] = dattem.f.T9[i2] * 1e9
        yetraj[i] = dat.f.t5out[i2, 4]
        vtraj[i] = dat.f.pos[i2, 0]
        atraj[i] = dat.f.pos[i2, 1]
    # ... dynamical ejecta
    for i1 in dynid:  # index of my original list
        i += 1  # index in the new list accounting for unprocessed trajs.
        i4 = np.where(dynidall == i1)[0]  # indices in Zeweis extended list of trajs.
        # if len(i2)<nsplit:
        #     print('missing dyn ejecta at i=',i,len(i2))
        weights = dat.f.mass[i4] * msol / (np.sum(dat.f.mass[i4]) * msol)
        xtraj[i, :] = np.sum(weights * xiso0[i4, :].T, axis=1)
        # ttraj[i] = sum(weights * dattem.f.T9[i2] * 1e9)
        # yetraj[i] = np.sum(weights * ye_summ_file[int(i1)])
        yetraj[i] = np.sum(weights * dat.f.t5out[i4, 4])
        vtraj[i] = np.sum(weights * dat.f.pos[i4, 0])
        atraj[i] = np.sum(weights * dat.f.pos[i4, 1])

    # now do the mapping using an SPH like interpolation
    # (see e.g. Price 2007, http://adsabs.harvard.edu/abs/2007PASA...24..159P,
    #  Price & Monaghan 2007, https://ui.adsabs.harvard.edu/abs/2007MNRAS.374.1347P,
    #  and Garcia-Senz 2009, https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..346G)

    # ... smoothing length prefactor and number of dimensions (see Eq. 10 of P2007)
    hsmeta = 1.01
    nu = 2
    # ... cylindrical coordinates of the particle positions
    rcyltraj, zcyltraj = np.zeros(ntraj), np.zeros(ntraj)
    for i in [int(j) for j in np.arange(ntraj)]:
        rcyltraj[i] = vtraj[i] * np.sin(atraj[i]) * cl * tsnap
        zcyltraj[i] = vtraj[i] * np.cos(atraj[i]) * cl * tsnap

    # ... cylindrical coordinates of the grid onto which we want to map
    vmax_cmps = vmax * cl
    nvr = numb_cells_ARTIS_radial
    nvz = numb_cells_ARTIS_z // eqsymfac  # number of mapping grid cells in z direction depends on equatorial symmetry

    wid_init_rcyl = vmax_cmps * tsnap / nvr
    pos_rcyl_min = np.array([vmax_cmps * tsnap / nvr * nr for nr in range(nvr)])
    pos_rcyl_mid = pos_rcyl_min + 0.5 * wid_init_rcyl
    pos_rcyl_max = pos_rcyl_min + wid_init_rcyl

    wid_init_z = (2.0 / eqsymfac) * vmax_cmps * tsnap / nvz
    if eqsymfac == 1:
        pos_z_min = np.array([-vmax_cmps * tsnap + 2.0 * vmax_cmps * tsnap / nvz * nz for nz in range(nvz)])
    else:
        pos_z_min = np.array([vmax_cmps * tsnap / nvz * nz for nz in range(nvz)])
    pos_z_mid = pos_z_min + 0.5 * wid_init_z
    # pos_z_max = pos_z_min + wid_init_z

    rgridc2d = np.array([pos_rcyl_mid[n_r] for n_r in range(nvr) for n_z in range(nvz)]).reshape(nvr, nvz)
    # the z-grid has to be shifted to starting from zero to keep consistency with Oli's script
    zgridc2d = np.array([pos_z_mid[n_z] for n_r in range(nvr) for n_z in range(nvz)]).reshape(nvr, nvz)

    volgrid2d = np.array([
        wid_init_z * np.pi * (pos_rcyl_max[n_r] ** 2 - pos_rcyl_min[n_r] ** 2)
        for n_r in range(nvr)
        for n_z in range(nvz)
    ]).reshape(nvr, nvz)

    # compute mass density and smoothing length of each particle
    # by solving Eq. 10 of P2007 where rho is replaced by the
    # 2D density rho_2D = rho_3D/(2 \pi R) = \sum_i m_i W_2D
    # with particle masses m_i and 2D interpolation kernel W_2D
    print("computing particle densities...")
    rho2dtraj = np.zeros(ntraj)  # this is the 2D density!!!
    hsmooth = np.zeros(ntraj)
    for i in [int(j) for j in np.arange(ntraj)]:
        # print(i)
        cont = True
        hl, hr = 0.00001 * cl * tsnap, 1.0 * cl * tsnap
        dist = np.sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
        ic = 0
        while cont:
            ic += 1
            h1 = 0.5 * (hl + hr)
            wsph = sphkernel(dist, h1, nu)
            rhos = np.sum(wsph * mtraj)
            fun = (mtraj[i] / ((h1 / hsmeta) ** nu) - rhos) / rhos
            if fun > 0.0:
                hl = h1
            else:
                hr = h1
            if abs(hr - hl) / hl < 1e-5:
                cont = False
                hsmooth[i] = 0.5 * (hl + hr)
                wsph = sphkernel(dist, 0.5 * (hl + hr), nu)
                rho2dtraj[i] = np.sum(wsph * mtraj)
            if ic > 50:
                print("Not good:", ic, hl, hr, fun)
                if ic > 60:
                    msg = "ic > 60"
                    raise AssertionError(msg)

    # f1 correction a la Garcia-Senz? (does not seem to make a significant difference)
    rho2dhat = rho2dtraj * f1corr(rcyltraj, hsmooth)

    # cross check: count number of neighbors within smoothing length
    neinum = np.zeros(ntraj)
    for i in [int(j) for j in np.arange(ntraj)]:
        dist = np.sqrt((rcyltraj[i] - rcyltraj) ** 2 + (zcyltraj[i] - zcyltraj) ** 2)
        neinum[i] = np.sum(np.where(dist / hsmooth < 2.0, 1.0, 0.0))
    neinumavg = np.sum(neinum * mtraj) / np.sum(mtraj)
    print("average number of neighbors:", neinumavg)

    # now interpolate all quantities onto the grid
    print("interpolating...")
    oa = np.add.outer
    distall = np.sqrt(oa(rgridc2d, -rcyltraj) ** 2 + oa(zgridc2d, -zcyltraj) ** 2)
    hall = np.multiply.outer(np.ones((nvr, nvz)), hsmooth)
    wall = sphkernel(distall, hall, nu)
    weight = wall * (mtraj / rho2dhat)
    weinor = (weight.T / (np.sum(weight, axis=2) + 1.0e-100).T).T
    hint = np.sum(weinor * hsmooth, axis=2)
    # ... density
    rho2d = np.sum(wall * mtraj * rho2dtraj / rho2dhat, axis=2)
    rhoint = rho2d / (
        2.0 * np.pi * np.clip(rgridc2d, 0.5 * hint, None)
    )  # limiting to 0.5*h seems to prevent artefacts near the axis
    # ... mass fractions
    xint = np.tensordot(xtraj.T, wall * mtraj, axes=(1, 2)) / (np.sum(wall * mtraj, axis=2) + 1e-100)
    xin2 = np.tensordot(xtraj.T, weinor, axes=(1, 2))  # for testing
    # ... temperature
    qinterpol = np.sum(weinor * qtraj, axis=2)
    yeinterpol = np.sum(weinor * yetraj, axis=2)

    # renormalize so that interpolated mass = sum of particle masses
    dmgrid = rhoint * volgrid2d
    print("total mass after interpolation (but BEFORE renormalization):", np.sum(dmgrid) / msol * eqsymfac)
    rescfac = np.sum(mtraj) / np.sum(dmgrid)
    dmgrid *= rescfac
    rhoint *= rescfac  # the densities are returned by means of rhoint
    mtot = np.sum(dmgrid)

    # test outputs
    print("===> mapped data")
    print("total mass                :", mtot / msol * eqsymfac)
    print("total element mass He Z=2 :", np.sum(np.sum(xint[iso[:, 1] == 2, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass Zr Z=40:", np.sum(np.sum(xint[iso[:, 1] == 40, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass Sn Z=50:", np.sum(np.sum(xint[iso[:, 1] == 50, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass Te Z=52:", np.sum(np.sum(xint[iso[:, 1] == 52, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass Xe Z=54:", np.sum(np.sum(xint[iso[:, 1] == 54, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass W  Z=74:", np.sum(np.sum(xint[iso[:, 1] == 74, :, :], axis=0) * dmgrid) * eqsymfac / msol)
    print("total element mass Pt Z=78:", np.sum(np.sum(xint[iso[:, 1] == 78, :, :], axis=0) * dmgrid) * eqsymfac / msol)

    print("===> tracer data")
    print("total mass                :", np.sum(dat.f.mass) * eqsymfac)
    print("total element mass He Z=2 :", np.sum(np.sum(xiso0[:, iso[:, 1] == 2], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass Zr Z=40:", np.sum(np.sum(xiso0[:, iso[:, 1] == 40], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass Sn Z=50:", np.sum(np.sum(xiso0[:, iso[:, 1] == 50], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass Te Z=52:", np.sum(np.sum(xiso0[:, iso[:, 1] == 52], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass Xe Z=54:", np.sum(np.sum(xiso0[:, iso[:, 1] == 54], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass W  Z=74:", np.sum(np.sum(xiso0[:, iso[:, 1] == 74], axis=1) * dat.f.mass) * eqsymfac)
    print("total element mass Pt Z=78:", np.sum(np.sum(xiso0[:, iso[:, 1] == 78], axis=1) * dat.f.mass) * eqsymfac)

    test = np.sum(xint, axis=0) - 1.0
    test = np.where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid    :", np.amax(np.where(test > -1, abs(test), 0.0)))

    test = np.sum(xin2, axis=0) - 1.0
    test = np.where(test > -1, test, 0.0)
    print("(X-1)_max over 2D grid 2  :", np.amax(np.where(test > -1, abs(test), 0.0)))

    # write file containing the contribution of each trajectory to each interpolated grid cell
    with (outputpath / Path("gridcontributions.txt")).open("w", encoding="utf-8") as fgridcontributions:
        fgridcontributions.write("particleid cellindex frac_of_cellmass frac_of_cellmass_includemissing" + "\n")
        for nz in np.arange(nvz):
            for nr in np.arange(nvr):
                cellid = nz * nvr + nr + 1
                if dmgrid[nr, nz] > (1e-100 * mtot):
                    # print(
                    # f"{nr} {nz} {temint[nr, nz]} {q_ergperg[nr, nz]} {rhoint[nr, nz]} {dmgrid[nr, nz]} {xint[nr, nz]}"
                    # )
                    wloc = wall[nr, nz, :] * rho2dtraj / rho2dhat
                    wloc /= np.sum(wloc)
                    pids = np.where(wloc > 1.0e-20)[0]
                    for pid in pids:
                        fgridcontributions.write(f"{pid:<10}  {cellid:<8} {wloc[pid]:25.15e} {wloc[pid]:25.15e}\n")

    return rgridc2d, zgridc2d, rhoint, xint, iso, qinterpol, yeinterpol, eqsymfac


def z_reflect(arr: npt.NDArray[np.floating | np.integer], sign: int = 1) -> npt.NDArray[np.floating | np.integer]:
    """Flatten an array and add a reflection in z."""
    _ngridrcyl = arr.shape[0]
    reflected = np.concatenate([sign * np.flip(arr[:, :], axis=1), arr[:, :]], axis=1)
    assert isinstance(reflected, np.ndarray)
    return reflected


# function added by Luke and Gerrit to create the ARTIS model.txt
def create_ARTIS_modelfile(
    ngridrcyl: int,
    ngridz: int,
    vmax: float,
    pos_t_s_grid_rad: npt.NDArray[np.floating | np.integer],
    pos_t_s_grid_z: npt.NDArray[np.floating | np.integer],
    rho_interpol: npt.NDArray[np.floating | np.integer],
    X_cells: npt.NDArray[np.floating | np.integer],
    isot_table: npt.NDArray[t.Any],
    q_ergperg: npt.NDArray[np.floating | np.integer],
    ye_traj: npt.NDArray[np.floating | np.integer],
    eqsymfac: int,
    outputpath: Path,
) -> None:
    numb_cells = ngridrcyl * ngridz
    import pandas as pd

    # now reflect the arrays if equatorial symmetry is assumed or otherwise not
    if eqsymfac == 1:
        dfmodel = pd.DataFrame({
            "inputcellid": range(1, numb_cells + 1),
            "pos_rcyl_mid": (pos_t_s_grid_rad).flatten(order="F"),
            "pos_z_mid": (pos_t_s_grid_z).flatten(order="F"),
            "rho": (rho_interpol).flatten(order="F"),
            "q": (q_ergperg).flatten(order="F"),
            # "cellYe": z_reflect(ye).flatten(order="F"),
        })
    else:
        # equatorial symmetry -> have to reflect
        pos_t_s_grid_rad = z_reflect(pos_t_s_grid_rad)
        pos_t_s_grid_z = z_reflect(pos_t_s_grid_z, sign=-1)
        rho_interpol = z_reflect(rho_interpol)
        q_ergperg = z_reflect(q_ergperg)
        ye_traj = z_reflect(ye_traj)
        dfmodel = pd.DataFrame({
            "inputcellid": range(1, numb_cells + 1),
            "pos_rcyl_mid": (pos_t_s_grid_rad).flatten(order="F"),
            "pos_z_mid": (pos_t_s_grid_z).flatten(order="F"),
            "rho": (rho_interpol).flatten(order="F"),
            "q": (q_ergperg).flatten(order="F"),
            "cellYe": (ye_traj).flatten(order="F"),
        })

    assert pos_t_s_grid_rad.shape == (ngridrcyl, ngridz)
    assert pos_t_s_grid_z.shape == (ngridrcyl, ngridz)
    assert rho_interpol.shape == (ngridrcyl, ngridz)
    assert q_ergperg.shape == (ngridrcyl, ngridz)

    # DF_model, DF_el_contribs, DF_contribs = at.inputmodel.rprocess_from_trajectory.add_abundancecontributions(
    #     dfgridcontributions=DF_gridcontributions,
    #     dfmodel=DF_model,
    #     t_model_days_incpremerger=0.1,
    #     traj_root=Path("./traj_PM/"),
    # )

    # add mass fraction columns
    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = pd.concat([dfmodel, pd.DataFrame({"X_Fegroup": np.ones(len(dfmodel))})], axis=1)
    # pdb.set_trace()

    dictabunds = {}
    dictelabunds = {"inputcellid": np.array(range(1, numb_cells + 1))}
    for tuple_idx, isot_tuple in enumerate(isot_table):
        if eqsymfac == 1:
            flat_isoabund = np.nan_to_num((X_cells[tuple_idx]).flatten(order="F"), nan=0.0)
        else:
            flat_isoabund = np.nan_to_num(z_reflect(X_cells[tuple_idx]).flatten(order="F"), nan=0.0)
        if np.any(flat_isoabund):
            elem_str = f"X_{at.get_elsymbol(isot_tuple[1])}"
            isotope_str = f"{elem_str}{isot_tuple[0] + isot_tuple[1]}"
            dictabunds[isotope_str] = flat_isoabund
            dictelabunds[elem_str] = (
                dictelabunds[elem_str] + flat_isoabund if elem_str in dictelabunds else flat_isoabund
            )

    print(f"Number of non-zero nuclides {len(dictabunds)}")
    dfmodel = pd.concat([dfmodel, pd.DataFrame(dictabunds)], axis=1)

    dfabundances = pd.DataFrame(dictelabunds).fillna(0.0)

    # create init abundance file
    at.inputmodel.save_initelemabundances(dfelabundances=pl.from_pandas(dfabundances), outpath=outputpath)

    # create modelmeta dictionary
    modelmeta = {
        "dimensions": 2,
        "ncoordgridrcyl": ngridrcyl,
        "ncoordgridz": ngridz,
        "t_model_init_days": tsnap / day,
        "vmax_cmps": vmax * cl,
    }

    # create model.txt
    at.inputmodel.save_modeldata(dfmodel=pl.from_pandas(dfmodel), modelmeta=modelmeta, outpath=outputpath)


def get_old_cell_indices(red_fact: int, new_r: int, new_z: int, N_cell_r_old: int) -> list[int]:
    # function to get old grid indices for a given new grid index
    old_r_indices = np.arange(red_fact * (new_r - 1) + 1, red_fact * new_r + 1)
    old_z_indices = np.arange(red_fact * (new_z - 1), red_fact * new_z) * N_cell_r_old
    indices = np.add.outer(old_r_indices, old_z_indices).flatten().tolist()
    return sorted(int(x) for x in indices)


def remap_mass_weighted_quantity(
    im_ARTIS_old: pl.DataFrame,
    fieldname: str,
    red_fact: int,
    N_cell_r_new: int,
    N_cell_z_new: int,
    N_cell_r_old: int,
    Delta_r: float,
    Delta_z: float,
) -> npt.NDArray[np.float64]:
    # function that remaps any mass-weighted quantity
    new_numb_cells = N_cell_r_new * N_cell_z_new
    new_field = np.zeros(new_numb_cells)

    # get arrays
    mass_arr = im_ARTIS_old["mass_g"].to_numpy()
    field_arr = im_ARTIS_old[fieldname].to_numpy()

    for new_z, new_r in itertools.product(range(1, N_cell_z_new + 1), range(1, N_cell_r_new + 1)):
        new_cell_idx = new_r + N_cell_r_new * (new_z - 1)

        old_idxs = get_old_cell_indices(red_fact, new_r, new_z, N_cell_r_old)
        old_idxs_np = np.array(old_idxs) - 1

        masses = mass_arr[old_idxs_np]
        values = field_arr[old_idxs_np]

        if fieldname == "mass_g":
            r_i = (new_r - 1) * Delta_r
            r_o = new_r * Delta_r
            V_new = np.pi * (r_o**2 - r_i**2) * Delta_z
            new_field[new_cell_idx - 1] = masses.sum() / V_new
        else:
            new_field[new_cell_idx - 1] = np.average(values, weights=masses)

    return new_field


def merge_ARTIS_cells(red_fact: int, N_r: int, N_z: int, v_max: float, outputpath: Path) -> None:
    """red_fact: number of cells to be merged."""
    import pandas as pd

    red_fact_1D = int(np.sqrt(red_fact))
    im_ARTIS_old = at.inputmodel.get_modeldata(modelpath=Path(), derived_cols=["mass_g"])[0].collect()
    N_cell_r_old, N_cell_z_old = N_r, N_z
    N_cell_r_new, N_cell_z_new = int(N_cell_r_old / red_fact_1D), int(N_cell_z_old / red_fact_1D)
    new_numb_cells = N_cell_r_new * N_cell_z_new
    r_max_snap = v_max * cl * tsnap

    # create new grid
    Delta_r = r_max_snap / N_cell_r_new
    Delta_z = 2 * r_max_snap / N_cell_z_new  # account for positive and negative z-values
    assert np.isclose(Delta_r, Delta_z, rtol=1e-20), "New grid cells not quadratic"
    r_mid_grid = np.array([(i + 0.5) * Delta_r for i in range(N_cell_r_new)])
    z_mid_grid = np.array([-r_max_snap + (i + 0.5) * Delta_z for i in range(N_cell_z_new)])

    print("Now remap masses, q and Ye")
    # remap density, integrated energy release and Y_e
    new_rho_arr = remap_mass_weighted_quantity(
        im_ARTIS_old, "mass_g", red_fact_1D, N_cell_r_new, N_cell_z_new, N_r, Delta_r, Delta_z
    )
    new_q_arr = remap_mass_weighted_quantity(
        im_ARTIS_old, "q", red_fact_1D, N_cell_r_new, N_cell_z_new, N_r, Delta_r, Delta_z
    )
    new_ye_arr = remap_mass_weighted_quantity(
        im_ARTIS_old, "Ye", red_fact_1D, N_cell_r_new, N_cell_z_new, N_r, Delta_r, Delta_z
    )
    print("  Done.")

    # new model data frame
    dfmodel = pd.DataFrame({
        "inputcellid": range(1, new_numb_cells + 1),
        "pos_rcyl_mid": np.tile(r_mid_grid, N_cell_z_new),
        "pos_z_mid": np.repeat(z_mid_grid, N_cell_r_new),
        "rho": new_rho_arr,
        "q": new_q_arr,
        "cellYe": new_ye_arr,
    })

    # now map the abundances
    dictabunds = {}
    element_abbrevs_list = [at.get_elsymbol(Z) for Z in range(1, 101)]  # Keep full list as in your code
    element_abbrevs_list_titled = [abbrev.title() for abbrev in element_abbrevs_list]
    el_mass_fracs = np.zeros((len(element_abbrevs_list), new_numb_cells))
    dictelabunds = {"inputcellid": np.array(range(1, new_numb_cells + 1))}

    nuclide_columns = [col for col in im_ARTIS_old.columns if col.startswith("X_")][1:]

    masses_all = im_ARTIS_old["mass_g"].to_numpy()
    abunds_all = {nuclide: im_ARTIS_old[nuclide].to_numpy() for nuclide in nuclide_columns}
    for nuclide in nuclide_columns:
        new_X_arr = np.zeros(new_numb_cells)
        for new_z in range(1, N_cell_z_new + 1):
            for new_r in range(1, N_cell_r_new + 1):
                new_cell_idx = new_r + N_cell_r_new * (new_z - 1)
                old_idxs = get_old_cell_indices(red_fact_1D, new_r, new_z, N_cell_r_old)
                old_idxs_np = np.array(old_idxs) - 1
                masses = masses_all[old_idxs_np]
                abunds = abunds_all[nuclide][old_idxs_np]
                new_X_arr[new_cell_idx - 1] = np.average(abunds, weights=masses)
        dictabunds[nuclide] = new_X_arr
        el = "".join([i for i in nuclide[2:] if not i.isdigit()])
        if el in element_abbrevs_list_titled:
            el_idx = element_abbrevs_list_titled.index(el)
            el_mass_fracs[el_idx] += new_X_arr

    for i, el in enumerate(element_abbrevs_list_titled):
        dictelabunds[f"X_{el}"] = el_mass_fracs[i]

    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = pd.concat([dfmodel, pd.DataFrame({"X_Fegroup": np.ones(len(dfmodel))})], axis=1)

    dfmodel = pd.concat([dfmodel, pd.DataFrame(dictabunds)], axis=1)
    dfabundances = pd.DataFrame(dictelabunds).fillna(0.0)

    at.inputmodel.save_initelemabundances(dfelabundances=pl.from_pandas(dfabundances), outpath=outputpath)
    modelmeta = {
        "dimensions": 2,
        "ncoordgridrcyl": N_cell_r_new,
        "ncoordgridz": N_cell_z_new,
        "t_model_init_days": tsnap / day,
        "vmax_cmps": v_max * cl,
    }

    at.inputmodel.save_modeldata(dfmodel=pl.from_pandas(dfmodel), modelmeta=modelmeta, outpath=outputpath)
    print(f"Successfully remapped model to {N_cell_r_new}x{N_cell_z_new} grid.")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default=".", help="Path of output ARTIS model file")

    parser.add_argument("-npz", required=True, help="Path to the model npz file")

    parser.add_argument("-iso", required=True, help="Path to the nuclide information npy (!) file")

    parser.add_argument(
        "-vmax", type=float, default=0.5, help="Maximum one-direction velocity in units of c the ARTIS model shall have"
    )

    parser.add_argument(
        "-ngridrcyl", type=int, default=25, help="Number of cells in radial direction the ARTIS model shall have"
    )

    parser.add_argument(
        "-ngridz", type=int, default=50, help="Number of cells in z direction the ARTIS model shall have"
    )

    parser.add_argument("--nodyn", action="store_true", help="Exclude dynamical ejecta from the model")

    parser.add_argument("--nohmns", action="store_true", help="Exclude neutrino wind ejecta from the model")

    parser.add_argument("--notorus", action="store_true", help="Exclude torus ejecta from the model")

    parser.add_argument(
        "-mergecells",
        type=int,
        default=None,
        help="Merge specified number of cells in postprocessing to keep precision in the mass fractions",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    numb_cells_ARTIS_radial = int(args.ngridrcyl)
    numb_cells_ARTIS_z = int(args.ngridz)
    pos_t_s_grid_rad, pos_t_s_grid_z, rho_interpol, X_cells, isot_table, q_ergperg, ye_traj, eqsymfac = get_grid(
        args.npz,
        args.iso,
        float(args.vmax),
        numb_cells_ARTIS_radial,
        numb_cells_ARTIS_z,
        args.nodyn,
        args.nohmns,
        args.notorus,
        args.outputpath,
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
        args.outputpath,
    )

    if args.mergecells is not None:
        # bunch of checks to assure proper cell merging
        assert isinstance(args.mergecells, int), "Number of cells to merge is not an integer!"
        # check if the number is a square number
        assert np.sqrt(args.mergecells).is_integer(), "Number of cells to merge is not a square number!"
        # check if the current number of cells is a multiple of the cells to merge
        assert (numb_cells_ARTIS_z / np.sqrt(args.mergecells)).is_integer(), (
            "Number of merged cells in z direction is no integer!"
        )
        assert (numb_cells_ARTIS_radial / np.sqrt(args.mergecells)).is_integer(), (
            "Number of merged cells in r direction is no integer!"
        )
        merge_ARTIS_cells(
            args.mergecells, numb_cells_ARTIS_radial, numb_cells_ARTIS_z, float(args.vmax), args.outputpath
        )


if __name__ == "__main__":
    main()
