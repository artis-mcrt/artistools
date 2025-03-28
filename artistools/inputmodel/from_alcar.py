"""Prepare data for ARTIS KN calculation from end-to-end hydro models. Original script by Oliver Just with modifications by Gerrit Leck for abundance mapping."""

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

# time of this snapshot
tsnap = 0.1 * day
vmax = 0.5  # maximum velocity in units of c


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


def get_grid() -> tuple[
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
    base = Path()
    dat = np.load(base / "kilonova_artis_input_138n1a6.npz")
    iso = np.load(base / "iso_table.npy")
    dattem = np.load(base / "kilonova_artis_input_138n1a6_rho_energy.npz")

    # first re-construct the original post-merger trajectories by merging the
    # split dynamical ejecta trajectories
    idx = np.array([int(i) for i in dat.f.idx])
    print(idx)
    isoA = iso[:, 0] + iso[:, 1]  # mass number = neutron number + proton number
    xiso0 = dat.f.nz[:, :] * isoA[:]
    state = np.array([round(i) for i in dat.f.state])
    dyncond = state == -1
    ndyn0 = np.sum(dyncond)
    ndyn = int(ndyn0 / 5.0)
    ntraj0 = len(idx)
    ntraj = ntraj0 - ndyn0 + ndyn
    mtraj = np.zeros(ntraj)  # final trajectory mass
    ncomp = len(xiso0[0, :])  # number of isotopes
    xtraj = np.zeros((ntraj, ncomp))  # final mass fractions for each isotope at t = tsnap
    ttraj = np.zeros(ntraj)  # final temperature in Kelvin
    vtraj = np.zeros(ntraj)  # final radial velocity
    atraj = np.zeros(ntraj)  # final polar angle
    dynid0 = idx[dyncond & (idx < 1e4)]
    nsplit = 5

    # ... first fill arrays with all data from non-dynamical ejecta
    i1: int | list[int]
    for i, i1 in ((i, np.nonzero(idx == i)[0][0]) for i in range(ntraj) if not np.isin(i, dynid0)):
        mtraj[i] = dat.f.mass[i1] * msol
        xtraj[i, :] = xiso0[i1, :]
        ttraj[i] = dattem.f.T9[i1] * 1e9
        vtraj[i] = dat.f.pos[i1, 0]
        atraj[i] = dat.f.pos[i1, 1]

    # ... now dynamical ejecta
    for i in dynid0:
        i1 = [np.nonzero(idx == int(i + n * 1e4))[0][0] for n in range(nsplit)]
        mtraj[i] = np.sum(dat.f.mass[i1]) * msol
        weights = dat.f.mass[i1] * msol / mtraj[i]
        xtraj[i, :] = np.sum(weights * xiso0[i1, :].T, 1)
        ttraj[i] = np.sum(weights * dattem.f.T9[i1] * 1e9)
        vtraj[i] = np.sum(weights * dat.f.pos[i1, 0])
        atraj[i] = np.sum(weights * dat.f.pos[i1, 1])

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
    vmax_cmps = vmax * 29979245800.0
    nvr = 25
    nvz = 50

    wid_init_rcyl = vmax_cmps * tsnap / nvr
    pos_rcyl_min = np.array([vmax_cmps * tsnap / nvr * nr for nr in range(nvr)])
    pos_rcyl_mid = pos_rcyl_min + 0.5 * wid_init_rcyl
    pos_rcyl_max = pos_rcyl_min + wid_init_rcyl

    wid_init_z = 2 * vmax_cmps * tsnap / nvz
    pos_z_min = np.array([-vmax_cmps * tsnap + 2.0 * vmax_cmps * tsnap / nvz * nz for nz in range(nvz)])
    pos_z_mid = pos_z_min + 0.5 * wid_init_z
    # pos_z_max = pos_z_min + wid_init_z

    rgridc2d = np.array([pos_rcyl_mid[n_r] for n_r in range(nvr) for n_z in range(nvz)]).reshape(nvr, nvz)
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
    print(f"computing particle densities...{ntraj} trajectories")
    rho2dtraj = np.zeros(ntraj)  # this is the 2D density!!!
    hsmooth = np.zeros(ntraj)
    for i in np.arange(ntraj):
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
    for i in np.arange(ntraj):
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
    xint = np.tensordot(xtraj.T, wall * mtraj, axes=(1, 2)) / np.sum(wall * mtraj, axis=2)
    # pdb.set_trace()
    # ... temperature
    temint = np.sum(weinor * ttraj, axis=2)

    # renormalize so that interpolated mass = sum of particle masses
    dmgrid = rhoint * volgrid2d
    print("mass after interpolation  :", np.sum(dmgrid) / msol)
    dmgrid = dmgrid / np.sum(dmgrid) * np.sum(mtraj)
    mtot = np.sum(dmgrid)
    print("mass after renormalization:", mtot / msol)

    CLIGHT = 2.99792458e10  # Speed of light [cm/s]
    STEBO = 5.670400e-5  # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4.]
    # Luke: get the energy per gram in the cell from the temperature by working backwards from:
    # T_initial = pow(CLIGHT / 4 / STEBO * rho_tmin * q_ergperg, 1. / 4.);
    q_ergperg = temint**4 * 4 * STEBO / CLIGHT / rhoint

    # write file containing the contribution of each trajectory to each interpolated grid cell
    with (base / "gridcontributions.txt").open("w", encoding="utf-8") as fgridcontributions:
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

    return nvr, nvz, rgridc2d, zgridc2d, rhoint, xint, iso, q_ergperg


def z_reflect(arr: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.floating | np.integer]:
    """Flatten an array and add a reflection in z."""
    _ngridrcyl, ngridz = arr.shape
    assert ngridz % 2 == 0
    reflected = np.concatenate([np.flip(arr[:, ngridz // 2 :], axis=1), arr[:, ngridz // 2 :]], axis=1).flatten(
        order="F"
    )
    assert isinstance(reflected, np.ndarray)
    return reflected


# function added by Luke and Gerrit to create the ARTIS model.txt
def create_ARTIS_modelfile(
    ngridrcyl: int,
    ngridz: int,
    pos_t_s_grid_rad: npt.NDArray[np.floating],
    pos_t_s_grid_z: npt.NDArray[np.floating],
    rho_interpol: npt.NDArray[np.floating],
    X_cells: npt.NDArray[np.floating],
    isot_table: npt.NDArray[t.Any],
    q_ergperg: npt.NDArray[np.floating],
    outpath: Path,
) -> None:
    assert pos_t_s_grid_rad.shape == (ngridrcyl, ngridz)
    assert pos_t_s_grid_z.shape == (ngridrcyl, ngridz)
    assert rho_interpol.shape == (ngridrcyl, ngridz)
    assert q_ergperg.shape == (ngridrcyl, ngridz)
    numb_cells = ngridrcyl * ngridz
    # DF_gridcontributions = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(".")
    # pdb.set_trace()
    # print("gridcontributions.txt read...done")
    dfmodel = pd.DataFrame({
        "inputcellid": range(1, numb_cells + 1),
        "pos_rcyl_mid": (pos_t_s_grid_rad).flatten(order="F"),
        "pos_z_mid": (pos_t_s_grid_z).flatten(order="F"),
        "rho": z_reflect(rho_interpol).flatten(order="F"),
        "q": z_reflect(q_ergperg).flatten(order="F"),
    })

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
    at.inputmodel.save_initelemabundances(dfelabundances=dfabundances, outpath=outpath)

    # create modelmeta dictionary
    modelmeta = {
        "dimensions": 2,
        "ncoordgridrcyl": ngridrcyl,
        "ncoordgridz": ngridz,
        "t_model_init_days": tsnap / day,
        "vmax_cmps": vmax * 29979245800.0,
    }

    # create model.txt
    at.inputmodel.save_modeldata(dfmodel=dfmodel, modelmeta=modelmeta, outpath=outpath)


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputpath", "-i", default=".", help="Path of snapshot files")
    parser.add_argument("-outputpath", "-o", default=".", help="Path of output ARTIS model file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    ngridrcyl, ngridz, pos_t_s_grid_rad, pos_t_s_grid_z, rho_interpol, X_cells, isot_table, q_ergperg = get_grid()

    create_ARTIS_modelfile(
        ngridrcyl,
        ngridz,
        pos_t_s_grid_rad,
        pos_t_s_grid_z,
        rho_interpol,
        X_cells,
        isot_table,
        q_ergperg,
        outpath=args.outputpath,
    )


if __name__ == "__main__":
    main()
