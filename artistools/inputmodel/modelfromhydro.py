#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import artistools as at

MSUN = 1.989e33
CLIGHT = 2.99792458e10


def read_ejectasnapshot(pathtosnapshot: str | Path) -> pd.DataFrame:
    column_names = [
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
        "rst",
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

    return pd.read_csv(
        Path(pathtosnapshot) / "ejectasnapshot.dat",
        sep=r"\s+",
        header=None,
        names=column_names,
        dtype={"id": int} | {col: float for col in column_names if col != "id"},
    )


def get_merger_time_geomunits(pathtogriddata: Path) -> float:
    mergertimefile = pathtogriddata / "tmerger.txt"
    if mergertimefile.exists():
        with mergertimefile.open("rt", encoding="utf-8") as fmergertimefile:
            comments = fmergertimefile.readline()
            assert comments.startswith("#")
            mergertime_geomunits = float(fmergertimefile.readline())
            print(f"Found simulation merger time to be {mergertime_geomunits} ({mergertime_geomunits * 4.926e-6} s) ")
        return mergertime_geomunits

    msg = 'Make file "tmerger.txt" with time of merger in geom units'
    raise FileNotFoundError(msg)


def get_snapshot_time_geomunits(pathtogriddata: Path | str) -> tuple[float, float]:
    pathtogriddata = Path(pathtogriddata)
    snapshotinfofiles = list(pathtogriddata.glob("*_info.dat*"))
    if not snapshotinfofiles:
        print("No info file found for dumpstep")
        sys.exit(1)

    if len(snapshotinfofiles) > 1:
        print("Too many sfho_info.dat files found")
        sys.exit(1)
    snapshotinfofile = Path(snapshotinfofiles[0])

    if snapshotinfofile.is_file():
        with snapshotinfofile.open("rt", encoding="utf-8") as fsnapshotinfo:
            line1 = fsnapshotinfo.readline()
            simulation_end_time_geomunits = float(line1.split()[2])
            print(
                f"Found simulation snapshot time to be {simulation_end_time_geomunits} "
                f"({simulation_end_time_geomunits * 4.926e-6} s)"
            )

        mergertime_geomunits = get_merger_time_geomunits(pathtogriddata)
        print(f"  time since merger {(simulation_end_time_geomunits - mergertime_geomunits) * 4.926e-6} s")

    else:
        print("Could not find snapshot info file to get simulation time")
        sys.exit(1)

    return simulation_end_time_geomunits, mergertime_geomunits


def read_griddat_file(
    pathtogriddata: str | Path, targetmodeltime_days: float | None = None
) -> tuple[pd.DataFrame, float, float, float, dict[str, t.Any]]:
    griddatfilepath = Path(pathtogriddata) / "grid.dat"

    # Get simulation time for ejecta snapshot
    simulation_end_time_geomunits, mergertime_geomunits = get_snapshot_time_geomunits(pathtogriddata)

    griddata = pd.read_csv(griddatfilepath, sep=r"\s+", comment="#", skiprows=3, dtype_backend="pyarrow").rename(
        columns={
            "gridindex": "inputcellid",
            "pos_x": "pos_x_min",
            "pos_y": "pos_y_min",
            "pos_z": "pos_z_min",
            "posx": "pos_x_min",  # for compatibility with fortran maptogrid script
            "posy": "pos_y_min",
            "posz": "pos_z_min",
        }
    )
    # griddata in geom units
    griddata["rho"] = griddata["rho"].fillna(0.0)

    if "cellYe" in griddata:
        griddata["cellYe"] = griddata["cellYe"].fillna(0.0)

    if "Q" in griddata:
        griddata["Q"] = griddata["Q"].fillna(0.0)

    factor_position = 1.478  # in km
    km_to_cm = 1e5
    griddata["pos_x_min"] = griddata["pos_x_min"] * factor_position * km_to_cm
    griddata["pos_y_min"] = griddata["pos_y_min"] * factor_position * km_to_cm
    griddata["pos_z_min"] = griddata["pos_z_min"] * factor_position * km_to_cm

    griddata["rho"] *= 6.176e17  # convert to g/cm³

    with griddatfilepath.open(encoding="utf-8") as gridfile:
        ngrid = int(gridfile.readline().split()[0])
        if ngrid != len(griddata["inputcellid"]):
            print("length of file and ngrid don't match")
            sys.exit(1)
        extratime_geomunits = float(gridfile.readline().split()[0])
        xmax = abs(float(gridfile.readline().split()[0]))
        xmax = (xmax * factor_position) * km_to_cm

    t_model_sec = (
        (simulation_end_time_geomunits - mergertime_geomunits) + extratime_geomunits
    ) * 4.926e-6  # in seconds
    # t_model of zero is the merger, but this was not time zero in the NSM simulation time
    t_mergertime_s = mergertime_geomunits * 4.926e-6
    vmax = xmax / t_model_sec  # cm/s

    t_model_days = t_model_sec / (24.0 * 3600)  # in days
    print(f"t_model in days {t_model_days} ({t_model_sec} s)")
    corner_vmax = math.sqrt(3 * vmax**2)
    print(
        f"vmax {vmax:.2e} cm/s ({vmax / 29979245800:.2f} * c) per component "
        f"real corner vmax {corner_vmax:.2e} cm/s ({corner_vmax / 29979245800:.2f} * c)"
    )

    if targetmodeltime_days is not None:
        griddata, modelmeta = at.inputmodel.scale_model_to_time(
            targetmodeltime_days=targetmodeltime_days, t_model_days=t_model_days, dfmodel=griddata
        )
        t_model_days = targetmodeltime_days
        xmax = -griddata.pos_x_min.min()

    ncoordgridx = round(len(griddata) ** (1.0 / 3.0))
    assert ncoordgridx**3 == len(griddata)
    wid_init = 2 * xmax / ncoordgridx
    print(f"Grid model is {ncoordgridx} x {ncoordgridx} x {ncoordgridx} = {len(griddata)} cells")
    griddata["mass_g"] = griddata["rho"] * wid_init**3

    print(f"Max tracers in a cell {max(griddata['tracercount'])}")

    modelmeta = {
        "dimensions": 3,
        "t_model_init_days": t_model_days,
        "vmax_cmps": vmax,
        "ncoordgridx": ncoordgridx,
        "ncoordgridy": ncoordgridx,
        "ncoordgridz": ncoordgridx,
        "wid_init_x": wid_init,
        "wid_init_y": wid_init,
        "wid_init_z": wid_init,
        "headercommentlines": [f"gridfolder: {Path(pathtogriddata).resolve().parts[-1]}"],
    }

    return griddata, t_model_days, t_mergertime_s, vmax, modelmeta


def read_mattia_grid_data_file(pathtogriddata: Path | str) -> tuple[pd.DataFrame, float, float]:
    # griddatfilepath = Path(pathtogriddata) / "q90_m0.01_v0.1.txt"
    griddatfilepath = Path(pathtogriddata) / "1D_m0.01_v0.1.txt"

    griddata = pd.read_csv(griddatfilepath, sep=r"\s+", comment="#", skiprows=1)
    with griddatfilepath.open(encoding="utf-8") as gridfile:
        t_model = float(gridfile.readline())
        print(f"t_model {t_model} seconds")
    xmax = max(griddata["posx"])
    vmax = xmax / t_model  # cm/s
    t_model /= 24.0 * 3600
    ngrid = len(griddata["posx"])

    griddata["rho"][griddata["rho"] <= 1e-50] = 0.0
    inputcellid = np.arange(1, ngrid + 1)
    griddata["inputcellid"] = inputcellid

    return griddata, t_model, vmax


def mirror_model_in_axis(griddata: pd.DataFrame) -> pd.DataFrame:
    grid = round(len(griddata) ** (1.0 / 3.0))

    rho = np.zeros((grid, grid, grid))
    cellYe = np.zeros((grid, grid, grid))
    tracercount = np.zeros((grid, grid, grid))
    Q = np.zeros((grid, grid, grid))

    i = 0
    for z in range(grid):
        for y in range(grid):
            for x in range(grid):
                rho[x, y, z] = griddata["rho"][i]
                cellYe[x, y, z] = griddata["cellYe"][i]
                tracercount[x, y, z] = griddata["tracercount"][i]
                Q[x, y, z] = griddata["Q"][i]
                i += 1

    for z in range(grid):
        z_mirror = grid - 1 - z
        for y in range(grid):
            for x in range(grid):
                if z < 50:
                    rho[x, y, z] = rho[x, y, z]
                    cellYe[x, y, z] = cellYe[x, y, z]
                    tracercount[x, y, z] = tracercount[x, y, z]
                    Q[x, y, z] = Q[x, y, z]
                if z >= 50:
                    rho[x, y, z] = rho[x, y, z_mirror]
                    cellYe[x, y, z] = cellYe[x, y, z_mirror]
                    tracercount[x, y, z] = tracercount[x, y, z_mirror]
                    Q[x, y, z] = Q[x, y, z_mirror]

    rho_1d_array = np.zeros(len(griddata))
    cellYe_1d_array = np.zeros(len(griddata))
    tracercount_1d_array = np.zeros(len(griddata))
    Q_1d_array = np.zeros(len(griddata))
    i = 0
    for z in range(grid):
        for y in range(grid):
            for x in range(grid):
                rho_1d_array[i] = rho[x, y, z]
                cellYe_1d_array[i] = cellYe[x, y, z]
                tracercount_1d_array[i] = tracercount[x, y, z]
                Q_1d_array[i] = Q[x, y, z]
                i += 1

    griddata["rho"] = rho_1d_array
    griddata["cellYe"] = cellYe_1d_array
    griddata["tracercount"] = tracercount_1d_array
    griddata["Q"] = Q_1d_array

    return griddata


def add_mass_to_center(
    griddata: pd.DataFrame,
    t_model_in_days: float,
    vmax: float,  # noqa: ARG001
    args: argparse.Namespace,  # noqa: ARG001
) -> pd.DataFrame:
    from scipy import integrate

    print(griddata)

    # Just (2021) Fig. 16 top left panel
    vel_hole = [0, 0.02, 0.05, 0.07, 0.09, 0.095, 0.1]
    mass_hole = [3e-4, 3e-4, 2e-4, 1e-4, 2e-5, 1e-5, 1e-9]
    mass_integrated = integrate.trapezoid(y=mass_hole, x=vel_hole)  # Msun

    # # Just (2021) Fig. 16 4th down, left panel
    # vel_hole = [0, 0.02, 0.05, 0.1, 0.15, 0.16]
    # mass_hole = [4e-3, 2e-3, 1e-3, 1e-4, 6e-6, 1e-9]
    # mass_integrated = integrate.trapezoid(y=mass_hole, x=vel_hole)  # Msun

    v_outer_hole = 0.1 * CLIGHT  # cm/s
    pos_outer_hole = v_outer_hole * t_model_in_days * (24.0 * 3600)  # cm
    vol_hole = 4 / 3 * np.pi * pos_outer_hole**3  # cm^3
    density_hole = (mass_integrated * MSUN) / vol_hole  # g / cm^3
    print(density_hole)

    for i, cellid in enumerate(griddata["inputcellid"]):
        # if pos < 0.1 c
        if (
            (np.sqrt(griddata["pos_x_min"][i] ** 2 + griddata["pos_y_min"][i] ** 2 + griddata["pos_z_min"][i] ** 2))
            / (t_model_in_days * (24.0 * 3600))
            / CLIGHT
        ) < 0.1:
            # if griddata['rho'][i] == 0:
            print("Inner empty cells")
            print(
                cellid, griddata["pos_x_min"][i], griddata["pos_y_min"][i], griddata["pos_z_min"][i], griddata["rho"][i]
            )
            griddata["rho"][i] += density_hole
            griddata["cellYe"][i] = max(griddata["cellYe"][i], 0.4)
            # print("Inner empty cells filled")
            print(
                cellid, griddata["pos_x_min"][i], griddata["pos_y_min"][i], griddata["pos_z_min"][i], griddata["rho"][i]
            )

    return griddata


def makemodelfromgriddata(
    gridfolderpath: Path | str,
    outputpath: Path | str,
    targetmodeltime_days: float | None = None,
    traj_root: Path | str | None = None,
    dimensions: int = 3,
    scalemass: float = 1.0,
    scalevelocity: float = 1.0,
    args: argparse.Namespace | None = None,
) -> None:
    if args is None:
        args = argparse.Namespace()
    pddfmodel, t_model_days, t_mergertime_s, vmax, modelmeta = at.inputmodel.modelfromhydro.read_griddat_file(
        pathtogriddata=gridfolderpath, targetmodeltime_days=targetmodeltime_days
    )

    if getattr(args, "fillcentralhole", False):
        pddfmodel = at.inputmodel.modelfromhydro.add_mass_to_center(pddfmodel, t_model_days, vmax, args)

    if getattr(args, "getcellopacityfromYe", False):
        at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, pddfmodel)

    dfgridcontributions = (
        at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(gridfolderpath)
        if Path(gridfolderpath, "gridcontributions.txt").is_file()
        else None
    )

    dfmodel = pl.from_pandas(pddfmodel).sort("inputcellid")
    assert dfmodel.schema["inputcellid"].is_integer()
    assert isinstance(dfmodel, pl.DataFrame)
    dfmodel = dfmodel.with_columns(pl.col("inputcellid").cast(pl.Int32))  # pylint: disable=no-member
    if scalemass != 1.0:
        origmass_msun = dfmodel["mass_g"].sum() / 2.99792458e33
        dfmodel = dfmodel.with_columns(rho=pl.col("rho") * scalemass, mass_g=pl.col("mass_g") * scalemass)
        newmass_msun = dfmodel["mass_g"].sum() / 2.99792458e33
        operationmsg = f"densities are scaled by factor of {scalemass} to increase total mass from {origmass_msun:.2e} to {newmass_msun:.2e} Msun"
        print(operationmsg)
        modelmeta["headercommentlines"].append(operationmsg)

    if scalevelocity != 1.0:
        dfmodel = dfmodel.with_columns(cs.starts_with("pos_") * scalevelocity, rho=pl.col("rho") * (scalevelocity**-3))
        vmax_cmps_old = modelmeta["vmax_cmps"]
        modelmeta["vmax_cmps"] *= scalevelocity
        operationmsg = f"velocities are scaled by a factor of {scalevelocity} (with density scaled by 1/f^3 to conserve mass). vmax/c changed from {vmax_cmps_old / 29979245800:.2f} to {modelmeta['vmax_cmps'] / 29979245800:.2f}"
        print(operationmsg)
        modelmeta["headercommentlines"].append(operationmsg)

    if traj_root is not None:
        print(f"Nuclear network abundances from {traj_root} will be used")
        modelmeta["headercommentlines"].append(f"trajfolder: {Path(traj_root).resolve().parts[-1]}")
        t_model_days_incpremerger = t_model_days + (t_mergertime_s / 86400)
        assert dfgridcontributions is not None
        (dfmodel, dfelabundances, dfgridcontributions) = (
            at.inputmodel.rprocess_from_trajectory.add_abundancecontributions(
                dfgridcontributions=dfgridcontributions,
                dfmodel=dfmodel,
                t_model_days_incpremerger=t_model_days_incpremerger,
                traj_root=traj_root,
            )
        )
    else:
        print("WARNING: No abundances will be set because no nuclear network trajectories folder was specified")
        dfelabundances = None

    if dimensions < 3:
        dfmodel, dfelabundances, dfgridcontributions, modelmeta = at.inputmodel.dimension_reduce_model(
            dfmodel=dfmodel,
            outputdimensions=dimensions,
            dfelabundances=dfelabundances,
            dfgridcontributions=dfgridcontributions,
            modelmeta=modelmeta,
        )

    if "cellYe" in dfmodel:
        at.inputmodel.opacityinputfile.write_Ye_file(outputpath, dfmodel)

    # if "Q" in dfmodel and args.makeenergyinputfiles:
    #     at.inputmodel.energyinputfiles.write_Q_energy_file(outputpath, dfmodel)

    if dfgridcontributions is not None:
        at.inputmodel.rprocess_from_trajectory.save_gridparticlecontributions(
            dfgridcontributions, Path(outputpath, "gridcontributions.txt")
        )

    if dfelabundances is not None:
        print(f"Writing to {Path(outputpath) / 'abundances.txt'}...")
        at.inputmodel.save_initelemabundances(
            dfelabundances=dfelabundances, outpath=outputpath, headercommentlines=modelmeta["headercommentlines"]
        )
    else:
        at.inputmodel.save_empty_abundance_file(outputfilepath=outputpath, npts_model=len(dfmodel))

    if "tracercount" in dfmodel:
        dfmodel = dfmodel.with_columns(pl.col("tracercount").cast(pl.Int32))

    print(f"Writing to {Path(outputpath) / 'model.txt'}...")
    at.inputmodel.save_modeldata(outpath=outputpath, dfmodel=dfmodel, modelmeta=modelmeta)


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-gridfolderpath", "-i", default=".", help="Path to folder containing grid.dat and gridcontributions.dat"
    )
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        default=None,
        help="Path to nuclear network trajectory folder, if abundances are required",
    )
    parser.add_argument(
        "-dimensions",
        "-d",
        default=3,
        type=int,
        help="Number of dimensions: 0 for one-zone spherical, 1 for spherically symmetric 1D, 2 for 2D cylindrical, 3 for 3D Cartesian",
    )
    parser.add_argument(
        "-targetmodeltime_days", "-t", type=float, default=0.1, help="Time in days for the output model snapshot"
    )
    parser.add_argument(
        "-scalemass",
        type=float,
        default=1.0,
        help="Multiply the total mass by scaling densities by some factor before writing the model file",
    )
    parser.add_argument(
        "-scalevelocity",
        type=float,
        default=1.0,
        help="Multiply ejecta velocities by some factor (adjusting density to conserve mass) before writing the model file",
    )
    parser.add_argument("-outputpath", "-o", default=None, help="Path for output model files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create ARTIS format model from grid.dat."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    gridfolderpath = args.gridfolderpath
    if not Path(gridfolderpath, "grid.dat").is_file():
        msg = "grid.dat is required. Run artistools maptogrid"
        raise FileNotFoundError(msg)

    outputpath = Path(f"artismodel_{args.dimensions}d") if args.outputpath is None else Path(args.outputpath)

    outputpath.mkdir(parents=True, exist_ok=True)

    makemodelfromgriddata(
        gridfolderpath=gridfolderpath,
        outputpath=outputpath,
        targetmodeltime_days=args.targetmodeltime_days,
        traj_root=args.trajectoryroot,
        dimensions=args.dimensions,
        scalemass=args.scalemass,
        scalevelocity=args.scalevelocity,
    )


if __name__ == "__main__":
    main()
