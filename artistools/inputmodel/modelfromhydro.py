# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS input model from the gridded output of a neutron star merger hydrodynamics simulation."""

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s as CLIGHT
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.constants import Msun_to_g as MSUN


def read_ejectasnapshot(
    pathtosnapshot: str | Path, usecols: list[str] | None, downsamplefactor: int | None
) -> pl.DataFrame:
    """Read an SPH ejecta snapshot, optionally keeping only every downsamplefactor-th particle."""
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
    dfsnapshot = at.read_wsv(
        Path(pathtosnapshot) / "ejectasnapshot.dat" if Path(pathtosnapshot).is_dir() else pathtosnapshot,
        has_header=False,
        new_columns=column_names,
        columns=usecols,
        schema_overrides={col: pl.Int64 if col == "id" else pl.Float64 for col in (usecols or column_names)},
    )

    if downsamplefactor is not None and downsamplefactor > 1:
        dfsnapshot = dfsnapshot.sample(len(dfsnapshot) // downsamplefactor)

    return dfsnapshot


def get_merger_time_geomunits(pathtogriddata: Path) -> float:
    """Return the merger time in geometric units read from tmerger.txt."""
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
    """Return the simulation end time and the merger time, both in geometric units."""
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
) -> tuple[pl.DataFrame, float, float, float, dict[str, t.Any]]:
    """Read grid.dat and return the cell data, the model and merger times, vmax, and the model metadata.

    When targetmodeltime_days is given, the grid is expanded homologously to that time.
    """
    griddatfilepath = Path(pathtogriddata) / "grid.dat"

    # Get simulation time for ejecta snapshot
    simulation_end_time_geomunits, mergertime_geomunits = get_snapshot_time_geomunits(pathtogriddata)

    griddata = at.read_wsv(griddatfilepath, comment_prefix="#", skip_rows=3).rename(
        {
            "gridindex": "inputcellid",
            "pos_x": "pos_x_min",
            "pos_y": "pos_y_min",
            "pos_z": "pos_z_min",
            "posx": "pos_x_min",  # for compatibility with fortran maptogrid script
            "posy": "pos_y_min",
            "posz": "pos_z_min",
        },
        strict=False,
    )

    factor_position = 1.478  # in km
    griddata = griddata.with_columns(
        # griddata in geom units
        cs.by_name("rho", "cellYe", "Q", require_all=False).fill_null(0.0)
    ).with_columns(
        cs.starts_with("pos_") * factor_position * km_to_cm,
        pl.col("rho") * 6.176e17,  # convert to g/cm³
    )

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
    corner_vmax = vmax * math.sqrt(3)
    print(
        f"vmax {vmax:.2e} cm/s ({vmax / CLIGHT:.2f} * c) per component "
        f"real corner vmax {corner_vmax:.2e} cm/s ({corner_vmax / CLIGHT:.2f} * c)"
    )

    if targetmodeltime_days is not None:
        griddata, modelmeta = at.inputmodel.scale_model_to_time(
            targetmodeltime_days=targetmodeltime_days, t_model_days=t_model_days, dfmodel=griddata
        )
        t_model_days = targetmodeltime_days
        min_pos_x = griddata["pos_x_min"].min()
        assert isinstance(min_pos_x, int | float)
        xmax = -min_pos_x

    ncoordgridx = round(len(griddata) ** (1.0 / 3.0))
    assert ncoordgridx**3 == len(griddata)
    wid_init = 2 * xmax / ncoordgridx
    print(f"Grid model is {ncoordgridx} x {ncoordgridx} x {ncoordgridx} = {len(griddata)} cells")
    griddata = griddata.with_columns(mass_g=pl.col("rho") * wid_init**3)

    max_tracercount = griddata["tracercount"].max()
    assert isinstance(max_tracercount, int | float)
    print(f"Max tracers in a cell {max_tracercount}")

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


def add_mass_to_center(
    griddata: pl.DataFrame,
    t_model_in_days: float,
    vmax: float,  # ruff:ignore[unused-function-argument]
    args: argparse.Namespace,  # ruff:ignore[unused-function-argument]
) -> pl.DataFrame:
    """Fill the low-velocity hole at the grid centre with the mass profile of Just et al. (2021) Fig. 16."""
    print(griddata)

    # Just (2021) Fig. 16 top left panel
    vel_hole = [0, 0.02, 0.05, 0.07, 0.09, 0.095, 0.1]
    mass_hole = [3e-4, 3e-4, 2e-4, 1e-4, 2e-5, 1e-5, 1e-9]
    mass_integrated = np.trapezoid(y=mass_hole, x=vel_hole)  # Msun

    # # Just (2021) Fig. 16 4th down, left panel
    # vel_hole = [0, 0.02, 0.05, 0.1, 0.15, 0.16]
    # mass_hole = [4e-3, 2e-3, 1e-3, 1e-4, 6e-6, 1e-9]
    # mass_integrated = np.trapezoid(y=mass_hole, x=vel_hole)  # Msun

    v_outer_hole = 0.1 * CLIGHT  # cm/s
    pos_outer_hole = v_outer_hole * t_model_in_days * (24.0 * 3600)  # cm
    vol_hole = 4 / 3 * np.pi * pos_outer_hole**3  # cm^3
    density_hole = (mass_integrated * MSUN) / vol_hole  # g / cm^3
    print(density_hole)

    # cells with velocity below 0.1 c get the hole density added and a Ye floor of 0.4
    inhole = (
        (pl.col("pos_x_min") ** 2 + pl.col("pos_y_min") ** 2 + pl.col("pos_z_min") ** 2).sqrt()
        / (t_model_in_days * (24.0 * 3600))
        / CLIGHT
    ) < 0.1

    showcols = ["inputcellid", "pos_x_min", "pos_y_min", "pos_z_min", "rho"]
    print("Inner empty cells")
    print(griddata.filter(inhole).select(showcols))

    griddata = griddata.with_columns(
        rho=pl.when(inhole).then(pl.col("rho") + density_hole).otherwise(pl.col("rho")),
        cellYe=pl.when(inhole).then(pl.max_horizontal(pl.col("cellYe"), pl.lit(0.4))).otherwise(pl.col("cellYe")),
    )

    print(griddata.filter(inhole).select(showcols))

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
    """Write an ARTIS model from grid.dat, taking abundances from the trajectories under traj_root if given."""
    if args is None:
        args = argparse.Namespace()
    dfmodel, t_model_days, t_mergertime_s, vmax, modelmeta = at.inputmodel.modelfromhydro.read_griddat_file(
        pathtogriddata=gridfolderpath, targetmodeltime_days=targetmodeltime_days
    )

    if getattr(args, "fillcentralhole", False):
        dfmodel = at.inputmodel.modelfromhydro.add_mass_to_center(dfmodel, t_model_days, vmax, args)

    if getattr(args, "getcellopacityfromYe", False):
        at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, dfmodel)

    dfgridcontributions = (
        at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(gridfolderpath)
        if Path(gridfolderpath, "gridcontributions.txt").is_file()
        else None
    )

    dfmodel = dfmodel.sort("inputcellid")
    assert dfmodel.schema["inputcellid"].is_integer()
    dfmodel = dfmodel.with_columns(pl.col("inputcellid").cast(pl.Int32))
    if scalemass != 1.0:
        origmass_msun = float(dfmodel["mass_g"].sum()) / MSUN
        dfmodel = dfmodel.with_columns(cs.by_name("rho", "mass_g", require_all=False) * scalemass)
        newmass_msun = float(dfmodel["mass_g"].sum()) / MSUN
        operationmsg = f"densities are scaled by factor of {scalemass} to increase total mass from {origmass_msun:.2e} to {newmass_msun:.2e} Msun"
        print(operationmsg)
        modelmeta["headercommentlines"].append(operationmsg)

    if scalevelocity != 1.0:
        dfmodel = dfmodel.with_columns(
            cs.starts_with("pos_", "vel_") * scalevelocity,
            cs.by_name("rho", "mass_g", require_all=False) * (scalevelocity**-3),
        )
        vmax_cmps_old = modelmeta["vmax_cmps"]
        for key in modelmeta:
            if key == "vmax_cmps" or key.startswith("wid_init_"):
                modelmeta[key] *= scalevelocity
        operationmsg = f"velocities are scaled by a factor of {scalevelocity} (with density scaled by 1/f^3 to conserve mass). vmax/c changed from {vmax_cmps_old / CLIGHT:.2f} to {modelmeta['vmax_cmps'] / CLIGHT:.2f}"
        print(operationmsg)
        modelmeta["headercommentlines"].append(operationmsg)

    if traj_root is not None:
        print(f"Nuclear network abundances from {traj_root} will be used")
        modelmeta["headercommentlines"].append(f"trajfolder: {Path(traj_root).resolve().parts[-1]}")
        t_model_days_incpremerger = t_model_days + (t_mergertime_s / day_to_s)
        assert dfgridcontributions is not None, (
            "gridcontributions.txt is required to set abundances from trajectories. Run artistools maptogrid"
        )
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
    """Add arguments to an argparse parser object."""
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
    at.add_outputpath_arg(parser, default=None, helptext="Path for output model files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create ARTIS format model from grid.dat."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

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
