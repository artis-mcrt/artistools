"""Merges models of any dimensions. Assumes cylindrical symmetry for 2D models which are getting mapped to 3D."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import math
import re
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at

cl_cmps = 29979245800.0
cl_kmps = 299792.458
day_cgs = 86400


def cell_weights(dim: int, cd1: npt.NDArray[np.float64], cd2: npt.NDArray[np.float64]) -> float:
    # computes the fraction of the volume of cell 1 which is also contained in cell 2
    # dim: cell dimension
    # cd1: cell data 1, cd2: cell data 2 (final grid)
    if dim == 1:
        return (
            np.heaviside(cd2[:, 1] - cd1[:, 0], 1)
            * np.heaviside(cd1[:, 1] - cd2[:, 0], 1)
            * (cd1[:, 1] ** 3 - cd2[:, 0] ** 3)
            / (cd1[:, 1] ** 3 - cd1[:, 0] ** 3)
        )
    if dim == 2:
        return (
            np.heaviside(cd2[:, 1] - cd1[:, 0], 1)
            * np.heaviside(cd1[:, 1] - cd2[:, 0], 1)
            * np.heaviside(cd2[:, 3] - cd1[:, 2], 1)
            * np.heaviside(cd1[:, 3] - cd2[:, 2], 1)
            * (cd1[:, 3] - cd2[:, 2])
            * (cd1[:, 1] ** 2 - cd2[:, 0] ** 2)
            / (cd1[:, 3] - cd1[:, 2])
            / (cd1[:, 1] ** 2 - cd1[:, 0] ** 2)
        )
    if dim == 3:
        return (
            np.heaviside(cd2[:, 1] - cd1[:, 0], 1)
            * np.heaviside(cd1[:, 1] - cd2[:, 0], 1)
            * np.heaviside(cd2[:, 3] - cd1[:, 2], 1)
            * np.heaviside(cd1[:, 3] - cd2[:, 2], 1)
            * np.heaviside(cd2[:, 5] - cd1[:, 4], 1)
            * np.heaviside(cd1[:, 5] - cd2[:, 4], 1)
            * (cd1[:, 1] - cd2[:, 0])
            / (cd1[:, 1] - cd1[:, 0])
            * (cd1[:, 3] - cd2[:, 2])
            / (cd1[:, 3] - cd1[:, 2])
            * (cd1[:, 5] - cd2[:, 4])
            / (cd1[:, 5] - cd1[:, 4])
        )
    return -1.0


def merge_models(
    mdf1: pl.DataFrame, mdf2: pl.DataFrame, final_grid: npt.NDArray[np.float64], dim: int, t_snap_s: float
) -> pl.DataFrame:
    """Map all model quantities here to that (assuming that models 1 and 2 have same dimension)."""
    if dim == 1:
        merged_model = pl.DataFrame({
            "inputcellid": np.arange(1, len(final_grid) + 1),
            "vel_r_max_kmps": final_grid,
        }).lazy()

        volumes_fg = (
            4
            / 3
            * np.pi
            * t_snap_s**3
            * np.array(
                [final_grid[0] ** 3] + [final_grid[i] ** 3 - final_grid[i - 1] ** 3 for i in range(1, len(final_grid))]
            )
        )

        # get minimum and maximum velocities of models and final grid
        v_max_m1 = mdf1.collect()["vel_r_max_kmps"].to_numpy()
        # v_min_m1 = v_max_m1 - v_max_m1[0]
        # cdm1 = np.column_stack((v_min_m1, v_max_m1))
        v_max_m2 = mdf2.collect()["vel_r_max_kmps"].to_numpy()
        # v_min_m2 = v_max_m2 - v_max_m2[0]
        # cdm2 = np.column_stack((v_min_m2, v_max_m2))
        # v_min_fg = final_grid - final_grid[0]
        # cdfg = np.column_stack((v_min_fg, final_grid))
        # weights_m1 = cell_weights(dim, cdm1, np.array([cdfg for i in range(len(cdfg))]))
        # weights_m2 = cell_weights(dim, cdm2, np.array([cdfg for i in range(len(cdfg))]))

        # get cell masses of models
        volumes_m1 = (
            4
            / 3
            * np.pi
            * t_snap_s**3
            * np.array([v_max_m1[0] ** 3] + [v_max_m1[i] ** 3 - v_max_m1[i - 1] ** 3 for i in range(1, len(v_max_m1))])
        )
        masses_m1 = volumes_m1 * 10 ** (mdf1.collect()["logrho"])
        volumes_m2 = (
            4
            / 3
            * np.pi
            * t_snap_s**3
            * np.array([v_max_m2[0] ** 3] + [v_max_m2[i] ** 3 - v_max_m2[i - 1] ** 3 for i in range(1, len(v_max_m2))])
        )
        masses_m2 = volumes_m2 * 10 ** (mdf2.collect()["logrho"])
    elif dim == 2:
        """
        import sys

        sys.exit("Not implemented yet. Abort")
        merged_model = pl.DataFrame({
            "inputcellid": np.arange(1, len(final_grid) + 1),
            "pos_rcyl_mid": final_grid[:, 0],
            "pos_z_mid": final_grid[:, 1],
        }).lazy()

        Delta_h = 2 * min([np.abs(h) for h in final_grid[:, 1]])

        # volumes_fg = Delta_h * np.pi *

        volumes_fg = np.pi * np.array(
            [final_grid[0] ** 3] + [final_grid[i] ** 3 - final_grid[i - 1] ** 3 for i in range(1, len(final_grid))]
        )
        """
    else:
        merged_model = pl.DataFrame({
            "inputcellid": np.arange(1, len(final_grid) + 1),
            "pos_x_min": final_grid[:, 0],
            "pos_y_min": final_grid[:, 1],
            "pos_z_min": final_grid[:, 2],
        }).lazy()

        # get uniform grid dimensions, volume the same for each cell
        Delta_x_fg = min(np.abs(x) for x in final_grid[:, 0] if x != 0.0)
        Delta_y_fg = min(np.abs(y) for y in final_grid[:, 1] if y != 0.0)
        Delta_z_fg = min(np.abs(z) for z in final_grid[:, 2] if z != 0.0)
        """
        cdfg = np.column_stack([
            final_grid[:, 0],
            final_grid[:, 0] + Delta_x_fg,
            final_grid[:, 1],
            final_grid[:, 1] + Delta_y_fg,
            final_grid[:, 2],
            final_grid[:, 2] + Delta_z_fg,
        ])
        """
        volumes_fg = np.array([Delta_x_fg * Delta_y_fg * Delta_z_fg for i in range(len(final_grid))])

        # following part commented out since equal grids are assumed for the moment
        """
        # cell data for models
        x_min_m1 = mdf1.collect()["pos_x_min"].to_numpy()
        sorted_x_min_m1 = sorted(set(x_min_m1))
        # Delta_x_m1 = sorted_x_min_m1[1] - sorted_x_min_m1[0]
        y_min_m1 = mdf1.collect()["pos_y_min"].to_numpy()
        sorted_y_min_m1 = sorted(set(y_min_m1))
        # Delta_y_m1 = sorted_y_min_m1[1] - sorted_y_min_m1[0]
        z_min_m1 = mdf1.collect()["pos_z_min"].to_numpy()
        sorted_z_min_m1 = sorted(set(z_min_m1))
        # Delta_z_m1 = sorted_z_min_m1[1] - sorted_z_min_m1[0]
        # x_max_m1 = x_min_m1 + Delta_x_m1
        # y_max_m1 = y_min_m1 + Delta_y_m1
        # z_max_m1 = z_min_m1 + Delta_z_m1
        # cdm1 = np.column_stack((x_min_m1, x_max_m1, y_min_m1, y_max_m1, z_min_m1, z_max_m1))
        x_min_m2 = mdf2.collect()["pos_x_min"].to_numpy()
        sorted_x_min_m2 = sorted(set(x_min_m2))
        Delta_x_m2 = sorted_x_min_m2[1] - sorted_x_min_m2[0]
        y_min_m2 = mdf2.collect()["pos_y_min"].to_numpy()
        sorted_y_min_m2 = sorted(set(y_min_m2))
        Delta_y_m2 = sorted_y_min_m2[1] - sorted_y_min_m2[0]
        z_min_m2 = mdf2.collect()["pos_z_min"].to_numpy()
        sorted_z_min_m2 = sorted(set(z_min_m2))
        Delta_z_m2 = sorted_z_min_m2[1] - sorted_z_min_m2[0]
        # x_max_m2 = x_min_m2 + Delta_x_m2
        # y_max_m2 = y_min_m2 + Delta_y_m2
        # z_max_m2 = z_min_m2 + Delta_z_m2
        # cdm2 = np.column_stack((x_min_m2, x_max_m2, y_min_m2, y_max_m2, z_min_m2, z_max_m2))
        # weights_m1 = cell_weights(dim, cdm1, cdfg)
        # weights_m2 = cell_weights(dim, cdm2, cdfg)
        """

    masses_m1 = mdf1.collect()["rho"].to_numpy() * volumes_fg
    masses_m2 = mdf2.collect()["rho"].to_numpy() * volumes_fg
    masses_fg = masses_m1 + masses_m2

    # density:
    rho = masses_fg / volumes_fg
    merged_model = merged_model.with_columns(pl.Series("rho", rho))

    # iron group mass fraction
    xfe_m1 = mdf1.collect()["X_Fegroup"].to_numpy()
    xfe_m2 = mdf2.collect()["X_Fegroup"].to_numpy()
    xfe_fg = (xfe_m1 * masses_m1 + xfe_m2 * masses_m2) / masses_fg
    merged_model = merged_model.with_columns(pl.Series("X_Fegroup", xfe_fg))

    # Ye
    ye_m1 = mdf1.collect()["cellYe"].to_numpy()
    ye_m2 = mdf2.collect()["cellYe"].to_numpy()
    ye_fg = (ye_m1 * masses_m1 + ye_m2 * masses_m2) / masses_fg
    merged_model = merged_model.with_columns(pl.Series("Ye", ye_fg))

    # q
    q_m1 = mdf1.collect()["q"].to_numpy()
    q_m2 = mdf2.collect()["q"].to_numpy()
    q_fg = (q_m1 * masses_m1 + q_m2 * masses_m2) / masses_fg
    merged_model = merged_model.with_columns(pl.Series("q", q_fg))

    # isotopic mass fractions
    col_names_1 = mdf1.collect().columns
    col_names_2 = mdf2.collect().columns
    all_isotopes = [X for X in list(set(col_names_1) | set(col_names_2)) if ("X_" in X and bool(re.search(r"\d", X)))]
    for col_name in all_isotopes:
        x_m1 = mdf1.collect()[col_name].to_numpy()
        x_m2 = mdf2.collect()[col_name].to_numpy()
        if col_name not in col_names_1:
            # isotope only in model 2
            x_m2 = mdf2.collect()[col_name].to_numpy()
            x_fg = x_m2 * masses_m2
        elif col_name not in col_names_2:
            # isotope only in model 1
            x_m1 = mdf1.collect()[col_name].to_numpy()
            x_fg = x_m1 * masses_m1
        else:
            # both models have the isotope
            x_m1 = mdf1.collect()[col_name].to_numpy()
            x_m2 = mdf2.collect()[col_name].to_numpy()
            x_fg = (x_m1 * masses_m1 + x_m2 * masses_m2) / masses_fg
        merged_model = merged_model.with_columns(pl.Series(col_name, x_fg))

    # x_col_list = [pl.Series(name, x_data[:, i]) for i, name in enumerate(all_isotopes)]
    # merged_model = merged_model.with_columns(x_col_list)

    return merged_model


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-vmax", required=True, help="Cut models at v = v_max in all coordinates (corners will have larger v!)."
    )

    parser.add_argument("modelpaths", type=str, default=["."], nargs="+", help=("Paths to models to be merged"))

    parser.add_argument("-dim", required=True, help="Dimension of the final merged model (e.g. 25, 25x50, 50x50x50)")

    parser.add_argument("-tsnap", required=True, help="Model snapshot time in days.")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    t_snap = float(args.tsnap)
    v_max = float(args.vmax)

    assert len(args.modelpaths) > 1, "Please provide at least two model files. Abort."

    # determine final model grid dimensions
    pattern = r"^\d+(x\d+)*$"
    assert re.match(pattern, args.dim) is not None, "Wrong format for final model dimensions. Abort."

    dims_list = [int(coord_dim) for coord_dim in args.dim.split("x")]
    N_cells_final = math.prod(dims_list)
    dim = len(dims_list)
    assert dim in set(1, 2, 3), "Model neither 1-, 2- or 3-dimensional. Abort."

    # set coordinate information for modelmeta and construct the final grid
    if dim == 1:
        coord_info_list = ["npts_model"]
        # final_grid: maximum radial velocity in km/s
        Delta_v_kmps = v_max * cl_kmps / dims_list[0]
        final_grid = np.arange(Delta_v_kmps, v_max * cl_kmps + Delta_v_kmps, Delta_v_kmps)
    elif dim == 2:
        coord_info_list = ["ncoordgridrcyl", "ncoordgridz"]
        cell_width = v_max / dims_list[0] * cl_cmps * t_snap * day_cgs
        # final_grid: cell midpoint coordinates in cm at t_snap
        pos_r_mid = np.array([cell_width * nr for nr in range(dims_list[0])]) + 0.5 * cell_width
        r_cellmid = np.array([pos_r_mid[n_r] for n_r in range(dims_list[0]) for n_z in range(dims_list[1])])
        pos_z_mid = (
            np.array([
                -v_max * cl_cmps * t_snap * day_cgs + 2.0 * v_max * cl_cmps * t_snap * day_cgs / dims_list[1] * nz
                for nz in range(dims_list[1])
            ])
            + 0.5 * cell_width
        )
        z_cellmid = np.array([pos_z_mid[n_z] for n_r in range(dims_list[0]) for n_z in range(dims_list[1])])

        final_grid = np.column_stack((r_cellmid, z_cellmid))
    else:
        # dim is now 3
        coord_info_list = ["ncoordgridx", "ncoordgridy", "ncoordgridz"]
        # final_grid: minimum coordinate positions of the cells at snapshot time
        cell_width = v_max / dims_list[0] * 2 * cl_cmps * t_snap * day_cgs  # no symmetry here!!
        pos_x_min = np.array([-v_max * cl_cmps * t_snap * day_cgs + nx * cell_width for nx in range(dims_list[0])])
        x_cellmin = np.array([
            pos_x_min[n_x] for n_z in range(dims_list[2]) for n_y in range(dims_list[1]) for n_x in range(dims_list[0])
        ])
        pos_y_min = np.array([-v_max * cl_cmps * t_snap * day_cgs + ny * cell_width for ny in range(dims_list[1])])
        y_cellmin = np.array([
            pos_y_min[n_y] for n_z in range(dims_list[2]) for n_y in range(dims_list[1]) for n_x in range(dims_list[0])
        ])
        pos_z_min = np.array([-v_max * cl_cmps * t_snap * day_cgs + nz * cell_width for nz in range(dims_list[2])])
        z_cellmin = np.array([
            pos_z_min[n_z] for n_z in range(dims_list[2]) for n_y in range(dims_list[1]) for n_x in range(dims_list[0])
        ])

        final_grid = np.column_stack((x_cellmin, y_cellmin, z_cellmin))

    model_list = []

    for mi, model_path in enumerate(args.modelpaths):
        mdf = at.inputmodel.get_modeldata(modelpath=Path(model_path))  # mdf: model data frame
        model_dim = mdf[1]["dimensions"]
        assert math.isclose(t_snap, mdf[1]["t_model_init_days"], rel_tol=1e-10), (
            f"Model {args.modelpath} has a different snapshot time. Abort."
        )
        if mi == 0:
            # first model: adapt dimensions only if necessary
            if model_dim > dim:
                model_list.append(at.inputmodel.dimension_reduce_model(mdf[0], dim))
            elif model_dim < dim:
                model_list.append(at.inputmodel.dimension_increase_model(mdf[0], dim, modelmeta=mdf[1]))
        # other models: merge with the previous one
        elif model_dim > dim:
            model_list.append(
                merge_models(
                    model_list[mi - 1],
                    at.inputmodel.dimension_reduce_model(mdf[0], dim),
                    final_grid,
                    dim,
                    t_snap * day_cgs,
                )
            )
        elif model_dim < dim:
            model_list.append(
                merge_models(
                    model_list[mi - 1],
                    at.inputmodel.dimension_increase_model(mdf[0], dim),
                    final_grid,
                    dim,
                    t_snap * day_cgs,
                )
            )
        else:
            model_list.append(merge_models(model_list[mi - 1], mdf[0], final_grid, dim, t_snap * day_cgs))

    element_abbrevs_list = [at.get_elsymbol(Z) for Z in range(1, 101)]  # Keep full list as in your code
    element_abbrevs_list_titled = [abbrev.title() for abbrev in element_abbrevs_list]

    nuclide_columns = [col for col in model_list[-1].collect_schema().names() if col.startswith("X_")][1:]

    """
    abunds_all = {
        nuclide: model_list[-1].select(pl.col(nuclide).cast(pl.Float32)).collect().to_series().to_numpy()
        for nuclide in nuclide_columns
    }
    """
    dfabundances = pl.DataFrame({"a": np.zeros(N_cells_final)}).lazy()

    for element in element_abbrevs_list_titled:
        dfabundances = dfabundances.with_columns(pl.lit(0).alias(element))
        pattern = re.escape(element) + r"\d"
        for nuclide in nuclide_columns:
            if re.search(pattern, nuclide):
                # now add upp the columns
                s = model_list[-1].select(pl.col(nuclide).cast(pl.Float32)).collect().to_series()
                dfabundances = dfabundances.with_columns((pl.col(element) + pl.lit(s)).alias(element))

    dfabundances_pd = dfabundances.collect().to_pandas()
    dfabundances_pd = dfabundances_pd.rename(columns={"a": "inputcellid"})
    dfabundances_pd["inputcellid"] = range(1, N_cells_final + 1)
    for column in dfabundances_pd.columns[1:]:
        dfabundances_pd = dfabundances_pd.rename(columns={column: f"X_{column}"})

    at.inputmodel.save_initelemabundances(dfelabundances=dfabundances_pd, outpath=Path("../"))

    # create new model.txt based on last element from model_list
    modelmeta = {"dimensions": dim, "t_model_init_days": t_snap, "vmax_cmps": v_max * cl_cmps}
    modelmeta.update(zip(coord_info_list, dims_list, strict=False))

    dfmodel_out_pd = model_list[-1].collect().to_pandas()
    dfmodel_out_pd["inputcellid"] = range(1, N_cells_final + 1)
    at.inputmodel.save_modeldata(dfmodel=dfmodel_out_pd, modelmeta=modelmeta, outpath=Path())
    print("Merged model saved!")


if __name__ == "__main__":
    main()
