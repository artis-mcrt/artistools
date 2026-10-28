"""Plot the binned radiation field estimators and their fitted dilute blackbody parameters."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import ticker

from artistools.constants import c_ang_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.constants import megaparsec_to_cm
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import scan_modeldata
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import exit_with_error
from artistools.misc import firstexisting
from artistools.misc import format_frame_path
from artistools.misc import format_range_list
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_times
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import print_saved
from artistools.misc import resolve_frameset_paths
from artistools.plottools import addarg_cellvelocity
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.radfield.core import get_binaverage_field
from artistools.radfield.core import get_binedges
from artistools.radfield.core import get_fitted_field
from artistools.radfield.core import get_fullspecfittedfield
from artistools.radfield.core import get_multibin_radfield_first_timestep
from artistools.radfield.core import read_radfield
from artistools.radfield.core import select_radfield_subset
from artistools.spectra import scan_spectra


def plot_line_estimators(
    axis: mplax.Axes,
    radfielddata: pl.DataFrame,
    modelgridindex: int | None = None,
    timestep: int | None = None,
    **plotkwargs: t.Any,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Plot the Jblue_lu values from the detailed line estimators on a spectrum, and return the drawn points.

    The points are the wavelengths in Angstroms and J_lambda.
    """
    # the detailed line estimators have bin_num < -1. Cell zero and timestep zero are falsy, so these filters must
    # test against None rather than truthiness
    radfielddataselected = select_radfield_subset(
        radfielddata, pl.col("bin_num") < -1, modelgridindex, timestep
    ).select("nu_upper", "J_nu_avg")
    if radfielddataselected.is_empty():
        print("No line estimators to plot")
        return np.array([]), np.array([])

    radfielddataselected = radfielddataselected.with_columns(
        lambda_angstroms=c_ang_per_s / pl.col("nu_upper"),
        Jb_lambda=pl.col("J_nu_avg") * (pl.col("nu_upper") ** 2) / c_ang_per_s,
    )

    arr_lambda = radfielddataselected["lambda_angstroms"].to_numpy()
    arr_jb_lambda = radfielddataselected["Jb_lambda"].to_numpy()
    axis.scatter(arr_lambda, arr_jb_lambda, label="Line estimators", s=0.2, **plotkwargs)
    return arr_lambda, arr_jb_lambda


def plot_specout(
    axis: mplax.Axes,
    modelpath: Path,
    timestep: int,
    peak_value: float | None = None,
    scale_factor: float | None = None,
    **plotkwargs: t.Any,
) -> tuple[str, pl.Series, pl.Series]:
    """Plot the ARTIS spectrum, and return the label, the wavelengths, and the drawn values.

    The caller gives the model path, because a run on a cluster writes spec.out to a subfolder of the
    model folder. The parent folder of that file is then the run folder and not the model folder.
    """
    dfspectrum = scan_spectra(modelpath=modelpath, timestepmin=timestep, directionbins=[-1])[-1].collect()
    label = "Emergent spectrum"
    if scale_factor is not None:
        label += " (scaled)"
        dfspectrum = dfspectrum.with_columns(pl.col("f_lambda") * scale_factor)

    if peak_value is not None:
        label += " (normalised)"
        dfspectrum = dfspectrum.with_columns(pl.col("f_lambda") / pl.col("f_lambda").max() * peak_value)

    axis.plot(dfspectrum["lambda_angstroms"], dfspectrum["f_lambda"], label=label, **plotkwargs)
    return label, dfspectrum["lambda_angstroms"], dfspectrum["f_lambda"]


def plot_celltimestep(
    modelpath: Path | str,
    radfielddata_cell: pl.DataFrame,
    timestep: int,
    outputfile: Path | str,
    xmin: float,
    xmax: float,
    modelgridindex: int,
    velocity_kmps: float,
    modelmeta: dict[str, t.Any],
    args: argparse.Namespace,
    normalised: bool = False,
    isframe: bool = False,
) -> bool:
    """Plot a cell at a timestep things like the bin edges, fitted field, and emergent spectrum (from all cells).

    radfielddata_cell holds the radiation field data of the cell at every timestep. velocity_kmps is the
    mid-point radial velocity of the cell, which the title gives.

    A plot that the merge takes in is one part of the product, thus --show and --open leave it alone.
    merge_pdf_files also deletes such a file, thus an application that opened it would hold nothing.
    """
    radfielddata = radfielddata_cell.filter(pl.col("timestep") == timestep)
    if radfielddata.is_empty():
        print(f"No data for timestep {timestep:d} modelgridindex {modelgridindex:d}")
        return False

    modelname = get_model_name(modelpath)
    time_days = get_timestep_times(modelpath)[timestep]
    print(f"Plotting {get_model_logname(modelpath)} timestep {timestep:d} (t={time_days:.3f}d)")
    T_R = radfielddata.filter(pl.col("bin_num") == -1).select("T_R").item()
    print(f"T_R = {T_R}")

    fig, axesgrid = make_frame_figure(args)
    axis = axesgrid[0][0]

    assert isinstance(axis, mplax.Axes)

    xlist, yvalues = get_fullspecfittedfield(radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep)

    label = r"Dilute blackbody model "
    axis.plot(xlist, yvalues, label=label, color="purple", linewidth=1.5)
    drawnseries: list[tuple[str, npt.ArrayLike, npt.ArrayLike]] = [(label.strip(), xlist, yvalues)]
    ymax = float(np.max(yvalues))

    if not args.nobandaverage:
        arr_lambda, yvalues = get_binaverage_field(radfielddata, modelgridindex=modelgridindex, timestep=timestep)
        axis.step(arr_lambda, yvalues, where="pre", label="Band-average field", color="green", linewidth=1.5)
        drawnseries.append(("Band-average field", arr_lambda, yvalues))
        ymax = np.max(
            [ymax] + [float(yval) for xval, yval in zip(arr_lambda, yvalues, strict=True) if xmin <= xval <= xmax]
        )

    firstmultibin = get_multibin_radfield_first_timestep(modelpath)
    firstnlteradfieldtimestep = None if firstmultibin is None else firstmultibin[0]
    if firstmultibin is not None:
        print(f"The artisoptions.h of the model gives {firstmultibin[1]} = {firstmultibin[0]}")
    else:
        print(
            "The artisoptions.h of the model gives no MULTIBIN_RADFIELD_FIRST_TIMESTEP and no older name of it, thus"
            " the radiation field model takes the fits of the bins. ARTIS takes the full-spectrum fit before that"
            " timestep"
        )
    arr_lambda_fitted, j_lambda_fitted = get_fitted_field(
        radfielddata,
        modelgridindex=modelgridindex,
        timestep=timestep,
        usebinfits=firstnlteradfieldtimestep is None or timestep >= firstnlteradfieldtimestep,
    )
    ymax = max(
        [ymax] + [yval for xval, yval in zip(arr_lambda_fitted, j_lambda_fitted, strict=True) if xmin <= xval <= xmax]
    )

    axis.plot(arr_lambda_fitted, j_lambda_fitted, label="Radiation field model", alpha=0.8, color="blue", linewidth=1.5)
    drawnseries.append(("Radiation field model", arr_lambda_fitted, j_lambda_fitted))

    arr_lambda_lines, jb_lambda_lines = plot_line_estimators(
        axis, radfielddata, modelgridindex=modelgridindex, timestep=timestep, zorder=-2, color="red"
    )
    drawnseries.append(("Line estimators", arr_lambda_lines, jb_lambda_lines))

    ymax = args.ymax if args.ymax is not None else max(ymax, float(np.max(jb_lambda_lines, initial=0.0)))
    try:
        print(f"Plotting {firstexisting('spec.out', folder=modelpath, tryzipped=True)}")
    except FileNotFoundError:
        print("Could not find spec.out")
        args.nospec = True

    if not args.nospec:
        plotkwargs: dict[str, t.Any] = {}
        if not normalised:
            # outer velocity
            v_surface = modelmeta["vmax_cmps"]
            r_surface = time_days * day_to_s * v_surface
            r_observer = megaparsec_to_cm
            scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
            print(
                "Scaling emergent spectrum flux at 1 Mpc to specific intensity "
                f"at surface (v={v_surface:.3e}, r={r_surface:.3e} {r_observer:.3e}) scale_factor: {scale_factor:.3e}"
            )
            plotkwargs["scale_factor"] = scale_factor
        else:
            plotkwargs["peak_value"] = ymax

        drawnseries.append(
            plot_specout(
                axis, Path(modelpath), timestep, zorder=-1, color="black", alpha=0.6, linewidth=1.0, **plotkwargs
            )
        )

    if args.showbinedges:
        binedges = get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5, color="red", label="", zorder=-1, alpha=0.4)

    figure_title = f"{modelname} {velocity_kmps:.0f} km/s at {time_days:.0f}d"

    set_plot_title(axis, figure_title, args)

    axis.set_xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    axis.set_ylabel(r"J$_\lambda$ [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")

    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))
    axis.set_xlim(left=xmin, right=xmax)
    # the parser accepts -ymin and -ymax, thus the axis must take what the user asked for. A radiation
    # field is not negative, thus zero is the default bottom
    axis.set_ylim(bottom=args.ymin if args.ymin is not None else 0.0, top=args.ymax if args.ymax is not None else ymax)

    set_exponent_label(axis)

    set_legend(axis, args, loc="best", handlelength=2, frameon=False, numpoints=1)

    if getattr(args, "print_data", False) or getattr(args, "write_data", False):
        output_data(args, {"": get_drawn_series_table(drawnseries, xmin, xmax)}, outputfile)
    # the suffix of the file name sets the format, e.g. .pdf or .png
    save_figure(fig, outputfile, args=args, isframe=isframe)
    return True


def get_drawn_series_table(
    drawnseries: Sequence[tuple[str, npt.ArrayLike, npt.ArrayLike]], xmin: float, xmax: float
) -> pl.DataFrame:
    """Return the points of each series of a plot of one cell from xmin to xmax, with one row for each point.

    Each series has its own wavelengths, thus a column gives the series of each row. The band-average field gives the
    points of its step line. The value of each point applies from the point before it to that point.
    """
    # a series with no point, e.g. a cell with no line estimators, needs the schema for the types of its columns
    schema = {
        "series": pl.String,
        "lambda_angstroms": pl.Float64,
        "J_lambda_erg_per_s_per_cm2_per_angstrom": pl.Float64,
    }
    dfseries = [
        pl.DataFrame(
            {
                "series": label,
                "lambda_angstroms": np.asarray(arr_lambda, dtype=np.float64),
                "J_lambda_erg_per_s_per_cm2_per_angstrom": np.asarray(j_lambda, dtype=np.float64),
            },
            schema=schema,
        )
        for label, arr_lambda, j_lambda in drawnseries
    ]
    if not dfseries:
        return pl.DataFrame(schema=schema)
    return pl.concat(dfseries).filter(pl.col("lambda_angstroms").is_between(xmin, xmax))


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser)

    addarg_modelgridindex(parser, helptext="Model grid cell to plot, or a range e.g. 3-7")

    addarg_cellvelocity(parser)

    parser.add_argument("--nospec", action="store_true", help="Don't plot the emergent spectrum")

    parser.add_argument("--showbinedges", action="store_true", help="Plot vertical lines at the bin edges")

    addarg_axislimits(
        parser,
        xmindefault=1000,
        xmaxdefault=20000,
        xminhelp="Plot range: minimum wavelength in Angstroms",
        xmaxhelp="Plot range: maximum wavelength in Angstroms",
        wavelength_aliases=True,
    )

    parser.add_argument("--normalised", action="store_true", help="Normalise the spectra to their peak values")

    addarg_notitle(parser)
    addarg_legend(parser)
    addarg_show(parser)
    addarg_darkmode(parser)
    addarg_verbose(parser)

    parser.add_argument("--nobandaverage", action="store_true", help="Suppress the band-average line")

    addarg_figscale(parser)

    addarg_output(
        parser, kind="file", helptext="Filename for the plot file. The suffix sets the format, e.g. .pdf or .png"
    )
    addarg_data_output(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radiation field estimators."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath = args.modelpath

    pdf_list: list[str] = []
    modelgridindexlist: list[int] = []

    if args.velocity_kmps is not None:
        modelgridindexlist = [get_mgi_of_velocity_kms(modelpath, args.velocity_kmps)]
    else:
        modelgridindexlist = args.modelgridindex or [0]

    # read_radfield parses a rank file on each call, thus one read of all the cells serves each cell and timestep
    radfielddata_allcells = read_radfield(modelpath, modelgridindex=modelgridindexlist)

    # a model whose last run stopped early has no radiation field data at the last timestep of the time grid. Thus
    # "last" is the last timestep that holds data
    timesteplast = (
        int(radfielddata_allcells.select(pl.col("timestep").max()).item())
        if not radfielddata_allcells.is_empty()
        else len(get_timestep_times(modelpath)) - 1
    )
    if args.timedays:
        timesteplist = [get_timestep_of_timedays(modelpath, args.timedays)]
    elif args.timestep is not None:
        timesteplist = parse_range_list(str(args.timestep), dictvars={"last": timesteplast})
    else:
        print(f"Using the last timestep with radiation field data: {timesteplast}")
        timesteplist = [timesteplast]

    # a merge makes one pdf of every plot, thus each plot is a part of the product and not the product
    frameset = resolve_frameset_paths(
        args.outputfile,
        framecount=len(modelgridindexlist) * len(timesteplist),
        framename="plotradfield_cell{cell:05d}_ts{timestep:03d}.pdf",
        combines=len(modelgridindexlist) * len(timesteplist) > 1,
    )

    # one query of the model gives the velocity of every cell. A query in each frame ran the derivation
    # of the model columns again for each cell and timestep
    dfmodel, modelmeta = scan_modeldata(modelpath)
    dfcellvelocities = (
        add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
        .filter(pl.col("modelgridindex").is_in(modelgridindexlist))
        .select("modelgridindex", "vel_r_mid")
        .collect()
    )
    velocity_kmps_of_cell = dict(
        zip(dfcellvelocities["modelgridindex"], dfcellvelocities["vel_r_mid"] / km_to_cm, strict=True)
    )

    for modelgridindex in modelgridindexlist:
        assert modelgridindex is not None
        radfielddata_cell = radfielddata_allcells.filter(pl.col("modelgridindex") == modelgridindex)
        for timestep in timesteplist:
            outputfile = format_frame_path(frameset.frametemplate, cell=modelgridindex, timestep=timestep)
            if plot_celltimestep(
                modelpath,
                radfielddata_cell,
                timestep,
                outputfile,
                xmin=args.xmin,
                xmax=args.xmax,
                modelgridindex=modelgridindex,
                # a cell that is not in the model has no radiation field data, thus no plot reads the NaN
                velocity_kmps=velocity_kmps_of_cell.get(modelgridindex, math.nan),
                modelmeta=modelmeta,
                args=args,
                normalised=args.normalised,
                isframe=frameset.combines,
            ):
                pdf_list.append(outputfile)

    if not pdf_list:
        exit_with_error(
            f"No radiation field data for the cells {format_range_list(modelgridindexlist)} at the timesteps"
            f" {format_range_list(timesteplist)}",
            f"Give a cell and a timestep with data. The last timestep with data is {timesteplast}",
        )

    # a run that holds data for one cell or one timestep alone makes one plot, and combine_frames
    # takes that plot for the product, because no plot of a merging run opened on its own
    productpath = frameset.finish(pdf_list, args)
    if len(pdf_list) == 1 and productpath is not None and Path(productpath) != Path(pdf_list[0]):
        # the table of the plot must stay beside the figure, which combine_frames moved to the name of the product
        frametable = Path(pdf_list[0]).with_suffix(".txt")
        if getattr(args, "write_data", False) and frametable.is_file():
            import shutil

            producttable = Path(productpath).with_suffix(".txt")
            shutil.move(str(frametable), str(producttable))
            print_saved(producttable)
