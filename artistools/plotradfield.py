"""Plot the binned radiation field estimators and their fitted dilute blackbody parameters."""

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import c_ang_per_s
from artistools.constants import day_to_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.constants import km_to_cm
from artistools.constants import megaparsec_to_cm
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_axislimits
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
from artistools.misc import get_artis_option
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import read_rank_outputfiles
from artistools.misc import resolve_frameset_paths
from artistools.misc.cliutils import format_range_list
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.spectra import get_spectra


def read_radfield(modelpath: Path | str, modelgridindex: int | Sequence[int] | None = None) -> pl.DataFrame:
    """Read radiation field data from a model folder, possibly with a modelgridindex filter."""
    return read_rank_outputfiles(modelpath, "radfield_{mpirank:04d}.out", modelgridindex=modelgridindex)


def select_radfield_subset(
    radfielddata: pl.DataFrame, binfilter: pl.Expr, modelgridindex: int | None, timestep: int | None
) -> pl.DataFrame:
    """Filter radfield rows by a bin_num condition and optionally by modelgridindex and timestep."""
    subset = radfielddata.filter(binfilter)
    if modelgridindex is not None:
        subset = subset.filter(pl.col("modelgridindex") == modelgridindex)
    if timestep is not None:
        subset = subset.filter(pl.col("timestep") == timestep)
    return subset


def get_binaverage_field(
    radfielddata: pl.DataFrame, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the dJ/dlambda constant average estimators of each bin.

    The estimator J of a bin does not depend on the fit of the bin, thus a bin with no fit (T_R < 0) still gives
    its J.
    """
    # exclude the global fit parameters and detailed lines with negative "bin_num"
    bindata = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    arr_lambda = np.array(get_binedges(bindata))

    yvalues = (
        bindata
        .with_columns(dlambda=c_ang_per_s * (1 / pl.col("nu_lower") - 1 / pl.col("nu_upper")))
        .select(pl.col("J") / pl.col("dlambda"))
        .to_series()
        .to_numpy()
    )

    # the first bin edge is a starting point with no field
    return arr_lambda, np.insert(yvalues, 0, 0.0)


def j_nu_dbb(arr_nu_hz: Sequence[float] | npt.NDArray[np.floating], W: float, T: float) -> list[float]:
    """Calculate the spectral energy density of a dilute blackbody radiation field.

    Parameters
    ----------
    arr_nu_hz : list
        A list of frequencies (in Hz) at which to calculate the spectral energy density.
    W : float
        The dilution factor of the blackbody radiation field.
    T : float
        The temperature of the blackbody radiation field (in Kelvin).

    Returns
    -------
    list
        A list of spectral energy density values (in CGS units) corresponding to the input frequencies.

    """
    if W <= 0.0:
        return [0.0 for _ in arr_nu_hz]

    # hnu/kT above this overflows math.expm1, and the Wien tail there is far below any plotted value, so those
    # frequencies contribute zero. Catching OverflowError around the whole comprehension instead would discard
    # every frequency's value, not just the ones that overflowed
    max_exponent = math.log(sys.float_info.max)

    def j_nu(nu_hz: float) -> float:
        exponent = h_erg_s * nu_hz / T / K_B_erg_per_K
        if exponent >= max_exponent:
            return 0.0
        return W * 1.4745007e-47 * pow(nu_hz, 3) / math.expm1(exponent)

    # iterate Python floats, since math.expm1 on numpy scalars is much slower
    return [j_nu(nu_hz) for nu_hz in np.asarray(arr_nu_hz, dtype=float).tolist()]


def get_fullspecfittedfield(
    radfielddata: pl.DataFrame, xmin: float, xmax: float, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the wavelengths and J_lambda of the full-spectrum dilute blackbody fit for one cell and timestep."""
    radfielddata = select_radfield_subset(radfielddata, pl.col("bin_num") == -1, modelgridindex, timestep)
    W = radfielddata.item(0, "W")
    assert isinstance(W, float)
    T_R = radfielddata.item(0, "T_R")
    assert isinstance(T_R, float)
    nu_lower = c_ang_per_s / xmin
    nu_upper = c_ang_per_s / xmax
    arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500, dtype=np.float64)
    arr_j_nu = j_nu_dbb(arr_nu_hz, W, T_R)

    arr_lambda = c_ang_per_s / arr_nu_hz
    arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

    return arr_lambda, arr_j_lambda


def get_fitted_field(
    radfielddata: pl.DataFrame, modelgridindex: int | None = None, timestep: int | None = None, usebinfits: bool = True
) -> tuple[list[float], list[float]]:
    """Return the radiation field model of ARTIS (list of lambda, list of J_lambda) made up of all bins.

    ARTIS radfield() takes the dilute blackbody fit of the bin. A bin with a negative W has no fit, and ARTIS
    then takes the full-spectrum dilute blackbody fit, which is bin -1 of the file. usebinfits=False gives every
    bin the full-spectrum fit, as ARTIS does before FIRST_NLTE_RADFIELD_TIMESTEP.
    """
    arr_lambda: list[float] = []
    j_lambda_fitted: list[float] = []

    fullspecfit = select_radfield_subset(radfielddata, pl.col("bin_num") == -1, modelgridindex, timestep)
    W_fullspec, T_R_fullspec = float(fullspecfit.item(0, "W")), float(fullspecfit.item(0, "T_R"))

    radfielddata_subset = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    for nu_lower, nu_upper, W_bin, T_R_bin in radfielddata_subset.select(
        "nu_lower", "nu_upper", "W", "T_R"
    ).iter_rows():
        W, T_R = (float(W_bin), float(T_R_bin)) if usebinfits and W_bin >= 0 else (W_fullspec, T_R_fullspec)
        arr_nu_hz_bin = np.linspace(nu_lower, nu_upper, num=200)
        arr_j_nu = j_nu_dbb(arr_nu_hz_bin, W, T_R)

        arr_lambda_bin = c_ang_per_s / arr_nu_hz_bin
        arr_j_lambda_bin = arr_j_nu * arr_nu_hz_bin / arr_lambda_bin

        arr_lambda += arr_lambda_bin.tolist()
        j_lambda_fitted += arr_j_lambda_bin.tolist()

    return arr_lambda, j_lambda_fitted


def get_first_nlte_radfield_timestep(modelpath: Path | str) -> int | None:
    """Return FIRST_NLTE_RADFIELD_TIMESTEP of artis/artisoptions.h in the folder of the run, or None.

    ARTIS uses the fits of the bins from this timestep. Before it, ARTIS uses the full-spectrum fit for every bin.
    """
    value = get_artis_option(modelpath, "FIRST_NLTE_RADFIELD_TIMESTEP")
    return int(value) if value is not None and value.isdigit() else None


def plot_line_estimators(
    axis: mplax.Axes,
    radfielddata: pl.DataFrame,
    modelgridindex: int | None = None,
    timestep: int | None = None,
    **plotkwargs: t.Any,
) -> float:
    """Plot the Jblue_lu values from the detailed line estimators on a spectrum."""
    # the detailed line estimators have bin_num < -1. Cell zero and timestep zero are falsy, so these filters must
    # test against None rather than truthiness
    radfielddataselected = select_radfield_subset(
        radfielddata, pl.col("bin_num") < -1, modelgridindex, timestep
    ).select("nu_upper", "J_nu_avg")
    if radfielddataselected.is_empty():
        print("No line estimators to plot")
        return 0.0

    radfielddataselected = radfielddataselected.with_columns(
        lambda_angstroms=c_ang_per_s / pl.col("nu_upper"),
        Jb_lambda=pl.col("J_nu_avg") * (pl.col("nu_upper") ** 2) / c_ang_per_s,
    )

    ymax = radfielddataselected["Jb_lambda"].max()
    assert isinstance(ymax, float)

    axis.scatter(
        radfielddataselected["lambda_angstroms"],
        radfielddataselected["Jb_lambda"],
        label="Line estimators",
        s=0.2,
        **plotkwargs,
    )
    return ymax


def plot_specout(
    axis: mplax.Axes,
    modelpath: Path,
    timestep: int,
    peak_value: float | None = None,
    scale_factor: float | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Plot the ARTIS spectrum.

    The caller gives the model path, because a run on a cluster writes spec.out to a subfolder of the
    model. The parent folder of that file is then the run folder and not the model.
    """
    dfspectrum = get_spectra(modelpath=modelpath, timestepmin=timestep, directionbins=[-1])[-1].collect()
    label = "Emergent spectrum"
    if scale_factor is not None:
        label += " (scaled)"
        dfspectrum = dfspectrum.with_columns(pl.col("f_lambda") * scale_factor)

    if peak_value is not None:
        label += " (normalised)"
        dfspectrum = dfspectrum.with_columns(pl.col("f_lambda") / pl.col("f_lambda").max() * peak_value)

    axis.plot(dfspectrum["lambda_angstroms"], dfspectrum["f_lambda"], label=label, **plotkwargs)


def get_binedges(radfielddata: pl.DataFrame) -> list[float]:
    """Return the radiation field bin boundaries as wavelengths [Angstroms]."""
    radfielddata = radfielddata.filter(pl.col("bin_num") >= 0)
    return [c_ang_per_s / radfielddata["nu_lower"].item(0), *list(c_ang_per_s / radfielddata["nu_upper"])]


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
    ymax = float(np.max(yvalues))

    if not args.nobandaverage:
        arr_lambda, yvalues = get_binaverage_field(radfielddata, modelgridindex=modelgridindex, timestep=timestep)
        axis.step(arr_lambda, yvalues, where="pre", label="Band-average field", color="green", linewidth=1.5)
        ymax = np.max(
            [ymax] + [float(yval) for xval, yval in zip(arr_lambda, yvalues, strict=True) if xmin <= xval <= xmax]
        )

    firstnlteradfieldtimestep = get_first_nlte_radfield_timestep(modelpath)
    if firstnlteradfieldtimestep is None:
        print(
            "The run holds no FIRST_NLTE_RADFIELD_TIMESTEP in artis/artisoptions.h, thus the radiation field model"
            " takes the fits of the bins. ARTIS takes the full-spectrum fit before that timestep"
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

    ymax3 = plot_line_estimators(
        axis, radfielddata, modelgridindex=modelgridindex, timestep=timestep, zorder=-2, color="red"
    )

    ymax = args.ymax if args.ymax is not None else max(ymax, ymax3)
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

        plot_specout(axis, Path(modelpath), timestep, zorder=-1, color="black", alpha=0.6, linewidth=1.0, **plotkwargs)

    if args.showbinedges:
        binedges = get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5, color="red", label="", zorder=-1, alpha=0.4)

    figure_title = f"{modelname} {velocity_kmps:.0f} km/s at {time_days:.0f}d"

    set_plot_title(axis, figure_title, args)

    axis.set_xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    axis.set_ylabel(r"J$_\lambda$ [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")
    from matplotlib import ticker

    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))
    axis.set_xlim(left=xmin, right=xmax)
    # the parser accepts -ymin and -ymax, thus the axis must take what the user asked for. A radiation
    # field is not negative, thus zero is the default bottom
    axis.set_ylim(bottom=args.ymin if args.ymin is not None else 0.0, top=args.ymax if args.ymax is not None else ymax)

    set_exponent_label(axis)

    set_legend(axis, args, loc="best", handlelength=2, frameon=False, numpoints=1)

    # the suffix of the file name sets the format, e.g. .pdf or .png
    save_figure(fig, outputfile, args=args, isframe=isframe)
    return True


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser)

    addarg_modelgridindex(parser, helptext="Model grid cell to plot, or a range e.g. 3-7")

    parser.add_argument("-velocity", "-v", type=float, default=-1, help="Specify cell by velocity")

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
    addarg_verbose(parser)

    parser.add_argument("--nobandaverage", action="store_true", help="Suppress the band-average line")

    addarg_figscale(parser)

    addarg_output(
        parser, kind="file", helptext="Filename for the plot file. The suffix sets the format, e.g. .pdf or .png"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radiation field estimators."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath = args.modelpath

    pdf_list: list[str] = []
    modelgridindexlist: list[int] = []

    if args.velocity >= 0.0:
        mgi = get_mgi_of_velocity_kms(modelpath, args.velocity)
        assert mgi is not None, f"Could not find a cell with velocity {args.velocity:.3f} km/s"
        modelgridindexlist = [mgi]
    else:
        modelgridindexlist = args.modelgridindex or [0]

    # read_radfield parses a rank file on each call, thus one read of all the cells serves each cell and timestep
    radfielddata_allcells = read_radfield(modelpath, modelgridindex=modelgridindexlist)

    # a run that stopped early holds no radiation field data at the last timestep of the time grid. Thus "last"
    # is the last timestep that holds data
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
    dfmodel, modelmeta = get_modeldata(modelpath)
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
            f"no radiation field data for the cells {format_range_list(modelgridindexlist)} at the timesteps"
            f" {format_range_list(timesteplist)}",
            f"Give a cell and a timestep with data. The last timestep with data is {timesteplast}",
        )

    # a run that holds data for one cell or one timestep alone makes one plot, and combine_frames
    # takes that plot for the product, because no plot of a merging run opened on its own
    frameset.finish(pdf_list, args)
