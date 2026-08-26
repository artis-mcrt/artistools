"""Plot the binned radiation field estimators and their fitted dilute blackbody parameters."""

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.commands import run_subcommand
from artistools.constants import c_ang_per_s
from artistools.constants import day_to_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.constants import km_to_cm
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_notitle
from artistools.misc import addarg_outputfile
from artistools.misc import addarg_quiet
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.plottools import get_figsize
from artistools.plottools import save_figure
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title


def read_files(modelpath: Path | str, timestep: int | None = None, modelgridindex: int | None = None) -> pl.DataFrame:
    """Read radiation field data from a model folder, possibly with timestep and modelgridindex filters."""
    return at.read_rank_outputfiles(
        modelpath, "radfield_{mpirank:04d}.out", timestep=timestep, modelgridindex=modelgridindex
    )


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
    """Get the dJ/dlambda constant average estimators of each bin."""
    # exclude the global fit parameters and detailed lines with negative "bin_num"
    bindata = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    arr_lambda = c_ang_per_s / bindata["nu_upper"].to_numpy()

    bindata = bindata.with_columns(dlambda=c_ang_per_s * (1 / pl.col("nu_lower") - 1 / pl.col("nu_upper")))

    yvalues = bindata.select(
        pl.when(pl.col("T_R") >= 0).then(pl.col("J") / pl.col("dlambda")).otherwise(0.0)
    ).to_numpy()

    # add the starting point
    arr_lambda = np.insert(arr_lambda, 0, c_ang_per_s / bindata["nu_lower"].item(0))
    yvalues = np.insert(yvalues, 0, 0.0)

    return arr_lambda, yvalues


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
    radfielddata: pl.DataFrame, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[list[float], list[float]]:
    """Return the fitted dilute blackbody (list of lambda, list of j_nu) made up of all bins."""
    arr_lambda: list[float] = []
    j_lambda_fitted: list[float] = []

    radfielddata_subset = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    for row in radfielddata_subset.iter_rows(named=True):
        nu_lower = row["nu_lower"]
        nu_upper = row["nu_upper"]

        if row["W"] >= 0:
            arr_nu_hz_bin = np.linspace(nu_lower, nu_upper, num=200)
            W, T_R = row["W"], row["T_R"]
            assert isinstance(W, float)
            assert isinstance(T_R, float)
            arr_j_nu = j_nu_dbb(arr_nu_hz_bin, W, T_R)

            arr_lambda_bin = c_ang_per_s / arr_nu_hz_bin
            arr_j_lambda_bin = arr_j_nu * arr_nu_hz_bin / arr_lambda_bin

            arr_lambda += arr_lambda_bin.tolist()
        else:
            arr_nu_hz_bin = np.array([nu_lower, nu_upper])
            arr_j_lambda_bin = np.array([0.0, 0.0])

            arr_lambda += [c_ang_per_s / nu for nu in arr_nu_hz_bin]
        j_lambda_fitted += arr_j_lambda_bin.tolist()

    return arr_lambda, j_lambda_fitted


def plot_line_estimators(
    axis: mplax.Axes,
    radfielddata: pl.DataFrame,
    modelgridindex: int | None = None,
    timestep: int | None = None,
    **plotkwargs: t.Any,
) -> float:
    """Plot the Jblue_lu values from the detailed line estimators on a spectrum."""
    ymax = -1

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

    if not radfielddataselected.is_empty():
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
    specfilename: str | Path,
    timestep: int,
    peak_value: float | None = None,
    scale_factor: float | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Plot the ARTIS spectrum."""
    print(f"Plotting {specfilename}")

    specfilename = Path(specfilename)
    if specfilename.is_dir():
        modelpath = specfilename
    elif specfilename.is_file():
        modelpath = Path(specfilename).parent

    dfspectrum = at.spectra.get_spectra(modelpath=modelpath, timestepmin=timestep)[-1].collect()
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
    timestep: int,
    outputfile: Path | str,
    xmin: float,
    xmax: float,
    modelgridindex: int,
    args: argparse.Namespace,
    normalised: bool = False,
) -> bool:
    """Plot a cell at a timestep things like the bin edges, fitted field, and emergent spectrum (from all cells)."""
    radfielddata = read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex)
    if radfielddata.select(pl.len()).item() == 0:
        print(f"No data for timestep {timestep:d} modelgridindex {modelgridindex:d}")
        return False

    modelname = at.get_model_name(modelpath)
    time_days = at.get_timestep_times(modelpath)[timestep]
    print(f"Plotting {modelname} timestep {timestep:d} (t={time_days:.3f}d)")
    T_R = radfielddata.filter(pl.col("bin_num") == -1).select("T_R").item()
    print(f"T_R = {T_R}")

    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=get_figsize(args), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )

    assert isinstance(axis, mplax.Axes)

    ymax = 0.0

    xlist, yvalues = get_fullspecfittedfield(radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep)

    label = r"Dilute blackbody model "
    # label += r'(T$_{\mathrm{R}}$'
    # label += f'= {row["T_R"]} K)')
    axis.plot(xlist, yvalues, label=label, color="purple", linewidth=1.5)
    ymax = float(np.max(yvalues))

    if not args.nobandaverage:
        arr_lambda, yvalues = get_binaverage_field(radfielddata, modelgridindex=modelgridindex, timestep=timestep)
        axis.step(arr_lambda, yvalues, where="pre", label="Band-average field", color="green", linewidth=1.5)
        ymax = np.max(
            [ymax] + [float(yval) for xval, yval in zip(arr_lambda, yvalues, strict=True) if xmin <= xval <= xmax]
        )

    arr_lambda_fitted, j_lambda_fitted = get_fitted_field(
        radfielddata, modelgridindex=modelgridindex, timestep=timestep
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
        specfilename = at.firstexisting("spec.out", folder=modelpath, tryzipped=True)
    except FileNotFoundError:
        print("Could not find spec.out")
        args.nospec = True

    modeldata, modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols="vel_r_mid")

    if not args.nospec:
        plotkwargs: dict[str, t.Any] = {}
        if not normalised:
            # outer velocity
            v_surface = modelmeta["vmax_cmps"]
            r_surface = time_days * day_to_s * v_surface
            r_observer = at.constants.megaparsec_to_cm
            scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
            print(
                "Scaling emergent spectrum flux at 1 Mpc to specific intensity "
                f"at surface (v={v_surface:.3e}, r={r_surface:.3e} {r_observer:.3e}) scale_factor: {scale_factor:.3e}"
            )
            plotkwargs["scale_factor"] = scale_factor
        else:
            plotkwargs["peak_value"] = ymax

        plot_specout(axis, specfilename, timestep, zorder=-1, color="black", alpha=0.6, linewidth=1.0, **plotkwargs)

    if args.showbinedges:
        binedges = get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5, color="red", label="", zorder=-1, alpha=0.4)

    velocity_kmps = (
        modeldata.filter(pl.col("modelgridindex") == modelgridindex).select("vel_r_mid").collect().item() / km_to_cm
    )

    figure_title = f"{modelname} {velocity_kmps:.0f} km/s at {time_days:.0f}d"
    # figure_title += '\ncell {modelgridindex} timestep {timestep}'

    set_plot_title(axis, figure_title, args)

    # axis.annotate(figure_title,
    #               xy=(0.02, 0.96), xycoords='axes fraction',
    #               horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    axis.set_ylabel(r"J$_\lambda$ [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")
    from matplotlib import ticker

    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))
    axis.set_xlim(left=xmin, right=xmax)
    # the parser accepts -ymin and -ymax, thus the axis must take what the user asked for. A radiation
    # field is not negative, thus zero is the default bottom
    axis.set_ylim(bottom=args.ymin if args.ymin is not None else 0.0, top=args.ymax if args.ymax is not None else ymax)

    at.plottools.set_exponent_label(axis)

    set_legend(axis, args, loc="best", handlelength=2, frameon=False, numpoints=1)

    save_figure(fig, outputfile, format="pdf", args=args)
    return True


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=".")

    addarg_timedays(parser, kind="str")

    addarg_timestep(parser, kind="strappend")

    addarg_modelgridindex(parser, kind="append", helptext="Model grid cell to plot, or a range e.g. 3-7")

    parser.add_argument("-velocity", "-v", type=float, default=-1, help="Specify cell by velocity")

    parser.add_argument("--nospec", action="store_true", help="Don't plot the emergent specrum")

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
    addarg_nolegend(parser)
    addarg_show(parser)
    addarg_quiet(parser)

    parser.add_argument("--nobandaverage", action="store_true", help="Suppress the band-average line")

    addarg_figscale(parser, figscaledefault=1.4)

    addarg_outputfile(parser, helptext="Filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radiation field estimators."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    at.set_mpl_style()

    defaultoutputfile = Path("plotradfield_cell{cell:05d}_ts{timestep:03d}.pdf")

    args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

    modelpath = args.modelpath

    pdf_list: list[str] = []
    modelgridindexlist: list[int] = []

    if args.velocity >= 0.0:
        mgi = at.inputmodel.get_mgi_of_velocity_kms(modelpath, args.velocity)
        assert mgi is not None, f"Could not find a cell with velocity {args.velocity:.3f} km/s"
        modelgridindexlist = [mgi]
    elif args.modelgridindex is None:
        modelgridindexlist = [0]
    else:
        modelgridindexlist = at.parse_range_list(args.modelgridindex)

    timesteplast = len(at.get_timestep_times(modelpath)) - 1
    if args.timedays:
        timesteplist = [at.get_timestep_of_timedays(modelpath, args.timedays)]
    elif args.timestep:
        timesteplist = at.parse_range_list(args.timestep, dictvars={"last": timesteplast})
    else:
        print("Using last timestep.")
        timesteplist = [timesteplast]

    for modelgridindex in modelgridindexlist:
        assert modelgridindex is not None
        for timestep in timesteplist:
            outputfile = str(args.outputfile).format(cell=modelgridindex, timestep=timestep)
            if plot_celltimestep(
                modelpath,
                timestep,
                outputfile,
                xmin=args.xmin,
                xmax=args.xmax,
                modelgridindex=modelgridindex,
                args=args,
                normalised=args.normalised,
            ):
                pdf_list.append(outputfile)

    if len(pdf_list) > 1:
        print(pdf_list)
        at.merge_pdf_files(pdf_list)


if __name__ == "__main__":
    run_subcommand("plotradfield")
