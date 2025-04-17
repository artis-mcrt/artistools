#!/usr/bin/env python3

import argparse
import math
import typing as t
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at

H = 6.6260755e-27  # Planck constant [erg s]
KB = 1.38064852e-16  # Boltzmann constant [erg/K]


@lru_cache(maxsize=4)
def read_files(modelpath: Path | str, timestep: int | None = None, modelgridindex: int | None = None) -> pd.DataFrame:
    """Read radiation field data from a list of file paths into a pandas DataFrame."""
    radfielddata_allfiles: list[pd.DataFrame] = []
    modelpath = Path(modelpath)

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    for folderpath in at.get_runfolders(modelpath, timestep=timestep):
        for mpirank in mpiranklist:
            radfieldfilename = f"radfield_{mpirank:04d}.out"
            radfieldfilepath = Path(folderpath, radfieldfilename)
            radfieldfilepath = at.firstexisting(radfieldfilename, folder=folderpath, tryzipped=True)

            if modelgridindex is not None:
                filesize = Path(radfieldfilepath).stat().st_size / 1024 / 1024
                print(f"Reading {Path(radfieldfilepath).relative_to(modelpath.parent)} ({filesize:.2f} MiB)")

            radfielddata_thisfile = pd.read_csv(radfieldfilepath, sep=r"\s+")
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)

            if timestep is not None:
                radfielddata_thisfile = radfielddata_thisfile.query("timestep==@timestep")

            if modelgridindex is not None:
                radfielddata_thisfile = radfielddata_thisfile.query("modelgridindex==@modelgridindex")

            if not radfielddata_thisfile.empty:
                if timestep is not None and modelgridindex is not None:
                    return radfielddata_thisfile
                radfielddata_allfiles.append(radfielddata_thisfile)

    return pd.concat(radfielddata_allfiles, ignore_index=True)


def select_bin(
    radfielddata: pd.DataFrame,
    nu: float | None = None,
    lambda_angstroms: float | None = None,
    modelgridindex: int | None = None,
    timestep: int | None = None,
) -> tuple[int, float, float]:
    assert nu is None or lambda_angstroms is None

    if lambda_angstroms is not None:
        nu = 2.99792458e18 / lambda_angstroms
    else:
        assert nu is not None
        lambda_angstroms = 2.99792458e18 / nu

    dfselected = radfielddata.query(
        ("modelgridindex == @modelgridindex and " if modelgridindex else "")
        + ("timestep == @timestep and " if timestep else "")
        + "nu_lower <= @nu and nu_upper >= @nu and bin_num > -1"
    )

    assert not dfselected.empty
    return dfselected.iloc[0].bin_num, dfselected.iloc[0].nu_lower, dfselected.iloc[0].nu_upper


def get_binaverage_field(
    radfielddata: pd.DataFrame, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the dJ/dlambda constant average estimators of each bin."""
    # exclude the global fit parameters and detailed lines with negative "bin_num"
    bindata = radfielddata.copy().query(
        "bin_num >= 0"
        + (" & modelgridindex==@modelgridindex" if modelgridindex else "")
        + (" & timestep==@timestep" if timestep else "")
    )

    arr_lambda = 2.99792458e18 / bindata["nu_upper"].to_numpy()

    bindata["dlambda"] = bindata.apply(lambda row: 2.99792458e18 * (1 / row["nu_lower"] - 1 / row["nu_upper"]), axis=1)

    yvalues = bindata.apply(
        lambda row: (
            row["J"] / row["dlambda"] if (not math.isnan(row["J"] / row["dlambda"]) and row["T_R"] >= 0) else 0.0
        ),
        axis=1,
    ).to_numpy()

    # add the starting point
    arr_lambda = np.insert(arr_lambda, 0, 2.99792458e18 / bindata["nu_lower"].iloc[0])
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
    if W > 0.0:
        try:
            return [W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(H * nu_hz / T / KB)) for nu_hz in arr_nu_hz]
        except OverflowError:
            print(f"WARNING: overflow error W {W}, T {T} (Did this happen in ARTIS too?)")

    return [0.0 for _ in arr_nu_hz]


def get_fullspecfittedfield(
    radfielddata: pd.DataFrame, xmin: float, xmax: float, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    row = (
        radfielddata.query(
            "bin_num == -1"
            + (" & modelgridindex==@modelgridindex" if modelgridindex else "")
            + (" & timestep==@timestep" if timestep else "")
        )
        .copy()
        .iloc[0]
    )
    nu_lower = 2.99792458e18 / xmin
    nu_upper = 2.99792458e18 / xmax
    arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
    arr_j_nu = j_nu_dbb(arr_nu_hz, row["W"], row["T_R"])

    arr_lambda = 2.99792458e18 / arr_nu_hz
    arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

    return arr_lambda, arr_j_lambda


def get_fitted_field(
    radfielddata: pd.DataFrame,
    modelgridindex: int | None = None,
    timestep: int | None = None,
    print_bins: bool = False,
    lambdamin: float | None = None,
    lambdamax: float | None = None,
) -> tuple[list[float], list[float]]:
    """Return the fitted dilute blackbody (list of lambda, list of j_nu) made up of all bins."""
    arr_lambda: list[float] = []
    j_lambda_fitted: list[float] = []

    radfielddata_subset = radfielddata.copy().query(
        "bin_num >= 0"
        + (" & modelgridindex==@modelgridindex" if modelgridindex else "")
        + (" & timestep==@timestep" if timestep else "")
    )

    if lambdamax is not None:
        nu_min = 2.99792458e18 / lambdamax

    if lambdamin is not None:
        nu_max = 2.99792458e18 / lambdamin

    for _, row in radfielddata_subset.iterrows():
        nu_lower = row["nu_lower"]
        nu_upper = row["nu_upper"]

        if lambdamax is not None:
            if nu_upper > nu_max:
                continue
            nu_lower = max(nu_lower, nu_min)
        if lambdamin is not None:
            if nu_lower < nu_min:
                continue
            nu_upper = min(nu_upper, nu_max)

        if row["W"] >= 0:
            arr_nu_hz_bin = np.linspace(nu_lower, nu_upper, num=200)
            arr_j_nu = j_nu_dbb(arr_nu_hz_bin, row["W"], row["T_R"])

            arr_lambda_bin = 2.99792458e18 / arr_nu_hz_bin
            arr_j_lambda_bin = arr_j_nu * arr_nu_hz_bin / arr_lambda_bin

            arr_lambda += list(arr_lambda_bin)
        else:
            arr_nu_hz_bin = np.array([nu_lower, nu_upper])
            arr_j_lambda_bin = np.array([0.0, 0.0])

            arr_lambda += [2.99792458e18 / nu for nu in arr_nu_hz_bin]
        j_lambda_fitted += list(arr_j_lambda_bin)

        lambda_lower = 2.99792458e18 / row["nu_upper"]
        lambda_upper = 2.99792458e18 / row["nu_lower"]
        if (
            print_bins
            and (lambdamax is None or lambda_lower < lambdamax)
            and (lambdamin is None or lambda_upper > lambdamin)
        ):
            print(
                f"Bin lambda_lower {lambda_lower:.1f} W {row['W']:.1e} "
                f"contribs {row['ncontrib']} J_nu_avg {row['J_nu_avg']:.1e}"
            )

    return arr_lambda, j_lambda_fitted


def plot_line_estimators(
    axis: mplax.Axes,
    radfielddata: pd.DataFrame,
    modelgridindex: int | None = None,
    timestep: int | None = None,
    **plotkwargs: t.Any,
) -> float:
    """Plot the Jblue_lu values from the detailed line estimators on a spectrum."""
    ymax = -1

    radfielddataselected = radfielddata.query(
        "bin_num < -1"
        + (" & modelgridindex==@modelgridindex" if modelgridindex else "")
        + (" & timestep==@timestep" if timestep else "")
    )[["nu_upper", "J_nu_avg"]]

    radfielddataselected["lambda_angstroms"] = 2.99792458e18 / radfielddataselected["nu_upper"]
    radfielddataselected["Jb_lambda"] = (
        radfielddataselected["J_nu_avg"] * (radfielddataselected["nu_upper"] ** 2) / 2.99792458e18
    )

    ymax = radfielddataselected["Jb_lambda"].max()
    assert isinstance(ymax, float)

    if not radfielddataselected.empty:
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

    dfspectrum = at.spectra.get_spectrum(modelpath=modelpath, timestepmin=timestep)[-1].to_pandas(
        use_pyarrow_extension_array=True
    )
    label = "Emergent spectrum"
    if scale_factor is not None:
        label += " (scaled)"
        dfspectrum["f_lambda"] *= scale_factor

    if peak_value is not None:
        label += " (normalised)"
        dfspectrum["f_lambda"] = dfspectrum["f_lambda"] / dfspectrum["f_lambda"].max() * peak_value

    dfspectrum.plot(x="lambda_angstroms", y="f_lambda", ax=axis, label=label, **plotkwargs)


def get_binedges(radfielddata: pd.DataFrame) -> list[float]:
    return [2.99792458e18 / radfielddata["nu_lower"].iloc[1], *list(2.99792458e18 / radfielddata["nu_upper"][1:])]


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
    if radfielddata.empty:
        print(f"No data for timestep {timestep:d} modelgridindex {modelgridindex:d}")
        return False

    modelname = at.get_model_name(modelpath)
    time_days = at.get_timestep_times(modelpath)[timestep]
    print(f"Plotting {modelname} timestep {timestep:d} (t={time_days:.3f}d)")
    T_R = radfielddata.query("bin_num == -1").iloc[0].T_R
    print(f"T_R = {T_R}")

    fig, axis = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        figsize=(
            args.figscale * at.get_config()["figwidth"],
            args.figscale * at.get_config()["figwidth"] * (0.25 + 0.4),
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
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

    ymax = args.ymax if args.ymax >= 0 else max(ymax, ymax3)
    try:
        specfilename = at.firstexisting("spec.out", folder=modelpath, tryzipped=True)
    except FileNotFoundError:
        print("Could not find spec.out")
        args.nospec = True

    if not args.nospec:
        plotkwargs: dict[str, t.Any] = {}
        if not normalised:
            _, modelmeta = at.inputmodel.get_modeldata(modelpath, getheadersonly=True)
            # outer velocity
            v_surface = modelmeta["vmax_cmps"]
            r_surface = time_days * 864000 * v_surface
            r_observer = 3.0857e24
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

    modeldata, _ = at.inputmodel.get_modeldata(modelpath, derived_cols="vel_r_mid")
    velocity_kmps = (
        modeldata.filter(pl.col("modelgridindex") == modelgridindex).select("vel_r_mid").collect().item() / 1e5
    )

    figure_title = f"{modelname} {velocity_kmps:.0f} km/s at {time_days:.0f}d"
    # figure_title += '\ncell {modelgridindex} timestep {timestep}'

    if not args.notitle:
        axis.set_title(figure_title, fontsize=11)

    # axis.annotate(figure_title,
    #               xy=(0.02, 0.96), xycoords='axes fraction',
    #               horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    axis.set_ylabel(r"J$_\lambda$ [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")
    from matplotlib import ticker

    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))
    axis.set_xlim(left=xmin, right=xmax)
    axis.set_ylim(bottom=0.0, top=ymax)

    axis.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(axis.get_ylabel()))

    axis.legend(loc="best", handlelength=2, frameon=False, numpoints=1, fontsize=9)

    print(f"Saving to {outputfile}")
    fig.savefig(str(outputfile), format="pdf")
    plt.close()
    return True


def plot_bin_fitted_field_evolution(
    axis: mplax.Axes,
    radfielddata: pd.DataFrame,
    nu_line: float,
    modelgridindex: int,
    **plotkwargs: dict[str, t.Any | None],
) -> None:
    bin_num, _nu_lower, _nu_upper = select_bin(radfielddata, nu=nu_line, modelgridindex=modelgridindex)
    # print(f"Selected bin_num {bin_num} to get a binned radiation field estimator")
    radfielddataselected: t.Any = radfielddata.query(
        f"bin_num == {bin_num} and modelgridindex == @modelgridindex and nu_lower <= @nu_line and nu_upper >= @nu_line"
    ).copy()

    radfielddataselected["Jb_nu_at_line"] = radfielddataselected.apply(
        lambda x: j_nu_dbb([nu_line], x.W, x.T_R)[0], axis=1
    )

    radfielddataselected = radfielddataselected.eval(
        "Jb_lambda_at_line = Jb_nu_at_line * (@nu_line ** 2) / 2.99792458e18"
    )
    lambda_angstroms = 2.99792458e18 / nu_line

    radfielddataselected.plot(
        x="timestep",
        y="Jb_lambda_at_line",
        ax=axis,
        label=f"Fitted field from bin at {lambda_angstroms:.1f} Å",
        **plotkwargs,
    )


def plot_global_fitted_field_evolution(
    axis: mplax.Axes,
    radfielddata: pd.DataFrame,
    nu_line: float,
    modelgridindex: int,  # noqa: ARG001
    **plotkwargs: t.Any,
) -> None:
    radfielddataselected = radfielddata.query("bin_num == -1 and modelgridindex == @modelgridindex").copy()

    radfielddataselected["J_nu_fullspec_at_line"] = radfielddataselected.apply(
        lambda x: j_nu_dbb([nu_line], x.W, x.T_R)[0], axis=1
    )

    radfielddataselected["J_lambda_fullspec_at_line"] = (
        radfielddataselected["J_nu_fullspec_at_line"] * (nu_line**2) / 2.99792458e18
    )
    lambda_angstroms = 2.99792458e18 / nu_line

    radfielddataselected.plot(
        x="timestep",
        y="J_lambda_fullspec_at_line",
        ax=axis,
        label=f"Full-spec fitted field at {lambda_angstroms:.1f} Å",
        **plotkwargs,
    )


def plot_line_estimator_evolution(
    axis: mplax.Axes,
    radfielddata: pd.DataFrame,
    bin_num: int,
    modelgridindex: int | None = None,
    timestep_min: int | None = None,
    timestep_max: int | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Plot the Jblue_lu values over time for a detailed line estimators."""
    radfielddataselected: t.Any = radfielddata.query(
        "bin_num == @bin_num"
        + (" & modelgridindex == @modelgridindex" if modelgridindex else "")
        + (" & timestep >= @timestep_min" if timestep_min else "")
        + (" & timestep <= @timestep_max" if timestep_max else "")
    )[["timestep", "nu_upper", "J_nu_avg"]]
    assert isinstance(radfielddataselected, pd.DataFrame)
    radfielddataselected = radfielddataselected.eval("lambda_angstroms = 2.99792458e18 / nu_upper")
    assert isinstance(radfielddataselected, pd.DataFrame)
    radfielddataselected = radfielddataselected.eval("Jb_lambda = J_nu_avg * (nu_upper ** 2) / 2.99792458e18")

    axis.plot(
        radfielddataselected["timestep"],
        radfielddataselected["Jb_lambda"],
        label=f"Jb_lu bin_num {bin_num}",
        **plotkwargs,
    )


def plot_timeevolution(
    modelpath: Path | str, outputfile: Path | str, modelgridindex: int, args: argparse.Namespace
) -> None:
    """Plot a estimator evolution over time for a cell. This is not well tested and should be checked."""
    print(f"Plotting time evolution of cell {modelgridindex:d}")

    radfielddata = read_files(modelpath, modelgridindex=modelgridindex)
    radfielddataselected = radfielddata.query("modelgridindex == @modelgridindex")

    nlinesplotted = 200
    fig, axes = plt.subplots(
        nlinesplotted,
        1,
        sharex=True,
        figsize=(
            args.figscale * at.get_config()["figwidth"],
            args.figscale * at.get_config()["figwidth"] * (0.25 + nlinesplotted * 0.35),
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if isinstance(axes, mplax.Axes):
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    timestep = at.get_timestep_of_timedays(modelpath, 330)
    time_days = at.get_timestep_time(modelpath, timestep)

    dftopestimators = radfielddataselected.query("timestep==@timestep and bin_num < -1").copy()
    dftopestimators["lambda_angstroms"] = 2.99792458e18 / dftopestimators["nu_upper"]
    dftopestimators["Jb_lambda"] = dftopestimators["J_nu_avg"] * (dftopestimators["nu_upper"] ** 2) / 2.99792458e18
    dftopestimators = dftopestimators.sort_values("Jb_lambda", ascending=False, inplace=False).iloc[:nlinesplotted]
    print(f"Top estimators at timestep {timestep} t={time_days:.1f}")
    print(dftopestimators)

    for ax, bin_num_estimator, nu_line in zip(
        axes, dftopestimators.bin_num.to_numpy(), dftopestimators.nu_upper.to_numpy(), strict=False
    ):
        lambda_angstroms = 2.99792458e18 / nu_line
        print(f"Selected line estimator with bin_num {bin_num_estimator}, lambda={lambda_angstroms:.1f}")
        plot_line_estimator_evolution(ax, radfielddataselected, bin_num_estimator, modelgridindex=modelgridindex)

        plot_bin_fitted_field_evolution(ax, radfielddata, nu_line, modelgridindex=modelgridindex)

        plot_global_fitted_field_evolution(ax, radfielddata, nu_line, modelgridindex=modelgridindex)
        ax.annotate(
            rf"$\lambda$={lambda_angstroms:.1f} Å in cell {modelgridindex:d}\n",
            xy=(0.02, 0.96),
            xycoords="axes fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=10,
        )

        ax.set_ylabel(r"J$_\lambda$ [erg/s/cm$^2$/$\mathrm{{\AA}}$]")
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1)

    axes[-1].set_xlabel(r"Timestep")
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    # axis.set_xlim(left=xmin, right=xmax)
    # axis.set_ylim(bottom=0.0, top=ymax)

    print(f"Saving to {outputfile}")
    fig.savefig(str(outputfile), format="pdf")
    plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-modelpath", default=".", type=Path, help="Path to ARTIS folder")

    parser.add_argument(
        "-xaxis", "-x", default="lambda", choices=["lambda", "timestep"], help="Horizontal axis variable."
    )

    parser.add_argument("-timedays", "-time", "-t", help="Time in days to plot")

    parser.add_argument("-timestep", "-ts", action="append", help="Timestep number to plot")

    parser.add_argument("-modelgridindex", "-cell", action="append", help="Modelgridindex to plot")

    parser.add_argument("-velocity", "-v", type=float, default=-1, help="Specify cell by velocity")

    parser.add_argument("--nospec", action="store_true", help="Don't plot the emergent specrum")

    parser.add_argument("--showbinedges", action="store_true", help="Plot vertical lines at the bin edges")

    parser.add_argument("-xmin", type=int, default=1000, help="Plot range: minimum wavelength in Angstroms")

    parser.add_argument("-xmax", type=int, default=20000, help="Plot range: maximum wavelength in Angstroms")

    parser.add_argument("-ymax", type=int, default=-1, help="Plot range: maximum J_nu")

    parser.add_argument("--normalised", action="store_true", help="Normalise the spectra to their peak values")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument("--nobandaverage", action="store_true", help="Suppress the band-average line")

    parser.add_argument(
        "-figscale", type=float, default=1.0, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("-o", action="store", dest="outputfile", type=Path, help="Filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radiation field estimators."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    at.set_mpl_style()

    defaultoutputfile = (
        Path("plotradfield_cell{modelgridindex:03d}_ts{timestep:03d}.pdf")
        if args.xaxis == "lambda"
        else Path("plotradfield_cell{modelgridindex:03d}_evolution.pdf")
    )

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif args.outputfile.is_dir():
        args.outputfile /= defaultoutputfile

    modelpath = args.modelpath

    pdf_list = []
    modelgridindexlist = []

    if args.velocity >= 0.0:
        modelgridindexlist = [at.inputmodel.get_mgi_of_velocity_kms(modelpath, args.velocity)]
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
        if args.xaxis == "lambda":
            for timestep in timesteplist:
                outputfile = str(args.outputfile).format(modelgridindex=modelgridindex, timestep=timestep)
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
        elif args.xaxis == "timestep":
            outputfile = str(args.outputfile).format(modelgridindex=modelgridindex)
            assert modelgridindex is not None
            plot_timeevolution(modelpath, outputfile, modelgridindex, args)
        else:
            print("Unknown plot type {args.plot}")
            raise AssertionError

    if len(pdf_list) > 1:
        print(pdf_list)
        at.merge_pdf_files(pdf_list)


if __name__ == "__main__":
    main()
