"""Artistools - spectra related functions."""

import argparse
import contextlib
import math
import os
import re
import typing as t
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at

fluxcontributiontuple = namedtuple(
    "fluxcontributiontuple", "fluxcontrib linelabel array_flambda_emission array_flambda_absorption color"
)
megaparsec_to_cm = 3.0856e24


def timeshift_fluxscale_co56law(scaletoreftime: float | None, spectime: float) -> float:
    if scaletoreftime is not None:
        # Co56 decay flux scaling
        assert spectime > 150
        return math.exp(float(spectime) / 113.7) / math.exp(scaletoreftime / 113.7)

    return 1.0


def get_exspec_bins(
    modelpath: str | Path | None = None,
    mnubins: int | None = None,
    nu_min_r: float | None = None,
    nu_max_r: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the wavelength bins for the emergent spectrum."""
    if modelpath is not None:
        dfspec = read_spec(modelpath, printwarningsonly=True)
        if mnubins is None:
            mnubins = dfspec.height

        nu_centre_min = dfspec.item(0, 0)
        nu_centre_max = dfspec.item(dfspec.height - 1, 0)

        # This is an exact solution for dlognu since we're assuming the bin centre spacing matches the bin edge spacing
        # but it's close enough for our purposes and avoids the difficulty of finding the exact solution (lots more algebra)
        dlognu = math.log(dfspec.item(1, 0) / dfspec.item(0, 0))

        if nu_min_r is None:
            nu_min_r = nu_centre_min / (1 + 0.5 * dlognu)

        if nu_max_r is None:
            nu_max_r = nu_centre_max * (1 + 0.5 * dlognu)

    assert nu_min_r is not None
    assert nu_max_r is not None
    assert mnubins is not None

    c_ang_s = 2.99792458e18

    dlognu = (math.log(nu_max_r) - math.log(nu_min_r)) / mnubins

    bins_nu_lower = np.array([math.exp(math.log(nu_min_r) + (m * (dlognu))) for m in range(mnubins)])
    # bins_nu_upper = np.array([math.exp(math.log(nu_min_r) + ((m + 1) * (dlognu))) for m in range(mnubins)])
    bins_nu_upper = bins_nu_lower * math.exp(dlognu)
    bins_nu_centre = 0.5 * (bins_nu_lower + bins_nu_upper)

    # np.flip is used to get an ascending wavelength array from an ascending nu array
    lambda_bin_edges = np.append(c_ang_s / np.flip(bins_nu_upper), c_ang_s / bins_nu_lower[0])
    lambda_bin_centres = c_ang_s / np.flip(bins_nu_centre)
    delta_lambdas = np.flip(c_ang_s / bins_nu_lower - c_ang_s / bins_nu_upper)

    return lambda_bin_edges, lambda_bin_centres, delta_lambdas


def stackspectra(
    spectra_and_factors: list[tuple[np.ndarray[t.Any, np.dtype[np.float64]], float]],
) -> np.ndarray[t.Any, np.dtype[np.float64]]:
    """Add spectra using weighting factors, i.e., specout[nu] = spec1[nu] * factor1 + spec2[nu] * factor2 + ...

    spectra_and_factors should be a list of tuples: spectra[], factor.
    """
    factor_sum = sum(factor for _, factor in spectra_and_factors)

    stackedspectrum = np.zeros_like(spectra_and_factors[0][0], dtype=float)
    for spectrum, factor in spectra_and_factors:
        stackedspectrum += spectrum * factor / factor_sum

    return stackedspectrum


def get_spectrum_at_time(
    modelpath: Path,
    timestep: int,
    time: float,
    args: argparse.Namespace | None,
    dirbin: int = -1,
    average_over_phi: bool | None = None,
    average_over_theta: bool | None = None,
) -> pd.DataFrame:
    if dirbin >= 0:
        if args is not None and args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
            return get_vspecpol_spectrum(modelpath, time, dirbin, args)
        assert average_over_phi is not None
        assert average_over_theta is not None
    else:
        average_over_phi = False
        average_over_theta = False

    return get_spectrum(
        modelpath=modelpath,
        directionbins=[dirbin],
        timestepmin=timestep,
        timestepmax=timestep,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
    )[dirbin]


def get_from_packets(
    modelpath: Path,
    timelowdays: float,
    timehighdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: None | float | np.ndarray = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    maxpacketfiles: int | None = None,
    getpacketcount: bool = False,
    directionbins: t.Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    nu_column: str = "nu_rf",
    fnufilterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
    nprocs_read_dfpackets: tuple[int, pl.DataFrame] | None = None,
) -> dict[int, pd.DataFrame]:
    """Get a spectrum dataframe using the packets files as input."""
    if directionbins is None:
        directionbins = [-1]

    assert use_time in {"arrival", "emission", "escape"}

    if use_time == "escape":
        modeldata, _ = at.inputmodel.get_modeldata(modelpath)
        vmax_beta = modeldata.iloc[-1].vel_r_max_kmps * 299792.458
        escapesurfacegamma = math.sqrt(1 - vmax_beta**2)
    else:
        escapesurfacegamma = None

    lambda_bin_edges: np.ndarray
    if delta_lambda is not None:
        lambda_bin_edges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        lambda_bin_centres = 0.5 * (lambda_bin_edges[:-1] + lambda_bin_edges[1:])  # bin centres
    else:
        lambda_bin_edges, lambda_bin_centres, delta_lambda = get_exspec_bins(modelpath=modelpath)
        lambda_min = lambda_bin_centres[0]
        lambda_max = lambda_bin_centres[-1]

    nu_min = 2.99792458e18 / lambda_max
    nu_max = 2.99792458e18 / lambda_min

    timelow = timelowdays * 86400.0
    timehigh = timehighdays * 86400.0

    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()
    ndirbins = at.get_viewingdirectionbincount()

    nprocs_read, dfpackets = nprocs_read_dfpackets or at.packets.get_packets_pl(
        modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if use_time == "arrival":
        dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(float(timelowdays), float(timehighdays)))
    elif use_time == "escape":
        assert escapesurfacegamma is not None
        dfpackets = dfpackets.filter(
            pl.col("escape_time").is_between(timelow / escapesurfacegamma, timehigh / escapesurfacegamma)
        )
    elif use_time == "emission":
        mean_correction = float(
            dfpackets.select((pl.col("em_time") - pl.col("t_arrive_d") * 86400.0).mean())
            .lazy()
            .collect()
            .to_numpy()[0][0]
        )

        em_time_low = float(timelowdays) * 86400.0 + mean_correction
        em_time_high = float(timehighdays) * 86400.0 + mean_correction
        dfpackets = dfpackets.filter(pl.col("em_time").is_between(em_time_low, em_time_high))

    dfpackets = dfpackets.filter(pl.col(nu_column).is_between(float(nu_min), float(nu_max)))

    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")

    encol = "e_cmf" if use_time == "escape" else "e_rf"
    getcols = {nu_column, encol}
    if directionbins != [-1]:
        if average_over_phi:
            getcols.add("costhetabin")
        elif average_over_theta:
            getcols.add("phibin")
        else:
            getcols.add("dirbin")

    dfpackets = dfpackets.select(getcols).lazy().collect().lazy()

    dfdict = {}
    for dirbin in directionbins:
        if dirbin == -1:
            solidanglefactor = 1.0
            pldfpackets_dirbin_lazy = dfpackets
        elif average_over_phi:
            assert not average_over_theta
            solidanglefactor = ncosthetabins
            pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("costhetabin") * 10 == dirbin)
        elif average_over_theta:
            solidanglefactor = nphibins
            pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("phibin") == dirbin)
        else:
            solidanglefactor = ndirbins
            pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("dirbin") == dirbin)

        pldfpackets_dirbin = pldfpackets_dirbin_lazy.with_columns(
            [(2.99792458e18 / pl.col(nu_column)).alias("lambda_angstroms")]
        ).select(["lambda_angstroms", encol])

        dfbinned = at.packets.bin_and_sum(
            pldfpackets_dirbin,
            bincol="lambda_angstroms",
            bins=list(lambda_bin_edges),
            sumcols=[encol],
            getcounts=getpacketcount,
        )
        array_flambda = (
            dfbinned[f"{encol}_sum"]
            / delta_lambda
            / (timehigh - timelow)
            / (4 * math.pi)
            * solidanglefactor
            / (megaparsec_to_cm**2)
            / nprocs_read
        ).to_numpy()

        if use_time == "escape":
            assert escapesurfacegamma is not None
            array_flambda /= escapesurfacegamma

        if fnufilterfunc:
            arr_nu = 2.99792458e18 / lambda_bin_centres
            array_f_nu = array_flambda * lambda_bin_centres / arr_nu
            array_f_nu = fnufilterfunc(array_f_nu)
            array_flambda = array_f_nu * arr_nu / lambda_bin_centres

        dfdict[dirbin] = pd.DataFrame(
            {
                "lambda_angstroms": lambda_bin_centres,
                "f_lambda": array_flambda,
            }
        )

        if getpacketcount:
            dfdict[dirbin]["packetcount"] = dfbinned["count"]

    return dfdict


@lru_cache(maxsize=16)
def read_spec(modelpath: Path, printwarningsonly: bool = False) -> pl.DataFrame:
    specfilename = at.firstexisting("spec.out", folder=modelpath, tryzipped=True)
    print(f"Reading {specfilename}")

    return (
        pl.read_csv(at.zopenpl(specfilename), separator=" ", infer_schema_length=0, truncate_ragged_lines=True)
        .with_columns(pl.all().cast(pl.Float64))
        .rename({"0": "nu"})
    )


@lru_cache(maxsize=16)
def read_spec_res(modelpath: Path) -> dict[int, pl.DataFrame]:
    """Return a dataframe of time-series spectra for every viewing direction."""
    specfilename = (
        modelpath
        if Path(modelpath).is_file()
        else at.firstexisting(["spec_res.out", "specpol_res.out"], folder=modelpath, tryzipped=True)
    )

    print(f"Reading {specfilename} (in read_spec_res)")
    res_specdata_in = pl.read_csv(at.zopenpl(specfilename), separator=" ", has_header=False, infer_schema_length=0)

    # drop last column of nulls (caused by trailing space on each line)
    if res_specdata_in[res_specdata_in.columns[-1]].is_null().all():
        res_specdata_in = res_specdata_in.drop(res_specdata_in.columns[-1])

    res_specdata = at.split_dataframe_dirbins(res_specdata_in, output_polarsdf=True)

    prev_dfshape = None
    for dirbin in res_specdata:
        assert isinstance(res_specdata[dirbin], pl.DataFrame)
        newcolnames = [str(x) for x in res_specdata[dirbin][0, :].to_numpy()[0]]
        newcolnames[0] = "nu"

        newcolnames_unique = set(newcolnames)
        oldcolnames = res_specdata[dirbin].columns
        if len(newcolnames) > len(newcolnames_unique):
            # for POL_ON, the time columns repeat for Q, U, and V stokes params.
            # here, we keep the first set (I) and drop the rest of the columns
            assert len(newcolnames) % len(newcolnames_unique) == 0  # must be an exact multiple
            newcolnames = newcolnames[: len(newcolnames_unique)]
            oldcolnames = oldcolnames[: len(newcolnames_unique)]
            res_specdata[dirbin] = res_specdata[dirbin].select(oldcolnames)

        res_specdata[dirbin] = (
            res_specdata[dirbin][1:]  # drop the first row that contains time headers
            .with_columns(pl.all().cast(pl.Float64))
            .rename(dict(zip(oldcolnames, newcolnames)))
        )

        # the number of timesteps and nu bins should match for all direction bins
        assert prev_dfshape is None or prev_dfshape == res_specdata[dirbin].shape
        prev_dfshape = res_specdata[dirbin].shape

    return res_specdata


@lru_cache(maxsize=200)
def read_emission_absorption_file(emabsfilename: str | Path) -> pl.DataFrame:
    """Read into a DataFrame one of: emission.out. emissionpol.out, emissiontrue.out, absorption.out."""
    try:
        emissionfilesize = Path(emabsfilename).stat().st_size / 1024 / 1024
        print(f" Reading {emabsfilename} ({emissionfilesize:.2f} MiB)")

    except AttributeError:
        print(f" Reading {emabsfilename}")

    dfemabs = pl.read_csv(
        at.zopenpl(emabsfilename), separator=" ", has_header=False, infer_schema_length=0
    ).with_columns(pl.all().cast(pl.Float32, strict=False))

    # drop last column of nulls (caused by trailing space on each line)
    if dfemabs[dfemabs.columns[-1]].is_null().all():
        dfemabs = dfemabs.drop(dfemabs.columns[-1])

    return dfemabs


@lru_cache(maxsize=4)
def get_spec_res(
    modelpath: Path,
    average_over_theta: bool = False,
    average_over_phi: bool = False,
) -> dict[int, pl.DataFrame]:
    res_specdata = read_spec_res(modelpath)
    if average_over_theta:
        res_specdata = at.average_direction_bins(res_specdata, overangle="theta")
    if average_over_phi:
        res_specdata = at.average_direction_bins(res_specdata, overangle="phi")

    return res_specdata


def get_spectrum(
    modelpath: Path,
    timestepmin: int,
    timestepmax: int | None = None,
    directionbins: t.Sequence[int] | None = None,
    fnufilterfunc: t.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] | None = None,
    average_over_theta: bool = False,
    average_over_phi: bool = False,
    stokesparam: t.Literal["I", "Q", "U"] = "I",
) -> dict[int, pd.DataFrame]:
    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax is None or timestepmax < 0:
        timestepmax = timestepmin

    if directionbins is None:
        directionbins = [-1]
    # keys are direction bins (or -1 for spherical average)
    specdata: dict[int, pl.DataFrame] = {}

    if any(dirbin != -1 for dirbin in directionbins):
        assert stokesparam == "I"
        try:
            specdata |= get_spec_res(
                modelpath=modelpath,
                average_over_theta=average_over_theta,
                average_over_phi=average_over_phi,
            )
        except FileNotFoundError:
            msg = "WARNING: Direction-resolved spectra not found. Getting only spherically averaged spectra instead."
            print(msg)
            directionbins = [-1]

    if -1 in directionbins:
        # spherically averaged spectra
        if stokesparam == "I":
            try:
                specdata[-1] = read_spec(modelpath=modelpath).to_pandas(use_pyarrow_extension_array=True)

            except FileNotFoundError:
                specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

        else:
            specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

    specdataout: dict[int, pd.DataFrame] = {}
    for dirbin in directionbins:
        arr_nu = specdata[dirbin]["nu"].to_numpy()
        arr_tdelta = at.get_timestep_times(modelpath, loc="delta")

        try:
            arr_f_nu = stackspectra(
                [
                    (specdata[dirbin][specdata[dirbin].columns[timestep + 1]].to_numpy(), arr_tdelta[timestep])
                    for timestep in range(timestepmin, timestepmax + 1)
                ]
            )
        except IndexError:
            print(" ERROR: data not available for timestep range")
            return specdataout

        # best to use the filter on this list because it
        # has regular sampling
        if fnufilterfunc:
            if dirbin == directionbins[0]:
                print("Applying filter to ARTIS spectrum")
            arr_f_nu = fnufilterfunc(arr_f_nu)

        c_ang_per_s = 2.99792458e18
        arr_lambda = c_ang_per_s / arr_nu
        arr_f_lambda = arr_f_nu * arr_nu / arr_lambda
        dfspectrum = pd.DataFrame({"lambda_angstroms": arr_lambda, "f_lambda": arr_f_lambda})
        dfspectrum = dfspectrum.sort_values(by="lambda_angstroms", ascending=True)

        specdataout[dirbin] = dfspectrum

    return specdataout


def make_virtual_spectra_summed_file(modelpath: Path) -> Path:
    nprocs = at.get_nprocs(modelpath)
    print("nprocs", nprocs)
    vspecpol_data_old: list[
        pd.DataFrame
    ] = []  # virtual packet spectra for each observer (all directions and opacity choices)
    vpktconfig = at.get_vpkt_config(modelpath)
    nvirtual_spectra = vpktconfig["nobsdirections"] * vpktconfig["nspectraperobs"]
    print(
        f"nobsdirections {vpktconfig['nobsdirections']} nspectraperobs {vpktconfig['nspectraperobs']} (total observers:"
        f" {nvirtual_spectra})"
    )
    for mpirank in range(nprocs):
        vspecpolfilename = f"vspecpol_{mpirank}-0.out"
        vspecpolpath = at.firstexisting(vspecpolfilename, folder=modelpath, tryzipped=True)
        print(f"Reading rank {mpirank} filename {vspecpolpath}")

        vspecpolfile = pd.read_csv(vspecpolpath, sep=r"\s+", header=None)
        # Where times of timesteps are written out a new virtual spectrum starts
        # Find where the time in row 0, column 1 repeats in any column 1
        index_of_new_spectrum = vspecpolfile.index[vspecpolfile.iloc[:, 1] == vspecpolfile.iloc[0, 1]]
        vspecpol_data = []  # list of all predefined vspectra
        for i, index_spectrum_starts in enumerate(index_of_new_spectrum[:nvirtual_spectra]):
            # TODO: this is different to at.split_dataframe_dirbins() -- could be made to be same format to not repeat code
            chunk = (
                vspecpolfile.iloc[index_spectrum_starts : index_of_new_spectrum[i + 1], :]
                if index_spectrum_starts != index_of_new_spectrum[-1]
                else vspecpolfile.iloc[index_spectrum_starts:, :]
            )
            vspecpol_data.append(chunk)

        if len(vspecpol_data_old) > 0:
            for i in range(len(vspecpol_data)):
                dftmp = vspecpol_data[i].copy()  # copy of vspectrum number i in a file
                # add copy to the same spectrum number from previous file
                # (don't need to copy row 1 = time or column 1 = freq)
                dftmp.iloc[1:, 1:] += vspecpol_data_old[i].iloc[1:, 1:]
                # spectrum i then equals the sum of all previous files spectrum number i
                vspecpol_data[i] = dftmp
        # update array containing sum of previous files
        vspecpol_data_old = vspecpol_data

    for spec_index, vspecpol in enumerate(vspecpol_data):
        outfile = modelpath / f"vspecpol_total-{spec_index}.out"
        print(f"Saved {outfile}")
        vspecpol.to_csv(outfile, sep=" ", index=False, header=False)

    return outfile


def make_averaged_vspecfiles(args: argparse.Namespace) -> None:
    filenames = [vspecfile for vspecfile in os.listdir(args.modelpath[0]) if vspecfile.startswith("vspecpol_total-")]

    def sorted_by_number(lst: list) -> list:
        def convert(text: str) -> int | str:
            return int(text) if text.isdigit() else text

        def alphanum_key(key: str) -> list[int | str]:
            return [convert(c) for c in re.split("([0-9]+)", key)]

        return sorted(lst, key=alphanum_key)

    filenames = sorted_by_number(filenames)

    for spec_index, filename in enumerate(filenames):  # vspecpol-total files
        vspecdata = [pd.read_csv(modelpath / filename, sep=r"\s+", header=None) for modelpath in args.modelpath]
        for i in range(1, len(vspecdata)):
            vspecdata[0].iloc[1:, 1:] += vspecdata[i].iloc[1:, 1:]

        vspecdata[0].iloc[1:, 1:] = vspecdata[0].iloc[1:, 1:] / len(vspecdata)
        vspecdata[0].to_csv(
            args.modelpath[0] / f"vspecpol_averaged-{spec_index}.out", sep=" ", index=False, header=False
        )


@lru_cache(maxsize=4)
def get_specpol_data(
    angle: int = -1, modelpath: Path | None = None, specdata: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    if specdata is None:
        assert modelpath is not None
        specfilename = (
            at.firstexisting("specpol.out", folder=modelpath, tryzipped=True)
            if angle == -1
            else at.firstexisting(f"specpol_res_{angle}.out", folder=modelpath, tryzipped=True)
        )

        print(f"Reading {specfilename}")
        specdata = pd.read_csv(specfilename, sep=r"\s+")

    return split_dataframe_stokesparams(specdata)


@lru_cache(maxsize=4)
def get_vspecpol_data(
    vspecangle: int | None = None, modelpath: Path | None = None, specdata: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    if specdata is None:
        assert modelpath is not None
        # alternatively use f'vspecpol_averaged-{angle}.out' ?
        vspecpath = modelpath
        if (modelpath / "vspecpol").is_dir():
            vspecpath = modelpath / "vspecpol"

        try:
            specfilename = at.firstexisting(f"vspecpol_total-{vspecangle}.out", folder=vspecpath, tryzipped=True)
        except FileNotFoundError:
            print(f"vspecpol_total-{vspecangle}.out does not exist. Generating all-rank summed vspec files..")
            specfilename = make_virtual_spectra_summed_file(modelpath=modelpath)

        print(f"Reading {specfilename}")
        specdata = pd.read_csv(specfilename, sep=r"\s+")

    return split_dataframe_stokesparams(specdata)


def split_dataframe_stokesparams(specdata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """DataFrames read from specpol*.out and vspecpol*.out are repeated over I, Q, U parameters. Split these into a dictionary of DataFrames."""
    specdata = specdata.rename({"0": "nu", "0.0": "nu"}, axis="columns")
    cols_to_split = [i for i, key in enumerate(specdata.keys()) if specdata.keys()[1] in key]
    stokes_params = {
        "I": pd.concat(
            [
                specdata["nu"],
                specdata.iloc[:, cols_to_split[0] : cols_to_split[1]],
            ],
            axis="columns",
        )
    }
    stokes_params["Q"] = pd.concat(
        [specdata["nu"], specdata.iloc[:, cols_to_split[1] : cols_to_split[2]]], axis="columns"
    )
    stokes_params["U"] = pd.concat([specdata["nu"], specdata.iloc[:, cols_to_split[2] :]], axis="columns")

    for param in ["Q", "U"]:
        stokes_params[param].columns = stokes_params["I"].keys()
        stokes_params[param + "/I"] = pd.concat(
            [specdata["nu"], stokes_params[param].iloc[:, 1:] / stokes_params["I"].iloc[:, 1:]], axis="columns"
        )
    return stokes_params


def get_vspecpol_spectrum(
    modelpath: Path | str,
    timeavg: float,
    angle: int,
    args: argparse.Namespace,
    fnufilterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    stokes_params = get_vspecpol_data(vspecangle=angle, modelpath=Path(modelpath))
    if "stokesparam" not in args:
        args.stokesparam = "I"
    vspecdata = stokes_params[args.stokesparam]

    nu = vspecdata.loc[:, "nu"].to_numpy()

    arr_tmid = [float(i) for i in vspecdata.columns.to_numpy()[1:] if i[-2] != "."]
    arr_tdelta = [l1 - l2 for l1, l2 in zip(arr_tmid[1:], arr_tmid[:-1])] + [arr_tmid[-1] - arr_tmid[-2]]

    def match_closest_time(reftime: float) -> str:
        return str(f"{min((float(x) for x in arr_tmid), key=lambda x: abs(x - reftime))}")

    # if 'timemin' and 'timemax' in args:
    #     timelower = match_closest_time(args.timemin)  # how timemin, timemax are used changed at some point
    #     timeupper = match_closest_time(args.timemax)  # to average over multiple timesteps needs to fix this
    # else:
    timelower = match_closest_time(timeavg)
    timeupper = match_closest_time(timeavg)
    timestepmin = vspecdata.columns.get_loc(timelower)
    timestepmax = vspecdata.columns.get_loc(timeupper)

    f_nu = stackspectra(
        [
            (vspecdata[vspecdata.columns[timestep + 1]].to_numpy(), arr_tdelta[timestep])
            for timestep in range(timestepmin - 1, timestepmax)
        ]
    )

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")
        f_nu = fnufilterfunc(f_nu)

    dfspectrum = pd.DataFrame({"nu": nu, "f_nu": f_nu})
    dfspectrum = dfspectrum.sort_values(by="nu", ascending=False)

    dfspectrum = dfspectrum.eval("lambda_angstroms = @c / nu", local_dict={"c": 2.99792458e18})
    return dfspectrum.eval("f_lambda = f_nu * nu / lambda_angstroms")


@lru_cache(maxsize=4)
def get_flux_contributions(
    modelpath: Path,
    filterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
    timestepmin: int = -1,
    timestepmax: int = -1,
    getemission: bool = True,
    getabsorption: bool = True,
    use_lastemissiontype: bool = True,
    directionbin: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[list[fluxcontributiontuple], npt.NDArray[np.float64]]:
    arr_tmid = at.get_timestep_times(modelpath, loc="mid")
    arr_tdelta = at.get_timestep_times(modelpath, loc="delta")
    arraynu = at.get_nu_grid(modelpath)
    arraylambda = 2.99792458e18 / arraynu
    if not Path(modelpath, "compositiondata.txt").is_file():
        print("WARNING: compositiondata.txt not found. Using output*.txt instead")
        elementlist = at.get_composition_data_from_outputfile(modelpath)
    else:
        elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)

    if directionbin is None:
        dbinlist = [-1]
    elif average_over_phi:
        assert not average_over_theta
        assert directionbin % at.get_viewingdirection_phibincount() == 0
        dbinlist = list(range(directionbin, directionbin + at.get_viewingdirection_phibincount()))
    elif average_over_theta:
        assert not average_over_phi
        assert directionbin < at.get_viewingdirection_phibincount()
        dbinlist = list(range(directionbin, at.get_viewingdirectionbincount(), at.get_viewingdirection_phibincount()))
    else:
        dbinlist = [directionbin]

    emissiondata: dict[int, pd.DataFrame] = {}
    absorptiondata: dict[int, pd.DataFrame] = {}
    maxion: int | None = None
    for dbin in dbinlist:
        if getemission:
            emissionfilenames = ["emission.out", "emissionpol.out"] if use_lastemissiontype else ["emissiontrue.out"]

            if dbin != -1:
                emissionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in emissionfilenames]

            emissionfilename = at.firstexisting(emissionfilenames, folder=modelpath, tryzipped=True)

            if "pol" in str(emissionfilename):
                print("This artis run contains polarisation data")
                # File contains I, Q and U and so times are repeated 3 times
                arr_tmid = np.array(arr_tmid.tolist() * 3)

            emissiondata[dbin] = read_emission_absorption_file(emissionfilename)

            maxion_float = (emissiondata[dbin].shape[1] - 1) / 2 / nelements  # also known as MIONS in ARTIS sn3d.h
            assert maxion_float.is_integer()
            if maxion is None:
                maxion = int(maxion_float)
                print(
                    f" inferred MAXION = {maxion} from emission file using nlements = {nelements} from"
                    " compositiondata.txt"
                )
            else:
                assert maxion == int(maxion_float)

            # check that the row count is product of timesteps and frequency bins found in spec.out
            assert emissiondata[dbin].shape[0] == len(arraynu) * len(arr_tmid)

        if getabsorption:
            absorptionfilenames = ["absorption.out", "absorptionpol.out"]
            if directionbin is not None:
                absorptionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in absorptionfilenames]

            absorptionfilename = at.firstexisting(absorptionfilenames, folder=modelpath, tryzipped=True)

            absorptiondata[dbin] = read_emission_absorption_file(absorptionfilename)
            absorption_maxion_float = absorptiondata[dbin].shape[1] / nelements
            assert absorption_maxion_float.is_integer()
            absorption_maxion = int(absorption_maxion_float)
            if maxion is None:
                maxion = absorption_maxion
                print(
                    f" inferred MAXION = {maxion} from absorption file using nlements = {nelements}from"
                    " compositiondata.txt"
                )
            else:
                assert absorption_maxion == maxion
            assert absorptiondata[dbin].shape[0] == len(arraynu) * len(arr_tmid)

    array_flambda_emission_total = np.zeros_like(arraylambda, dtype=float)
    contribution_list = []
    if filterfunc:
        print("Applying filter to ARTIS spectrum")

    assert maxion is not None
    for element in range(nelements):
        nions = elementlist.nions[element]
        # nions = elementlist.iloc[element].uppermost_ion_stage - elementlist.iloc[element].lowermost_ion_stage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist.lowermost_ion_stage[element]
            ionserieslist: list[tuple[int, str]] = [
                (element * maxion + ion, "bound-bound"),
                (nelements * maxion + element * maxion + ion, "bound-free"),
            ]

            if element == ion == 0:
                ionserieslist.append((2 * nelements * maxion, "free-free"))

            for selectedcolumn, emissiontypeclass in ionserieslist:
                # if linelabel.startswith('Fe ') or linelabel.endswith("-free"):
                #     continue
                if getemission:
                    array_fnu_emission = stackspectra(
                        [
                            (
                                emissiondata[dbin][timestep :: len(arr_tmid), selectedcolumn].to_numpy(),
                                arr_tdelta[timestep] / len(dbinlist),
                            )
                            for timestep in range(timestepmin, timestepmax + 1)
                            for dbin in dbinlist
                        ]
                    )
                else:
                    array_fnu_emission = np.zeros_like(arraylambda, dtype=float)

                if absorptiondata and selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = stackspectra(
                        [
                            (
                                absorptiondata[dbin][timestep :: len(arr_tmid), selectedcolumn].to_numpy(),
                                arr_tdelta[timestep] / len(dbinlist),
                            )
                            for timestep in range(timestepmin, timestepmax + 1)
                            for dbin in dbinlist
                        ]
                    )
                else:
                    array_fnu_absorption = np.zeros_like(arraylambda, dtype=float)

                # best to use the filter on fnu (because it hopefully has regular sampling)
                if filterfunc:
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn <= nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                array_flambda_emission_total += array_flambda_emission
                fluxcontribthisseries = abs(np.trapz(array_fnu_emission, x=arraynu)) + abs(
                    np.trapz(array_fnu_absorption, x=arraynu)
                )

                if emissiontypeclass == "bound-bound":
                    linelabel = at.get_ionstring(elementlist.Z[element], ion_stage)
                elif emissiontypeclass == "free-free":
                    linelabel = "free-free"
                else:
                    linelabel = f"{at.get_ionstring(elementlist.Z[element], ion_stage)} {emissiontypeclass}"

                contribution_list.append(
                    fluxcontributiontuple(
                        fluxcontrib=fluxcontribthisseries,
                        linelabel=linelabel,
                        array_flambda_emission=array_flambda_emission,
                        array_flambda_absorption=array_flambda_absorption,
                        color=None,
                    )
                )

    return contribution_list, array_flambda_emission_total


def get_flux_contributions_from_packets(
    modelpath: Path,
    timelowdays: float,
    timehighdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: None | float | np.ndarray = None,
    getemission: bool = True,
    getabsorption: bool = True,
    maxpacketfiles: int | None = None,
    filterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
    groupby: t.Literal["ion", "line"] = "ion",
    maxseriescount: int | None = None,
    fixedionlist: list[str] | None = None,
    modelgridindex: int | None = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    use_lastemissiontype: bool = True,
    emissionvelocitycut: float | None = None,
    directionbin: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[list[fluxcontributiontuple], np.ndarray, np.ndarray]:
    assert groupby in {"ion", "line"}

    if directionbin is None:
        directionbin = -1

    emtypecolumn = "emissiontype" if use_lastemissiontype else "trueemissiontype"

    linelistlazy = at.get_linelist_pldf(modelpath=modelpath, get_ion_str=True)
    bflistlazy = at.get_bflist(modelpath, get_ion_str=True)

    nprocs_read, lzdfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if emissionvelocitycut is not None:
        lzdfpackets = at.packets.add_derived_columns_lazy(lzdfpackets)
        lzdfpackets = lzdfpackets.filter(pl.col("emission_velocity") > emissionvelocitycut)

    lzdfpackets = lzdfpackets.filter(pl.col("t_arrive_d").is_between(float(timelowdays), float(timehighdays)))

    cols = {"e_rf"}
    cols.add({"arrival": "t_arrive_d", "emission": "em_time", "escape": "excape_time"}[use_time])

    expr_linelist_to_str = (
        pl.col("ion_str")
        if groupby == "ion"
        else pl.format(
            "{} λ{} {}-{}",
            pl.col("ion_str"),
            pl.col("lambda_angstroms_air").sub(0.5).round(0).cast(pl.String).str.strip_suffix(".0"),
            pl.col("upperlevelindex"),
            pl.col("lowerlevelindex"),
        )
    )

    if getemission:
        cols |= {"emissiontype_str", "nu_rf"}
        bflistlazy = bflistlazy.with_columns((-1 - pl.col("bfindex").cast(pl.Int32)).alias(emtypecolumn))
        expr_bflist_to_str = (
            pl.col("ion_str") + " bound-free"
            if groupby == "ion"
            else pl.format("{} bound-free {}-{}", pl.col("ion_str"), pl.col("lowerlevel"), pl.col("upperionlevel"))
        )

        emtypestrings = pl.concat(
            [
                linelistlazy.select(
                    [
                        pl.col("lineindex").alias(emtypecolumn),
                        expr_linelist_to_str.alias("emissiontype_str"),
                    ]
                ),
                pl.DataFrame(
                    {emtypecolumn: [-9999999], "emissiontype_str": ["free-free"]},
                    schema_overrides={emtypecolumn: pl.Int32, "emissiontype_str": pl.String},
                ).lazy(),
                bflistlazy.select(
                    [
                        pl.col(emtypecolumn),
                        expr_bflist_to_str.alias("emissiontype_str"),
                    ]
                ),
            ],
        ).with_columns(pl.col("emissiontype_str").cast(pl.Categorical))

        lzdfpackets = lzdfpackets.join(emtypestrings, on=emtypecolumn, how="left")

    if getabsorption:
        cols |= {"absorptiontype_str", "absorption_freq"}

        abstypestrings = pl.concat(
            [
                linelistlazy.select(
                    [
                        pl.col("lineindex").alias("absorption_type"),
                        expr_linelist_to_str.alias("absorptiontype_str"),
                    ]
                ),
                pl.DataFrame(
                    {"absorption_type": [-1, -2], "absorptiontype_str": ["free-free", "bound-free"]},
                    schema_overrides={"absorption_type": pl.Int32, "absorptiontype_str": pl.String},
                ).lazy(),
            ],
        ).with_columns(pl.col("absorptiontype_str").cast(pl.Categorical))

        lzdfpackets = lzdfpackets.join(abstypestrings, on="absorption_type", how="left")

    if directionbin != -1:
        cols |= {"costhetabin", "phibin", "dirbin"}

    nu_min = 2.99792458e18 / lambda_max
    nu_max = 2.99792458e18 / lambda_min
    dfpackets = (
        lzdfpackets.filter(
            (pl.col("nu_rf").is_between(nu_min, nu_max)) | (pl.col("absorption_freq").is_between(nu_min, nu_max))
        )
        .select([col for col in cols if col in lzdfpackets.columns])
        .collect()
    )

    emissiongroups = (
        dict(dfpackets.filter(pl.col("nu_rf").is_between(nu_min, nu_max)).group_by("emissiontype_str"))
        if getemission
        else {}
    )

    absorptiongroups = (
        dict(dfpackets.filter(pl.col("absorption_freq").is_between(nu_min, nu_max)).group_by("absorptiontype_str"))
        if getabsorption
        else {}
    )

    allgroupnames = set(emissiongroups.keys()) | set(absorptiongroups.keys())

    if maxseriescount is None:
        maxseriescount = len(allgroupnames)

    # group small contributions together to avoid the cost of binning individual spectra for them
    grouptotals: list[tuple[float, str]] = []
    for groupname in allgroupnames:
        groupemiss = emissiongroups[groupname]["e_rf"].sum() if groupname in emissiongroups else 0.0
        groupabs = absorptiongroups[groupname]["e_rf"].sum() if groupname in absorptiongroups else 0.0
        grouptotal = groupemiss + groupabs

        if groupname is not None:
            assert isinstance(groupname, str)
            grouptotals.append((grouptotal, groupname))
        else:
            with contextlib.suppress(KeyError):
                del emissiongroups[groupname]
                del absorptiongroups[groupname]

    allgroupnames = set(emissiongroups.keys()) | set(absorptiongroups.keys())

    if fixedionlist is not None and (unrecognised_items := [x for x in fixedionlist if x not in allgroupnames]):
        print(f"WARNING: (packets) did not find {len(unrecognised_items)} items in fixedionlist: {unrecognised_items}")

    def sortkey(x: tuple[float, str]) -> tuple[int, float]:
        (grouptotal, groupname) = x

        if fixedionlist is None:
            return (0, -grouptotal)

        return (fixedionlist.index(groupname), 0) if groupname in fixedionlist else (len(fixedionlist) + 1, -grouptotal)

    sorted_grouptotals = sorted(grouptotals, key=sortkey)
    other_groups = sorted_grouptotals[maxseriescount:]

    if other_groups:
        allgroupnames.add("Other")

        if emdfs := [emissiongroups[groupname] for _, groupname in other_groups if groupname in emissiongroups]:
            emissiongroups["Other"] = pl.concat(emdfs)

        if absdfs := [absorptiongroups[groupname] for _, groupname in other_groups if groupname in absorptiongroups]:
            absorptiongroups["Other"] = pl.concat(absdfs)

        for grouptotal, groupname in other_groups:
            with contextlib.suppress(KeyError):
                del emissiongroups[groupname]
                del absorptiongroups[groupname]
            allgroupnames.remove(groupname)

    array_flambda_emission_total = None
    contribution_list = []
    array_lambda = None
    for groupname in allgroupnames:
        array_flambda_emission = None

        if groupname in emissiongroups:
            spec_group = get_from_packets(
                modelpath=modelpath,
                timelowdays=timelowdays,
                timehighdays=timehighdays,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                use_time=use_time,
                delta_lambda=delta_lambda,
                fnufilterfunc=filterfunc,
                nprocs_read_dfpackets=(nprocs_read, emissiongroups[groupname]),
                directionbins=[directionbin],
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
            )[directionbin]

            if array_lambda is None:
                array_lambda = spec_group["lambda_angstroms"].to_numpy()

            array_flambda_emission = spec_group["f_lambda"].to_numpy()

            if array_flambda_emission_total is None:
                array_flambda_emission_total = np.zeros_like(array_flambda_emission, dtype=float)

            array_flambda_emission_total += array_flambda_emission

        if groupname in absorptiongroups:
            spec_group = get_from_packets(
                modelpath=modelpath,
                timelowdays=timelowdays,
                timehighdays=timehighdays,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                delta_lambda=delta_lambda,
                nu_column="absorption_freq",
                fnufilterfunc=filterfunc,
                nprocs_read_dfpackets=(nprocs_read, absorptiongroups[groupname]),
            )[-1]

            if array_lambda is None:
                array_lambda = spec_group["lambda_angstroms"].to_numpy()

            array_flambda_absorption = spec_group["f_lambda"].to_numpy()
        else:
            array_flambda_absorption = np.zeros_like(array_flambda_emission, dtype=float)

        if array_flambda_emission is None:
            array_flambda_emission = np.zeros_like(array_flambda_absorption, dtype=float)

        fluxcontribthisseries = abs(np.trapz(array_flambda_emission, x=array_lambda)) + abs(
            np.trapz(array_flambda_absorption, x=array_lambda)
        )

        if fluxcontribthisseries > 0.0:
            contribution_list.append(
                fluxcontributiontuple(
                    fluxcontrib=fluxcontribthisseries,
                    linelabel=str(groupname),
                    array_flambda_emission=array_flambda_emission,
                    array_flambda_absorption=array_flambda_absorption,
                    color=None,
                )
            )

    if array_flambda_emission_total is None:
        array_flambda_emission_total = np.zeros_like(array_lambda, dtype=float)

    assert array_lambda is not None

    return contribution_list, array_flambda_emission_total, array_lambda


def sort_and_reduce_flux_contribution_list(
    contribution_list_in: list[fluxcontributiontuple],
    maxseriescount: int,
    arraylambda_angstroms: np.ndarray,
    fixedionlist: list[str] | None = None,
    hideother: bool = False,
    greyscale: bool = False,
) -> list[fluxcontributiontuple]:
    if fixedionlist:
        if unrecognised_items := [x for x in fixedionlist if x not in [y.linelabel for y in contribution_list_in]]:
            print(f"WARNING: did not understand these items in fixedionlist: {unrecognised_items}")

        # sort in manual order
        def sortkey(x: fluxcontributiontuple) -> tuple[int, float]:
            assert fixedionlist is not None
            return (
                fixedionlist.index(x.linelabel) if x.linelabel in fixedionlist else len(fixedionlist) + 1,
                -x.fluxcontrib,
            )

    else:
        # sort descending by flux contribution
        def sortkey(x: fluxcontributiontuple) -> tuple[int, float]:
            return (0, -x.fluxcontrib)

    contribution_list = sorted(contribution_list_in, key=sortkey)

    color_list: list[t.Any]
    if greyscale:
        hatches = at.spectra.plotspectra.hatches
        seriescount = len(fixedionlist) if fixedionlist else maxseriescount
        colorcount = math.ceil(seriescount / 1.0 / len(hatches))
        greylist = [str(x) for x in np.linspace(0.4, 0.9, colorcount, endpoint=True)]
        color_list = []
        for c in range(colorcount):
            for _h in hatches:
                color_list.append(greylist[c])
        # color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))
        mpl.rcParams["hatch.linewidth"] = 0.1
        # TODO: remove???
        color_list = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))
    else:
        color_list = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))

    # combine the items past maxseriescount or not in manual list into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_fluxcontrib = 0

    contribution_list_out = []
    numotherprinted = 0
    maxnumotherprinted = 20
    entered_other = False
    plotted_ion_list = []
    index = 0

    for row in contribution_list:
        if row.linelabel != "Other" and fixedionlist and row.linelabel in fixedionlist:
            contribution_list_out.append(row._replace(color=color_list[fixedionlist.index(row.linelabel)]))
        elif row.linelabel != "Other" and not fixedionlist and index < maxseriescount:
            contribution_list_out.append(row._replace(color=color_list[index]))
            plotted_ion_list.append(row.linelabel)
        else:
            remainder_fluxcontrib += row.fluxcontrib
            remainder_flambda_emission += row.array_flambda_emission
            remainder_flambda_absorption += row.array_flambda_absorption
            if row.linelabel != "Other" and not entered_other:
                print(f"  Other (top {maxnumotherprinted}):")
                entered_other = True

        if row.linelabel != "Other":
            index += 1

        if numotherprinted < maxnumotherprinted and row.linelabel != "Other":
            integemiss = abs(np.trapz(row.array_flambda_emission, x=arraylambda_angstroms))
            integabsorp = abs(np.trapz(-row.array_flambda_absorption, x=arraylambda_angstroms))
            if integabsorp > 0.0 and integemiss > 0.0:
                print(
                    f"{row.fluxcontrib:.1e}, emission {integemiss:.1e}, "
                    f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'"
                )
            elif integemiss > 0.0:
                print(f"  emission {integemiss:.1e} [erg/s/cm^2]: '{row.linelabel}'")
            else:
                print(f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'")

            if entered_other:
                numotherprinted += 1

    if not fixedionlist:
        cmdarg = "'" + "' '".join(plotted_ion_list) + "'"
        print("To reuse this ion/process contribution list, pass the following command-line argument: ")
        print(f"     -fixedionlist {cmdarg}")
        print("Or in python: ")
        print(f"     fixedionlist={plotted_ion_list}")

    if remainder_fluxcontrib > 0.0 and not hideother:
        contribution_list_out.append(
            fluxcontributiontuple(
                fluxcontrib=remainder_fluxcontrib,
                linelabel="Other",
                array_flambda_emission=remainder_flambda_emission,
                array_flambda_absorption=remainder_flambda_absorption,
                color="grey",
            )
        )

    return contribution_list_out


def print_integrated_flux(
    arr_f_lambda: np.ndarray | pd.Series, arr_lambda_angstroms: np.ndarray | pd.Series, distance_megaparsec: float = 1.0
) -> float:
    integrated_flux = abs(np.trapz(np.nan_to_num(arr_f_lambda, nan=0.0), x=arr_lambda_angstroms))
    print(
        f" integrated flux ({arr_lambda_angstroms.min():.1f} to "
        f"{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e} erg/s/cm2"
    )
    return integrated_flux


def get_reference_spectrum(filename: Path | str) -> tuple[pd.DataFrame, dict[t.Any, t.Any]]:
    if Path(filename).is_file():
        filepath = Path(filename)
    else:
        filepath = Path(at.get_config()["path_artistools_dir"], "data", "refspectra", filename)

        if not filepath.is_file():
            filepathxz = filepath.with_suffix(f"{filepath.suffix}.xz")
            if filepathxz.is_file():
                filepath = filepathxz
            else:
                filepathgz = filepath.with_suffix(f"{filepath.suffix}.gz")
                if filepathgz.is_file():
                    filepath = filepathgz

    metadata = at.get_file_metadata(filepath)

    flambdaindex = metadata.get("f_lambda_columnindex", 1)

    specdata = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["lambda_angstroms", "f_lambda"],
        usecols=[0, flambdaindex],
    )

    # new_lambda_angstroms = []
    # binned_flux = []
    #
    # wavelengths = specdata['lambda_angstroms']
    # fluxes = specdata['f_lambda']
    # nbins = 10
    #
    # for i in np.arange(start=0, stop=len(wavelengths) - nbins, step=nbins):
    #     new_lambda_angstroms.append(wavelengths[i + int(nbins / 2)])
    #     sum_flux = 0
    #     for j in range(i, i + nbins):
    #
    #         if not math.isnan(fluxes[j]):
    #             print(fluxes[j])
    #             sum_flux += fluxes[j]
    #     binned_flux.append(sum_flux / nbins)
    #
    # filtered_specdata = pd.DataFrame(new_lambda_angstroms, columns=['lambda_angstroms'])
    # filtered_specdata['f_lamba'] = binned_flux
    # print(filtered_specdata)
    # plt.plot(specdata['lambda_angstroms'], specdata['f_lambda'])
    # plt.plot(new_lambda_angstroms, binned_flux)
    #
    # filtered_specdata.to_csv('/Users/ccollins/artis_nebular/artistools/artistools/data/refspectra/' + name,
    #                          index=False, header=False, sep=' ')

    if "a_v" in metadata or "e_bminusv" in metadata:
        print("Correcting for reddening")
        from extinction import apply
        from extinction import ccm89

        specdata["f_lambda"] = apply(
            ccm89(specdata["lambda_angstroms"].to_numpy(), a_v=-metadata["a_v"], r_v=metadata["r_v"], unit="aa"),
            specdata["f_lambda"].to_numpy(),
        )

    if "z" in metadata:
        print("Correcting for redshift")
        specdata["lambda_angstroms"] /= 1 + metadata["z"]

    return specdata, metadata
