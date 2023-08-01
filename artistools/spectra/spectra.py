"""Artistools - spectra related functions."""
from __future__ import annotations

import argparse
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
from astropy import constants as const
from astropy import units as u
from typeguard import typechecked

import artistools as at

fluxcontributiontuple = namedtuple(
    "fluxcontributiontuple", "fluxcontrib linelabel array_flambda_emission array_flambda_absorption color"
)


def timeshift_fluxscale_co56law(scaletoreftime: float | None, spectime: float) -> float:
    if scaletoreftime is not None:
        # Co56 decay flux scaling
        assert spectime > 150
        return math.exp(float(spectime) / 113.7) / math.exp(scaletoreftime / 113.7)

    return 1.0


def get_exspec_bins() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    MNUBINS = 1000
    NU_MIN_R = 1e13
    NU_MAX_R = 5e16

    print(
        f" assuming {MNUBINS=} {NU_MIN_R=:.1e} {NU_MAX_R=:.1e}. Check artisoptions.h if you want to exactly match"
        " exspec binning."
    )

    c_ang_s = 2.99792458e18

    dlognu = (math.log(NU_MAX_R) - math.log(NU_MIN_R)) / MNUBINS

    bins_nu_lower = np.array([math.exp(math.log(NU_MIN_R) + (m * (dlognu))) for m in range(MNUBINS)])
    # bins_nu_upper = np.array(
    #     [math.exp(math.log(NU_MIN_R) + ((m + 1) * (dlognu))) for m in range(MNUBINS)])
    bins_nu_upper = bins_nu_lower * math.exp(dlognu)
    bins_nu_centre = 0.5 * (bins_nu_lower + bins_nu_upper)

    array_lambdabinedges = np.append(c_ang_s / np.flip(bins_nu_upper), c_ang_s / bins_nu_lower[0])
    array_lambda = c_ang_s / np.flip(bins_nu_centre)
    delta_lambda = np.flip(c_ang_s / bins_nu_lower - c_ang_s / bins_nu_upper)

    return array_lambdabinedges, array_lambda, delta_lambda


def stackspectra(
    spectra_and_factors: list[tuple[np.ndarray[t.Any, np.dtype[np.float64]], float]]
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


@typechecked
def get_from_packets(
    modelpath: Path,
    timelowdays: float,
    timehighdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: None | float | np.ndarray = None,
    use_escapetime: bool = False,
    maxpacketfiles: int | None = None,
    useinternalpackets: bool = False,
    getpacketcount: bool = False,
    directionbins: t.Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    fnufilterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[int, pd.DataFrame]:
    """Get a spectrum dataframe using the packets files as input."""
    assert not useinternalpackets
    if directionbins is None:
        directionbins = [-1]

    if use_escapetime:
        modeldata, _ = at.inputmodel.get_modeldata(modelpath)
        vmax_beta = modeldata.iloc[-1].velocity_outer * 299792.458
        escapesurfacegamma = math.sqrt(1 - vmax_beta**2)
    else:
        escapesurfacegamma = None

    nu_min = 2.99792458e18 / lambda_max
    nu_max = 2.99792458e18 / lambda_min

    array_lambdabinedges: np.ndarray
    if delta_lambda:
        array_lambdabinedges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        array_lambda = 0.5 * (array_lambdabinedges[:-1] + array_lambdabinedges[1:])  # bin centres
    else:
        array_lambdabinedges, array_lambda, delta_lambda = get_exspec_bins()

    timelow = timelowdays * 86400.0
    timehigh = timehighdays * 86400.0

    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()
    ndirbins = at.get_viewingdirectionbincount()

    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if not use_escapetime:
        dfpackets = dfpackets.filter(
            (float(timelowdays) <= pl.col("t_arrive_d")) & (pl.col("t_arrive_d") <= float(timehighdays))
        )
    else:
        dfpackets = dfpackets.filter(
            (timelow <= (pl.col("escape_time") * escapesurfacegamma))
            & ((pl.col("escape_time") * escapesurfacegamma) <= timehigh)
        )
    dfpackets = dfpackets.filter((float(nu_min) <= pl.col("nu_rf")) & (pl.col("nu_rf") <= float(nu_max)))

    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")

    encol = "e_cmf" if use_escapetime else "e_rf"
    getcols = ["nu_rf", encol]
    if directionbins != [-1]:
        if average_over_phi:
            getcols.append("costhetabin")
        elif average_over_theta:
            getcols.append("phibin")
        else:
            getcols.append("dirbin")
    dfpackets = dfpackets.select(getcols).collect().lazy()

    dfdict = {}
    megaparsec_to_cm = 3.085677581491367e24
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
            [(2.99792458e18 / pl.col("nu_rf")).alias("lambda_angstroms")]
        ).select(["lambda_angstroms", encol])

        dfbinned = at.packets.bin_and_sum(
            pldfpackets_dirbin,
            bincol="lambda_angstroms",
            bins=list(array_lambdabinedges),
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

        if use_escapetime:
            assert escapesurfacegamma is not None
            array_flambda /= escapesurfacegamma

        if fnufilterfunc:
            arr_nu = 2.99792458e18 / array_lambda
            array_f_nu = array_flambda * array_lambda / arr_nu
            array_f_nu = fnufilterfunc(array_f_nu)
            array_flambda = array_f_nu * arr_nu / array_lambda

        dfdict[dirbin] = pd.DataFrame(
            {
                "lambda_angstroms": array_lambda,
                "f_lambda": array_flambda,
            }
        )

        if getpacketcount:
            dfdict[dirbin]["packetcount"] = dfbinned["count"]

    return dfdict


@lru_cache(maxsize=16)
def read_spec_res(modelpath: Path) -> dict[int, pl.DataFrame]:
    """Return a dataframe of time-series spectra for every viewing direction."""
    specfilename = (
        modelpath
        if Path(modelpath).is_file()
        else at.firstexisting(["spec_res.out", "specpol_res.out"], folder=modelpath, tryzipped=True)
    )

    print(f"Reading {specfilename} (in read_spec_res)")
    res_specdata_in = pl.read_csv(at.zopen(specfilename, "rb"), separator=" ", has_header=False, infer_schema_length=0)

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
        at.zopen(emabsfilename, "rb").read(), separator=" ", has_header=False, infer_schema_length=0
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


@typechecked
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
    specdata: dict[int, pd.DataFrame] = {}

    if -1 in directionbins:
        # spherically averaged spectra
        if stokesparam == "I":
            try:
                specfilename = at.firstexisting("spec.out", folder=modelpath, tryzipped=True)

                print(f"Reading {specfilename}")

                specdata[-1] = (
                    pl.read_csv(at.zopen(specfilename, mode="rb"), separator=" ", infer_schema_length=0)
                    .with_columns(pl.all().cast(pl.Float64))
                    .rename({"0": "nu"})
                    .to_pandas()
                )

            except FileNotFoundError:
                specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

        else:
            specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

    if any(dirbin != -1 for dirbin in directionbins):
        assert stokesparam == "I"
        specdata |= get_spec_res(
            modelpath=modelpath,
            average_over_theta=average_over_theta,
            average_over_phi=average_over_phi,
        )

    specdataout: dict[int, pd.DataFrame] = {}
    for dirbin in directionbins:
        arr_nu = specdata[dirbin]["nu"].to_numpy()
        arr_tdelta = at.get_timestep_times(modelpath, loc="delta")

        arr_f_nu = stackspectra(
            [
                (specdata[dirbin][specdata[dirbin].columns[timestep + 1]].to_numpy(), arr_tdelta[timestep])
                for timestep in range(timestepmin, timestepmax + 1)
            ]
        )

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
    vspecpol_data_old: list[pd.DataFrame] = (
        []
    )  # virtual packet spectra for each observer (all directions and opacity choices)
    vpktconfig = at.get_vpkt_config(modelpath)
    nvirtual_spectra = vpktconfig["nobsdirections"] * vpktconfig["nspectraperobs"]
    print(
        f"nobsdirections {vpktconfig['nobsdirections']} nspectraperobs {vpktconfig['nspectraperobs']} (total observers:"
        f" {nvirtual_spectra})"
    )
    for mpirank in range(nprocs):
        vspecpolfilename = f"vspecpol_{mpirank}-0.out"
        print(f"Reading rank {mpirank} filename {vspecpolfilename}")
        vspecpolpath = Path(modelpath, vspecpolfilename)
        if not vspecpolpath.is_file():
            vspecpolpath = Path(modelpath, vspecpolfilename + ".gz")
            if not vspecpolpath.is_file():
                print(f"Warning: Could not find {vspecpolpath.relative_to(modelpath.parent)}")
                continue

        vspecpolfile = pd.read_csv(vspecpolpath, delim_whitespace=True, header=None)
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
            for i, _ in enumerate(vspecpol_data):
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
    filenames = []
    for vspecfile in os.listdir(args.modelpath[0]):
        if vspecfile.startswith("vspecpol_total-"):
            filenames.append(vspecfile)

    def sorted_by_number(l: list) -> list:
        def convert(text: str) -> int | str:
            return int(text) if text.isdigit() else text

        def alphanum_key(key: str) -> list[int | str]:
            return [convert(c) for c in re.split("([0-9]+)", key)]

        return sorted(l, key=alphanum_key)

    filenames = sorted_by_number(filenames)

    for spec_index, filename in enumerate(filenames):  # vspecpol-total files
        vspecdata = [
            pd.read_csv(modelpath / filename, delim_whitespace=True, header=None) for modelpath in args.modelpath
        ]
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
        specdata = pd.read_csv(specfilename, delim_whitespace=True)

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
        specdata = pd.read_csv(specfilename, delim_whitespace=True)

    return split_dataframe_stokesparams(specdata)


def split_dataframe_stokesparams(specdata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """DataFrames read from specpol*.out and vspecpol*.out are repeated over I, Q, U
    parameters. Split these into a dictionary of DataFrames.
    """
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
            (vspecdata[vspecdata.columns[timestep + 1]], arr_tdelta[timestep])
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
    averageoverphi: bool = False,
    averageovertheta: bool = False,
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
    elif averageoverphi:
        assert not averageovertheta
        assert directionbin % at.get_viewingdirection_phibincount() == 0
        dbinlist = list(range(directionbin, directionbin + at.get_viewingdirection_phibincount()))
    elif averageovertheta:
        assert not averageoverphi
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
        # nions = elementlist.iloc[element].uppermost_ionstage - elementlist.iloc[element].lowermost_ionstage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist.lowermost_ionstage[element]
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


@lru_cache(maxsize=4)
def get_flux_contributions_from_packets(
    modelpath: Path,
    timelowerdays: float,
    timeupperdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: None | float | np.ndarray = None,
    getemission: bool = True,
    getabsorption: bool = True,
    maxpacketfiles: int | None = None,
    filterfunc: t.Callable[[np.ndarray], np.ndarray] | None = None,
    groupby: t.Literal["ion", "line", "upperterm", "terms"] | None = "ion",
    modelgridindex: int | None = None,
    use_escapetime: bool = False,
    use_lastemissiontype: bool = True,
    useinternalpackets: bool = False,
    emissionvelocitycut: float | None = None,
) -> tuple[list[fluxcontributiontuple], np.ndarray, np.ndarray]:
    assert groupby in [None, "ion", "line", "upperterm", "terms"]

    if groupby in ["terms", "upperterm"]:
        adata = at.atomic.get_levels(modelpath)

    def get_emprocesslabel(
        linelist: dict[int, at.linetuple], bflist: dict[int, tuple[int, int, int, int]], emtype: int
    ) -> str:
        if emtype >= 0:
            line = linelist[emtype]

            if groupby == "line":
                return (
                    f"{at.get_ionstring(line.atomic_number, line.ionstage)} "
                    f"λ{line.lambda_angstroms:.0f} "
                    f"({line.upperlevelindex}-{line.lowerlevelindex})"
                )

            if groupby == "terms":
                upper_config = (
                    adata.query("Z == @line.atomic_number and ion_stage == @line.ionstage", inplace=False)
                    .iloc[0]
                    .levels.iloc[line.upperlevelindex]
                    .levelname
                )
                upper_term_noj = upper_config.split("_")[-1].split("[")[0]
                lower_config = (
                    adata.query("Z == @line.atomic_number and ion_stage == @line.ionstage", inplace=False)
                    .iloc[0]
                    .levels.iloc[line.lowerlevelindex]
                    .levelname
                )
                lower_term_noj = lower_config.split("_")[-1].split("[")[0]
                return f"{at.get_ionstring(line.atomic_number, line.ionstage)} {upper_term_noj}->{lower_term_noj}"

            if groupby == "upperterm":
                upper_config = (
                    adata.query("Z == @line.atomic_number and ion_stage == @line.ionstage", inplace=False)
                    .iloc[0]
                    .levels.iloc[line.upperlevelindex]
                    .levelname
                )
                upper_term_noj = upper_config.split("_")[-1].split("[")[0]
                return f"{at.get_ionstring(line.atomic_number, line.ionstage)} {upper_term_noj}"

            return f"{at.get_ionstring(line.atomic_number, line.ionstage)} bound-bound"

        if emtype == -9999999:
            return "free-free"

        bfindex = -emtype - 1
        if bfindex in bflist:
            (atomic_number, ionstage, level) = bflist[bfindex][:3]
            if groupby == "line":
                return f"{at.get_ionstring(atomic_number, ionstage)} bound-free {level}"
            return f"{at.get_ionstring(atomic_number, ionstage)} bound-free"

        return f"? bound-free (bfindex={bfindex})"

    def get_absprocesslabel(linelist: dict[int, at.linetuple], abstype: int) -> str:
        if abstype >= 0:
            line = linelist[abstype]
            if groupby == "line":
                return (
                    f"{at.get_ionstring(line.atomic_number, line.ionstage)} "
                    f"λ{line.lambda_angstroms:.0f} "
                    f"({line.upperlevelindex}-{line.lowerlevelindex})"
                )
            return f"{at.get_ionstring(line.atomic_number, line.ionstage)} bound-bound"
        if abstype == -1:
            return "free-free"
        return "bound-free" if abstype == -2 else "? other absorp."

    array_lambdabinedges: np.ndarray
    if delta_lambda is not None:
        array_lambdabinedges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        array_lambda = 0.5 * (array_lambdabinedges[:-1] + array_lambdabinedges[1:])  # bin centres
    else:
        array_lambdabinedges, array_lambda, delta_lambda = get_exspec_bins()
    assert delta_lambda is not None

    if use_escapetime:
        modeldata, _ = at.inputmodel.get_modeldata(modelpath)
        vmax = modeldata.iloc[-1].velocity_outer * u.km / u.s
        betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles)

    linelist = at.get_linelist_dict(modelpath=modelpath)

    energysum_spectrum_emission_total = np.zeros_like(array_lambda, dtype=float)
    array_energysum_spectra = {}

    timelow = timelowerdays * 86400.0
    timehigh = timeupperdays * 86400.0

    nprocs_read = len(packetsfiles)
    c_cgs = 29979245800.0
    nu_min = 2.99792458e18 / lambda_max
    nu_max = 2.99792458e18 / lambda_min

    emtypecolumn = (
        "emissiontype" if useinternalpackets else "emissiontype" if use_lastemissiontype else "trueemissiontype"
    )

    for _index, packetsfile in enumerate(packetsfiles):
        if useinternalpackets:
            # if we're using packets*.out files, these packets are from the last timestep
            t_seconds = at.get_timestep_times(modelpath, loc="start")[-1] * 86400.0

            if modelgridindex is not None:
                v_inner = at.inputmodel.get_modeldata_tuple(modelpath)[0]["velocity_inner"].iloc[modelgridindex] * 1e5
                v_outer = at.inputmodel.get_modeldata_tuple(modelpath)[0]["velocity_outer"].iloc[modelgridindex] * 1e5
            else:
                v_inner = 0.0
                v_outer = at.inputmodel.get_modeldata_tuple(modelpath)[0]["velocity_outer"].iloc[-1] * 1e5

            r_inner = t_seconds * v_inner
            r_outer = t_seconds * v_outer

            dfpackets = at.packets.readfile(packetsfile, packet_type="TYPE_RPKT")
            print("Using non-escaped internal r-packets")
            dfpackets = dfpackets.query(f'type_id == {at.packets.type_ids["TYPE_RPKT"]} and @nu_min <= nu_rf < @nu_max')
            if modelgridindex is not None:
                assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
                # dfpackets.eval(f'velocity = sqrt(posx ** 2 + posy ** 2 + posz ** 2) / @t_seconds', inplace=True)
                # dfpackets.query(f'@v_inner <= velocity <= @v_outer',
                #                 inplace=True)
                dfpackets = dfpackets.query("where in @assoc_cells[@modelgridindex]")
            print(f"  {len(dfpackets)} internal r-packets matching frequency range")
        else:
            dfpackets = at.packets.readfile(packetsfile, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
            dfpackets = dfpackets.query(
                "@nu_min <= nu_rf < @nu_max and trueemissiontype >= 0 and "
                + (
                    "@timelow < escape_time * @betafactor < @timehigh"
                    if use_escapetime
                    else "@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh"
                ),
            )
            print(f"  {len(dfpackets)} escaped r-packets matching frequency and arrival time ranges")

            if emissionvelocitycut:
                dfpackets = at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])

                dfpackets = dfpackets.query("(emission_velocity / 1e5) > @emissionvelocitycut")

        if np.isscalar(delta_lambda):
            dfpackets = dfpackets.eval("xindex = floor((2.99792458e18 / nu_rf - @lambda_min) / @delta_lambda)")
            if getabsorption:
                dfpackets = dfpackets.eval(
                    "xindexabsorbed = floor((2.99792458e18 / absorption_freq - @lambda_min) / @delta_lambda)",
                )
        else:
            dfpackets["xindex"] = (
                np.digitize(2.99792458e18 / dfpackets.nu_rf, bins=array_lambdabinedges, right=True) - 1
            )
            if getabsorption:
                dfpackets["xindexabsorbed"] = (
                    np.digitize(2.99792458e18 / dfpackets.absorption_freq, bins=array_lambdabinedges, right=True) - 1
                )

        bflist = at.get_bflist(modelpath)
        for _, packet in dfpackets.iterrows():
            xindex = int(packet.xindex)
            assert xindex >= 0

            pkt_en = packet.e_cmf / betafactor if use_escapetime else packet.e_rf

            energysum_spectrum_emission_total[xindex] += pkt_en

            if getemission:
                # if emtype >= 0 and linelist[emtype].upperlevelindex <= 80:
                #     continue
                # emprocesskey = get_emprocesslabel(packet.emissiontype)
                emprocesskey = get_emprocesslabel(linelist, bflist, packet[emtypecolumn])
                # print('packet lambda_cmf: {2.99792458e18 / packet.nu_cmf}.1f}, lambda_rf {lambda_rf:.1f}, {emprocesskey}')

                if emprocesskey not in array_energysum_spectra:
                    array_energysum_spectra[emprocesskey] = (
                        np.zeros_like(array_lambda, dtype=float),
                        np.zeros_like(array_lambda, dtype=float),
                    )

                array_energysum_spectra[emprocesskey][0][xindex] += pkt_en

            if getabsorption:
                abstype = packet.absorption_type
                if abstype > 0:
                    absprocesskey = get_absprocesslabel(linelist, abstype)

                    xindexabsorbed = int(packet.xindexabsorbed)  # bin by absorption wavelength
                    # xindexabsorbed = xindex  # bin by final escaped wavelength

                    if absprocesskey not in array_energysum_spectra:
                        array_energysum_spectra[absprocesskey] = (
                            np.zeros_like(array_lambda, dtype=float),
                            np.zeros_like(array_lambda, dtype=float),
                        )

                    array_energysum_spectra[absprocesskey][1][xindexabsorbed] += pkt_en

    if useinternalpackets:
        volume = 4 / 3.0 * math.pi * (r_outer**3 - r_inner**3)
        if modelgridindex:
            volume_shells = volume
            assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
            volume = (
                at.get_wid_init_at_tmin(modelpath) * t_seconds / (at.get_inputparams(modelpath)["tmin"] * 86400.0)
            ) ** 3 * len(assoc_cells[modelgridindex])
            print("volume", volume, "shell volume", volume_shells, "-------------------------------------------------")
        normfactor = c_cgs / 4 / math.pi / delta_lambda / volume / nprocs_read
    else:
        megaparsec_to_cm = 3.085677581491367e24
        normfactor = 1.0 / delta_lambda / (timehigh - timelow) / 4 / math.pi / (megaparsec_to_cm**2) / nprocs_read

    array_flambda_emission_total = energysum_spectrum_emission_total * normfactor

    contribution_list = []
    for groupname, (energysum_spec_emission, energysum_spec_absorption) in array_energysum_spectra.items():
        array_flambda_emission = energysum_spec_emission * normfactor

        array_flambda_absorption = energysum_spec_absorption * normfactor

        fluxcontribthisseries = abs(np.trapz(array_flambda_emission, x=array_lambda)) + abs(
            np.trapz(array_flambda_absorption, x=array_lambda)
        )

        linelabel = groupname.replace(" bound-bound", "")

        contribution_list.append(
            fluxcontributiontuple(
                fluxcontrib=fluxcontribthisseries,
                linelabel=linelabel,
                array_flambda_emission=array_flambda_emission,
                array_flambda_absorption=array_flambda_absorption,
                color=None,
            )
        )

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
        unrecognised_items = [x for x in fixedionlist if x not in [y.linelabel for y in contribution_list_in]]
        if unrecognised_items:
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

    # combine the items past maxseriescount or not in manual list into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_fluxcontrib = 0

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

    contribution_list_out = []
    numotherprinted = 0
    maxnumotherprinted = 20
    entered_other = False
    plotted_ion_list = []
    for index, row in enumerate(contribution_list):
        if fixedionlist and row.linelabel in fixedionlist:
            contribution_list_out.append(row._replace(color=color_list[fixedionlist.index(row.linelabel)]))
        elif not fixedionlist and index < maxseriescount:
            contribution_list_out.append(row._replace(color=color_list[index]))
            plotted_ion_list.append(row.linelabel)
        else:
            remainder_fluxcontrib += row.fluxcontrib
            remainder_flambda_emission += row.array_flambda_emission
            remainder_flambda_absorption += row.array_flambda_absorption
            if not entered_other:
                print(f"  Other (top {maxnumotherprinted}):")
                entered_other = True

        if numotherprinted < maxnumotherprinted:
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
    arr_f_lambda: np.ndarray, arr_lambda_angstroms: np.ndarray, distance_megaparsec: float = 1.0
) -> float:
    integrated_flux = (
        abs(np.trapz(np.nan_to_num(arr_f_lambda, nan=0.0), x=arr_lambda_angstroms)) * u.erg / u.s / (u.cm**2)
    )
    print(
        f" integrated flux ({arr_lambda_angstroms.min():.1f} to "
        f"{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e}"
    )
    # luminosity = integrated_flux * 4 * math.pi * (distance_megaparsec * u.megaparsec ** 2)
    # print(f'(L={luminosity.to("Lsun"):.3e})')
    return integrated_flux


def get_reference_spectrum(filename: Path | str) -> tuple[pd.DataFrame, dict[str, t.Any]]:
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
        delim_whitespace=True,
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


def write_flambda_spectra(modelpath: Path, args: argparse.Namespace) -> None:
    """Write out spectra to text files.

    Writes lambda_angstroms and f_lambda to .txt files for all timesteps and create
    a text file containing the time in days for each timestep.
    """
    outdirectory = Path(modelpath, "spectra")

    outdirectory.mkdir(parents=True, exist_ok=True)

    if Path(modelpath, "specpol.out").is_file():
        specfilename = modelpath / "specpol.out"
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.to_numpy()[1:] if i[-2] != "."]
    else:
        specfilename = at.firstexisting("spec.out", folder=modelpath, tryzipped=True)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.to_numpy()[1:]

    number_of_timesteps = len(timearray)

    if not args.timestep:
        args.timestep = f"0-{number_of_timesteps - 1}"

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )

    with (outdirectory / "spectra_list.txt").open("w+") as spectra_list:
        arr_tmid = at.get_timestep_times(modelpath, loc="mid")

        for timestep in range(timestepmin, timestepmax + 1):
            dfspectrum = get_spectrum(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[-1]
            tmid = arr_tmid[timestep]

            outfilepath = outdirectory / f"spectrum_ts{timestep:02.0f}_{tmid:.2f}d.txt"

            with outfilepath.open("w") as spec_file:
                spec_file.write("#lambda f_lambda_1Mpc\n")
                spec_file.write("#[A] [erg/s/cm2/A]\n")

                dfspectrum.to_csv(
                    spec_file, header=False, sep=" ", index=False, columns=["lambda_angstroms", "f_lambda"]
                )

            spectra_list.write(str(outfilepath.absolute()) + "\n")

    with (outdirectory / "time_list.txt").open("w+") as time_list:
        for time in arr_tmid:
            time_list.write(f"{time} \n")

    print(f"Saved in {outdirectory}")
