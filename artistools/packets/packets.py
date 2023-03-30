#!/usr/bin/env python3
import calendar
import gzip
import math
import multiprocessing
import os
import sys
from collections.abc import Collection
from collections.abc import Sequence
from functools import lru_cache
from functools import partial
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

import artistools as at

# import matplotlib.patches as mpatches
# from collections import namedtuple

CLIGHT = 2.99792458e10
DAY = 86400

types = {
    10: "TYPE_GAMMA",
    11: "TYPE_RPKT",
    20: "TYPE_NTLEPTON",
    32: "TYPE_ESCAPE",
}

type_ids = dict((v, k) for k, v in types.items())

EMTYPE_NOTSET = -9999000
EMTYPE_FREEFREE = -9999999


@lru_cache(maxsize=16)
def get_column_names_artiscode(modelpath: Union[str, Path]) -> Optional[list[str]]:
    modelpath = Path(modelpath)
    if Path(modelpath, "artis").is_dir():
        print("detected artis code directory")
        packet_properties = []
        inputfilename = at.firstexisting(["packet_init.cc", "packet_init.c"], folder=(modelpath / "artis"))
        print(f"found {inputfilename}: getting packet column names from artis code")
        with open(inputfilename) as inputfile:
            packet_print_lines = [line.split(",") for line in inputfile if "fprintf(packets_file," in line]
            for line in packet_print_lines:
                for element in line:
                    if "pkt[i]." in element:
                        packet_properties.append(element)

        for i, element in enumerate(packet_properties):
            packet_properties[i] = element.split(".")[1].split(")")[0]

        columns = packet_properties
        replacements_dict = {
            "type": "type_id",
            "pos[0]": "posx",
            "pos[1]": "posy",
            "pos[2]": "posz",
            "dir[0]": "dirx",
            "dir[1]": "diry",
            "dir[2]": "dirz",
            "escape_type": "escape_type_id",
            "em_pos[0]": "em_posx",
            "em_pos[1]": "em_posy",
            "em_pos[2]": "em_posz",
            "absorptiontype": "absorption_type",
            "absorptionfreq": "absorption_freq",
            "absorptiondir[0]": "absorptiondirx",
            "absorptiondir[1]": "absorptiondiry",
            "absorptiondir[2]": "absorptiondirz",
            "stokes[0]": "stokes1",
            "stokes[1]": "stokes2",
            "stokes[2]": "stokes3",
            "pol_dir[0]": "pol_dirx",
            "pol_dir[1]": "pol_diry",
            "pol_dir[2]": "pol_dirz",
            "trueemissionvelocity": "true_emission_velocity",
        }

        for i, column_name in enumerate(columns):
            if column_name in replacements_dict:
                columns[i] = replacements_dict[column_name]

        return columns

    return None


def add_derived_columns(
    dfpackets: pd.DataFrame,
    modelpath: Path,
    colnames: Sequence[str],
    allnonemptymgilist: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    cm_to_km = 1e-5
    day_in_s = 86400
    if dfpackets.empty:
        return dfpackets

    colnames = at.makelist(colnames)

    def em_modelgridindex(packet) -> Union[int, float]:
        return at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )

    def emtrue_modelgridindex(packet) -> Union[int, float]:
        return at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.true_emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )

    def em_timestep(packet) -> int:
        return at.get_timestep_of_timedays(modelpath, packet.em_time / day_in_s)

    def emtrue_timestep(packet) -> int:
        return at.get_timestep_of_timedays(modelpath, packet.trueem_time / day_in_s)

    if "emission_velocity" in colnames:
        dfpackets.eval("emission_velocity = sqrt(em_posx ** 2 + em_posy ** 2 + em_posz ** 2) / em_time", inplace=True)

        dfpackets.eval("em_velx = em_posx / em_time", inplace=True)
        dfpackets.eval("em_vely = em_posy / em_time", inplace=True)
        dfpackets.eval("em_velz = em_posz / em_time", inplace=True)

    if "em_modelgridindex" in colnames:
        if "emission_velocity" not in dfpackets.columns:
            dfpackets = add_derived_columns(
                dfpackets, modelpath, ["emission_velocity"], allnonemptymgilist=allnonemptymgilist
            )
        dfpackets["em_modelgridindex"] = dfpackets.apply(em_modelgridindex, axis=1)

    if "emtrue_modelgridindex" in colnames:
        dfpackets["emtrue_modelgridindex"] = dfpackets.apply(emtrue_modelgridindex, axis=1)

    if "em_timestep" in colnames:
        dfpackets["em_timestep"] = dfpackets.apply(em_timestep, axis=1)

    if any([x in colnames for x in ["angle_bin", "dirbin", "costhetabin", "phibin"]]):
        dfpackets = bin_packet_directions(modelpath, dfpackets)

    return dfpackets


def readfile_text(packetsfile: Union[Path, str], modelpath: Path = Path(".")) -> pl.DataFrame:
    usecols_nodata = None  # print a warning for missing columns if the source code columns can't be read

    skiprows: int = 0
    column_names: Optional[list[str]] = None
    try:
        fpackets = at.zopen(packetsfile, "rt")

        datastartpos = fpackets.tell()  # will be updated if this was actually the start of a header
        firstline = fpackets.readline()

        if firstline.lstrip().startswith("#"):
            column_names = firstline.lstrip("#").split()
            assert column_names is not None
            column_names.append("ignore")
            # get the column count from the first data line to check header matched
            datastartpos = fpackets.tell()
            dataline = fpackets.readline()
            inputcolumncount = len(dataline.split())
            skiprows = 1
        else:
            inputcolumncount = len(firstline.split())

        fpackets.seek(datastartpos)  # go to first data line

    except gzip.BadGzipFile:
        print(f"\nBad Gzip File: {packetsfile}")
        raise gzip.BadGzipFile

    try:
        dfpackets = pd.read_csv(
            fpackets,
            sep=" ",
            header=None,
            skiprows=skiprows,
            names=column_names,
            skip_blank_lines=True,
            engine="pyarrow",
        )
    except Exception as e:
        print(f"Error occured in file {packetsfile}")
        raise e

    # space at the end of line made an extra column of Nones
    if dfpackets[dfpackets.columns[-1]].isnull().all():
        dfpackets.drop(labels=dfpackets.columns[-1], axis=1, inplace=True)

    if hasattr(dfpackets.columns[0], "startswith") and dfpackets.columns[0].startswith("#"):
        dfpackets.rename(columns={dfpackets.columns[0]: dfpackets.columns[0].lstrip("#")}, inplace=True)

    elif dfpackets.columns[0] in ["C0", 0]:
        inputcolumncount = len(dfpackets.columns)
        column_names = get_column_names_artiscode(modelpath)
        if column_names:  # found them in the artis code files
            assert len(column_names) == inputcolumncount

        else:  # infer from column positions
            # new artis added extra columns to the end of this list, but they may be absent in older versions
            # the packets file may have a truncated set of columns, but we assume that they
            # are only truncated, i.e. the columns with the same index have the same meaning
            columns_full = [
                "number",
                "where",
                "type_id",
                "posx",
                "posy",
                "posz",
                "dirx",
                "diry",
                "dirz",
                "last_cross",
                "tdecay",
                "e_cmf",
                "e_rf",
                "nu_cmf",
                "nu_rf",
                "escape_type_id",
                "escape_time",
                "scat_count",
                "next_trans",
                "interactions",
                "last_event",
                "emissiontype",
                "trueemissiontype",
                "em_posx",
                "em_posy",
                "em_posz",
                "absorption_type",
                "absorption_freq",
                "nscatterings",
                "em_time",
                "absorptiondirx",
                "absorptiondiry",
                "absorptiondirz",
                "stokes1",
                "stokes2",
                "stokes3",
                "pol_dirx",
                "pol_diry",
                "pol_dirz",
                "originated_from_positron",
                "true_emission_velocity",
                "trueem_time",
                "pellet_nucindex",
            ]

            assert len(columns_full) >= inputcolumncount
            usecols_nodata = [n for n in columns_full if columns_full.index(n) >= inputcolumncount]
            column_names = columns_full[:inputcolumncount]

        dfpackets.columns = column_names

    # except Exception as ex:
    #     print(f'Problem with file {packetsfile}')
    #     print(f'ERROR: {ex}')
    #     sys.exit(1)

    if usecols_nodata:
        print(f"WARNING: no data in packets file for columns: {usecols_nodata}")
        for col in usecols_nodata:
            dfpackets[col] = float("NaN")

    return pl.from_pandas(dfpackets)


def readfile(
    packetsfile: Union[Path, str],
    type: Optional[str] = None,
    escape_type: Optional[Literal["TYPE_RPKT", "TYPE_GAMMA"]] = None,
) -> pd.DataFrame:
    """Read a packet file into a Pandas DataFrame."""
    return readfile_lazypolars(packetsfile, type=type, escape_type=escape_type).collect().to_pandas()


def readfile_lazypolars(
    packetsfile: Union[Path, str],
    modelpath: Union[None, Path, str] = None,
    type: Optional[str] = None,
    escape_type: Optional[Literal["TYPE_RPKT", "TYPE_GAMMA"]] = None,
) -> pl.LazyFrame:
    """Read a packet file into a Polars lazy DataFrame."""
    packetsfile = Path(packetsfile)

    packetsfileparquet = (
        packetsfile
        if packetsfile.suffixes == [".out", ".parquet"]
        else at.stripallsuffixes(packetsfile).with_suffix(".out.parquet")
    )
    packetsfiletext = (
        packetsfile
        if packetsfile.suffixes in [[".out"], [".out", ".gz"], [".out", ".xz"]]
        else at.firstexisting([at.stripallsuffixes(packetsfile).with_suffix(".out")])
    )
    write_parquet = True

    dfpackets = None
    if packetsfile == packetsfileparquet and os.path.getmtime(packetsfileparquet) > calendar.timegm(
        (2023, 3, 22, 0, 0, 0)
    ):
        try:
            dfpackets = pl.scan_parquet(packetsfileparquet)
            write_parquet = False
        except Exception as e:
            print(f"Error occured in file {packetsfile}. Reading from text version.")

    if dfpackets is None:
        dfpackets = readfile_text(packetsfiletext).lazy()

    if "t_arrive_d" not in dfpackets.columns:
        dfpackets = dfpackets.with_columns(
            [
                (
                    (
                        pl.col("escape_time")
                        - (
                            pl.col("posx") * pl.col("dirx")
                            + pl.col("posy") * pl.col("diry")
                            + pl.col("posz") * pl.col("dirz")
                        )
                        / 29979245800.0
                    )
                    / 86400.0
                ).alias("t_arrive_d"),
            ]
        )
        write_parquet = True

    if write_parquet:
        print(f"Saving {packetsfileparquet}")
        dfpackets = dfpackets.sort(by=["type_id", "escape_type_id", "t_arrive_d"])
        dfpackets.collect().write_parquet(packetsfileparquet, compression="lz4", use_pyarrow=True, statistics=True)

    if escape_type is not None:
        assert type is None or type == "TYPE_ESCAPE"
        dfpackets = dfpackets.filter(
            (pl.col("type_id") == type_ids["TYPE_ESCAPE"]) & (pl.col("escape_type_id") == type_ids[escape_type])
        )
    elif type is not None and type != "":
        dfpackets = dfpackets.filter(pl.col("type_id") == type_ids["TYPE_ESCAPE"])

    if modelpath is not None:
        dfpackets = bin_packet_directions_lazypolars(modelpath, dfpackets)

    return dfpackets


def get_packetsfilepaths(modelpath: Union[str, Path], maxpacketfiles: Optional[int] = None) -> list[Path]:
    def preferred_alternative(f: Path) -> bool:
        f_nosuffixes = at.stripallsuffixes(f)

        suffix_priority = [[".out", ".gz"], [".out", ".xz"], [".out", ".parquet"]]
        if f.suffixes in suffix_priority:
            startindex = suffix_priority.index(f.suffixes) + 1
        else:
            startindex = 0

        if any(f_nosuffixes.with_suffix("".join(s)).is_file() for s in suffix_priority[startindex:]):
            return True
        return False

    packetsfiles = sorted(
        list(Path(modelpath).glob("packets00_*.out*")) + list(Path(modelpath, "packets").glob("packets00_*.out*"))
    )

    # strip out duplicates in the case that some are stored as binary and some are text files
    packetsfiles = [f for f in packetsfiles if not preferred_alternative(f)]

    if maxpacketfiles is not None and maxpacketfiles > 0 and len(packetsfiles) > maxpacketfiles:
        print(f"Reading from the first {maxpacketfiles} of {len(packetsfiles)} packets files")
        packetsfiles = packetsfiles[:maxpacketfiles]
    else:
        print(f"Reading from {len(packetsfiles)} packets files")

    return packetsfiles


def get_pldfpackets(
    modelpath: Union[str, Path],
    maxpacketfiles: Optional[int] = None,
    type: Optional[str] = None,
    escape_type: Optional[Literal["TYPE_RPKT", "TYPE_GAMMA"]] = None,
) -> tuple[int, pl.LazyFrame]:
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles)

    nprocs_read = len(packetsfiles)
    # allescrpktfile = Path(modelpath) / "packets_rpkt_escaped.parquet"
    # write_allescrpkt_parquet = False

    # if maxpacketfiles is None and type == "TYPE_ESCAPE" and escape_type == "TYPE_RPKT":
    #     if allescrpktfile.is_file():
    #         print(f"Reading from {allescrpktfile}")
    #         # use_statistics is causing some weird errors! (zero flux spectra)
    #         pllfpackets = pl.scan_parquet(allescrpktfile, use_statistics=False)
    #         return nprocs_read, pllfpackets
    #     else:
    #         write_allescrpkt_parquet = True

    pllfpackets = pl.concat(
        [
            at.packets.readfile_lazypolars(packetsfile, modelpath=modelpath, type=type, escape_type=escape_type)
            for packetsfile in packetsfiles
        ],
        how="vertical",
        rechunk=False,
    )

    # pllfpackets = pl.scan_parquet(Path(modelpath) / "packets" / "packets00_*.out.parquet")

    # Luke: this turned out to be slower than reading 960 or 3840 parquet files separately
    # if write_allescrpkt_parquet:
    #     print(f"Saving {allescrpktfile}")
    #     # sorting can be extremely slow (and exceed RAM) but it probably speeds up access
    #     # pllfpackets = pllfpackets.sort(by=["type_id", "escape_type_id", "t_arrive_d"])
    #     pllfpackets.sink_parquet(allescrpktfile, compression="lz4", row_group_size=1024 * 1024)

    return nprocs_read, pllfpackets


def get_directionbin(
    dirx: float, diry: float, dirz: float, nphibins: int, ncosthetabins: int, syn_dir: Sequence[float]
) -> int:
    dirmag = np.sqrt(dirx**2 + diry**2 + dirz**2)
    pkt_dir = [dirx / dirmag, diry / dirmag, dirz / dirmag]
    costheta = np.dot(pkt_dir, syn_dir)
    thetabin = int((costheta + 1.0) / 2.0 * ncosthetabins)

    xhat = np.array([1.0, 0.0, 0.0])
    vec1 = np.cross(pkt_dir, syn_dir)
    vec2 = np.cross(xhat, syn_dir)
    cosphi = np.dot(vec1, vec2) / at.vec_len(vec1) / at.vec_len(vec2)

    vec3 = np.cross(vec2, syn_dir)
    testphi = np.dot(vec1, vec3)

    if testphi > 0:
        phibin = int(math.acos(cosphi) / 2.0 / np.pi * nphibins)
    else:
        phibin = int((math.acos(cosphi) + np.pi) / 2.0 / np.pi * nphibins)

    na = (thetabin * nphibins) + phibin
    return na


def bin_packet_directions_lazypolars(modelpath: Union[Path, str], dfpackets: pl.LazyFrame) -> pl.LazyFrame:
    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()

    syn_dir = at.get_syn_dir(Path(modelpath))
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)

    dfpackets = dfpackets.with_columns(
        (pl.col("dirx") ** 2 + pl.col("diry") ** 2 + pl.col("dirz") ** 2).sqrt().alias("dirmag"),
    )
    dfpackets = dfpackets.with_columns(
        (
            (pl.col("dirx") * syn_dir[0] + pl.col("diry") * syn_dir[1] + pl.col("dirz") * syn_dir[2]) / pl.col("dirmag")
        ).alias("costheta"),
    )
    dfpackets = dfpackets.with_columns(
        ((pl.col("costheta") + 1) / 2.0 * ncosthetabins).cast(pl.Int64).alias("costhetabin"),
    )
    dfpackets = dfpackets.with_columns(
        ((pl.col("diry") * syn_dir[2] - pl.col("dirz") * syn_dir[1]) / pl.col("dirmag")).alias("vec1_x"),
        ((pl.col("dirz") * syn_dir[0] - pl.col("dirx") * syn_dir[2]) / pl.col("dirmag")).alias("vec1_y"),
        ((pl.col("dirx") * syn_dir[1] - pl.col("diry") * syn_dir[0]) / pl.col("dirmag")).alias("vec1_z"),
    )

    dfpackets = dfpackets.with_columns(
        (
            (pl.col("vec1_x") * vec2[0] + pl.col("vec1_y") * vec2[1] + pl.col("vec1_z") * vec2[2])
            / (pl.col("vec1_x") ** 2 + pl.col("vec1_y") ** 2 + pl.col("vec1_z") ** 2).sqrt()
            / float(np.linalg.norm(vec2))
        ).alias("cosphi"),
    )

    # vec1 = dir cross syn_dir
    dfpackets = dfpackets.with_columns(
        ((pl.col("diry") * syn_dir[2] - pl.col("dirz") * syn_dir[1]) / pl.col("dirmag")).alias("vec1_x"),
        ((pl.col("dirz") * syn_dir[0] - pl.col("dirx") * syn_dir[2]) / pl.col("dirmag")).alias("vec1_y"),
        ((pl.col("dirx") * syn_dir[1] - pl.col("diry") * syn_dir[0]) / pl.col("dirmag")).alias("vec1_z"),
    )

    vec3 = np.cross(vec2, syn_dir)

    # arr_testphi = np.dot(arr_vec1, vec3)
    dfpackets = dfpackets.with_columns(
        (
            (pl.col("vec1_x") * vec3[0] + pl.col("vec1_y") * vec3[1] + pl.col("vec1_z") * vec3[2]) / pl.col("dirmag")
        ).alias("testphi"),
    )

    dfpackets = dfpackets.with_columns(
        (
            pl.when(pl.col("testphi") > 0)
            .then(pl.col("cosphi").arccos() / 2.0 / np.pi * nphibins)
            .otherwise((pl.col("cosphi").arccos() + np.pi) / 2.0 / np.pi * nphibins)
        )
        .cast(pl.Int64)
        .alias("phibin"),
    )
    dfpackets = dfpackets.with_columns(
        (pl.col("costhetabin") * nphibins + pl.col("phibin")).alias("dirbin"),
    )

    return dfpackets


def bin_packet_directions(modelpath: Union[Path, str], dfpackets: pd.DataFrame) -> pd.DataFrame:
    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()

    syn_dir = at.get_syn_dir(Path(modelpath))
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)

    pktdirvecs = dfpackets[["dirx", "diry", "dirz"]].values

    # normalise. might not be needed
    dirmags = np.linalg.norm(pktdirvecs, axis=1)
    pktdirvecs /= np.array([dirmags, dirmags, dirmags]).transpose()

    costheta = np.dot(pktdirvecs, syn_dir)
    arr_costhetabin = ((costheta + 1) / 2.0 * ncosthetabins).astype(int)
    dfpackets["costhetabin"] = arr_costhetabin

    arr_vec1 = np.cross(pktdirvecs, syn_dir)
    arr_cosphi = np.dot(arr_vec1, vec2) / np.linalg.norm(arr_vec1, axis=1) / np.linalg.norm(vec2)
    vec3 = np.cross(vec2, syn_dir)
    arr_testphi = np.dot(arr_vec1, vec3)

    arr_phibin = np.zeros(len(pktdirvecs), dtype=int)
    filta = arr_testphi > 0
    arr_phibin[filta] = np.arccos(arr_cosphi[filta]) / 2.0 / np.pi * nphibins
    filtb = np.invert(filta)
    arr_phibin[filtb] = (np.arccos(arr_cosphi[filtb]) + np.pi) / 2.0 / np.pi * nphibins
    dfpackets["phibin"] = arr_phibin
    dfpackets["arccoscosphi"] = np.arccos(arr_cosphi)

    dfpackets["dirbin"] = (arr_costhetabin * nphibins) + arr_phibin

    assert np.all(dfpackets["dirbin"] < at.get_viewingdirectionbincount())

    return dfpackets


def make_3d_histogram_from_packets(modelpath, timestep_min, timestep_max=None, em_time=True):
    if timestep_max is None:
        timestep_max = timestep_min
    modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)

    timeminarray = at.get_timestep_times_float(modelpath=modelpath, loc="start")
    timedeltaarray = at.get_timestep_times_float(modelpath=modelpath, loc="delta")
    timemaxarray = at.get_timestep_times_float(modelpath=modelpath, loc="end")

    # timestep = 63 # 82 73 #63 #54 46 #27
    # print([(ts, time) for ts, time in enumerate(timeminarray)])
    if em_time:
        print("Binning by packet emission time")
    else:
        print("Binning by packet arrival time")

    packetsfiles = at.packets.get_packetsfilepaths(modelpath)

    emission_position3d = [[], [], []]
    e_rf = []
    e_cmf = []

    for npacketfile in range(0, len(packetsfiles)):
        # for npacketfile in range(0, 1):
        dfpackets = at.packets.readfile(packetsfiles[npacketfile])
        at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])
        dfpackets = dfpackets.dropna(subset=["emission_velocity"])  # drop rows where emission_vel is NaN

        only_packets_0_scatters = False
        if only_packets_0_scatters:
            print("Only using packets with 0 scatters")
            # print(dfpackets[['scat_count', 'interactions', 'nscatterings']])
            dfpackets.query("nscatterings == 0", inplace=True)

        # print(dfpackets[['emission_velocity', 'em_velx', 'em_vely', 'em_velz']])
        # select only type escape and type r-pkt (don't include gamma-rays)
        dfpackets.query(
            f'type_id == {type_ids["TYPE_ESCAPE"]} and escape_type_id == {type_ids["TYPE_RPKT"]}', inplace=True
        )
        if em_time:
            dfpackets.query("@timeminarray[@timestep_min] < em_time/@DAY < @timemaxarray[@timestep_max]", inplace=True)
        else:  # packet arrival time
            dfpackets.query("@timeminarray[@timestep_min] < t_arrive_d < @timemaxarray[@timestep_max]", inplace=True)

        emission_position3d[0].extend([em_velx / CLIGHT for em_velx in dfpackets["em_velx"]])
        emission_position3d[1].extend([em_vely / CLIGHT for em_vely in dfpackets["em_vely"]])
        emission_position3d[2].extend([em_velz / CLIGHT for em_velz in dfpackets["em_velz"]])

        e_rf.extend([e_rf for e_rf in dfpackets["e_rf"]])
        e_cmf.extend([e_cmf for e_cmf in dfpackets["e_cmf"]])

    emission_position3d = np.array(emission_position3d)
    weight_by_energy = True
    if weight_by_energy:
        e_rf = np.array(e_rf)
        e_cmf = np.array(e_cmf)
        # weights = e_rf
        weights = e_cmf
    else:
        weights = None

    print(emission_position3d.shape)
    print(emission_position3d[0].shape)

    # print(emission_position3d)
    grid_3d, _, _, _ = make_3d_grid(modeldata, vmax_cms)
    print(grid_3d.shape)
    # https://stackoverflow.com/questions/49861468/binning-random-data-to-regular-3d-grid-with-unequal-axis-lengths
    hist, _ = np.histogramdd(emission_position3d.T, [np.append(ax, np.inf) for ax in grid_3d], weights=weights)
    # print(hist.shape)
    if weight_by_energy:
        # Divide binned energies by number of processes and by length of timestep
        hist = (
            hist / len(packetsfiles) / (timemaxarray[timestep_max] - timeminarray[timestep_min])
        )  # timedeltaarray[timestep]  # histogram weighted by energy
    # - need to divide by number of processes
    # and length of timestep(s)

    # # print histogram coordinates
    # coords = np.nonzero(hist)
    # for i, j, k in zip(*coords):
    #     print(f'({grid_3d[0][i]}, {grid_3d[1][j]}, {grid_3d[2][k]}): {hist[i][j][k]}')

    return hist


def make_3d_grid(modeldata, vmax_cms):
    # modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)
    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    xgrid = np.zeros(grid)
    vmax = vmax_cms / CLIGHT
    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    x, y, z = np.meshgrid(xgrid, xgrid, xgrid)
    grid_3d = np.array([xgrid, xgrid, xgrid])
    # grid_Te = np.zeros((grid, grid, grid))
    # print(grid_Te.shape)
    return grid_3d, x, y, z


def get_mean_packet_emission_velocity_per_ts(
    modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT", maxpacketfiles=None, escape_angles=None
) -> pd.DataFrame:
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    timearray = at.get_timestep_times_float(modelpath=modelpath, loc="mid")
    arr_timedelta = at.get_timestep_times_float(modelpath=modelpath, loc="delta")
    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    dfpackets_escape_velocity_and_arrive_time = pd.DataFrame
    emission_data = pd.DataFrame(
        {"t_arrive_d": timearray, "mean_emission_velocity": np.zeros_like(timearray, dtype=float)}
    )

    for i, packetsfile in enumerate(packetsfiles):
        dfpackets = at.packets.readfile(packetsfile, type=packet_type, escape_type=escape_type)
        at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])
        if escape_angles is not None:
            dfpackets = at.packets.bin_packet_directions(modelpath, dfpackets)
            dfpackets.query("dirbin == @escape_angles", inplace=True)

        if i == 0:  # make new df
            dfpackets_escape_velocity_and_arrive_time = dfpackets[["t_arrive_d", "emission_velocity"]]
        else:  # append to df
            # dfpackets_escape_velocity_and_arrive_time = dfpackets_escape_velocity_and_arrive_time.append(
            #     other=dfpackets[["t_arrive_d", "emission_velocity"]], ignore_index=True
            # )
            dfpackets_escape_velocity_and_arrive_time = pd.concat(
                [dfpackets_escape_velocity_and_arrive_time, dfpackets[["t_arrive_d", "emission_velocity"]]],
                ignore_index=True,
            )

    print(dfpackets_escape_velocity_and_arrive_time)
    binned = pd.cut(
        dfpackets_escape_velocity_and_arrive_time["t_arrive_d"], timearrayplusend, labels=False, include_lowest=True
    )
    for binindex, emission_velocity in (
        dfpackets_escape_velocity_and_arrive_time.groupby(binned)["emission_velocity"].mean().iteritems()
    ):
        emission_data["mean_emission_velocity"][binindex] += emission_velocity  # / 2.99792458e10

    return emission_data


def bin_and_sum(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    bincol: str,
    bins: list[Union[float, int]],
    sumcols: list[str] = [],
    getcounts: bool = False,
) -> pd.DataFrame:
    """bins is a list of lower edges, and the final upper edge"""

    # dfout = pl.DataFrame(pl.Series(bincol + "_bin", np.arange(0, len(bins) - 1)))
    dfout = pd.DataFrame({bincol + "_bin": np.arange(0, len(bins) - 1)})

    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=True)

    # POLARS METHOD (slower than pandas for some reason)
    # df = df.with_columns(
    #     [
    #         (
    #             df[bincol]
    #             .cut(
    #                 bins=list(bins),
    #                 category_label=bincol + "_bin",
    #                 maintain_order=True,
    #             )
    #             .get_column(bincol + "_bin")
    #             .cast(pl.Int64)
    #             - 1
    #         )
    #     ]
    # )
    # aggs = [pl.col(col).sum().alias(col + "_sum") for col in sumcols]

    # if getcounts:
    #     aggs.append(pl.col(bincol).count().alias("count"))

    # wlbins = df.groupby(bincol + "_bin").agg(aggs)

    # # now we will include the empty bins
    # dfout = dfout.join(wlbins, how="left", on=bincol + "_bin").fill_null(0.0)

    # PANDAS METHOD

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    pdbins = pd.cut(
        x=df[bincol],
        bins=bins,
        right=True,
        labels=range(len(bins) - 1),
        include_lowest=True,
    )

    if sumcols is not None:
        for col in sumcols:
            # dfout = dfout.with_columns(
            #     [pl.Series(col + "_sum", df[col].groupby(pdbins).sum().values) for col in sumcols]
            # )
            dfout[col + "_sum"] = df[col].groupby(pdbins).sum().values

    if getcounts:
        # dfout = dfout.with_columns([pl.Series("count", df[bincol].groupby(pdbins).count().values)])
        dfout["count"] = df[bincol].groupby(pdbins).count().values

    return dfout
