import calendar
import gzip
import math
import multiprocessing
import typing as t
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import artistools as at

# for the parquet files
time_parquetschemachange = (2024, 2, 16, 11, 0, 0)

CLIGHT = 2.99792458e10
DAY = 86400

types = {
    10: "TYPE_GAMMA",
    11: "TYPE_RPKT",
    20: "TYPE_NTLEPTON",
    32: "TYPE_ESCAPE",
}

type_ids = {v: k for k, v in types.items()}

EMTYPE_NOTSET = -9999000
EMTYPE_FREEFREE = -9999999

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


@lru_cache(maxsize=16)
def get_column_names_artiscode(modelpath: str | Path) -> list[str] | None:
    modelpath = Path(modelpath)
    if Path(modelpath, "artis").is_dir():
        print("detected artis code directory")
        packet_properties: list[str] = []
        inputfilename = at.firstexisting(["packet_init.cc", "packet_init.c"], folder=modelpath / "artis")
        print(f"found {inputfilename}: getting packet column names from artis code:")
        with inputfilename.open() as inputfile:
            packet_print_lines = [line.split(",") for line in inputfile if "fprintf(packets_file," in line]
            for line in packet_print_lines:
                packet_properties.extend(element for element in line if "pkt[i]." in element)
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
        print(columns)

        return columns

    return None


def add_derived_columns(
    dfpackets: pd.DataFrame,
    modelpathin: Path | str,
    colnames: t.Sequence[str],
    allnonemptymgilist: t.Sequence[int] | None = None,
) -> pd.DataFrame:
    """Add columns to a packets DataFrame that are derived from the values that are stored in the packets files."""
    modelpath = Path(modelpathin)
    cm_to_km = 1e-5
    day_in_s = 86400
    if dfpackets.empty:
        return dfpackets

    colnames = at.makelist(colnames)
    dimensions = at.get_inputparams(modelpath)["n_dimensions"]

    def em_modelgridindex(packet) -> int | float:
        assert dimensions == 1

        return at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )

    def emtrue_modelgridindex(packet) -> int | float:
        assert dimensions == 1

        return at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.true_emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )

    def em_timestep(packet) -> int:
        return at.get_timestep_of_timedays(modelpath, packet.em_time / day_in_s)

    def emtrue_timestep(packet) -> int:
        return at.get_timestep_of_timedays(modelpath, packet.trueem_time / day_in_s)

    if "emission_velocity" in colnames:
        dfpackets["emission_velocity"] = (
            np.sqrt(dfpackets["em_posx"] ** 2 + dfpackets["em_posy"] ** 2 + dfpackets["em_posz"] ** 2)
            / dfpackets["em_time"]
        )

        dfpackets["em_velx"] = dfpackets["em_posx"] / dfpackets["em_time"]
        dfpackets["em_vely"] = dfpackets["em_posy"] / dfpackets["em_time"]
        dfpackets["em_velz"] = dfpackets["em_posz"] / dfpackets["em_time"]

    if "em_modelgridindex" in colnames:
        if "emission_velocity" not in dfpackets.columns:
            dfpackets = add_derived_columns(
                dfpackets, modelpath, ["emission_velocity"], allnonemptymgilist=allnonemptymgilist
            )
        dfpackets["em_modelgridindex"] = dfpackets.apply(em_modelgridindex, axis=1)

    if "emtrue_modelgridindex" in colnames:
        dfpackets["emtrue_modelgridindex"] = dfpackets.apply(emtrue_modelgridindex, axis=1)

    if "emtrue_timestep" in colnames:
        dfpackets["emtrue_timestep"] = dfpackets.apply(emtrue_timestep, axis=1)

    if "em_timestep" in colnames:
        dfpackets["em_timestep"] = dfpackets.apply(em_timestep, axis=1)

    if any(x in colnames for x in ["angle_bin", "dirbin", "costhetabin", "phibin"]):
        dfpackets = bin_packet_directions(modelpath, dfpackets)

    return dfpackets


def add_derived_columns_lazy(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    modelmeta: dict[str, t.Any] | None = None,
    dfmodel: pd.DataFrame | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Add columns to a packets DataFrame that are derived from the values that are stored in the packets files.

    We might as well add everything, since the columns only get calculated when they are actually used (polars LazyFrame).
    """
    dfpackets = dfpackets.lazy().with_columns(
        [
            (
                (pl.col("em_posx") ** 2 + pl.col("em_posy") ** 2 + pl.col("em_posz") ** 2).sqrt() / pl.col("em_time")
            ).alias("emission_velocity")
        ]
    )

    dfpackets = dfpackets.with_columns(
        [
            (
                (
                    (pl.col("em_posx") * pl.col("dirx")) ** 2
                    + (pl.col("em_posy") * pl.col("diry")) ** 2
                    + (pl.col("em_posz") * pl.col("dirz")) ** 2
                ).sqrt()
                / pl.col("em_time")
            ).alias("emission_velocity_lineofsight")
        ]
    )

    if modelmeta is None:
        return dfpackets

    if modelmeta["dimensions"] > 1:
        t_model_s = modelmeta["t_model_init_days"] * 86400.0
        vmax = modelmeta["vmax_cmps"]

        if modelmeta["dimensions"] == 2:
            vwidthrcyl = modelmeta["wid_init_rcyl"] / t_model_s
            vwidthz = modelmeta["wid_init_z"] / t_model_s
            dfpackets = dfpackets.with_columns(
                [
                    ((pl.col("em_posx").pow(2) + pl.col("em_posy").pow(2)).sqrt() / pl.col("em_time") / vwidthrcyl)
                    .cast(pl.Int32)
                    .alias("coordpointnumrcyl"),
                    ((pl.col("em_posz") / pl.col("em_time") + vmax) / vwidthz).cast(pl.Int32).alias("coordpointnumz"),
                ]
            )
            dfpackets = dfpackets.with_columns(
                [
                    (pl.col("coordpointnumz") * modelmeta["ncoordgridrcyl"] + pl.col("coordpointnumrcyl")).alias(
                        "em_modelgridindex"
                    )
                ]
            )

        elif modelmeta["dimensions"] == 3:
            vwidth = modelmeta["wid_init"] / t_model_s
            dfpackets = dfpackets.with_columns(
                [
                    ((pl.col(f"em_pos{ax}") / pl.col("em_time") + vmax) / vwidth)
                    .cast(pl.Int32)
                    .alias(f"coordpointnum{ax}")
                    for ax in ["x", "y", "z"]
                ]
            )
            dfpackets = dfpackets.with_columns(
                [
                    (
                        pl.col("coordpointnumz") * modelmeta["ncoordgridy"] * modelmeta["ncoordgridx"]
                        + pl.col("coordpointnumy") * modelmeta["ncoordgridx"]
                        + pl.col("coordpointnumx")
                    ).alias("em_modelgridindex")
                ]
            )
    elif modelmeta["dimensions"] == 1:
        assert dfmodel is not None, "dfmodel must be provided for 1D models to set em_modelgridindex"
        velbins = (dfmodel.select("vel_r_max_kmps").lazy().collect()["vel_r_max_kmps"] * 1000.0).to_list()
        dfpackets = dfpackets.with_columns(
            (
                pl.col("emission_velocity")
                .cut(breaks=list(velbins), labels=[str(x) for x in range(-1, len(velbins))])
                .cast(str)
                .cast(pl.Int32)
            ).alias("em_modelgridindex")
        )

    return dfpackets


def readfile_text(packetsfile: Path | str, modelpath: Path = Path()) -> pl.DataFrame:
    """Read a packets*.out(.xz/.zstd) space-separated text file into a polars DataFrame."""
    print(f"Reading {packetsfile}")
    column_names: list[str] | None = None
    try:
        fpackets = at.zopen(packetsfile, mode="r")

        datastartpos = fpackets.tell()  # will be updated if this was actually the start of a header
        firstline = fpackets.readline().decode()

        if firstline.lstrip().startswith("#"):
            column_names = firstline.lstrip("#").split()
            assert column_names is not None

            # get the column count from the first data line to check header matched
            datastartpos = fpackets.tell()
            dataline = fpackets.readline().decode()
            inputcolumncount = len(dataline.split())
            assert inputcolumncount == len(column_names)
        else:
            inputcolumncount = len(firstline.split())
            column_names = get_column_names_artiscode(modelpath)
            if column_names:  # found them in the artis code files
                assert len(column_names) == inputcolumncount

            else:  # infer from column positions
                assert len(columns_full) >= inputcolumncount
                column_names = columns_full[:inputcolumncount]

        fpackets.seek(datastartpos)  # go to first data line

    except gzip.BadGzipFile:
        print(f"\nBad Gzip File: {packetsfile}")
        raise

    dtype_overrides = {
        "absorption_freq": pl.Float32,
        "absorption_type": pl.Int32,
        "absorptiondirx": pl.Float32,
        "absorptiondiry": pl.Float32,
        "absorptiondirz": pl.Float32,
        "e_cmf": pl.Float64,
        "e_rf": pl.Float64,
        "em_posx": pl.Float32,
        "em_posy": pl.Float32,
        "em_posz": pl.Float32,
        "em_time": pl.Float32,
        "emissiontype": pl.Int32,
        "escape_time": pl.Float32,
        "escape_type_id": pl.Int32,
        "interactions": pl.Int32,
        "last_event": pl.Int32,
        "nscatterings": pl.Int32,
        "nu_cmf": pl.Float32,
        "nu_rf": pl.Float32,
        "number": pl.Int32,
        "originated_from_positron": pl.Int32,
        "pellet_nucindex": pl.Int32,
        "pol_dirx": pl.Float32,
        "pol_diry": pl.Float32,
        "pol_dirz": pl.Float32,
        "scat_count": pl.Int32,
        "stokes1": pl.Float32,
        "stokes2": pl.Float32,
        "stokes3": pl.Float32,
        "t_decay": pl.Float32,
        "true_emission_velocity": pl.Float32,
        "trueem_time": pl.Float32,
        "trueemissiontype": pl.Int32,
        "type_id": pl.Int32,
    }

    try:
        dfpackets = pl.read_csv(
            at.zopenpl(packetsfile),
            separator=" ",
            has_header=False,
            comment_prefix="#",
            new_columns=column_names,
            infer_schema_length=20000,
            dtypes=dtype_overrides,
        )

    except Exception:
        print(f"Error occured in file {packetsfile}")
        raise

    dfpackets = dfpackets.drop(["next_trans", "last_cross"])

    # drop last column of nulls (caused by trailing space on each line)
    if dfpackets[dfpackets.columns[-1]].is_null().all():
        dfpackets = dfpackets.drop(dfpackets.columns[-1])

    if "true_emission_velocity" in dfpackets.columns:
        # some packets don't have this set, which confused read_csv to mark it as str
        dfpackets = dfpackets.with_columns([pl.col("true_emission_velocity").cast(pl.Float32)])

    if "originated_from_positron" in dfpackets.columns:
        dfpackets = dfpackets.with_columns([pl.col("originated_from_positron").cast(pl.Boolean)])

    # Luke: packet energies in ergs can be huge (>1e39) which is too large for Float32
    return dfpackets.with_columns(
        [pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Float64).exclude(["e_rf", "e_cmf"]).cast(pl.Float32)]
    )


def readfile(
    packetsfile: Path | str,
    packet_type: str | None = None,
    escape_type: t.Literal["TYPE_RPKT", "TYPE_GAMMA"] | None = None,
    use_pyarrow_extension_array=True,
) -> pd.DataFrame:
    """Read a packet file into a Pandas DataFrame."""
    dfpackets = pl.read_parquet(packetsfile)

    if escape_type is not None:
        assert packet_type is None or packet_type == "TYPE_ESCAPE"
        dfpackets = dfpackets.filter(
            (pl.col("type_id") == type_ids["TYPE_ESCAPE"]) & (pl.col("escape_type_id") == type_ids[escape_type])
        )
    elif packet_type is not None and packet_type:
        dfpackets = dfpackets.filter(pl.col("type_id") == type_ids[packet_type])

    return dfpackets.to_pandas(use_pyarrow_extension_array=use_pyarrow_extension_array)


def convert_text_to_parquet(
    packetsfiletext: Path | str,
) -> Path:
    packetsfiletext = Path(packetsfiletext)
    packetsfileparquet = at.stripallsuffixes(packetsfiletext).with_suffix(".out.parquet")

    dfpackets = readfile_text(packetsfiletext).lazy()
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
            )
            .cast(pl.Float32)
            .alias("t_arrive_d"),
        ]
    )

    syn_dir = next(
        (at.get_syn_dir(p) for p in packetsfiletext.parents if Path(p, "syn_dir.txt").is_file()),
        (0.0, 0.0, 1.0),
    )
    dfpackets = add_packet_directions_lazypolars(dfpackets, syn_dir)
    dfpackets = bin_packet_directions_lazypolars(dfpackets)

    # print(f"Saving {packetsfileparquet}")
    dfpackets = dfpackets.sort(by=["type_id", "escape_type_id", "t_arrive_d"])
    dfpackets.collect().write_parquet(packetsfileparquet, compression="zstd", statistics=True, compression_level=6)

    return packetsfileparquet


def get_packetsfilepaths(
    modelpath: str | Path, maxpacketfiles: int | None = None, printwarningsonly: bool = False
) -> list[Path]:
    """Get a list of Paths to parquet-formatted packets files, (which are generated from text files if needed)."""
    nprocs = at.get_nprocs(modelpath)

    searchfolders = [Path(modelpath, "packets"), Path(modelpath)]
    # in descending priority (based on speed of reading)
    suffix_priority = [".out.zst", ".out.zst", ".out", ".out.gz", ".out.xz"]
    t_lastschemachange = calendar.timegm(time_parquetschemachange)

    parquetpacketsfiles = []
    parquetrequiredfiles = []

    for rank in range(nprocs + 1):
        name_nosuffix = f"packets00_{rank:04d}"
        found_rank = False

        for folderpath in searchfolders:
            filepath = (folderpath / name_nosuffix).with_suffix(".out.parquet")
            if filepath.is_file():
                if filepath.stat().st_mtime < t_lastschemachange:
                    filepath.unlink(missing_ok=True)
                    print(f"{filepath} is out of date.")
                else:
                    if rank < nprocs:
                        parquetpacketsfiles.append(filepath)
                    found_rank = True

        if not found_rank:
            for suffix in suffix_priority:
                for folderpath in searchfolders:
                    filepath = (folderpath / name_nosuffix).with_suffix(suffix)
                    if filepath.is_file():
                        if rank < nprocs:
                            parquetrequiredfiles.append(filepath)
                        found_rank = True
                        break

                if found_rank:
                    break

        if found_rank and rank >= nprocs:
            print(f"WARNING: nprocs is {nprocs} but file {filepath} exists")
        elif not found_rank and rank < nprocs:
            print(f"WARNING: packets file for rank {rank} was not found.")

        if maxpacketfiles is not None and (len(parquetpacketsfiles) + len(parquetrequiredfiles)) >= maxpacketfiles:
            break

    if len(parquetrequiredfiles) >= 20:
        with multiprocessing.get_context("spawn").Pool(processes=at.get_config()["num_processes"]) as pool:
            convertedparquetpacketsfiles = pool.map(convert_text_to_parquet, parquetrequiredfiles)
            pool.close()
            pool.join()
    else:
        convertedparquetpacketsfiles = [convert_text_to_parquet(p) for p in parquetrequiredfiles]

    parquetpacketsfiles += list(convertedparquetpacketsfiles)

    if not printwarningsonly:
        if maxpacketfiles is not None and nprocs > maxpacketfiles:
            print(f"Reading from the first {maxpacketfiles} of {nprocs} packets files")
        else:
            print(f"Reading from {len(parquetpacketsfiles)} packets files")

    return parquetpacketsfiles


def get_packets_pl(
    modelpath: str | Path,
    maxpacketfiles: int | None = None,
    packet_type: str | None = None,
    escape_type: t.Literal["TYPE_RPKT", "TYPE_GAMMA"] | None = None,
) -> tuple[int, pl.LazyFrame]:
    if escape_type is not None:
        assert packet_type in {None, "TYPE_ESCAPE"}
        if packet_type is None:
            packet_type = "TYPE_ESCAPE"

    packetsfiles = get_packetsfilepaths(modelpath, maxpacketfiles)

    nprocs_read = len(packetsfiles)
    packetsdatasize_gb = nprocs_read * Path(packetsfiles[0]).stat().st_size / 1024 / 1024 / 1024
    print(f" data size is {packetsdatasize_gb:.1f} GB ({nprocs_read} * size of {packetsfiles[0].parts[-1]})")

    pldfpackets = pl.scan_parquet(packetsfiles)

    if escape_type is not None:
        assert packet_type is None or packet_type == "TYPE_ESCAPE"
        pldfpackets = pldfpackets.filter(
            (pl.col("type_id") == type_ids["TYPE_ESCAPE"]) & (pl.col("escape_type_id") == type_ids[escape_type])
        )
    elif packet_type is not None and packet_type:
        pldfpackets = pldfpackets.filter(pl.col("type_id") == type_ids[packet_type])

    return nprocs_read, pldfpackets


def get_directionbin(
    dirx: float, diry: float, dirz: float, nphibins: int, ncosthetabins: int, syn_dir: tuple[float, float, float]
) -> int:
    dirmag = np.sqrt(dirx**2 + diry**2 + dirz**2)
    pkt_dir = [dirx / dirmag, diry / dirmag, dirz / dirmag]
    costheta = np.dot(pkt_dir, syn_dir)
    costhetabin = int((costheta + 1.0) / 2.0 * ncosthetabins)

    xhat = np.array([1.0, 0.0, 0.0])
    vec1 = np.cross(pkt_dir, syn_dir)
    vec2 = np.cross(xhat, syn_dir)
    cosphi = np.dot(vec1, vec2) / at.vec_len(vec1) / at.vec_len(vec2)

    vec3 = np.cross(vec2, syn_dir)
    testphi = np.dot(vec1, vec3)

    phibin = (
        int(math.acos(cosphi) / 2.0 / math.pi * nphibins)
        if testphi >= 0
        else int((math.acos(cosphi) + math.pi) / 2.0 / math.pi * nphibins)
    )

    return (costhetabin * nphibins) + phibin


def add_packet_directions_lazypolars(
    dfpackets: pl.LazyFrame | pl.DataFrame, syn_dir: tuple[float, float, float]
) -> pl.LazyFrame:
    dfpackets = dfpackets.lazy()
    assert len(syn_dir) == 3
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)  # -yhat if syn_dir is zhat

    if "dirmag" not in dfpackets.columns:
        dfpackets = dfpackets.with_columns(
            (pl.col("dirx") ** 2 + pl.col("diry") ** 2 + pl.col("dirz") ** 2).sqrt().alias("dirmag"),
        )

    if "costheta" not in dfpackets.columns:
        dfpackets = dfpackets.with_columns(
            (
                (pl.col("dirx") * syn_dir[0] + pl.col("diry") * syn_dir[1] + pl.col("dirz") * syn_dir[2])
                / pl.col("dirmag")
            )
            .cast(pl.Float32)
            .alias("costheta"),
        )

    if "phi" not in dfpackets.columns:
        # vec1 = dir cross syn_dir
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
            )
            .cast(pl.Float32)
            .alias("cosphi"),
        )

        vec3 = np.cross(vec2, syn_dir)  # -xhat if syn_dir is zhat

        # arr_testphi = np.dot(arr_vec1, vec3)
        dfpackets = dfpackets.with_columns(
            ((pl.col("vec1_x") * vec3[0] + pl.col("vec1_y") * vec3[1] + pl.col("vec1_z") * vec3[2]) / pl.col("dirmag"))
            .cast(pl.Float32)
            .alias("testphi"),
        )

        dfpackets = dfpackets.with_columns(
            (
                pl.when(pl.col("testphi") >= 0)
                .then(2 * math.pi - pl.col("cosphi").arccos())
                .otherwise(pl.col("cosphi").arccos())
            )
            .cast(pl.Float32)
            .alias("phi"),
        )

    return dfpackets.drop(["dirmag", "vec1_x", "vec1_y", "vec1_z"])


def bin_packet_directions_lazypolars(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    nphibins: int | None = None,
    ncosthetabins: int | None = None,
    phibintype: t.Literal["phidescending", "phiascending"] = "phidescending",
) -> pl.LazyFrame:
    dfpackets = dfpackets.lazy()
    if nphibins is None:
        nphibins = at.get_viewingdirection_phibincount()

    if ncosthetabins is None:
        ncosthetabins = at.get_viewingdirection_costhetabincount()

    dfpackets = dfpackets.with_columns(
        ((pl.col("costheta") + 1) / 2.0 * ncosthetabins).fill_nan(0.0).cast(pl.Int32).alias("costhetabin"),
    )

    if phibintype == "phiascending":
        dfpackets = dfpackets.with_columns(
            (pl.col("phi") / 2.0 / math.pi * nphibins).fill_nan(0.0).cast(pl.Int32).alias("phibin"),
        )
    else:
        # for historical consistency, this binning method decreases phi angle with increasing bin index
        dfpackets = dfpackets.with_columns(
            (
                pl.when(pl.col("testphi") > 0)
                .then(pl.col("cosphi").arccos() / (2 * math.pi) * nphibins)
                .otherwise((pl.col("cosphi").arccos() + math.pi) / (2 * math.pi) * nphibins)
            )
            .fill_nan(0.0)
            .cast(pl.Int32)
            .alias("phibin"),
        )

    return dfpackets.with_columns(
        (pl.col("costhetabin") * nphibins + pl.col("phibin")).cast(pl.Int32).alias("dirbin"),
    )


def bin_packet_directions(
    modelpath: Path | str, dfpackets: pd.DataFrame, syn_dir: tuple[float, float, float] | None = None
) -> pd.DataFrame:
    """Avoid this slow pandas function and use bin_packet_directions_lazypolars instead for new code."""
    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()

    syn_dir = at.get_syn_dir(Path(modelpath)) if syn_dir is None else syn_dir
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)

    pktdirvecs = dfpackets[["dirx", "diry", "dirz"]].to_numpy()

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
    filta = arr_testphi >= 0
    arr_phibin[filta] = np.arccos(arr_cosphi[filta]) / 2.0 / math.pi * nphibins
    filtb = np.invert(filta)
    arr_phibin[filtb] = (np.arccos(arr_cosphi[filtb]) + math.pi) / 2.0 / math.pi * nphibins
    dfpackets["phibin"] = arr_phibin
    dfpackets["arccoscosphi"] = np.arccos(arr_cosphi)

    dfpackets["dirbin"] = (arr_costhetabin * nphibins) + arr_phibin

    assert np.all(dfpackets["dirbin"] < at.get_viewingdirectionbincount())

    return dfpackets


def make_3d_histogram_from_packets(modelpath, timestep_min, timestep_max=None, em_time=True):
    if timestep_max is None:
        timestep_max = timestep_min
    modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)

    timeminarray = at.get_timestep_times(modelpath=modelpath, loc="start")
    # timedeltaarray = at.get_timestep_times(modelpath=modelpath, loc="delta")
    timemaxarray = at.get_timestep_times(modelpath=modelpath, loc="end")

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

    only_packets_0_scatters = False
    for packetsfile in packetsfiles:
        # for npacketfile in range(0, 1):
        dfpackets = readfile(packetsfile)
        at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])
        dfpackets = dfpackets.dropna(subset=["emission_velocity"])  # drop rows where emission_vel is NaN

        if only_packets_0_scatters:
            print("Only using packets with 0 scatters")
            # print(dfpackets[['scat_count', 'interactions', 'nscatterings']])
            dfpackets = dfpackets.query("nscatterings == 0")

        # print(dfpackets[['emission_velocity', 'em_velx', 'em_vely', 'em_velz']])
        # select only type escape and type r-pkt (don't include gamma-rays)
        dfpackets = dfpackets.query(
            f'type_id == {type_ids["TYPE_ESCAPE"]} and escape_type_id == {type_ids["TYPE_RPKT"]}'
        )
        if em_time:
            dfpackets = dfpackets.query("@timeminarray[@timestep_min] < em_time/@DAY < @timemaxarray[@timestep_max]")
        else:  # packet arrival time
            dfpackets = dfpackets.query("@timeminarray[@timestep_min] < t_arrive_d < @timemaxarray[@timestep_max]")

        emission_position3d[0].extend(list(dfpackets["em_velx"] / CLIGHT))
        emission_position3d[1].extend(list(dfpackets["em_vely"] / CLIGHT))
        emission_position3d[2].extend(list(dfpackets["em_velz"] / CLIGHT))

        e_rf.extend(list(dfpackets["e_rf"]))
        e_cmf.extend(list(dfpackets["e_cmf"]))

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
    for _z in range(grid):
        for _y in range(grid):
            for x in range(grid):
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

    timearray = at.get_timestep_times(modelpath=modelpath, loc="mid")
    arr_timedelta = at.get_timestep_times(modelpath=modelpath, loc="delta")
    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    dfpackets_escape_velocity_and_arrive_time = pd.DataFrame()
    emission_data = pd.DataFrame(
        {"t_arrive_d": timearray, "mean_emission_velocity": np.zeros_like(timearray, dtype=float)}
    )

    for i, packetsfile in enumerate(packetsfiles):
        dfpackets = readfile(packetsfile, packet_type=packet_type, escape_type=escape_type)
        at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])
        if escape_angles is not None:
            dfpackets = at.packets.bin_packet_directions(modelpath, dfpackets)
            dfpackets = dfpackets.query("dirbin == @escape_angles")

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
    df: pl.DataFrame | pl.LazyFrame,
    bincol: str,
    bins: list[float | int],
    sumcols: list[str] | None = None,
    getcounts: bool = False,
) -> pl.DataFrame:
    """Bins is a list of lower edges, and the final upper edge."""
    # Polars method

    df = df.with_columns(
        (
            pl.col(bincol)
            .cut(breaks=list(bins), labels=[str(x) for x in range(-1, len(bins))])
            .cast(str)
            .cast(pl.Int32)
        ).alias(f"{bincol}_bin")
    )

    aggs = [pl.col(col).sum().alias(col + "_sum") for col in sumcols] if sumcols is not None else []

    if getcounts:
        aggs.append(pl.col(bincol).count().alias("count"))

    wlbins = df.group_by(f"{bincol}_bin").agg(aggs).lazy().collect()

    # now we will include the empty bins
    dfout = pl.DataFrame(pl.Series(name=f"{bincol}_bin", values=np.arange(0, len(bins) - 1), dtype=pl.Int32))
    return dfout.join(wlbins, how="left", on=f"{bincol}_bin").fill_null(0)
