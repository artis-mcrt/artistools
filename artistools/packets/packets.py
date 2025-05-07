import calendar
import math
import tempfile
import time
import typing as t
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at

CLIGHT = 2.99792458e10
DAY = 86400

types = {10: "TYPE_GAMMA", 11: "TYPE_RPKT", 20: "TYPE_NTLEPTON", 32: "TYPE_ESCAPE"}

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
        with inputfilename.open(encoding="utf-8") as inputfile:
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
    colnames: Sequence[str],
    allnonemptymgilist: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Add columns to a packets DataFrame that are derived from the values that are stored in the packets files.

    DEPRECATED: Use add_derived_columns_lazy instead.
    """
    modelpath = Path(modelpathin)
    cm_to_km = 1e-5
    day_in_s = 86400
    if dfpackets.empty:
        return dfpackets

    colnames = at.makelist(colnames)
    dimensions = at.get_inputparams(modelpath)["n_dimensions"]

    def em_modelgridindex(packet: t.Any) -> int | float:
        assert dimensions == 1

        mgi = at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )
        return math.nan if mgi is None else mgi

    def emtrue_modelgridindex(packet: t.Any) -> int | float:
        assert dimensions == 1

        mgi = at.inputmodel.get_mgi_of_velocity_kms(
            modelpath, packet.true_emission_velocity * cm_to_km, mgilist=allnonemptymgilist
        )
        return math.nan if mgi is None else mgi

    def em_timestep(packet: t.Any) -> int:
        return at.get_timestep_of_timedays(modelpath, packet.em_time / day_in_s)

    def emtrue_timestep(packet: t.Any) -> int:
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

    if any(x in colnames for x in ("angle_bin", "dirbin", "costhetabin", "phibin")):
        dfpackets = bin_packet_directions(dfpackets)

    return dfpackets


def add_derived_columns_lazy(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    modelmeta: dict[str, t.Any] | None = None,
    dfmodel: pd.DataFrame | pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Add columns to a packets DataFrame that are derived from the values that are stored in the packets files.

    We might as well add everything, since the columns only get calculated when they are actually used (polars LazyFrame).
    """
    if isinstance(dfmodel, pd.DataFrame):
        dfmodel = pl.from_pandas(dfmodel).lazy()

    assert isinstance(dfmodel, pl.LazyFrame)

    dfpackets = dfpackets.lazy().with_columns(
        emission_velocity=(
            (pl.col("em_posx") ** 2 + pl.col("em_posy") ** 2 + pl.col("em_posz") ** 2).sqrt() / pl.col("em_time")
        ),
        emission_velocity_lineofsight=(
            (
                (pl.col("em_posx") * pl.col("dirx")) ** 2
                + (pl.col("em_posy") * pl.col("diry")) ** 2
                + (pl.col("em_posz") * pl.col("dirz")) ** 2
            ).sqrt()
            / pl.col("em_time")
        ),
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
                coordpointnumrcyl=(
                    (pl.col("em_posx").pow(2) + pl.col("em_posy").pow(2)).sqrt() / pl.col("em_time") / vwidthrcyl
                ).cast(pl.Int32),
                coordpointnumz=((pl.col("em_posz") / pl.col("em_time") + vmax) / vwidthz).cast(pl.Int32),
            ).with_columns(
                em_modelgridindex=(pl.col("coordpointnumz") * modelmeta["ncoordgridrcyl"] + pl.col("coordpointnumrcyl"))
            )

        elif modelmeta["dimensions"] == 3:
            vwidth = modelmeta["wid_init"] / t_model_s
            dfpackets = dfpackets.with_columns([
                ((pl.col(f"em_pos{ax}") / pl.col("em_time") + vmax) / vwidth).cast(pl.Int32).alias(f"coordpointnum{ax}")
                for ax in ("x", "y", "z")
            ]).with_columns(
                em_modelgridindex=(
                    pl.col("coordpointnumz") * modelmeta["ncoordgridy"] * modelmeta["ncoordgridx"]
                    + pl.col("coordpointnumy") * modelmeta["ncoordgridx"]
                    + pl.col("coordpointnumx")
                )
            )

    elif modelmeta["dimensions"] == 1:
        assert dfmodel is not None, "dfmodel must be provided for 1D models to set em_modelgridindex"

        velbins = (dfmodel.select(pl.col("vel_r_max_kmps")).lazy().collect()["vel_r_max_kmps"] * 1000.0).to_list()
        dfpackets = dfpackets.with_columns(
            em_modelgridindex=(
                pl.col("emission_velocity")
                .cut(breaks=velbins, labels=[str(x) for x in range(-1, len(velbins))])
                .cast(str)
                .cast(pl.Int32)
            )
        )

    return dfpackets


def get_packets_text_columns(packetsfile: Path | str, modelpath: Path = Path()) -> list[str]:
    column_names: list[str] = []
    with at.zopen(packetsfile, mode="rt", encoding="utf-8") as fpackets:
        firstline = fpackets.readline()

        if firstline.lstrip().startswith("#"):
            column_names = firstline.lstrip("#").split()
            assert column_names is not None

            # get the column count from the first data line to check header matched
            dataline = fpackets.readline()
            inputcolumncount = len(dataline.split())
            assert inputcolumncount == len(column_names)
        else:
            inputcolumncount = len(firstline.split())
            column_names_artis = get_column_names_artiscode(modelpath)
            if column_names_artis is not None:  # found them in the artis code files
                column_names = column_names_artis
                assert len(column_names) == inputcolumncount
            else:  # infer from column positions
                assert len(columns_full) >= inputcolumncount
                column_names = columns_full[:inputcolumncount]

    return column_names


def readfile(
    packetsfile: Path | str,
    packet_type: str | None = None,
    escape_type: t.Literal["TYPE_RPKT", "TYPE_GAMMA"] | None = None,
) -> pd.DataFrame:
    """Read a packet file into a Pandas DataFrame."""
    dfpackets = readfile_text(packetsfile, column_names=get_packets_text_columns(packetsfile))

    if escape_type is not None:
        assert packet_type is None or packet_type == "TYPE_ESCAPE"
        dfpackets = dfpackets.filter(
            (pl.col("type_id") == type_ids["TYPE_ESCAPE"]) & (pl.col("escape_type_id") == type_ids[escape_type])
        )
    elif packet_type is not None and packet_type:
        dfpackets = dfpackets.filter(pl.col("type_id") == type_ids[packet_type])

    return dfpackets.to_pandas(use_pyarrow_extension_array=True)


def readfile_text(packetsfiletext: Path | str, column_names: list[str]) -> pl.DataFrame:
    """Read a packets*.out(.xz/.zstd) space-separated text file into a polars DataFrame."""
    packetsfiletext = Path(packetsfiletext)
    print(f"  reading {packetsfiletext}")
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
            at.zopenpl(packetsfiletext),
            separator=" ",
            has_header=False,
            comment_prefix="#",
            new_columns=column_names,
            infer_schema_length=20000,
            schema_overrides=dtype_overrides,
        )

    except Exception:
        print(f"Error occurred in file {packetsfiletext}")
        raise

    mpirank = int(packetsfiletext.name.split("_")[-1].split(".")[0])
    dfpackets = dfpackets.drop(
        [
            "next_trans",
            "last_event",
            "last_cross",
            "absorptiondirx",
            "absorptiondiry",
            "absorptiondirz",
            "interactions",
            "pol_dirx",
            "pol_diry",
            "pol_dirz",
        ],
        strict=False,
    ).with_columns(mpirank=pl.lit(mpirank, dtype=pl.Int32))

    # drop last column of nulls (caused by trailing space on each line)
    if dfpackets[dfpackets.columns[-1]].is_null().all():
        dfpackets = dfpackets.drop(dfpackets.columns[-1])

    if "originated_from_positron" in dfpackets.columns:
        dfpackets = dfpackets.with_columns([pl.col("originated_from_positron").cast(pl.Boolean)])

    # Luke: packet energies in ergs can be huge (>1e39) which is too large for Float32
    return dfpackets.with_columns([
        pl.col(pl.Int64).cast(pl.Int32),
        pl.col(pl.Float64).exclude(["e_rf", "e_cmf"]).cast(pl.Float32),
    ])


def read_virtual_packets_text_file(vpacketsfiletext: Path | str, column_names: list[str]) -> pl.DataFrame:
    vpacketsfiletext = Path(vpacketsfiletext)
    mpirank = int(vpacketsfiletext.name.split("_")[-1].split(".")[0])

    return pl.read_csv(
        vpacketsfiletext,
        separator=" ",
        has_header=False,
        comment_prefix="#",
        new_columns=column_names,
        schema_overrides={
            "emissiontype": pl.Int32,
            "trueemissiontype": pl.Int32,
            "absorption_type": pl.Int32,
            "absorption_freq": pl.Float64,
        }
        | {col: pl.Float64 for col in column_names if col.endswith("_nu_rf") or "_e_rf" in col}
        | {col: pl.Float32 for col in column_names if col.endswith("_t_arrive_d")},
    ).with_columns(mpirank=pl.lit(mpirank, dtype=pl.Int32))


def get_packets_text_paths(modelpath: str | Path, maxpacketfiles: int | None = None) -> list[Path]:
    """Get a list of Paths to packets*.out files."""
    modelpath = Path(modelpath)
    nprocs_read = at.get_nprocs(modelpath)
    if maxpacketfiles is not None:
        nprocs_read = min(nprocs_read, maxpacketfiles)

    return [
        at.firstexisting(f"packets00_{rank:04d}.out", folder=modelpath, tryzipped=True, search_subfolders=True)
        for rank in range(nprocs_read)
    ]


def get_vpackets_text_columns(vpacketsfiletext: Path) -> list[str]:
    firstline: str = at.zopen(vpacketsfiletext, mode="rt", encoding="utf-8").readline()
    assert firstline.lstrip().startswith("#")
    return firstline.lstrip("#").split()


def get_rankbatch_parquetfile(
    modelpath: Path | str, batch_mpiranks: Sequence[int], batchindex: int, virtual: bool
) -> Path:
    """Get the path to a parquet file containing packets for a specific batch of MPI ranks. If the file does not exists or is outdated, generate it first from the text files."""
    modelpath = Path(modelpath)
    strpacket = "vpackets" if virtual else "packets"
    packetdir = Path(modelpath, strpacket)
    packetdir.mkdir(exist_ok=True, parents=True)

    parquetfilename = (
        f"{strpacket}batch{batchindex:02d}_{batch_mpiranks[0]:04d}_{batch_mpiranks[-1]:04d}.out.parquet.tmp"
    )
    parquetfilepath = packetdir / parquetfilename

    # time when the schema for the parquet files last changed (e.g. new computed columns added or data types changed)
    time_parquetschemachange = (2024, 4, 23, 9, 0, 0)
    t_lastschemachange = calendar.timegm(time_parquetschemachange)

    text_filenames = [
        (f"vpackets_{rank:04d}.out" if virtual else f"packets00_{rank:04d}.out") for rank in batch_mpiranks
    ]

    conversion_needed = True
    if parquetfilepath.is_file():
        parquet_mtime = parquetfilepath.stat().st_mtime
        if text_filepath := at.anyexist(text_filenames[-1], folder=modelpath, tryzipped=True, search_subfolders=True):
            last_textfile_mtime = text_filepath.stat().st_mtime

            if parquet_mtime > last_textfile_mtime and parquet_mtime > t_lastschemachange:
                conversion_needed = False
            else:
                msg = f"ERROR: outdated file: {parquetfilepath}. Delete it to regenerate."
                raise AssertionError(msg)
        else:
            conversion_needed = False

    if conversion_needed:
        time_start_load = time.perf_counter()
        print(f"  generating {parquetfilepath.relative_to(modelpath)}...")

        text_file_paths = [
            at.firstexisting(filename, folder=modelpath, tryzipped=True, search_subfolders=True)
            for filename in text_filenames
        ]

        column_names = (
            get_vpackets_text_columns(text_file_paths[0])
            if virtual
            else get_packets_text_columns(text_file_paths[0], modelpath=modelpath)
        )

        ftextreader = read_virtual_packets_text_file if virtual else readfile_text

        pldf_batch = pl.concat(
            (ftextreader(text_file_path, column_names=column_names).lazy() for text_file_path in text_file_paths),
            how="vertical",
        )

        assert pldf_batch is not None

        if virtual:
            pldf_batch = pldf_batch.sort(by=["dir0_t_arrive_d"])
        else:
            pldf_batch = pldf_batch.with_columns(
                t_arrive_d=(
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
                ).cast(pl.Float32)
            ).sort(by=["type_id", "escape_type_id", "t_arrive_d"])

            pldf_batch = add_packet_directions_lazypolars(pldf_batch)
            pldf_batch = bin_packet_directions_lazypolars(
                pldf_batch, nphibins=10, ncosthetabins=10, phibintype="phibinhistoricaldescendingdiscont"
            )

        print(
            f"   took {time.perf_counter() - time_start_load:.1f} seconds. Writing parquet file...", end="", flush=True
        )
        time_start_write = time.perf_counter()
        tempparquetfilepath = Path(
            tempfile.mkstemp(dir=packetdir, prefix=f"{parquetfilename}.partial", suffix=".tmp")[1]
        )
        pldf_batch.sink_parquet(tempparquetfilepath, compression="zstd", statistics=True, compression_level=8)
        if parquetfilepath.exists():
            tempparquetfilepath.unlink()
        else:
            tempparquetfilepath.rename(parquetfilepath)
        print(f"took {time.perf_counter() - time_start_write:.1f} seconds")
    else:
        print(f"  scanning {parquetfilepath.relative_to(modelpath)}")

    return parquetfilepath


def get_packets_batch_parquet_paths(
    modelpath: str | Path, maxpacketfiles: int | None = None, printwarningsonly: bool = False, virtual: bool = False
) -> tuple[int, list[Path]]:
    """Get a list of Paths to parquet-formatted packets files, (which are generated from text files if needed)."""
    nprocs = at.get_nprocs(modelpath)

    mpirank_groups_all = list(enumerate(at.misc.batched(range(nprocs), 100)))
    mpirank_groups = [
        (batchindex, batch_mpiranks)
        for batchindex, batch_mpiranks in mpirank_groups_all
        if maxpacketfiles is None or batch_mpiranks[-1] < maxpacketfiles
    ]

    if not mpirank_groups:
        msg = f"No packets batches selected. Set maxpacketfiles to at least {mpirank_groups_all[0][1][-1] + 1}"
        raise ValueError(msg)

    if not printwarningsonly:
        if maxpacketfiles is not None and nprocs > maxpacketfiles:
            nprocs_read = mpirank_groups[-1][1][-1] + 1
            print(f"Reading packets from the first {nprocs_read} of {nprocs} ranks")
        else:
            print(f"Reading packets from {nprocs} ranks")

    parquetpacketsfiles = [
        get_rankbatch_parquetfile(modelpath, batch_mpiranks=batch_mpiranks, batchindex=batchindex, virtual=virtual)
        for batchindex, batch_mpiranks in mpirank_groups
    ]
    assert bool(parquetpacketsfiles)
    nprocs_read = sum(len(batch_mpiranks) for _, batch_mpiranks in mpirank_groups)
    return nprocs_read, parquetpacketsfiles


def get_virtual_packets_pl(modelpath: str | Path, maxpacketfiles: int | None = None) -> tuple[int, pl.LazyFrame]:
    nprocs_read, vpacketparquetfiles = get_packets_batch_parquet_paths(
        modelpath, maxpacketfiles=maxpacketfiles, virtual=True
    )

    nbatches_read = len(vpacketparquetfiles)
    packetsdatasize_gb = nbatches_read * Path(vpacketparquetfiles[0]).stat().st_size / 1024 / 1024 / 1024
    print(f"  data size is {packetsdatasize_gb:.1f} GB ({nbatches_read} * size of {vpacketparquetfiles[0].parts[-1]})")

    # add some extra columns to imitate the real packets
    dfpackets = pl.scan_parquet(vpacketparquetfiles).with_columns(
        type_id=type_ids["TYPE_ESCAPE"], escape_type_id=type_ids["TYPE_RPKT"]
    )

    npkts_total = dfpackets.select(pl.count("dir0_t_arrive_d")).collect().item(0, 0)
    print(f"  files contain {npkts_total:.2e} virtual packet events (shared among directions and opacity choices)")

    return nprocs_read, dfpackets


def get_packets_pl(
    modelpath: str | Path,
    maxpacketfiles: int | None = None,
    packet_type: str | None = None,
    escape_type: str | None = None,
) -> tuple[int, pl.LazyFrame]:
    if escape_type is not None:
        assert packet_type in {None, "TYPE_ESCAPE"}
        if packet_type is None:
            packet_type = "TYPE_ESCAPE"

    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)

    nprocs_read, packetsparquetfiles = get_packets_batch_parquet_paths(modelpath, maxpacketfiles)

    nbatches_read = len(packetsparquetfiles)
    packetsdatasize_gb = nbatches_read * Path(packetsparquetfiles[0]).stat().st_size / 1024 / 1024 / 1024
    print(f"  data size is {packetsdatasize_gb:.1f} GB ({nbatches_read} * size of {packetsparquetfiles[0].parts[-1]})")

    pldfpackets = pl.scan_parquet(packetsparquetfiles).rename(
        {"originated_from_positron": "originated_from_particlenotgamma"}, strict=False
    )

    npkts_total = pldfpackets.select(pl.count("e_rf")).collect().item(0, 0)
    print(f"  files contain {npkts_total:.2e} packets")

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


def add_packet_directions_lazypolars(dfpackets: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    dfpackets = dfpackets.lazy()
    syn_dir = np.array([0.0, 0.0, 1.0])
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)  # -yhat if syn_dir is zhat

    colnames = dfpackets.collect_schema().names()

    if "dirmag" not in colnames:
        dfpackets = dfpackets.with_columns(
            (pl.col("dirx") ** 2 + pl.col("diry") ** 2 + pl.col("dirz") ** 2).sqrt().alias("dirmag")
        )

    if "costheta" not in colnames:
        dfpackets = dfpackets.with_columns(
            (
                (pl.col("dirx") * syn_dir[0] + pl.col("diry") * syn_dir[1] + pl.col("dirz") * syn_dir[2])
                / pl.col("dirmag")
            )
            .cast(pl.Float32)
            .alias("costheta")
        )

    if "phi" not in colnames:
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
            .alias("cosphi")
        )

        vec3 = np.cross(vec2, syn_dir)  # -xhat if syn_dir is zhat

        # arr_testphi = np.dot(arr_vec1, vec3)
        dfpackets = dfpackets.with_columns(
            ((pl.col("vec1_x") * vec3[0] + pl.col("vec1_y") * vec3[1] + pl.col("vec1_z") * vec3[2]) / pl.col("dirmag"))
            .cast(pl.Float32)
            .alias("testphi")
        )

        dfpackets = dfpackets.with_columns(
            (
                pl.when(pl.col("testphi") >= 0)
                .then(2 * math.pi - pl.col("cosphi").arccos())
                .otherwise(pl.col("cosphi").arccos())
            )
            .cast(pl.Float32)
            .alias("phi")
        )

    return dfpackets.drop(["dirmag", "vec1_x", "vec1_y", "vec1_z"])


def bin_packet_directions_lazypolars(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    nphibins: int | None = None,
    ncosthetabins: int | None = None,
    phibintype: t.Literal[
        "phibinhistoricaldescendingdiscont", "phibinmonotonicasc"
    ] = "phibinhistoricaldescendingdiscont",
) -> pl.LazyFrame:
    dfpackets = dfpackets.lazy()
    if nphibins is None:
        nphibins = at.get_viewingdirection_phibincount()

    if ncosthetabins is None:
        ncosthetabins = at.get_viewingdirection_costhetabincount()

    dfpackets = dfpackets.with_columns(
        ((pl.col("costheta") + 1) / 2.0 * ncosthetabins).fill_nan(0.0).cast(pl.Int32).alias("costhetabin")
    )

    if phibintype == "phibinmonotonicasc":
        dfpackets = dfpackets.with_columns(
            (pl.col("phi") / 2.0 / math.pi * nphibins).fill_nan(0.0).cast(pl.Int32).alias("phibinmonotonicasc")
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
            .alias("phibin")
        ).with_columns((pl.col("costhetabin") * nphibins + pl.col("phibin")).cast(pl.Int32).alias("dirbin"))

    return dfpackets


def bin_packet_directions(dfpackets: pd.DataFrame) -> pd.DataFrame:
    """Avoid this slow pandas function and use bin_packet_directions_lazypolars instead for new code."""
    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()

    syn_dir = np.array([0.0, 0.0, 1.0])
    xhat = np.array([1.0, 0.0, 0.0])
    vec2 = np.cross(xhat, syn_dir)

    pktdirvecs = dfpackets[["dirx", "diry", "dirz"]].to_numpy().copy()

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


def make_3d_histogram_from_packets(
    modelpath: str | Path, timestep_min: int, timestep_max: int | None = None, em_time: bool = True
) -> npt.NDArray[np.floating]:
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

    _, packetsfiles = at.packets.get_packets_batch_parquet_paths(modelpath)

    emission_position3d_lists: list[list[float]] = [[], [], []]
    e_cmf: list[float] = []

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
            f"type_id == {type_ids['TYPE_ESCAPE']} and escape_type_id == {type_ids['TYPE_RPKT']}"
        )
        if em_time:
            dfpackets = dfpackets.query("@timeminarray[@timestep_min] < em_time/@DAY < @timemaxarray[@timestep_max]")
        else:  # packet arrival time
            dfpackets = dfpackets.query("@timeminarray[@timestep_min] < t_arrive_d < @timemaxarray[@timestep_max]")

        emission_position3d_lists[0].extend(list(dfpackets["em_velx"] / CLIGHT))
        emission_position3d_lists[1].extend(list(dfpackets["em_vely"] / CLIGHT))
        emission_position3d_lists[2].extend(list(dfpackets["em_velz"] / CLIGHT))

        e_cmf.extend(list(dfpackets["e_cmf"]))

    emission_position3d = np.array(emission_position3d_lists)
    weight_by_energy = True
    weights = np.array(e_cmf) if weight_by_energy else None

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

    return np.array(hist, dtype=np.float64)


def make_3d_grid(
    modeldata: pd.DataFrame, vmax_cms: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    # modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)
    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    xgrid = np.zeros(grid)
    vmax = vmax_cms / CLIGHT
    i = 0
    for _z in range(grid):
        for _y in range(grid):
            for nx in range(grid):
                xgrid[nx] = -vmax + 2 * nx * vmax / grid
                i += 1

    x, y, z = np.meshgrid(xgrid, xgrid, xgrid)
    grid_3d = np.array([xgrid, xgrid, xgrid])
    # grid_Te = np.zeros((grid, grid, grid))
    # print(grid_Te.shape)
    return grid_3d, x, y, z


def get_mean_packet_emission_velocity_per_ts(
    modelpath: str | Path,
    packet_type: str = "TYPE_ESCAPE",
    escape_type: t.Literal["TYPE_RPKT", "TYPE_GAMMA"] = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    escape_angles: int | None = None,
) -> pd.DataFrame:
    nprocs_read, packetsfiles = at.packets.get_packets_batch_parquet_paths(modelpath, maxpacketfiles=maxpacketfiles)
    assert nprocs_read > 0

    timearray = at.get_timestep_times(modelpath=modelpath, loc="mid")
    arr_timedelta = at.get_timestep_times(modelpath=modelpath, loc="delta")
    timearrayplusend = [*timearray, timearray[-1] + arr_timedelta[-1]]

    dfpackets_escape_velocity_and_arrive_time = pd.DataFrame()
    emission_data = pd.DataFrame({
        "t_arrive_d": timearray,
        "mean_emission_velocity": np.zeros_like(timearray, dtype=float),
    })

    for i, packetsfile in enumerate(packetsfiles):
        dfpackets = readfile(packetsfile, packet_type=packet_type, escape_type=escape_type)
        at.packets.add_derived_columns(dfpackets, modelpath, ["emission_velocity"])
        if escape_angles is not None:
            dfpackets = at.packets.bin_packet_directions(dfpackets)
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
) -> pl.LazyFrame:
    """Bins is a list of lower edges, and the final upper edge."""
    # Polars method

    dfcut = df.lazy().with_columns(
        (pl.col(bincol).cut(breaks=bins, labels=[str(x) for x in range(-1, len(bins))]))
        .alias(f"{bincol}_bin")
        .cast(pl.Int32)
    )

    aggs = [pl.col(col).sum().alias(col + "_sum") for col in sumcols] if sumcols is not None else []

    if getcounts:
        aggs.append(pl.col(bincol).count().alias("count"))

    wlbins = dfcut.group_by(f"{bincol}_bin").agg(aggs)

    # now we will include the empty bins
    return (
        pl.DataFrame({f"{bincol}_bin": range(len(bins) - 1)}, schema={f"{bincol}_bin": pl.Int32})
        .lazy()
        .join(wlbins, how="left", on=f"{bincol}_bin", coalesce=True)
        .fill_null(0)
        .sort(by=f"{bincol}_bin")
    )
