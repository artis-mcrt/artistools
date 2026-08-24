"""Read ARTIS packets and virtual packets files, caching them as parquet, and bin them by viewing direction."""

import calendar
import math
import time
import typing as t
from collections.abc import Sequence
from functools import lru_cache
from itertools import batched
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s as CLIGHT
from artistools.constants import day_to_s

type_ids = {"TYPE_GAMMA": 10, "TYPE_RPKT": 11, "TYPE_NTLEPTON": 20, "TYPE_ESCAPE": 32}

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
    "stokes_i",
    "stokes_q",
    "stokes_u",
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
    """Return the packet column names parsed from the ARTIS source in the model folder, or None if it is absent."""
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
            "stokes[0]": "stokes_i",
            "stokes[1]": "stokes_q",
            "stokes[2]": "stokes_u",
            "pol_dir[0]": "pol_dirx",
            "pol_dir[1]": "pol_diry",
            "pol_dir[2]": "pol_dirz",
            "trueemissionvelocity": "true_emission_velocity",
        }

        for i, column_name in enumerate(packet_properties):
            if column_name in replacements_dict:
                packet_properties[i] = replacements_dict[column_name]
        print(packet_properties)

        return packet_properties

    return None


def add_derived_columns_lazy(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    modelmeta: dict[str, t.Any] | None = None,
    dfmodel: pl.DataFrame | pl.LazyFrame | None = None,
    modelpath: Path | str | None = None,
) -> pl.LazyFrame:
    """Add columns to a packets DataFrame that are derived from the values that are stored in the packets files.

    We might as well add everything, since the columns only get calculated when they are actually used (polars LazyFrame).
    """
    if isinstance(dfmodel, pl.DataFrame):
        dfmodel = dfmodel.lazy()

    if dfmodel is None:
        assert modelpath is not None, "modelpath must be provided if dfmodel is not provided"
        dfmodel, modelmeta_read = at.get_modeldata(modelpath=modelpath)
        if modelmeta is None:
            modelmeta = modelmeta_read

    dfpackets = dfpackets.lazy()

    if modelpath is not None:
        timebins = [tstart * day_to_s for tstart in at.get_timestep_times(modelpath, loc="start")] + [
            at.get_timestep_times(modelpath, loc="end")[-1] * day_to_s
        ]
        dfpackets = dfpackets.with_columns(
            (pl.col("em_time").cut(breaks=timebins).to_physical().cast(pl.Int32) - 1).alias("em_timestep")
        )

    if "trueem_posx" in dfpackets.collect_schema().names():
        dfpackets = dfpackets.with_columns(
            true_emission_velocity=(
                (pl.col("trueem_posx") ** 2 + pl.col("trueem_posy") ** 2 + pl.col("trueem_posz") ** 2).sqrt()
                / pl.col("trueem_time")
            )
        )

    dfpackets = dfpackets.with_columns(
        emission_velocity=(
            (pl.col("em_posx") ** 2 + pl.col("em_posy") ** 2 + pl.col("em_posz") ** 2).sqrt() / pl.col("em_time")
        ),
        emission_velocity_lineofsight=(
            (
                (pl.col("em_posx") * pl.col("dirx"))
                + (pl.col("em_posy") * pl.col("diry"))
                + (pl.col("em_posz") * pl.col("dirz"))
            )
            / pl.col("em_time")
        ),
    )

    if modelmeta is None:
        return dfpackets

    if modelmeta["dimensions"] > 1:
        t_model_s = modelmeta["t_model_init_days"] * day_to_s
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

        velbins = [0.0, *(dfmodel.select(pl.col("vel_r_max_kmps") * 100000.0).collect().to_series().to_list())]

        def velocity_to_mgi(velcol: str) -> pl.Expr:
            return pl.col(velcol).cut(breaks=velbins).to_physical().cast(pl.Int32) - 1

        dfpackets = dfpackets.with_columns(em_modelgridindex=velocity_to_mgi("emission_velocity"))
        if "true_emission_velocity" in dfpackets.collect_schema().names():
            dfpackets = dfpackets.with_columns(emtrue_modelgridindex=velocity_to_mgi("true_emission_velocity"))

    return dfpackets


def get_packets_text_columns(packetsfile: Path | str, modelpath: Path | str = ".") -> list[str]:
    """Return the column names of a packets file, from its header, the ARTIS source, or the historical defaults."""
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


def readfile_text(packetsfiletext: Path | str, column_names: list[str]) -> pl.DataFrame:
    """Read a packets*.out(.xz/.zst) space-separated text file into a polars DataFrame."""
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
        "stokes_i": pl.Float32,
        "stokes_q": pl.Float32,
        "stokes_u": pl.Float32,
        "t_decay": pl.Float32,
        "true_emission_velocity": pl.Float32,
        "trueem_posx": pl.Float32,
        "trueem_posy": pl.Float32,
        "trueem_posz": pl.Float32,
        "trueem_time": pl.Float32,
        "trueemissiontype": pl.Int32,
        "type_id": pl.Int32,
    }

    try:
        dfpackets = pl.read_csv(
            at.polars_source(packetsfiletext),
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

    dfpackets = at.drop_trailing_null_column(dfpackets)

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
            "stokes0",
            "stokes_i",
        ],
        strict=False,
    ).with_columns(mpirank=pl.lit(mpirank, dtype=pl.Int32))

    if "originated_from_positron" in dfpackets.columns:
        dfpackets = dfpackets.with_columns([pl.col("originated_from_positron").cast(pl.Boolean)])

    # Luke: packet energies in ergs can be huge (>1e39) which is too large for Float32
    return dfpackets.with_columns([
        pl.col(pl.Int64).cast(pl.Int32, strict=True),
        pl.col(pl.Float64).exclude(["e_rf", "e_cmf"]).cast(pl.Float32, strict=True),
    ])


def read_virtual_packets_text_file(vpacketsfiletext: Path | str, column_names: list[str]) -> pl.DataFrame:
    """Read one rank's virtual packets text file, adding the MPI rank taken from the filename."""
    vpacketsfiletext = Path(vpacketsfiletext)
    mpirank = int(vpacketsfiletext.name.split("_")[-1].split(".")[0])

    # the caller resolves the path with tryzipped=True, thus polars_source only has to open the
    # .xz case, which polars cannot read from a path
    dfvpackets = pl.read_csv(
        at.polars_source(vpacketsfiletext),
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
    )

    return at.drop_trailing_null_column(dfvpackets).with_columns(mpirank=pl.lit(mpirank, dtype=pl.Int32))


def get_vpackets_text_columns(vpacketsfiletext: Path) -> list[str]:
    """Return the column names from the header line of a virtual packets file."""
    with at.zopen(vpacketsfiletext, mode="rt", encoding="utf-8") as f:
        firstline: str = f.readline()
    assert firstline.lstrip().startswith("#")
    return firstline.lstrip("#").split()


def format_timestamp(timestamp: float) -> str:
    """Return a UTC time as a string. Log messages use it to compare the modification times of files."""
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(timestamp))


def get_packets_rankbatch_parquetfile(
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

    # The time of the last change to the parquet schema. A schema change adds a column or changes a data type.
    # The code makes a new cache file if the cache is older than this time.
    # Increase this time only for a change that makes an older cache file incorrect.
    time_parquetschemachange = (2024, 4, 23, 9, 0, 0)
    t_lastschemachange = calendar.timegm(time_parquetschemachange)

    text_filenames = [
        (f"vpackets_{rank:04d}.out" if virtual else f"packets00_{rank:04d}.out") for rank in batch_mpiranks
    ]

    conversion_needed = True
    outdatedparquet: tuple[int, int] | None = None
    if parquetfilepath.is_file():
        parquetstat = parquetfilepath.stat()
        parquet_mtime = parquetstat.st_mtime
        # only the last rank's file is checked, on the assumption that a run writes all of its ranks together. An
        # individually-updated earlier file will not invalidate the cached parquet
        if text_filepath := at.firstexisting_or_none(
            text_filenames[-1], folder=modelpath, tryzipped=True, search_subfolders=True
        ):
            last_textfile_mtime = text_filepath.stat().st_mtime

            if parquet_mtime > last_textfile_mtime and parquet_mtime > t_lastschemachange:
                conversion_needed = False
            else:
                # the identity comes from the stat that showed the file is outdated, so only that exact
                # file can be replaced by this rank's rewrite
                outdatedparquet = at.get_file_identity(parquetstat)
                # leave the outdated file in place: write_parquet_atomic() puts the new one at the path in
                # one step, so the path always resolves to a complete parquet. Deleting it first opens a
                # window in which a concurrent reader (another rank, or another pytest-xdist worker) finds
                # it missing or half-swapped
                reasons = []
                if parquet_mtime <= last_textfile_mtime:
                    reasons.append(
                        f"{text_filepath.relative_to(modelpath)} was modified later"
                        f" ({format_timestamp(last_textfile_mtime)})"
                    )
                if parquet_mtime <= t_lastschemachange:
                    reasons.append(f"the parquet schema changed later ({format_timestamp(t_lastschemachange)})")

                print(
                    f"  {parquetfilepath.relative_to(modelpath)} was written"
                    f" {format_timestamp(parquet_mtime)} but {' and '.join(reasons)}."
                    " File will be regenerated..."
                )
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
                        / CLIGHT
                    )
                    / day_to_s
                ).cast(pl.Float32)
            ).sort(by=["type_id", "escape_type_id", "t_arrive_d"])

            pldf_batch = add_packet_directions_lazypolars(pldf_batch)
            pldf_batch = bin_packet_directions_polars(
                pldf_batch,
                nphibins=at.get_viewingdirection_phibincount(),
                ncosthetabins=at.get_viewingdirection_costhetabincount(),
                phibintype="phibinhistoricaldescendingdiscont",
            )

        print(
            f"   took {time.perf_counter() - time_start_load:.1f} seconds. Writing parquet file...", end="", flush=True
        )
        time_start_write = time.perf_counter()
        at.write_parquet_atomic(pldf_batch, parquetfilepath, compression_level=12, replaces=outdatedparquet)
        print(f"took {time.perf_counter() - time_start_write:.1f} seconds")

    return parquetfilepath


def get_packets_batch_parquet_paths(
    modelpath: str | Path, maxpacketfiles: int | None = None, printwarningsonly: bool = False, virtual: bool = False
) -> tuple[int, list[Path]]:
    """Get a list of Paths to parquet-formatted packets files, (which are generated from text files if needed)."""
    nprocs = at.get_nprocs(modelpath)

    mpirank_groups_all = list(enumerate(batched(range(nprocs), 100, strict=False)))
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
        get_packets_rankbatch_parquetfile(
            modelpath, batch_mpiranks=batch_mpiranks, batchindex=batchindex, virtual=virtual
        )
        for batchindex, batch_mpiranks in mpirank_groups
    ]
    assert bool(parquetpacketsfiles)
    nprocs_read = sum(len(batch_mpiranks) for _, batch_mpiranks in mpirank_groups)
    return nprocs_read, parquetpacketsfiles


def get_virtual_packets(modelpath: str | Path, maxpacketfiles: int | None = None) -> tuple[int, pl.LazyFrame]:
    """Return the number of MPI ranks read and a lazy frame of all of the model's virtual packets."""
    nprocs_read, vpacketparquetfiles = get_packets_batch_parquet_paths(
        modelpath, maxpacketfiles=maxpacketfiles, virtual=True
    )

    nbatches_read = len(vpacketparquetfiles)
    packetsdatasize_gb = sum(f.stat().st_size for f in vpacketparquetfiles) / 1024 / 1024 / 1024
    print(f"  total parquet size is {packetsdatasize_gb:.1f} GB (from {nbatches_read} batches)")

    # add some extra columns to imitate the real packets
    dfpackets = pl.scan_parquet(vpacketparquetfiles).with_columns(
        type_id=type_ids["TYPE_ESCAPE"], escape_type_id=type_ids["TYPE_RPKT"]
    )

    npkts_total = dfpackets.select(pl.len()).collect().item()
    print(f"  files contain {npkts_total:.2e} virtual packet events (shared among directions and opacity choices)")

    return nprocs_read, dfpackets


def get_packets(
    modelpath: str | Path,
    maxpacketfiles: int | None = None,
    packet_type: str | None = None,
    escape_type: str | None = None,
) -> tuple[int, pl.LazyFrame]:
    """Return the number of MPI ranks read and a lazy frame of the model's packets, filtered by type if given."""
    if escape_type is not None:
        assert packet_type in {None, "TYPE_ESCAPE"}
        if packet_type is None:
            packet_type = "TYPE_ESCAPE"

    nprocs_read, packetsparquetfiles = get_packets_batch_parquet_paths(modelpath, maxpacketfiles)

    nbatches_read = len(packetsparquetfiles)
    packetsdatasize_gb = sum(f.stat().st_size for f in packetsparquetfiles) / 1024 / 1024 / 1024
    print(f"  total parquet size is {packetsdatasize_gb:.1f} GB (from {nbatches_read} batches)")

    pldfpackets = pl.scan_parquet(packetsparquetfiles).rename(
        {
            "originated_from_positron": "originated_from_particlenotgamma",
            "stokes0": "stokes_i",
            "stokes1": "stokes_q",
            "stokes2": "stokes_u",
        },
        strict=False,
    )

    npkts_total = pldfpackets.select(pl.len()).collect().item()
    print(f"  files contain {npkts_total:.2e} packets from {nprocs_read} ranks")

    if escape_type is not None:
        if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
            msg = f"Unknown escape type {escape_type}"
            raise ValueError(msg)
        assert packet_type is None or packet_type == "TYPE_ESCAPE"
        pldfpackets = pldfpackets.filter(
            (pl.col("type_id") == type_ids["TYPE_ESCAPE"]) & (pl.col("escape_type_id") == type_ids[escape_type])
        )
    elif packet_type is not None and packet_type:
        pldfpackets = pldfpackets.filter(pl.col("type_id") == type_ids[packet_type])

    return nprocs_read, pldfpackets


def get_directionbin(
    dirx: float,
    diry: float,
    dirz: float,
    nphibins: int,
    ncosthetabins: int,
    syn_dir: tuple[float | int, float | int, float | int],
) -> int:
    """Return the viewing direction bin index for a single packet direction vector."""
    dirmag = np.sqrt(dirx**2 + diry**2 + dirz**2)
    pkt_dir = [dirx / dirmag, diry / dirmag, dirz / dirmag]
    costheta = np.dot(pkt_dir, syn_dir)
    costhetabin = min(int((costheta + 1.0) / 2.0 * ncosthetabins), ncosthetabins - 1)

    vec1 = np.cross(pkt_dir, syn_dir)
    if at.vec_len(vec1) == 0.0:
        # if the direction is parallel to the syn_dir, we cannot determine phi
        phibin = 0
    else:
        xhat = np.array([1.0, 0.0, 0.0])
        vec2 = np.cross(xhat, syn_dir)
        cosphi = np.dot(vec1, vec2) / at.vec_len(vec1) / at.vec_len(vec2)

        vec3 = np.cross(vec2, syn_dir)
        testphi = np.dot(vec1, vec3)

        # acos(cosphi) + pi reaches exactly 2 pi when cosphi == -1, which would otherwise land in the first phi bin of
        # the next costheta ring, so clamp to the last bin
        phibin = min(
            int(math.acos(cosphi) / 2.0 / math.pi * nphibins)
            if testphi > 0
            else int((math.acos(cosphi) + math.pi) / 2.0 / math.pi * nphibins),
            nphibins - 1,
        )

    return (costhetabin * nphibins) + phibin


def add_packet_directions_lazypolars(dfpackets: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Add the normalised direction vector and the costheta and phi angles of each packet."""
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

        # arr_testphi = np.dot(arr_vec1, vec3). vec1 was already normalised by dirmag above, and only the sign of
        # testphi is used, so there is no further division here (matching get_directionbin)
        dfpackets = dfpackets.with_columns(
            (pl.col("vec1_x") * vec3[0] + pl.col("vec1_y") * vec3[1] + pl.col("vec1_z") * vec3[2])
            .cast(pl.Float32)
            .alias("testphi")
        )

        dfpackets = dfpackets.with_columns(
            (
                pl
                .when(pl.col("testphi") > 0)
                .then(2 * math.pi - pl.col("cosphi").arccos())
                .otherwise(pl.col("cosphi").arccos())
            )
            .cast(pl.Float32)
            .alias("phi")
        )

    return dfpackets.drop(["dirmag", "vec1_x", "vec1_y", "vec1_z"])


def bin_packet_directions_polars(
    dfpackets: pl.LazyFrame | pl.DataFrame,
    nphibins: int | None = None,
    ncosthetabins: int | None = None,
    phibintype: t.Literal[
        "phibinhistoricaldescendingdiscont", "phibinmonotonicasc"
    ] = "phibinhistoricaldescendingdiscont",
) -> pl.LazyFrame:
    """Add the costheta, phi, and combined viewing direction bin index of each packet.

    phibintype selects between the historical descending-and-discontinuous phi bins and monotonically ascending ones.
    """
    dfpackets = dfpackets.lazy()
    if nphibins is None:
        nphibins = at.get_viewingdirection_phibincount()

    if ncosthetabins is None:
        ncosthetabins = at.get_viewingdirection_costhetabincount()

    dfpackets = dfpackets.with_columns(
        pl.min_horizontal(
            ((pl.col("costheta") + 1) / 2.0 * ncosthetabins).fill_nan(0).cast(pl.Int32), ncosthetabins - 1
        ).alias("costhetabin")
    )

    if phibintype == "phibinmonotonicasc":
        # phi reaches exactly 2 pi when cosphi == 1 and testphi > 0, so clamp to the last bin
        dfpackets = dfpackets.with_columns(
            pl.min_horizontal(
                (pl.col("phi") / 2.0 / math.pi * nphibins).fill_nan(0.0).cast(pl.Int32), nphibins - 1
            ).alias("phibinmonotonicasc")
        )
    else:
        # for historical consistency, this binning method decreases phi angle with increasing bin index
        # acos(cosphi) + pi reaches exactly 2 pi when cosphi == -1, which would otherwise land in the first phi bin of
        # the next costheta ring, so clamp to the last bin
        dfpackets = dfpackets.with_columns(
            pl.min_horizontal(
                (
                    pl
                    .when(pl.col("testphi") > 0)
                    .then(pl.col("cosphi").arccos() / (2 * math.pi) * nphibins)
                    .otherwise((pl.col("cosphi").arccos() + math.pi) / (2 * math.pi) * nphibins)
                )
                .fill_nan(0)
                .cast(pl.Int32),
                nphibins - 1,
            ).alias("phibin")
        ).with_columns((pl.col("costhetabin") * nphibins + pl.col("phibin")).cast(pl.Int32).alias("dirbin"))

    return dfpackets


def filter_packets_dirbin(
    dfpackets: pl.LazyFrame, dirbin: int, average_over_phi: bool = False, average_over_theta: bool = False
) -> tuple[pl.LazyFrame, float]:
    """Filter packets to a viewing direction bin, returning the filtered frame and the solid-angle factor (4 pi / solidangle).

    dirbin -1 selects all directions. When averaging over phi or theta angle, dirbin must be the first bin of its averaging group.
    """
    if dirbin == -1:
        return dfpackets, 1.0

    if average_over_phi:
        assert not average_over_theta
        nphibins = at.get_viewingdirection_phibincount()
        return (
            dfpackets.filter(pl.col("costhetabin") * nphibins == dirbin),
            float(at.get_viewingdirection_costhetabincount()),
        )

    if average_over_theta:
        return dfpackets.filter(pl.col("phibin") == dirbin), float(at.get_viewingdirection_phibincount())

    return dfpackets.filter(pl.col("dirbin") == dirbin), float(at.get_viewingdirectionbincount())


def bin_and_sum(
    df: pl.DataFrame | pl.LazyFrame,
    bincol: str,
    bins: Sequence[float | int],
    sumcols: list[str] | None = None,
    getcounts: bool = False,
    otheraggs: pl.Expr | list[pl.Expr] | None = None,
) -> pl.LazyFrame:
    """Bins is a list of lower edges, and the final upper edge."""
    # Polars method

    nbins = len(bins) - 1
    dfcut = (
        df
        .lazy()
        .filter(pl.col(bincol).is_between(bins[0], bins[-1], closed="both"))
        .with_columns(
            # each bin is [lower, upper), except the last one, which also includes its upper edge. cut() would put a
            # value sitting exactly on that final edge into the overflow bin, so clamp it back into the last bin
            pl.min_horizontal(
                pl.col(bincol).cut(breaks=bins, left_closed=True).to_physical().cast(pl.Int32) - 1, nbins - 1
            ).alias(f"{bincol}_bin")
        )
    )

    if sumcols is None:
        sumcols = []

    aggs = [pl.col(col).sum().alias(col + "_sum") for col in sumcols]

    if getcounts:
        aggs.append(pl.col(bincol).count().alias("count"))

    if otheraggs is None:
        otheraggs = []
    elif isinstance(otheraggs, pl.Expr):
        otheraggs = [otheraggs]
    aggs.extend(otheraggs)

    wlbins = dfcut.group_by(f"{bincol}_bin").agg(aggs)

    # now we will include the empty bins
    return (
        pl
        .LazyFrame({f"{bincol}_bin": range(nbins)}, schema={f"{bincol}_bin": pl.Int32})
        .join(wlbins, how="left", on=f"{bincol}_bin", coalesce=True)
        # fill nulls with 0 for sum columns
        .with_columns(pl.col(f"{sumcol}_sum").fill_null(0) for sumcol in sumcols)
        .with_columns(cs.by_name("count", require_all=False).fill_null(0))
        .sort(by=f"{bincol}_bin")
    )
