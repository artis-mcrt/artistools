"""ARTIS model folder information: input parameters, run folders, and MPI rank mappings."""

import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import day_to_s
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import readnoncommentline
from artistools.misc.fileio import zopen
from artistools.misc.fileio import zopenpl


def get_vpkt_config(modelpath: Path | str) -> dict[str, t.Any]:
    filename = Path(modelpath, "vpkt.txt")

    with filename.open(encoding="utf-8") as vpkt_txt:
        vpkt_config: dict[str, t.Any] = {
            "nobsdirections": int(vpkt_txt.readline()),
            "cos_theta": [float(x) for x in vpkt_txt.readline().split()],
            "phi": [float(x) for x in vpkt_txt.readline().split()],
        }
        assert isinstance(vpkt_config["cos_theta"], t.Sized)
        assert vpkt_config["nobsdirections"] == len(vpkt_config["cos_theta"])
        assert isinstance(vpkt_config["phi"], t.Sized)
        assert len(vpkt_config["cos_theta"]) == len(vpkt_config["phi"])

        speclistline = vpkt_txt.readline().split()
        nspecflag = int(speclistline[0])

        if nspecflag == 1:
            vpkt_config["nspectraperobs"] = int(speclistline[1])
            vpkt_config["z_excludelist"] = [int(x) for x in speclistline[2:]]
        else:
            vpkt_config["nspectraperobs"] = 1
            vpkt_config["z_excludelist"] = [0]

        timesline = vpkt_txt.readline().split()
        vpkt_config["time_limits_enabled"], vpkt_config["initial_time"], vpkt_config["final_time"] = (
            int(timesline[0]),
            float(timesline[1]),
            float(timesline[2]),
        )

    return vpkt_config


def get_grid_mapping(modelpath: Path | str) -> tuple[dict[int, list[int]], dict[int, int], bool]:
    """Get a bi-directional mapping between model cells and propagation grid cells. These can be different, e.g. 1D input model with a 3D grid.

    Returns a tuple with:
    - dict[modelgridindex] = list of associated propagation cellindices,
    - dict[cellindex] = modelgridindex,
    - bool indicating if the mapping is direct (one-to-one).
    """
    modelpath = Path(modelpath)
    filename = firstexisting("grid.out", tryzipped=True, folder=modelpath)
    dfgrid = pl.read_csv(
        zopenpl(filename),
        separator=" ",
        has_header=False,
        comment_prefix="#",
        schema={"cellindex": pl.Int32, "modelgridindex": pl.Int32},
    )
    assoc_cells: dict[int, list[int]] = dict(
        dfgrid
        .group_by("modelgridindex")
        .agg(pl.col("cellindex"))
        .select([pl.col("modelgridindex"), pl.col("cellindex")])
        .iter_rows()
    )

    mgi_of_propcells: dict[int, int] = dict(dfgrid.select([pl.col("cellindex"), pl.col("modelgridindex")]).iter_rows())
    direct_model_propgrid_map = all(
        len(propcells) == 1 and mgi == propcells[0] for mgi, propcells in assoc_cells.items()
    )

    return assoc_cells, mgi_of_propcells, direct_model_propgrid_map


def get_wid_init_at_tmodel(
    modelpath: Path | str | None = None,
    ngridpoints: int | None = None,
    t_model_days: float | None = None,
    xmax: float | None = None,
) -> float:
    """Return the Cartesian cell width [cm] at the model snapshot time."""
    if ngridpoints is None or t_model_days is None or xmax is None:
        # Luke: ngridpoint only equals the number of model cells if the model is 3D
        assert modelpath is not None
        from artistools.inputmodel import get_modeldata

        _, modelmeta = get_modeldata(modelpath)
        assert modelmeta["dimensions"] == 3
        ngridpoints = modelmeta["npts_model"]
        xmax = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * day_to_s
    assert ngridpoints is not None
    ncoordgridx: int = round(ngridpoints ** (1.0 / 3.0))

    assert xmax is not None
    return 2.0 * xmax / ncoordgridx


@lru_cache(maxsize=16)
def get_nu_grid(modelpath: Path) -> npt.NDArray[np.floating]:
    """Return an array of frequencies at which the ARTIS spectra are binned by exspec."""
    specdata = pl.read_csv(
        firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True),
        separator=" ",
        has_header=False,
        skip_rows=1,
        columns=[0],
        new_columns=["nu"],
    )
    return specdata["nu"].to_numpy()


@lru_cache(maxsize=8)
def get_model_name(path: Path | str, maxlen: int | None = 50) -> str:
    """Get the name of an ARTIS model from the path to any file inside it.

    Name will be either from a special plotlabel.txt file if it exists or the enclosing directory name
    """
    path = Path(path)
    if not path.exists() and path.parts[0] == "codecomparison":
        return str(path)

    abspath = path.resolve()

    modelpath = abspath if abspath.is_dir() else abspath.parent

    try:
        plotlabelfile = Path(modelpath, "plotlabel.txt")
        with plotlabelfile.open(encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        foldername = Path(modelpath).name
        return foldername if (maxlen is None or len(foldername) <= maxlen) else f"...{foldername[-maxlen:]}"


@lru_cache(maxsize=8)
def get_npts_model(modelpath: Path) -> int:
    """Return the number of cell in the model.txt."""
    modelfilepath = (
        Path(modelpath) if Path(modelpath).is_file() else firstexisting("model.txt", folder=modelpath, tryzipped=True)
    )
    with zopen(modelfilepath) as modelfile:
        nptsline = readnoncommentline(modelfile).split(maxsplit=1)
        if len(nptsline) == 1:
            return int(nptsline[0])
        return int(nptsline[0]) * int(nptsline[1])


def get_inputfilepath(modelpath: Path | str) -> Path:
    """Return the path to input.txt, raising a helpful error if it does not exist."""
    inputfilepath = Path(modelpath, "input.txt")
    if not inputfilepath.is_file():
        msg = f"{inputfilepath} not found. Is {Path(modelpath).resolve()} an ARTIS folder?"
        raise FileNotFoundError(msg)
    return inputfilepath


@lru_cache(maxsize=8)
def get_nprocs(modelpath: Path) -> int:
    """Return the number of MPI processes specified in input.txt."""
    return int(get_inputfilepath(modelpath).read_text(encoding="utf-8").split("\n")[21].split("#")[0])


@lru_cache(maxsize=8)
def get_inputparams(modelpath: Path) -> dict[str, t.Any]:
    """Return parameters specified in input.txt."""
    params: dict[str, t.Any] = {}
    with get_inputfilepath(modelpath).open("r", encoding="utf-8") as inputfile:
        params["pre_zseed"] = int(readnoncommentline(inputfile).split("#")[0])

        # number of time steps
        params["ntstep"] = int(readnoncommentline(inputfile).split("#")[0])

        # number of start and end time step
        params["itstep"], params["ftstep"] = (int(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        params["tmin"], params["tmax"] = (float(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        MeV_in_Hz = 2.417989242084918e20
        params["nusyn_min"], params["nusyn_max"] = (
            float(x) * MeV_in_Hz for x in readnoncommentline(inputfile).split("#")[0].split()
        )

        # number of times for synthesis
        params["nsyn_time"] = int(readnoncommentline(inputfile).split("#")[0])

        # start and end times for synthesis
        params["nsyn_time_start"], params["nsyn_time_end"] = (
            float(x) for x in readnoncommentline(inputfile).split("#")[0].split()
        )

        params["n_dimensions"] = int(readnoncommentline(inputfile).split("#")[0])

        # there are more parameters in the file that are not read yet...

    return params


@lru_cache(maxsize=16)
def get_runfolder_timesteps(folderpath: Path | str) -> tuple[int, ...]:
    """Get the set of timesteps covered by the output files in an ARTIS run folder."""
    if estimparquetfiles := sorted(Path(folderpath).glob("estimbatch*.out.parquet*")):
        # if there are estimators in parquet format, read the timesteps from there
        dfestfile = pl.scan_parquet(estimparquetfiles[0])
        timesteps_contained = (
            dfestfile.select(pl.col("timestep")).unique().sort("timestep").collect().to_series().to_list()
        )
        # the first timestep of a restarted run is duplicate and should be ignored
        restart_timestep = None if 0 in timesteps_contained else timesteps_contained[0]
        return tuple(ts for ts in timesteps_contained if ts != restart_timestep)
    if estimfiles := sorted(Path(folderpath).glob("estimators_*.out*")):
        with zopen(estimfiles[0]) as estfile:
            timesteps_contained = sorted({int(line.split()[1]) for line in estfile if line.startswith("timestep ")})
            # the first timestep of a restarted run is duplicate and should be ignored
            restart_timestep = None if 0 in timesteps_contained else timesteps_contained[0]
            return tuple(ts for ts in timesteps_contained if ts != restart_timestep)

    return ()


def get_runfolders(
    modelpath: Path | str, timestep: int | None = None, timesteps: Sequence[int] | None = None
) -> Sequence[Path]:
    """Get a list of folders containing ARTIS output files from a modelpath, optionally with a timestep restriction.

    The folder list may include non-ARTIS folders if a timestep is not specified.
    """
    folderlist_all = (*sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]), Path(modelpath))
    if (timestep is not None and timestep > -1) or (timesteps is not None and len(timesteps) > 0):
        folder_list_matching = []
        for folderpath in folderlist_all:
            folder_timesteps = get_runfolder_timesteps(folderpath)
            if timesteps is None and timestep is not None and timestep in folder_timesteps:
                return (folderpath,)  # return a single folder if only one timestep is specified
            if timesteps is not None and any(ts in folder_timesteps for ts in timesteps):
                folder_list_matching.append(folderpath)

        return tuple(folder_list_matching)

    return [folderpath for folderpath in folderlist_all if get_runfolder_timesteps(folderpath)]


def get_mpiranklist(
    modelpath: Path | str, modelgridindex: Iterable[int] | int | None = None, only_ranks_withgridcells: bool = False
) -> Sequence[int]:
    """Get a list of rank ids.

    - modelpath:
        pathlib.Path() to ARTIS model folder
    - modelgridindex:
        give a cell number to only return the rank number that updates this cell (and outputs its estimators)
    - only_ranks_withgridcells:
        set True to skip ranks that only update packets (i.e. that don't update any grid cells/output estimators).
    """

    def all_ranks() -> range:
        if only_ranks_withgridcells:
            return range(
                min(
                    get_nprocs(modelpath),
                    get_mpirankofcell(modelpath=modelpath, modelgridindex=get_npts_model(modelpath) - 1) + 1,
                )
            )
        return range(get_nprocs(modelpath))

    if modelgridindex is None or modelgridindex == []:
        return all_ranks()

    if isinstance(modelgridindex, Iterable):
        mpiranklist = set()
        for mgi in modelgridindex:
            assert isinstance(mgi, int)
            if mgi < 0:
                return all_ranks()

            mpiranklist.add(get_mpirankofcell(mgi, modelpath=modelpath))

        return sorted(mpiranklist)

    # in case modelgridindex is a single number rather than an iterable
    if modelgridindex < 0:
        return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))

    return [get_mpirankofcell(modelgridindex, modelpath=modelpath)]


def read_rank_outputfiles(
    modelpath: Path | str, filenameformat: str, timestep: int | None = None, modelgridindex: int | None = None
) -> pl.DataFrame:
    """Read per-MPI-rank whitespace-separated output files (e.g. radfield_{mpirank:04d}.out) from the run folders into one DataFrame.

    When a timestep or model grid cell is given, only the run folders and ranks that could contain it are read,
    and the rows are filtered to that selection (negative values mean no filter).
    """
    import pandas as pd

    filepaths = [
        firstexisting(filenameformat.format(mpirank=mpirank), folder=folderpath, tryzipped=True)
        for folderpath in get_runfolders(modelpath, timestep=timestep)
        for mpirank in get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    ]
    assert filepaths, f"No {filenameformat} files found in {modelpath}"

    dfout = (
        pl
        .concat(pl.from_pandas(pd.read_csv(filepath, sep=r"\s+", dtype_backend="pyarrow")) for filepath in filepaths)
        .rename({"ionstage": "ion_stage"}, strict=False)
        .with_columns(pl.col("modelgridindex").cast(pl.Int64), pl.col("timestep").cast(pl.Int64))
    )

    if modelgridindex is not None and modelgridindex >= 0:
        dfout = dfout.filter(pl.col("modelgridindex") == modelgridindex)
    if timestep is not None and timestep >= 0:
        dfout = dfout.filter(pl.col("timestep") == timestep)

    return dfout


def read_rank_outputfiles_lazy(
    modelpath: Path | str, filenameformat: str, timestep: int | None = None, modelgridindex: int | None = None
) -> pl.LazyFrame:
    """Read per-MPI-rank whitespace-separated output files (e.g. radfield_{mpirank:04d}.out) from the run folders into one LazyFrame.

    When a timestep or model grid cell is given, only the run folders and ranks that could contain it are read,
    and the rows are filtered to that selection (negative values mean no filter).
    """
    filepaths = [
        firstexisting(filenameformat.format(mpirank=mpirank), folder=folderpath, tryzipped=True)
        for folderpath in get_runfolders(modelpath, timestep=timestep)
        for mpirank in get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    ]
    assert filepaths, f"No {filenameformat} files found in {modelpath}"

    # Lazy path: use scan_csv and perform lazy transformations
    lazy_frames = [
        pl.scan_csv(zopenpl(filepath), separator=" ", comment_prefix="#", truncate_ragged_lines=True)
        for filepath in filepaths
    ]
    dfout = pl.concat(lazy_frames)
    dfout = dfout.rename({"ionstage": "ion_stage"}, strict=False).with_columns(
        pl.col("modelgridindex").cast(pl.Int64), pl.col("timestep").cast(pl.Int64)
    )

    if modelgridindex is not None and modelgridindex >= 0:
        dfout = dfout.filter(pl.col("modelgridindex") == modelgridindex)
    if timestep is not None and timestep >= 0:
        dfout = dfout.filter(pl.col("timestep") == timestep)

    return dfout


def get_cellsofmpirank(mpirank: int, modelpath: Path | str) -> Iterable[int]:
    """Return an iterable of the cell numbers processed by a given MPI rank."""
    npts_model = get_npts_model(modelpath)
    nprocs = get_nprocs(modelpath)

    assert mpirank < nprocs

    nblock = npts_model // nprocs
    n_leftover = npts_model % nprocs

    if mpirank < n_leftover:
        ndo = nblock + 1
        nstart = mpirank * (nblock + 1)
    else:
        ndo = nblock
        nstart = n_leftover + mpirank * nblock

    return list(range(nstart, nstart + ndo))


@lru_cache(maxsize=16)
def get_dfrankassignments(modelpath: Path | str) -> pl.LazyFrame | None:
    filerankassignments = Path(modelpath, "modelgridrankassignments.out")
    if filerankassignments.is_file():
        return pl.scan_csv(filerankassignments, has_header=True, separator=" ").rename(
            lambda column_name: column_name.removeprefix("#")
        )
    return None


def get_mpirankofcell(modelgridindex: int, modelpath: Path | str) -> int:
    """Return the rank number of the MPI process responsible for handling a specified cell's updating and output."""
    modelpath = Path(modelpath)
    npts_model = get_npts_model(modelpath)
    assert modelgridindex < npts_model

    dfrankassignments = get_dfrankassignments(modelpath)
    if dfrankassignments is not None:
        dfselected = dfrankassignments.filter(
            (pl.col("ndo") > 0)
            & (pl.col("nstart") <= modelgridindex)
            & ((pl.col("nstart") + pl.col("ndo") - 1) >= modelgridindex)
        ).collect()
        assert dfselected.height == 1
        return int(dfselected["rank"].item())

    nprocs = get_nprocs(modelpath)

    if nprocs > npts_model:
        mpirank = modelgridindex
    else:
        nblock = npts_model // nprocs
        n_leftover = npts_model % nprocs

        mpirank = (
            modelgridindex // (nblock + 1)
            if modelgridindex <= n_leftover * (nblock + 1)
            else n_leftover + (modelgridindex - n_leftover * (nblock + 1)) // nblock
        )

    assert modelgridindex in get_cellsofmpirank(mpirank, modelpath)

    return mpirank
