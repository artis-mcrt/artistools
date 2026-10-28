"""ARTIS model folder information: input parameters, run folders, and MPI rank mappings."""

import contextlib
import dataclasses as dc
import math
import re
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import day_to_s
from artistools.constants import h_ev_s
from artistools.misc.console import print_detail
from artistools.misc.console import print_warning
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import firstexisting_or_none
from artistools.misc.fileio import natural_sort_key
from artistools.misc.fileio import polars_source_open
from artistools.misc.fileio import read_wsv
from artistools.misc.fileio import readnoncommentline
from artistools.misc.fileio import zopen
from artistools.misc.modelpaths import modelpath_cache
from artistools.misc.modelpaths import path_is_codecomparison
from artistools.misc.modelpaths import resolve_modelpath
from artistools.misc.remote import check_local_path
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import on_model_host

if t.TYPE_CHECKING:
    from collections.abc import Mapping


@on_model_host
def get_vpkt_config(modelpath: Path) -> dict[str, t.Any]:
    """Return the virtual packet settings from a model's vpkt.txt."""
    config = parse_vpkt_input(Path(modelpath, "vpkt.txt").read_text(encoding="utf-8"))

    # ARTIS runs one full-opacity spectrum per observer when the file gives no exclusion list
    z_excludelist = config.opacityexclusions or [0]

    return {
        "nobsdirections": len(config.directions_costheta_phi),
        "cos_theta": [costheta for costheta, _ in config.directions_costheta_phi],
        "phi": [phi for _, phi in config.directions_costheta_phi],
        "nspectraperobs": len(z_excludelist),
        "z_excludelist": z_excludelist,
        "time_limits_enabled": int(config.override_tminmax),
        "initial_time": config.vspec_tmin_in_days,
        "final_time": config.vspec_tmax_in_days,
        # an empty list means the single range of the constants of the ARTIS build, which the file does not give
        "custom_lambda_ranges": config.custom_lambda_ranges,
    }


def get_grid_mapping(modelpath: Path | str) -> tuple[dict[int, list[int]], dict[int, int], bool]:
    """Return the maps from the model cells to the propagation grid cells, and back.

    The two grids can be different, e.g. a 1D input model with a 3D grid.

    Returns a tuple with:
    - dict[modelgridindex] = list of associated propagation cellindices,
    - dict[cellindex] = modelgridindex,
    - bool indicating if the mapping is direct (one-to-one).
    """
    modelpath = Path(modelpath)
    filename = firstexisting("grid.out", tryzipped=True, folder=modelpath)
    with polars_source_open(filename) as source:
        dfgrid = pl.read_csv(
            source,
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
        # artistools.inputmodel imports artistools.misc, thus this import runs at call time
        from artistools.inputmodel import get_modeldata

        _, modelmeta = get_modeldata(modelpath)
        assert modelmeta["dimensions"] == 3
        ngridpoints = modelmeta["npts_model"]
        xmax = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * day_to_s
    assert ngridpoints is not None
    ncoordgridx: int = round(ngridpoints ** (1.0 / 3.0))

    assert xmax is not None
    return 2.0 * xmax / ncoordgridx


def get_nu_grid(modelpath: Path | str) -> npt.NDArray[np.floating]:
    """Return an array of frequencies at which the ARTIS spectra are binned by exspec."""
    return get_nu_grid_cached(resolve_modelpath(modelpath))


@lru_cache(maxsize=16)
@on_model_host
def get_nu_grid_cached(modelpath: Path) -> npt.NDArray[np.floating]:
    """Return the frequency grid of the model at an absolute path."""
    specfile = firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True)
    with polars_source_open(specfile) as source:
        specdata = pl.read_csv(
            source,
            separator=" ",
            has_header=False,
            skip_rows=1,
            columns=[0],
            new_columns=["nu"],
            extra_columns="ignore",
        )
    return specdata["nu"].to_numpy()


def shorten_middle(text: str, maxlen: int | None) -> str:
    """Return text of no more than maxlen characters, with the middle replaced by an ellipsis.

    The name of a model folder often carries the name of the model at the start, and the date, the packet count, and the
    machine at the end. Thus a cut of either end alone loses what the reader needs most.
    """
    if maxlen is None or len(text) <= maxlen:
        return text

    if maxlen <= 3:
        return text[:maxlen]

    keep = maxlen - 3
    head = (keep + 1) // 2
    return f"{text[:head]}...{text[len(text) - (keep - head) :]}"


def get_model_name(path: Path | str) -> str:
    """Get the name of an ARTIS model from the path to any file inside it.

    Name will be either from a special plotlabel.txt file if it exists or the enclosing directory name
    """
    path = Path(path)
    if path_is_codecomparison(path):
        return str(path)

    # the cache key holds the absolute path, see ModelpathCache
    return get_model_name_cached(resolve_modelpath(path))


@lru_cache(maxsize=8)
@on_model_host
def get_model_name_cached(abspath: Path) -> str:
    """Return the name of the ARTIS model at an absolute path."""
    modelpath = abspath if abspath.is_dir() else abspath.parent

    try:
        plotlabelfile = Path(modelpath, "plotlabel.txt")
        with plotlabelfile.open(encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        foldername = Path(modelpath).name
        return shorten_middle(foldername, maxlen=50)


def print_modelpath(modelpath: Path | str) -> None:
    """Print the folder of the model below a heading, as the other plot commands do.

    The name of a model says nothing about the folder that holds it, and a user runs a command over
    many folders. The full path answers that, because "." says nothing when the command runs in the model folder.
    """
    print_detail(f"modelpath: {resolve_modelpath(modelpath)}")


def get_artis_source_text(modelpath: Path | str, filename: str) -> str | None:
    """Return the text of a file in the copy of the ARTIS source in the artis folder of the model, or None."""
    sourcepath = Path(modelpath) / "artis" / filename
    return sourcepath.read_text(encoding="utf-8") if sourcepath.is_file() else None


# a value option, e.g. "constexpr double MINTEMP = 3500.;" or a value on the next line, an option with no value, e.g.
# "constexpr std::optional<double> GAMMA_USE_KAPPA_GREY;", and the start of a function option, e.g.
# "constexpr bool LEVEL_IS_NLTE(int element_z, int ionstage, int level) {"
# C++ also permits a direct initialisation of a value, e.g. "constexpr int MAXITER{4};" or "constexpr int MAXITER(4);".
# The type is the text up to the name, e.g. "std::array<int, 2>" or "const char*". A string literal in a value
# can hold ";", e.g. "a;b", thus the value is a sequence of literals and other characters. A lone "'" is a digit
# separator, e.g. 1'000. The pattern tries the function option before the parenthesis form, because both start with
# "(". The parenthesis form holds no "{", thus a function with a trailing return type is not a value
CPP_LITERAL_PATTERN = r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'"
ARTIS_OPTION_PATTERN = re.compile(
    rf"^[ \t]*constexpr\s+(?:enum\s+)?([^=;(\n{{]+?)[\s*&]+(\w+)\s*"
    rf"(\([^)]*\)\s*\{{|\((?:{CPP_LITERAL_PATTERN}|[^;\"{{])*\)\s*;|\{{|=(?:{CPP_LITERAL_PATTERN}|[^;\"])*;|;)",
    re.MULTILINE,
)
# an optional value in a constructor form, e.g. "std::optional<int>{12}" or "std::make_optional(12)"
OPTIONAL_CONSTRUCTOR_PATTERN = re.compile(r"(?:std::optional\s*<.*>|std::make_optional)\s*(\{.*\}|\(.*\))", re.DOTALL)


def get_value_without_outer_brackets(value: str) -> str:
    """Return the value inside one outer pair of braces or parentheses, e.g. "12" for "{12}", or the value itself.

    A list of more than one element, e.g. "{26, 2}" of a std::array, keeps its braces, because the braces are part of
    the value. A comma inside a string literal or inside an inner bracket does not separate two elements. The first
    and the last bracket must be one pair, thus "(a) * (b)" stays as it is.
    """
    if len(value) < 2 or (value[0], value[-1]) not in {("{", "}"), ("(", ")")}:
        return value
    inner = value[1:-1]
    depth = 0
    for char in re.sub(CPP_LITERAL_PATTERN, "", inner):
        if char in "{([":
            depth += 1
        elif char in "})]":
            depth -= 1
        if depth < 0 or (char == "," and depth == 0):
            return value
    return inner.strip()


def get_artis_option_value(typename: str, initialiser: str) -> str:
    """Return the text of the value of an option from its initialiser, in one form for each meaning.

    The initialiser is the text after the name, e.g. "= 12", "{12}", "(12)", "= {12}", or "" for no initialiser. All
    of these forms give "12". A std::optional with no value gives "std::nullopt", e.g. for "", "{}", or "= {}". Any
    other type with no initialiser or with empty brackets gives "{}", its value-initialised state.
    """
    value = get_value_without_outer_brackets(" ".join(initialiser.removeprefix("=").split()))
    is_optional = typename.startswith("std::optional")
    if match := OPTIONAL_CONSTRUCTOR_PATTERN.fullmatch(value):
        is_optional = True
        value = get_value_without_outer_brackets(match.group(1))
    if value in {"", "{}", "()"}:
        return "std::nullopt" if is_optional else "{}"
    return value


def parse_artis_options(text: str) -> dict[str, str]:
    """Return each constexpr option of the text of an artisoptions.h and the text of its value.

    A function option gives its parameters and its body. The body of ION_NLEVELS_EXCITED_NLTE can hold an if block,
    thus the body ends at the brace that closes its first brace.
    """
    # a disabled alternative can be in a comment, and it must not replace the option that ARTIS compiles. A string
    # literal can hold "//", e.g. a URL, thus the pattern finds the literals first and keeps them
    text = re.sub(
        rf"{CPP_LITERAL_PATTERN}|/\*.*?\*/|//[^\n]*",
        lambda match: match.group(0) if match.group(0)[0] in "\"'" else "",
        text,
        flags=re.DOTALL,
    )
    options: dict[str, str] = {}
    for match in ARTIS_OPTION_PATTERN.finditer(text):
        typename, name, valuetext = match.groups()
        if valuetext.endswith("{"):
            # a function body or a brace initialiser ends at the brace that closes its first brace
            depth, end = 1, match.end()
            while depth > 0 and end < len(text):
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            if valuetext == "{":
                options[name] = get_artis_option_value(typename.strip(), text[match.start(3) : end])
            else:
                options[name] = " ".join(text[match.start(3) : end].split())
        else:
            options[name] = get_artis_option_value(typename.strip(), valuetext.removesuffix(";"))
    return options


def get_artis_options(modelpath: Path | str) -> dict[str, str]:
    """Return each constexpr option of the artisoptions.h of a model and the text of its value, or an empty dict.

    on_model_host finds a remote path only in a Path argument, thus a str path becomes a Path first.
    """
    return read_artis_options(resolve_modelpath(modelpath))


@on_model_host
def read_artis_options(modelpath: Path) -> dict[str, str]:
    """Return each constexpr option of the artisoptions.h in a model folder and the text of its value.

    The model folder of an ARTIS test holds artisoptions.h. The model folder of a production model holds the ARTIS
    source in the folder artis. For a local test, artis can be a link to the ARTIS repository, which holds a different
    preset. Thus the copy in the model folder comes first. A commented line gives no option. A function option, e.g.
    LEVEL_IS_NLTE, gives its parameters and its body.
    """
    optionspath = next(
        (path for path in (modelpath / "artisoptions.h", modelpath / "artis" / "artisoptions.h") if path.is_file()),
        None,
    )
    if optionspath is None:
        return {}
    return parse_artis_options(optionspath.read_text(encoding="utf-8"))


def get_artis_option(modelpath: Path | str, name: str) -> str | None:
    """Return the value text of one constexpr option of the artisoptions.h of a model, or None."""
    return get_artis_options(modelpath).get(name)


@lru_cache(maxsize=16)
@on_model_host
def get_remote_model_logname(path: Path, label: str | None) -> str:
    """Return the log name of a remote model. The host gives it, and the cache keeps it for each plot of a viewer."""
    return get_model_logname(path, label)


def get_model_logname(path: Path | str, label: str | None = None) -> str:
    """Return the label of an ARTIS model and the name of its folder, for a log message.

    The label comes from the caller (e.g. the --label argument) or from get_model_name.
    """
    path = Path(path)
    if is_remote_path(path):
        return get_remote_model_logname(path, label)
    modelname = get_model_name(path)
    label = label or modelname
    if path_is_codecomparison(path):
        return label

    abspath = path.resolve()
    foldername = (abspath if abspath.is_dir() else abspath.parent).name
    # a label that is the folder name needs no second copy
    return label if label == modelname == foldername else f"{label} (folder {foldername})"


def parse_npts_line(line: str, modelfilepath: Path | str) -> list[int]:
    """Return the one or two cell counts from the first line of model.txt after the comments.

    One number is the cell count. Two numbers are the 2D counts along the cylindrical radius and along
    the z axis. The line can end with an inline comment, as the lines of input.txt do.
    """
    cellcounts: list[int] = []
    # a token that is not an integer leaves the list empty, and the error below then names the file
    with contextlib.suppress(ValueError):
        cellcounts = [int(token) for token in line.split("#", 1)[0].split()]
    if len(cellcounts) not in {1, 2}:
        msg = f"The first line of {modelfilepath} after the comments must hold one or two integers, not {line!r}"
        raise ValueError(msg)
    return cellcounts


@modelpath_cache(maxsize=8)
def get_npts_model(modelpath: Path) -> int:
    """Return the number of cells in the model.txt."""
    modelfilepath = (
        Path(modelpath) if Path(modelpath).is_file() else firstexisting("model.txt", folder=modelpath, tryzipped=True)
    )
    with zopen(modelfilepath) as modelfile:
        return math.prod(parse_npts_line(readnoncommentline(modelfile), modelfilepath))


def get_inputfilepath(modelpath: Path | str) -> Path:
    """Return the path to input.txt, raising a helpful error if it does not exist."""
    check_local_path(modelpath)
    inputfilepath = Path(modelpath, "input.txt")
    if not inputfilepath.is_file():
        msg = f"{inputfilepath} not found. Is {Path(modelpath).resolve()} an ARTIS folder?"
        raise FileNotFoundError(msg)
    return inputfilepath


@modelpath_cache(maxsize=8)
def get_nprocs(modelpath: Path) -> int:
    """Return the number of MPI processes specified in input.txt.

    ARTIS counts only the lines that hold a value, and it keeps comment lines when it rewrites input.txt.
    """
    lines = get_inputfilepath(modelpath).read_text(encoding="utf-8").splitlines()
    valuelines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return int(valuelines[21].split("#")[0])


@modelpath_cache(maxsize=8)
@on_model_host
def get_inputparams(modelpath: Path) -> dict[str, t.Any]:
    """Return the parameters that input.txt gives.

    Do not change the dictionary that this function returns. The next caller gets the same object.
    """
    params: dict[str, t.Any] = {}
    with get_inputfilepath(modelpath).open("r", encoding="utf-8") as inputfile:
        params["pre_zseed"] = int(readnoncommentline(inputfile).split("#")[0])

        # number of timesteps
        params["ntstep"] = int(readnoncommentline(inputfile).split("#")[0])

        # the first timestep and the last timestep
        params["itstep"], params["ftstep"] = (int(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        params["tmin"], params["tmax"] = (float(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        MeV_in_Hz = 1e6 / h_ev_s
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


def get_runfolder_timesteps(folderpath: Path | str) -> tuple[int, ...]:
    """Return the sorted timesteps of the estimator files in an ARTIS run folder.

    A restarted run repeats the last timestep of the folder before it, and the result holds that timestep. Only
    get_runfolders knows the order of the folders, thus it gives such a timestep to the earlier folder.
    """
    return get_runfolder_timesteps_cached(resolve_modelpath(folderpath))


# a window checks every run folder for each plot, and a model with many restarts has more than 16 of them
@lru_cache(maxsize=1024)
def get_runfolder_timesteps_cached(folderpath: Path) -> tuple[int, ...]:
    """Return the timesteps of the run folder at an absolute path."""
    return tuple(read_runfolder_timesteps(folderpath))


def read_runfolder_timesteps(folderpath: Path) -> Sequence[int]:
    """Return the sorted timesteps of the estimators of a run folder, with the repeated timestep of a restart."""
    # this import runs at call time, because artistools.estimators imports artistools.misc
    from artistools.estimators import estimbatch_parquet_is_current
    from artistools.estimators import get_allranks_timesteps
    from artistools.rustext import estimtimesteps

    if (allranks_timesteps := get_allranks_timesteps(folderpath)) is not None:
        return allranks_timesteps

    # a stale cache can hold fewer timesteps than the text files, e.g. while ARTIS still runs.
    # Thus the function reads only a current cache. For a stale cache, it reads the text files
    estimparquetfiles = sorted(Path(folderpath).glob("estimbatch*.out.parquet*"))
    if estimparquetfiles and estimbatch_parquet_is_current(estimparquetfiles[0], folderpath):
        dfestfile = pl.scan_parquet(estimparquetfiles[0])
        return dfestfile.select(pl.col("timestep")).unique().sort("timestep").collect().to_series().to_list()
    # a leftover sibling such as estimators_0000.out.bak sorts before estimators_0000.out.zst, thus
    # the glob alone cannot pick the file. firstexisting_or_none applies the precedence that zopen reads
    # the estimator file of all ranks, estimators_allranks.out, has no rank number and is not a file of a rank
    estimstems = sorted({
        name[: name.index(".out") + len(".out")]
        for name in (path.name for path in Path(folderpath).glob("estimators_*.out*"))
        if ".out" in name and name[len("estimators_") : name.index(".out")].isdigit()
    })
    estimfilepath = next(
        (
            found
            for stem in estimstems
            if (found := firstexisting_or_none(stem, folder=folderpath, tryzipped=True, search_subfolders=False))
            is not None
        ),
        None,
    )
    if estimfilepath is not None:
        # estimtimesteps reads a compressed file that a stopped job cut, as the parser does. Thus such a file stops no
        # command that looks for the run folders
        return estimtimesteps(estimfilepath)

    return ()


def get_run_subfolders(modelpath: Path | str) -> tuple[Path, ...]:
    """Return the subfolders of a model folder in the order of the runs, and then the model folder.

    A run folder carries the number of the job, thus a lexical order puts a run of 10000001 in front
    of a run of 9876543. The natural order gives the folders in the order of the runs.
    """
    return (
        *sorted((child for child in Path(modelpath).iterdir() if child.is_dir()), key=natural_sort_key),
        Path(modelpath),
    )


def get_runfolders(
    modelpath: Path | str, timestep: int | None = None, timesteps: Sequence[int] | None = None
) -> Sequence[Path]:
    """Get a list of folders containing ARTIS output files from a modelpath, optionally with a timestep restriction.

    Without a timestep, the list holds each folder whose estimator files give at least one timestep. A restarted run
    repeats the last timestep of the folder before it, and the earlier folder keeps each timestep that two folders
    hold, as in scan_estimators. A folder with no earlier folder keeps its first timestep, e.g. a model whose first run
    folder is gone.
    """
    singletimestep = timestep if timesteps is None and timestep is not None and timestep > -1 else None
    ownedtimesteps_of_folder: dict[Path, set[int]] = {}
    earliertimesteps: set[int] = set()
    for folderpath in get_run_subfolders(modelpath):
        foldertimesteps = get_runfolder_timesteps(folderpath)
        # the earliest folder that holds a timestep keeps it, thus a later folder cannot change the result
        if singletimestep is not None and singletimestep in foldertimesteps:
            return (folderpath,)
        ownedtimesteps_of_folder[folderpath] = set(foldertimesteps) - earliertimesteps
        earliertimesteps.update(foldertimesteps)

    if singletimestep is not None:
        return ()

    if timesteps is not None and (len(timesteps) > 0 or (timestep is not None and timestep > -1)):
        return tuple(
            folderpath for folderpath, owned in ownedtimesteps_of_folder.items() if not owned.isdisjoint(timesteps)
        )

    return [folderpath for folderpath, owned in ownedtimesteps_of_folder.items() if owned]


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

    if modelgridindex is None:
        return all_ranks()

    if isinstance(modelgridindex, Iterable):
        # a comparison of a numpy array with an empty list raises, thus the size of the array decides
        cells = np.fromiter(modelgridindex, dtype=np.int64)
        if cells.size == 0 or (cells < 0).any():
            return all_ranks()

        return sorted(set(get_mpiranks_of_cells(modelpath, cells).tolist()))

    # in case modelgridindex is a single number rather than an iterable
    if modelgridindex < 0:
        return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))

    return [get_mpirankofcell(modelgridindex, modelpath=modelpath)]


def get_rank_outputfiles(folderpath: Path, filenameformat: str) -> dict[int, Path]:
    """Return the output file of each MPI rank in a run folder, keyed by rank, e.g. nlte_0001.out.zst for rank 1.

    filenameformat names the file of a rank, e.g. "nlte_{mpirank:04d}.out". One glob gives the names, and the
    precedence of firstexisting then selects one file for each rank: the plain file before a compressed file. A test
    of each name with stat took 3.4 s on Lustre for the 15368 rank files of a run. Each stat there is a request to a
    server.
    """
    prefix, suffix = re.split(r"\{mpirank[^}]*\}", filenameformat)
    files_by_name = {path.name: path for path in folderpath.glob(f"{prefix}*{suffix}*")}
    # ARTIS pads the rank to a minimum width of four, thus a rank of 10000 or more has more digits
    ranks = {
        int(rankstr)
        for rankstr in (name.removeprefix(prefix).split(".")[0] for name in files_by_name)
        if rankstr.isdigit()
    }
    rankfiles: dict[int, Path] = {}
    for rank in sorted(ranks):
        stem = filenameformat.format(mpirank=rank)
        candidates = (stem, *(stem + ext for ext in COMPRESSED_EXTENSIONS))
        if (name := next((name for name in candidates if name in files_by_name), None)) is not None:
            rankfiles[rank] = files_by_name[name]
    return rankfiles


# the number of lines of a rank file that give the types of its columns when a filter keeps no row
FIRST_LINES_FOR_TYPES = 100


def read_rank_outputfiles(
    modelpath: Path | str,
    filenameformat: str,
    timestep: int | None = None,
    modelgridindex: int | Sequence[int] | None = None,
) -> pl.DataFrame:
    """Read the output files of each MPI rank, e.g. radfield_{mpirank:04d}.out, in the run folders into one DataFrame.

    Each row of the files starts with the timestep and the model grid cell. A timestep, a model grid cell, or a
    sequence of cells selects the rows, and a negative value selects all of them.

    With the option WRITE_COMBINED_ALLRANK_OUT_FILES, ARTIS writes the rows of all ranks into one file of a run
    folder, e.g. radfield_allranks.out. The script scripts/combine_allrank_out_files.py of ARTIS makes the same file
    from the files of the ranks, and it keeps them by default. A read of some cells takes the files of their ranks,
    because a file of all ranks holds every cell. A read of all cells takes the file of all ranks, unless a file of a
    rank is newer. When the file of a rank is missing, the read also takes the file of all ranks.
    """
    prefix, suffix = re.split(r"\{mpirank[^}]*\}", filenameformat)
    # the format holds a field for the rank, thus a message names the family rather than one file
    filefamily = f"{prefix}*{suffix}"
    cells: list[int] = [modelgridindex] if isinstance(modelgridindex, int) else list(modelgridindex or [])
    # a negative cell number selects all cells
    selectedcells: list[int] = cells if all(mgi >= 0 for mgi in cells) else []
    # get_mpiranklist also stops with an error for a cell that is not in the model
    mpiranks = get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    # a rank whose cells all hold no matter writes no file, and a file of all ranks has no row for its cells
    nonemptycounts = get_nonempty_cellcounts(modelpath)
    ranks_with_matter = [rank for rank in mpiranks if nonemptycounts is None or nonemptycounts.get(rank) != 0]
    if selectedcells and not ranks_with_matter:
        cellstext = f"Cell {selectedcells[0]} holds" if len(selectedcells) == 1 else f"The cells {selectedcells} hold"
        msg = f"{cellstext} no matter. ARTIS writes no {filefamily} data for a cell with no matter"
        raise ValueError(msg)

    runfolders = get_runfolders(modelpath, timestep=timestep)
    if not runfolders and timestep is not None and timestep >= 0 and (allfolders := get_runfolders(modelpath)):
        heldtimesteps = sorted({ts for folder in allfolders for ts in get_runfolder_timesteps(folder)})
        msg = (
            f"No run folder of {modelpath} holds timestep {timestep}. The estimator files of the run folders give "
            f"timesteps {heldtimesteps[0]} to {heldtimesteps[-1]}"
        )
        raise ValueError(msg)

    filepathsofeachfolder: list[list[Path]] = []
    for folderpath in runfolders:
        rankfiles = get_rank_outputfiles(folderpath, filenameformat)
        missingranks = [rank for rank in ranks_with_matter if rank not in rankfiles]
        allranksfile = firstexisting_or_none(
            f"{prefix}allranks{suffix}", folder=folderpath, tryzipped=True, search_subfolders=False
        )
        if allranksfile is not None and (
            missingranks
            or (
                not selectedcells
                and max((rankfiles[rank].stat().st_mtime for rank in ranks_with_matter), default=0.0)
                <= allranksfile.stat().st_mtime
            )
        ):
            filepathsofeachfolder.append([allranksfile])
        elif missingranks:
            # the rank handles a cell that holds matter, thus the file is missing. firstexisting
            # names every compressed form that it looked for
            firstexisting(
                filenameformat.format(mpirank=missingranks[0]),
                folder=folderpath,
                tryzipped=True,
                search_subfolders=False,
            )
        elif ranks_with_matter:
            filepathsofeachfolder.append([rankfiles[rank] for rank in ranks_with_matter])

    if not filepathsofeachfolder:
        msg = f"No {filefamily} files found in {modelpath}"
        raise FileNotFoundError(msg)

    # A file of all ranks holds every cell. Thus a filter of the raw lines on the timestep and the cell keeps the
    # memory of a read small. The filter keeps the header line, which starts with "timestep modelgridindex"
    leadingfields = pl.col("line").str.extract_groups(r"^[ \t]*(?<timestep>[^ \t]+)[ \t]+(?<cell>[^ \t]+)")
    rowconditions: list[pl.Expr] = []
    if timestep is not None and timestep >= 0:
        rowconditions.append(leadingfields.struct.field("timestep") == str(timestep))
    if selectedcells:
        rowconditions.append(leadingfields.struct.field("cell").is_in([str(mgi) for mgi in selectedcells]))
    line_filter = (
        (leadingfields.struct.field("timestep") == "timestep") | pl.all_horizontal(rowconditions)
        if rowconditions
        else None
    )

    def read_file(filepath: Path, folderindex: int, line_filter: pl.Expr | None) -> pl.DataFrame | None:
        dffile = read_rank_file(filepath, line_filter=line_filter)
        if dffile is None:
            return None
        return dffile.rename({"ionstage": "ion_stage"}, strict=False).with_columns(
            pl.col("modelgridindex").cast(pl.Int64),
            pl.col("timestep").cast(pl.Int64),
            pl.lit(folderindex, dtype=pl.Int32).alias("folderindex"),
        )

    dfsoffiles = [
        dffile
        for folderindex, folderfilepaths in enumerate(filepathsofeachfolder)
        for filepath in folderfilepaths
        if (dffile := read_file(filepath, folderindex, line_filter)) is not None
    ]
    if not dfsoffiles:
        msg = f"Each {filefamily} file of {modelpath} is empty"
        raise ValueError(msg)

    # A file with the header line and no row gives a String column for each value. A relaxed concat with such a
    # frame would make each value column of the result a String
    nonemptyframes = [dffile for dffile in dfsoffiles if not dffile.is_empty()]
    if not nonemptyframes:
        # no row holds the timestep or the cells. The first lines of a file give the types, thus the empty result has
        # the types of a full read, and a filter such as pl.col("Z") == 26 does not compare a String with a number
        firstlines = pl.int_range(pl.len()) <= FIRST_LINES_FOR_TYPES
        nonemptyframes = [
            dfsample.clear()
            for folderindex, folderfilepaths in enumerate(filepathsofeachfolder)
            for filepath in folderfilepaths
            if (dfsample := read_file(filepath, folderindex, firstlines)) is not None and not dfsample.is_empty()
        ][:1] or dfsoffiles[:1]
    dfout = pl.concat(nonemptyframes, how="vertical_relaxed")

    # the first timestep of a restarted run repeats the last timestep of the folder before it.
    # scan_estimators keeps the first row of each cell and timestep, thus this keeps it as well.
    # A later folder can hold a cell that the earlier folder never wrote. Thus the pair of the
    # timestep and the cell decides, and not the timestep alone, and the earliest folder of the pair keeps its rows
    if len(filepathsofeachfolder) > 1:
        dfout = dfout.filter(pl.col("folderindex") == pl.col("folderindex").min().over(["timestep", "modelgridindex"]))

    return dfout.drop("folderindex")


def read_rank_file(filepath: Path, line_filter: pl.Expr | None = None) -> pl.DataFrame | None:
    """Return the rows of an output file of one rank or of all ranks, or None when the file is empty.

    A job that stopped before its first write leaves an empty file. line_filter keeps only the lines for which it is
    true, see read_wsv.
    """
    try:
        return read_wsv(filepath, line_filter=line_filter)
    except pl.exceptions.NoDataError:
        print_warning(f"{filepath} is empty, thus it gives no rows. The job stopped before its first write")
        return None


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


@modelpath_cache(maxsize=16)
def get_dfrankassignments(modelpath: Path) -> pl.LazyFrame | None:
    """Return the cell-to-MPI-rank assignments, or None when modelgridrankassignments.out is absent."""
    filerankassignments = firstexisting_or_none(
        "modelgridrankassignments.out", folder=modelpath, tryzipped=True, search_subfolders=False
    )
    if filerankassignments is not None:
        with polars_source_open(filerankassignments) as source:
            return pl.scan_csv(source, has_header=True, separator=" ").rename(
                lambda column_name: column_name.removeprefix("#")
            )
    return None


def get_rankassignments(modelpath: Path | str) -> pl.DataFrame | None:
    """Return the cell-to-MPI-rank assignments, or None when the file is absent.

    The clone is cheap, because polars shares the data of a frame, and it keeps a caller that changes
    the columns in place from changing what the next caller reads.
    """
    dfrankassignments = get_rankassignments_cached(resolve_modelpath(modelpath))

    return dfrankassignments.clone() if dfrankassignments is not None else None


@lru_cache(maxsize=16)
def get_rankassignments_cached(modelpath: Path) -> pl.DataFrame | None:
    """Return the rank assignments of the model, and keep the frame for the next caller.

    get_mpirankofcell asks for one cell at a time, thus a scan that each call collects reads the file
    again for each cell. Do not change the frame that this function returns.
    """
    lzrankassignments = get_dfrankassignments(modelpath)

    return lzrankassignments.collect() if lzrankassignments is not None else None


@modelpath_cache(maxsize=16)
def get_nonempty_cellcounts(modelpath: Path) -> "Mapping[int, int] | None":
    """Return the count of cells that hold matter for each rank, or None without the assignments file.

    ARTIS assigns no 3D cell to a shell that holds no matter. A rank whose count is zero handles such
    cells alone, thus it writes no output file, and the absence of that file is not a fault.
    """
    dfranks = get_rankassignments(modelpath)
    if dfranks is None:
        return None

    if "ndo_nonempty" not in dfranks.columns:
        return None

    return MappingProxyType(dict(zip(dfranks["rank"], dfranks["ndo_nonempty"], strict=True)))


def get_mpiranks_of_cells(modelpath: Path | str, cells: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the rank of each cell with one search for all the cells.

    A lookup for each cell took 18 to 81 us, thus 2 to 10 s for the 125 000 cells of a 3D snapshot.
    """
    return get_ranks_of_cells_in_layout(
        cells, get_npts_model(modelpath), get_rankassignments(modelpath), lambda: get_nprocs(Path(modelpath))
    )


def get_ranks_of_cells_in_layout(
    cells: npt.NDArray[np.int64],
    npts_model: int,
    dfrankassignments: pl.DataFrame | None,
    get_nprocs_of_layout: "Callable[[], int]",
) -> npt.NDArray[np.int64]:
    """Return the rank of each cell from modelgridrankassignments.out, or from the blocks of ARTIS for nprocs ranks.

    A restart can use a different number of ranks, thus a run folder can give its own assignments and rank count.
    """
    if outsidecells := [int(cell) for cell in cells if not 0 <= cell < npts_model]:
        # a model of one cell makes "1 cells" and "0 to 0" read badly, thus name the range alone
        msg = f"Cell {outsidecells[0]} is not in this model. Its cells are 0 to {npts_model - 1}"
        raise ValueError(msg)

    if dfrankassignments is not None:
        dfranks = dfrankassignments.filter(pl.col("ndo") > 0).sort("nstart")
        nstart = dfranks["nstart"].to_numpy()
        blockindex = np.searchsorted(nstart, cells, side="right") - 1
        if (cells >= nstart[blockindex] + dfranks["ndo"].to_numpy()[blockindex]).any():
            msg = "modelgridrankassignments.out gives no rank to some of the cells"
            raise ValueError(msg)
        return dfranks["rank"].to_numpy().astype(np.int64)[blockindex]

    nprocs = get_nprocs_of_layout()
    if nprocs > npts_model:
        return cells

    nblock, n_leftover = divmod(npts_model, nprocs)
    return np.where(
        cells <= n_leftover * (nblock + 1),
        cells // (nblock + 1),
        n_leftover + (cells - n_leftover * (nblock + 1)) // nblock,
    )


def get_mpirankofcell(modelgridindex: int, modelpath: Path | str) -> int:
    """Return the rank number of the MPI process responsible for handling a specified cell's updating and output."""
    mpirank = int(get_mpiranks_of_cells(modelpath, np.array([modelgridindex], dtype=np.int64))[0])
    # get_cellsofmpirank knows the blocks of the formula alone, and not modelgridrankassignments.out
    assert get_rankassignments(modelpath) is not None or modelgridindex in get_cellsofmpirank(mpirank, modelpath)
    return mpirank


@dc.dataclass
class VpktConfig:
    """The settings held in a vpkt.txt file, in the order they appear."""

    directions_costheta_phi: list[tuple[float, float]] = dc.field(
        default_factory=lambda: [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
    )
    opacityexclusions: list[int] = dc.field(default_factory=list)
    override_tminmax: bool = False
    vspec_tmin_in_days: float = 0.2
    vspec_tmax_in_days: float = 1.5
    custom_lambda_ranges: list[tuple[float, float]] = dc.field(default_factory=list)
    override_thickcell_tau: bool = True
    cell_is_optically_thick_vpkt: float = 100.0
    tau_max_vpkt: float = 10.0
    vgrid_on: bool = False
    tmin_vgrid_in_days: float = 0.2
    tmax_vgrid_in_days: float = 1.5
    vgrid_lambda_ranges: list[tuple[float, float]] = dc.field(default_factory=lambda: [(3500.0, 6000.0)])


class VpktTokenReader:
    """Hand out whitespace-separated tokens, so that any file ARTIS reads with fscanf is accepted."""

    def __init__(self, contents: str) -> None:
        """Split the file contents into tokens, ready to be consumed in order."""
        self.tokens = contents.split()
        self.pos = 0

    def next_value[T](self, converter: Callable[[str], T], typename: str, description: str) -> T:
        """Return the next token converted by converter, naming what was expected on failure."""
        if self.pos >= len(self.tokens):
            msg = f"vpkt.txt ended while reading {description}"
            raise ValueError(msg)

        token = self.tokens[self.pos]
        self.pos += 1
        try:
            return converter(token)
        except ValueError as exc:
            msg = f"vpkt.txt has {token!r} where {description} should be a {typename}"
            raise ValueError(msg) from exc

    def next_int(self, description: str) -> int:
        """Return the next token as an integer."""
        return self.next_value(int, "integer", description)

    def next_float(self, description: str) -> float:
        """Return the next token as a float."""
        return self.next_value(float, "number", description)

    def next_lambda_ranges(self, count: int, description: str) -> list[tuple[float, float]]:
        """Return count consecutive lambdamin/lambdamax pairs."""
        return [
            (self.next_float(f"{description} lambdamin"), self.next_float(f"{description} lambdamax"))
            for _ in range(count)
        ]

    @property
    def exhausted(self) -> bool:
        """Whether every token has been consumed."""
        return self.pos >= len(self.tokens)


def parse_vpkt_input(contents: str) -> VpktConfig:
    """Parse the contents of a vpkt.txt file, following the same order as the ARTIS reader."""
    reader = VpktTokenReader(contents)

    ndirections = reader.next_int("the number of observer directions")
    if ndirections < 1:
        msg = f"vpkt.txt must have at least one observer direction, but declares {ndirections}"
        raise ValueError(msg)
    costhetas = [reader.next_float(f"costheta of direction {i}") for i in range(ndirections)]
    phis = [reader.next_float(f"phi of direction {i}") for i in range(ndirections)]

    # any flag other than 1 means one full-opacity spectrum, which an empty list stands for here
    opacityexclusions: list[int] = []
    if reader.next_int("the custom opacity list flag") == 1:
        nspectraperobsdir = reader.next_int("the number of opacity choices")
        opacityexclusions = [reader.next_int(f"opacity choice {i}") for i in range(nspectraperobsdir)]

    override_tminmax = reader.next_int("the time window override flag")
    vspec_tmin_in_days = reader.next_float("the time window start")
    vspec_tmax_in_days = reader.next_float("the time window end")

    custom_lambda_ranges: list[tuple[float, float]] = []
    if reader.next_int("the custom wavelength range flag") == 1:
        nwavelengthranges = reader.next_int("the number of wavelength ranges")
        custom_lambda_ranges = reader.next_lambda_ranges(nwavelengthranges, "wavelength range")

    override_thickcell_tau = reader.next_int("the thick cell override flag")
    cell_is_optically_thick_vpkt = reader.next_float("the thick cell optical depth")
    tau_max_vpkt = reader.next_float("tau_max_vpkt")
    vgrid_on = bool(reader.next_int("the velocity grid map flag"))

    config = VpktConfig(
        directions_costheta_phi=list(zip(costhetas, phis, strict=True)),
        opacityexclusions=opacityexclusions,
        override_tminmax=bool(override_tminmax),
        vspec_tmin_in_days=vspec_tmin_in_days,
        vspec_tmax_in_days=vspec_tmax_in_days,
        custom_lambda_ranges=custom_lambda_ranges,
        override_thickcell_tau=bool(override_thickcell_tau),
        cell_is_optically_thick_vpkt=cell_is_optically_thick_vpkt,
        tau_max_vpkt=tau_max_vpkt,
        vgrid_on=vgrid_on,
    )

    # ARTIS stops reading here when the map is off, so the block may legitimately be absent
    if vgrid_on or not reader.exhausted:
        config.tmin_vgrid_in_days = reader.next_float("the velocity grid map start time")
        config.tmax_vgrid_in_days = reader.next_float("the velocity grid map end time")
        ngridranges = reader.next_int("the number of velocity grid wavelength ranges")
        config.vgrid_lambda_ranges = reader.next_lambda_ranges(ngridranges, "velocity grid wavelength range")

    return config
