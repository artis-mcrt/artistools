"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""

import contextlib
import datetime
import string
import textwrap
import time
import typing as t
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

import polars as pl
from polars import selectors as cs

import artistools as at
from artistools.constants import K_B_ev_per_K
from artistools.misc import path_is_codecomparison

if t.TYPE_CHECKING:
    import os


# Units by the end of a name. A derived column takes its units from its end and not from its start, e.g.
# vel_r_min_kmps and init_kinetic_en_erg. These are tried before the prefixes below.
UNITS_BY_SUFFIX: dict[str, str] = {"_kmps": "km/s", "_on_c": "c", "_en_erg": "erg", "_days": "days"}

# Suffixes that make a variant of the same quantity in other units, e.g. vel_r_mid, vel_r_mid_on_c, and
# vel_r_mid_kmps. The listing names the variants one time rather than once for each base name.
VARIANT_SUFFIXES = ("", "_on_c", "_kmps")


VARIABLEUNITS: dict[str, str] = {
    "time": "days",
    "tdays": "days",
    "gamma_NT": "s^-1",
    # a recombination rate coefficient, and that coefficient times the electron density
    "Alpha_R": "cm$^3$/s",
    "Alpha_R*nne": "s^-1",
    "RRC_LTE_Nahar": "cm$^3$/s",
    "gamma_R": "s^-1",
    "gamma_R_bfest": "s^-1",
    "TR": "K",
    "Te": "K",
    "TJ": "K",
    "nne": "e$^-$/cm$^3$",
    "nniso": "cm$^{-3}$",
    "nnion": "cm$^{-3}$",
    "nnelement": "cm$^{-3}$",
    "nntot": "cm$^{-3}$",
    "deposition": "erg/s/cm$^3$",
    "emission_ana": "erg/s/cm$^3$",
    "total_dep": "erg/s/cm$^3$",
    "heating": "erg/s/cm$^3$",
    "heating_dep/total_dep": "Ratio",
    "cooling": "erg/s/cm$^3$",
    "rho": "g/cm$^3$",
    "logrho": "log10(g/cm$^3$)",
    "init_rho": "g/cm$^3$",
    "init_logrho": "log10(g/cm$^3$)",
    "init_X": "mass fraction",
    "kinetic_en_erg": "erg",
    "init_kinetic_en_erg": "erg",
    "mass": "g",
    "volume": "cm$^3$",
    "deltavol_deltat": "cm$^3$ days",
    "velocity": "km/s",
    "vel": "cm/s",
    "beta": "v/c",
    **{f"vel_{ax}_mid": "cm/s" for ax in ["x", "y", "z", "r", "rcyl"]},
    **{f"vel_{ax}_mid_on_c": "c" for ax in ["x", "y", "z", "r", "rcyl"]},
}


def get_units(name: str, *, latex: bool = True) -> str | None:
    """Return the units of an estimator variable, or None when it carries none.

    The name can be a whole column name such as nniso_Fe56, or the prefix of a family such as nniso.
    A trailing underscore of a prefix makes no difference. With latex=False the units come back without
    the mathematics marks that a plot label needs.
    """
    key = name.rstrip("_").removesuffix("_prevtimestep")

    units = VARIABLEUNITS.get(key)
    if units is None:
        units = next((value for suffix, value in UNITS_BY_SUFFIX.items() if key.endswith(suffix)), None)
    parts = key.split("_")
    if units is None:
        # try each shorter prefix, e.g. gamma_NT_Ar_I finds the entry gamma_NT
        units = next(
            (found for stop in range(len(parts) - 1, 0, -1) if (found := VARIABLEUNITS.get("_".join(parts[:stop])))),
            None,
        )
    if units is None:
        # then each shorter ending, because a derived name can carry the quantity there, e.g. init_volume
        units = next(
            (found for start in range(1, len(parts)) if (found := VARIABLEUNITS.get("_".join(parts[start:])))), None
        )

    if units is None:
        return None

    return units if latex else units.translate(str.maketrans("", "", "${}"))


def get_variablelongunits(key: str) -> str | None:
    """Return the full axis label for an estimator variable, or None when only the short unit is wanted."""
    return {"heating_dep/total_dep": "", "TR": "Temperature [K]", "Te": "Temperature [K]", "TJ": "Temperature [K]"}.get(
        key
    )


# Columns that share a prefix, with a description of what the group holds. The listing gives one line for
# each group rather than one line for each column. A longer prefix wins, thus emission_ana_ beats emission_.
# Columns that carry no unit. get_units returns None both for these and for a name that it does
# not know, thus the listing needs this map to tell the two apart.
DIMENSIONLESS: dict[str, str] = {
    "W": "dilution factor",
    "Ye": "electrons per nucleon",
    "grey_depth": "grey optical depth",
    "heating_dep/total_dep": "ratio",
    "inputcellid": "index",
    "modelgridindex": "index",
    "thick": "flag",
    "timestep": "index",
    "titeration": "index",
}


PREFIX_GROUPS: dict[str, str] = {
    "cooling_": "cooling rate of each process",
    "deposition_": "energy deposition rate of each particle",
    "emission_ana_": "analytic energy emission rate of each particle",
    "heating_": "heating rate of each process",
    "init_": "value of the model snapshot model.txt, before the first timestep",
    "vel_": "velocity coordinates of the cell",
}


def parse_species(suffix: str) -> str | None:
    """Return the species that the suffix names, or None when it names no element, ion, or isotope."""
    elsymbol, sep, rest = suffix.partition("_")
    if elsymbol not in at.get_elsymbolslist():
        # an isotope joins the mass number to the symbol, e.g. Ni56
        stem = suffix.rstrip(string.digits)
        return suffix if not sep and stem != suffix and stem in at.get_elsymbolslist() else None

    if not sep:
        return elsymbol

    # an ion stage takes a space, e.g. nnion_Fe_II names the ion "Fe II"
    if at.decode_roman_numeral(rest) > 0:
        return f"{elsymbol} {rest}"

    # any other suffix keeps its underscore, because that is how the column name joins it
    return suffix if rest == "otherstable" else None


def split_species_suffix(colname: str) -> tuple[str, str] | None:
    """Return the family and the species of a column such as nnion_Fe_II, or None when it has no species.

    The families that name one species for each column are the largest part of the estimator columns, thus
    a listing that repeats every one of them is hard to read.

    Each split of the name is tried, from the longest suffix to the shortest. One regular expression is
    not enough, because a symbol such as C or V is also a Roman numeral: in init_X_C the first reading
    takes X as the element and _C as the ion stage, and that reading has to give way to init_X and C.
    """
    parts = colname.split("_")
    for index in range(1, len(parts)):
        if species := parse_species("_".join(parts[index:])):
            return ("_".join(parts[:index]), species)

    return None


def summarise_ions(species: Collection[str]) -> str:
    """Return a compact listing of ions, e.g. "Fe I-V", or the plain names when they are not all ions."""
    stages: dict[str, list[int]] = defaultdict(list)
    for name in species:
        elsymbol, _, stage = name.partition(" ")
        stagenumber = at.decode_roman_numeral(stage)
        if stagenumber < 1:
            return ", ".join(sorted(species))
        stages[elsymbol].append(stagenumber)

    parts = []
    for elsymbol in sorted(stages):
        low, high = min(stages[elsymbol]), max(stages[elsymbol])
        romans = at.roman_numerals
        parts.append(f"{elsymbol} {romans[low]}" if low == high else f"{elsymbol} {romans[low]}-{romans[high]}")

    return ", ".join(parts)


def format_units(name: str) -> str:
    """Return the units of a variable in square brackets, ready for a terminal, or an empty string.

    A variable that carries no unit gives the note of DIMENSIONLESS instead.
    """
    units = get_units(name, latex=False) or DIMENSIONLESS.get(name.rstrip("_"))

    return f" [{units}]" if units else ""


# Above this count, a family of nuclides gives a summary rather than every name. A kilonova model holds
# more than 2000 of them, which no terminal can show.
MAXSPECIES_LISTED = 60


def species_placeholder(species: Collection[str]) -> str:
    """Return the name of what a family takes, e.g. "ion", or "element or nuclide" for a mixed family.

    The family init_X_ takes the mass fraction of an element such as init_X_Fe, and also of one nuclide
    such as init_X_Fe52, thus one word cannot name what it takes.
    """
    kinds = {"ion" if " " in name else "element" if name in at.get_elsymbolslist() else "nuclide" for name in species}

    return " or ".join(sorted(kinds)) if kinds else "species"


def summarise_nuclides(species: Collection[str]) -> str:
    """Return one line that counts the species of a family and names the elements that they cover."""
    symbols = {name.removesuffix("_otherstable").rstrip(string.digits) for name in species}
    known = sorted((one for one in symbols if one in at.get_elsymbolslist()), key=at.get_atomic_number)
    across = f", {known[0]} to {known[-1]}" if known else ""

    # a family that names an element as well as a nuclide counts both, and the count of the elements
    # then makes the phrase "of N elements" repeat itself
    bare = sum(name in at.get_elsymbolslist() for name in species)
    counts = (
        f"{bare} elements and {len(species) - bare} nuclides"
        if bare
        else f"{len(species)} nuclides of {len(symbols)} elements"
    )

    return f"{counts}{across}. For a full list pass --listnuclides"


def summarise_columns(columns: Collection[str], *, fullnuclides: bool = False) -> str:
    """Return a listing of the estimator columns, with each family and each group on one line."""
    families: dict[str, list[str]] = defaultdict(list)
    groups: dict[str, list[str]] = defaultdict(list)
    plain: list[str] = []
    prefixes = sorted(PREFIX_GROUPS, key=len, reverse=True)
    for colname in columns:
        if split := split_species_suffix(colname):
            families[split[0]].append(split[1])
        elif prefix := next((name for name in prefixes if colname.startswith(name)), None):
            groups[prefix].append(colname.removeprefix(prefix))
        else:
            plain.append(colname)

    def wrap(text: str) -> str:
        return textwrap.fill(text, width=110, initial_indent="    ", subsequent_indent="    ")

    lines = [
        f"{len(columns)} estimator variables:",
        "",
        f"  ({len(plain)}): one value for each cell and timestep",
        wrap(", ".join(f"{name}{format_units(name)}" for name in sorted(plain))),
    ]

    for prefix in sorted(groups):
        members = sorted(groups[prefix])
        # a variant gives the same quantity in other units, thus name the variants in the heading and
        # leave the base names alone, e.g. vel_<name>, vel_<name>_on_c, and vel_<name>_kmps
        bases: dict[str, list[str]] = defaultdict(list)
        for name in members:
            suffix = next(one for one in reversed(VARIANT_SUFFIXES) if name.endswith(one))
            bases[name.removesuffix(suffix)].append(suffix)

        variants = [one for one in VARIANT_SUFFIXES if any(one in found for found in bases.values())]
        names = ", ".join(f"{prefix}<name>{one}" for one in variants)

        if len(variants) > 1:
            # each variant gives the same quantity in one unit, thus the heading names them in order and
            # the base names stay bare
            headingunits = " " + ", ".join(
                format_units(prefix + next(base for base, found in bases.items() if one in found) + one).strip()
                for one in variants
            )
            memberlist = sorted(bases)
        else:
            # otherwise the members can disagree, e.g. init_rho and init_kinetic_en_erg, thus one heading
            # cannot name their units and each member takes its own
            memberunits = {format_units(prefix + name) for name in members}
            headingunits = memberunits.pop() if len(memberunits) == 1 else ""
            memberlist = [name + ("" if headingunits else format_units(prefix + name)) for name in members]

        lines.extend([
            "",
            f"  {names}  ({len(members)}){headingunits}: {PREFIX_GROUPS[prefix]}",
            wrap(", ".join(memberlist)),
        ])

    for family in sorted(families):
        species = families[family]
        listing = summarise_ions(species)
        # summarise_ions gives back the plain names when the species are nuclides and not ions, thus a
        # family of thousands of nuclides would fill the terminal. A family of bare element symbols stays
        # whole, because 83 of them take three lines and no name of them is a nuclide
        nuclides = [one for one in species if one not in at.get_elsymbolslist()]
        if not fullnuclides and nuclides and len(species) > MAXSPECIES_LISTED and listing == ", ".join(sorted(species)):
            listing = summarise_nuclides(species)

        placeholder = species_placeholder(species)
        lines.extend(["", f"  {family}_<{placeholder}>  ({len(species)}){format_units(family)}:", wrap(listing)])

    return "\n".join(lines)


def get_varname_formatted(varname: str) -> str:
    """Return the LaTeX-formatted name of an estimator variable, or the name unchanged if there is no mapping."""
    return {
        "nne": r"n$_{\rm e}$",
        "lognne": r"Log n$_{\rm e}$",
        "rho": r"$\rho$",
        "Te": r"T$_{\rm e}$",
        "TR": r"T$_{\rm R}$",
        "TJ": r"T$_{\rm J}$",
        "gamma_NT": r"$\Gamma_{\rm non-thermal}$ [s$^{-1}$]",
        "gamma_R_bfest": r"$\Gamma_{\rm phot}$ [s$^{-1}$]",
        "heating_dep/total_dep": "Heating fraction",
        # the horizontal axis variables. Without an entry here they keep the lower-case name that -x takes
        "velocity": "Velocity",
        "beta": r"$\beta$",
        "time": "Time",
        "timestep": "Timestep",
        # -x cellid and -x modelgridindex both plot the modelgridindex column, thus they share one label
        "cellid": "Model grid index",
        "modelgridindex": "Model grid index",
        **{f"vel_{ax}_mid": f"$v_{{{ax}}}$" for ax in ["x", "y", "z", "r", "rcyl"]},
        **{f"vel_{ax}_mid_on_c": f"$v_{{{ax}}}$" for ax in ["x", "y", "z", "r", "rcyl"]},
    }.get(varname, varname)


def get_units_string(variable: str) -> str:
    """Return an estimator variable's units in square brackets, or an empty string when it has none."""
    return f" [{units}]" if (units := get_units(variable)) else ""


def _estimator_colsortkey(col: str) -> str:
    """Sort timestep, modelgridindex, and titeration first, then the remaining columns alphabetically."""
    return f"-{col!r}" if col in {"timestep", "modelgridindex", "titeration"} else col


def get_estimators_rankbatch_parquetfile(
    folderpath: Path | str,
    batch_mpiranks: Sequence[int],
    batchindex: int,
    modelpath: Path | str | None = None,
    verbose: bool = False,
) -> Path:
    """Return the parquet cache for one batch of MPI ranks' estimator files, creating it if it is missing or stale."""

    def printornot(msg: str) -> None:
        if verbose:
            print(msg)

    modelpath = Path(folderpath).parent if modelpath is None else Path(modelpath)
    folderpath = Path(folderpath)
    parquetfilename = f"estimbatch{batchindex:02d}_{batch_mpiranks[0]:04d}_{batch_mpiranks[-1]:04d}.out.parquet.tmp"
    parquetfilepath = folderpath / parquetfilename

    textsource_mtime: float | int | None = None
    with contextlib.suppress(StopIteration):
        textsource_mtime = next(folderpath.glob("estimators_????.out*")).stat().st_mtime

    assert len(batch_mpiranks) == max(batch_mpiranks) - min(batch_mpiranks) + 1, (
        "batch_mpiranks must be a contiguous range of ranks"
    )
    assert len(set(batch_mpiranks)) == len(batch_mpiranks), "batch_mpiranks must not contain duplicates"

    parquetstat: os.stat_result | None = None
    with contextlib.suppress(FileNotFoundError):
        parquetstat = parquetfilepath.stat()

    outdatedparquet: tuple[int, int] | None = None
    if parquetstat is None:
        generate_parquet = True
    elif textsource_mtime and textsource_mtime > parquetstat.st_mtime:
        # leave the stale file in place: write_parquet_atomic() puts the new one at the path in one step, so
        # the path always resolves to a complete parquet. Deleting it first opens a window in which a
        # concurrent reader finds it missing or half-swapped. The identity comes from the stat that showed
        # the file is stale, so only that exact file can be replaced by this rewrite
        outdatedparquet = at.get_file_identity(parquetstat)
        print(
            f"  {parquetfilepath.relative_to(modelpath.parent)} is older than the estimator text files."
            " File will be regenerated..."
        )
        generate_parquet = True
    else:
        generate_parquet = False

    if generate_parquet:
        print(f"  generating {parquetfilepath.relative_to(modelpath.parent)}...")

        time_start = time.perf_counter()

        print(
            f"    reading {len(batch_mpiranks)} estimator files in {folderpath.relative_to(Path(folderpath).parent)}...",
            end="",
            flush=True,
        )

        pldf_batch = at.rustext.estimparse(folderpath, min(batch_mpiranks), max(batch_mpiranks))

        pldf_batch = pldf_batch.with_columns(
            cs.by_name("titeration", "timestep", "modelgridindex", require_all=False).cast(pl.Int32)
        )

        sortedcols: list[str] = sorted(pldf_batch.columns, key=_estimator_colsortkey)
        pldf_batch = pldf_batch.select(sortedcols)
        print(f"took {time.perf_counter() - time_start:.1f} s. Writing parquet file...", end="", flush=True)
        time_start = time.perf_counter()

        assert pldf_batch is not None
        at.write_parquet_atomic(
            pldf_batch,
            parquetfilepath,
            metadata={
                "creationtimeutc": str(datetime.datetime.now(datetime.UTC)),
                "textsource_mtime": str(textsource_mtime),
                "batch_rank_min": str(min(batch_mpiranks)),
                "batch_rank_max": str(max(batch_mpiranks)),
                "batchindex": str(batchindex),
            },
            replaces=outdatedparquet,
        )

        print(f"took {time.perf_counter() - time_start:.1f} s.")

    filesize = parquetfilepath.stat().st_size / 1024 / 1024
    try:
        printornot(f"  scanning {parquetfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)")
    except ValueError:
        printornot(f"  scanning {parquetfilepath} ({filesize:.2f} MiB)")

    return parquetfilepath


def join_cell_modeldata(
    estimators: pl.LazyFrame, modelpath: Path | str, verbose: bool = False
) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    """Join the estimator data with data from model.txt and derived quantities, e.g. density, volume, etc."""
    assert estimators is not None
    estimators = estimators.join(
        at
        .get_timesteps(modelpath)
        .select("timestep", "tmid_days", "twidth_days")
        .with_columns(tmid_days_prevtimestep=pl.col("tmid_days").shift(1)),
        on="timestep",
        how="left",
        coalesce=True,
    )
    dfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["ALL"], get_elemabundances=True, printwarningsonly=not verbose
    )

    dfmodel = dfmodel.rename({
        colname: f"init_{colname}"
        for colname in dfmodel.collect_schema().names()
        if not colname.startswith("vel_") and colname not in {"inputcellid", "modelgridindex", "mass_g"}
    })
    return estimators.join(dfmodel, on="modelgridindex", suffix="_initmodel").with_columns(
        rho=pl.col("init_rho") * (modelmeta["t_model_init_days"] / pl.col("tmid_days")) ** 3,
        volume=pl.col("init_volume") * (pl.col("tmid_days") / modelmeta["t_model_init_days"]) ** 3,
        volume_prevtimestep=pl.col("init_volume")
        * (pl.col("tmid_days_prevtimestep") / modelmeta["t_model_init_days"]) ** 3,
    ), modelmeta


def lazyframe_from_estimator_dict(estimators: dict[tuple[int, int], t.Any]) -> pl.LazyFrame:
    """Return a LazyFrame of the estimators of a back-end that reads into a dict keyed by (timestep, cell).

    The index columns take the same dtype as the ARTIS file reader gives them, so that a join with the model
    data matches on either path.
    """
    return pl.LazyFrame(
        [{"timestep": ts, "modelgridindex": mgi, **estimvals} for (ts, mgi), estimvals in estimators.items()],
        orient="row",
    ).with_columns(pl.col("timestep").cast(pl.Int32), pl.col("modelgridindex").cast(pl.Int32))


def add_derived_estimator_columns(pldflazy: pl.LazyFrame) -> pl.LazyFrame:
    """Add quantities derived from the estimator columns that were read from file."""
    colnames = pldflazy.collect_schema().names()

    if "heating_gamma/gamma_dep" in colnames:
        pldflazy = pldflazy.with_columns(gamma_dep=pl.col("heating_gamma") / pl.col("heating_gamma/gamma_dep"))

    if "deposition_gamma" in colnames:
        # sum up the gamma, elec, positron, alpha deposition contributions
        pldflazy = pldflazy.with_columns(total_dep=pl.sum_horizontal(cs.starts_with("deposition_")))
    elif "heating_heating_dep/total_dep" in colnames:
        # for older files with no deposition data, take heating part of deposition and heating fraction
        pldflazy = pldflazy.with_columns(total_dep=pl.col("heating_dep") / pl.col("heating_heating_dep/total_dep"))

    # only fill the number density columns: ARTIS omits zero-abundance ions and isotopes from the estimator files,
    # so a missing number density means zero. The file reader already fills these with zero for cells that skip them
    # within one file, and this makes the columns that a whole rank omitted agree. Every other column (Te, TR, nne,
    # ...) must keep its nulls so that missing data isn't silently read as a real zero
    # a selector that matches nothing makes this a no-op, so no guard is needed here
    pldflazy = pldflazy.with_columns(cs.starts_with("nnelement_", "nnion_", "nniso_").fill_null(0))

    # a back-end that read a real total number density from file keeps it. Deriving nntot there would
    # replace it with a sum over only the elements that the back-end happened to supply.
    if "nntot" not in colnames and any(col.startswith("nnelement_") for col in colnames):
        pldflazy = pldflazy.with_columns(nntot=pl.sum_horizontal(cs.starts_with("nnelement_")))

    return pldflazy


def scan_estimators(
    modelpath: Path | str = ".",
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
    join_modeldata: bool = False,
    verbose: bool = False,
    classicartis: bool = False,
) -> pl.LazyFrame:
    """Read estimator files into a polars LazyFrame with columns for timestep, modelgridindex, and estimator values.

    Selecting particular timesteps or modelgrid cells will speed this up by reducing the number of files that must be read.
    """
    modelpath = Path(modelpath)
    match_modelgridindex: Sequence[int] | None
    if modelgridindex is None:
        match_modelgridindex = None
    elif isinstance(modelgridindex, int):
        match_modelgridindex = (modelgridindex,)
    else:
        match_modelgridindex = tuple(modelgridindex)

    match_timestep: Sequence[int] | None
    if timestep is None:
        match_timestep = None
    elif isinstance(timestep, int):
        match_timestep = (timestep,)
    else:
        match_timestep = tuple(timestep)

    # a codecomparison path has no ARTIS run folders to scan, so build the frame from the reference file and
    # fall through to the shared filter/derive/join tail rather than returning early and skipping it
    is_codecomparison = path_is_codecomparison(modelpath)

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")
    if is_codecomparison:
        pldflazy = lazyframe_from_estimator_dict(
            at.codecomparison.read_reference_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        )
    elif classicartis:
        from artistools.estimators.estimators_classic import read_classic_estimators

        estimatorsdict = read_classic_estimators(modelpath)
        assert estimatorsdict is not None
        pldflazy = lazyframe_from_estimator_dict(estimatorsdict)
    else:
        pldflazy = _scan_artis_estimators(
            modelpath, match_modelgridindex=match_modelgridindex, match_timestep=match_timestep, verbose=verbose
        )

    if match_modelgridindex is not None:
        pldflazy = pldflazy.filter(pl.col("modelgridindex").is_in(match_modelgridindex))

    if match_timestep is not None:
        pldflazy = pldflazy.filter(pl.col("timestep").is_in(match_timestep))

    pldflazy = add_derived_estimator_columns(pldflazy)

    if join_modeldata:
        pldflazy, _ = join_cell_modeldata(estimators=pldflazy, modelpath=modelpath, verbose=verbose)

    return pldflazy


def _scan_artis_estimators(
    modelpath: Path, match_modelgridindex: Sequence[int] | None, match_timestep: Sequence[int] | None, verbose: bool
) -> pl.LazyFrame:
    """Scan the parquet estimator caches of an ARTIS run, or cross join model cells with timesteps if there are none."""
    mpiranklist = at.get_mpiranklist(modelpath, only_ranks_withgridcells=True)
    mpiranks_matched = (
        {at.get_mpirankofcell(modelpath=modelpath, modelgridindex=mgi) for mgi in match_modelgridindex}
        if match_modelgridindex
        else set(mpiranklist)
    )
    mpirank_groups = [
        (batchindex, mpiranks)
        for batchindex, mpiranks in enumerate(batched(mpiranklist, 100, strict=False))
        if mpiranks_matched.intersection(mpiranks)
    ]

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep)
    if runfolders:
        parquetfiles = [
            get_estimators_rankbatch_parquetfile(
                modelpath=modelpath,
                folderpath=runfolder,
                batch_mpiranks=mpiranks,
                batchindex=batchindex,
                verbose=verbose,
            )
            for runfolder in runfolders
            for batchindex, mpiranks in mpirank_groups
        ]

        assert bool(parquetfiles)
        if not verbose:
            datasize_GB = sum(pfile.stat().st_size for pfile in parquetfiles) / 1024 / 1024 / 1024
            str_runfolders = ", ".join([Path(x).relative_to(modelpath).as_posix() for x in runfolders])
            print(
                f"  scanning {len(parquetfiles)} parquet estimator files ({datasize_GB:.1f} GB) from {str_runfolders}..."
            )
        pldflazy = (
            pl
            .concat([pl.scan_parquet(pfile) for pfile in parquetfiles], how="diagonal_relaxed")
            .unique(["timestep", "modelgridindex"], maintain_order=True, keep="first")
            .lazy()
        )
    else:
        print(
            f"WARNING: No run folders found in {modelpath}. Enabling fallback to cross join of all model data and timesteps."
        )
        pldflazy = (
            at
            .get_timesteps(modelpath)
            .select("timestep", "tmid_days", "twidth_days")
            .join(at.inputmodel.get_modeldata(modelpath)[0].select("modelgridindex"), how="cross")
        )

    return pldflazy


def read_estimators(
    modelpath: Path | str = ".",
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
) -> dict[tuple[int, int], dict[str, t.Any]]:
    """Read ARTIS estimator data into a dictionary keyed by (timestep, modelgridindex).

    When collecting many cells and timesteps, this is very slow, and it's almost always better to use scan_estimators instead.
    """
    # scan_estimators already applies the modelgridindex and timestep filters
    pldfestimators = scan_estimators(modelpath, modelgridindex, timestep).collect()

    estimators: dict[tuple[int, int], dict[str, t.Any]] = {}
    for estimtsmgi in pldfestimators.iter_rows(named=True):
        ts, mgi = estimtsmgi["timestep"], estimtsmgi["modelgridindex"]
        estimators[ts, mgi] = {
            k: v for k, v in estimtsmgi.items() if k not in {"timestep", "modelgridindex"} and v is not None
        }

    return estimators


def get_averageexcitation(
    modelpath: Path | str, atomic_number: int, ion_stage: int, dftexc: pl.LazyFrame
) -> pl.LazyFrame:
    """Return the population-weighted mean level excitation energy [eV] of an ion per timestep and cell.

    dftexc gives the excitation temperature (columns timestep, modelgridindex, T_exc) used to spread
    the superlevel population over the levels it stands in for.
    """
    dfpops = (
        at.nltepops
        .read_files(modelpath)
        .lazy()
        .filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage))
    )

    adata = at.atomic.get_levels(modelpath)
    dfionlevels = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage))["levels"].item()
    if dfionlevels is None:
        msg = f"No level data for Z={atomic_number} ion_stage={ion_stage}"
        raise ValueError(msg)
    dflevels = dfionlevels.lazy().select(
        level=pl.col("levelindex").cast(pl.Int64), energy_ev=pl.col("energy_ev"), g=pl.col("g")
    )

    groupcols = ["timestep", "modelgridindex"]

    # resolved levels contribute their own energy; the superlevel is added below
    dfresolved = (
        dfpops
        .filter(pl.col("level") >= 0)
        .join(dflevels, on="level", how="inner")
        .group_by(groupcols)
        .agg(energypopsum=(pl.col("energy_ev") * pl.col("n_NLTE")).sum(), levelnumber_sl=pl.col("level").max() + 1)
    )

    dfionpopsum = dfpops.group_by(groupcols).agg(ionpopsum=pl.col("n_NLTE").sum())

    dfsuperlevel = (
        dfpops
        .filter(pl.col("level") < 0)
        .group_by(groupcols)
        .agg(n_NLTE_sl=pl.col("n_NLTE").sum())
        .join(dfresolved.select([*groupcols, "levelnumber_sl"]), on=groupcols, how="inner")
        .join(dftexc, on=groupcols, how="inner")
        .collect()
    )

    dfsuperlevelenergy = _superlevel_energy(dfsuperlevel, dflevels.collect(), groupcols)

    # inner join on dftexc, so the result covers exactly the cells a temperature was given for and a
    # missing one cannot silently drop the superlevel term
    return (
        dfresolved
        .join(dftexc.select(groupcols), on=groupcols, how="inner")
        .join(dfionpopsum, on=groupcols, how="inner")
        .join(dfsuperlevelenergy.lazy(), on=groupcols, how="left")
        .with_columns(
            averageexcitation=(pl.col("energypopsum") + pl.col("energypopsum_sl").fill_null(0.0)) / pl.col("ionpopsum")
        )
        .select([*groupcols, "averageexcitation"])
    )


def _superlevel_energy(dfsuperlevel: pl.DataFrame, dflevels: pl.DataFrame, groupcols: list[str]) -> pl.DataFrame:
    """Return the energy the superlevel population contributes, Boltzmann-distributed at T_exc."""
    schema: dict[str, pl.DataType] = {
        **{col: dfsuperlevel.schema[col] for col in groupcols},
        "energypopsum_sl": pl.Float64(),
    }
    if dfsuperlevel.is_empty():
        return pl.DataFrame(schema=schema)

    # levelnumber_sl is the same for every cell in practice, so this loops once
    contributions = []
    for levelnumber_sl in dfsuperlevel["levelnumber_sl"].unique().sort():
        dflevels_above = dflevels.filter(pl.col("level") >= levelnumber_sl)
        if dflevels_above.is_empty():
            continue

        contributions.append(
            dfsuperlevel
            .filter(pl.col("levelnumber_sl") == levelnumber_sl)
            .join(dflevels_above, how="cross")
            .with_columns(boltzfac=pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("T_exc")).exp())
            .group_by(groupcols)
            .agg(
                energypopsum_sl=pl.col("n_NLTE_sl").first()
                * (pl.col("energy_ev") * pl.col("boltzfac")).sum()
                / pl.col("boltzfac").sum()
            )
        )

    return pl.concat(contributions) if contributions else pl.DataFrame(schema=schema)
