"""Element symbols, ion stages, nuclide data, and atomic line/bound-free transition tables."""

import string
import typing as t
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

from artistools.commands import get_path
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import write_parquet_atomic
from artistools.misc.fileio import zopen

roman_numerals = (
    "",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)


@lru_cache(maxsize=8)
def get_composition_data(filename: Path | str) -> pl.DataFrame:
    """Return a DataFrame containing details of included elements and ions."""
    filename = Path(filename, "compositiondata.txt") if Path(filename).is_dir() else Path(filename)

    rows = []
    with filename.open(encoding="utf-8") as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        for _ in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            row = [int(x) for x in linesplit[:5]] + [float(x) for x in linesplit[5:]]

            rows.append(row)

    return pl.DataFrame(
        rows,
        schema=[
            ("Z", pl.Int32),
            ("nions", pl.Int32),
            ("lowermost_ion_stage", pl.Int32),
            ("uppermost_ion_stage", pl.Int32),
            ("nlevelsmax_readin", pl.Int32),
            ("abundance", pl.Float64),
            ("mass", pl.Float64),
        ],
        orient="row",
    )


def get_composition_data_from_outputfile(modelpath: Path | str) -> pl.DataFrame:
    """Read ion list from output file."""
    element_Z = []
    lowermost_ion_stage: list[int | None] = []
    uppermost_ion_stage: list[int | None] = []

    with Path(modelpath, "output_0-0.txt").open(encoding="utf-8") as foutput:
        Z: int | None = None
        elementindex = -1
        for row in foutput:
            if row.split()[0] == "[input.c]":
                split_row = row.split()
                if split_row[1] == "element":
                    Z = int(split_row[4])
                    elementindex += 1
                    element_Z.append(Z)
                    lowermost_ion_stage.append(None)
                    uppermost_ion_stage.append(None)
                elif split_row[1] == "ion":
                    assert Z is not None
                    ion_stage = int(split_row[2])
                    if lowermost_ion_stage[-1] is None:
                        lowermost_ion_stage[-1] = ion_stage
                    else:
                        lowermost_ion_stage[-1] = min(lowermost_ion_stage[-1], ion_stage)
                    if uppermost_ion_stage[-1] is None:
                        uppermost_ion_stage[-1] = ion_stage
                    else:
                        uppermost_ion_stage[-1] = max(uppermost_ion_stage[-1], ion_stage)

    return pl.DataFrame(
        zip(element_Z, lowermost_ion_stage, uppermost_ion_stage, strict=True),
        schema=[("Z", pl.Int32), ("lowermost_ion_stage", pl.Int32), ("uppermost_ion_stage", pl.Int32)],
        orient="row",
    ).with_columns(nions=pl.col("uppermost_ion_stage") - pl.col("lowermost_ion_stage") + 1)


def get_z_a_nucname(nucname: str) -> tuple[int, int]:
    """Return atomic number and mass number from a string like 'Pb208', 'X_Pb208', or "nniso_Pb208' (returns 82, 208)."""
    if "_" in nucname:
        nucname = nucname.split("_")[1]

    z = get_atomic_number(nucname.rstrip(string.digits))
    assert z > 0

    a = int(nucname.lower().lstrip(string.ascii_lowercase))

    return z, a


@lru_cache(maxsize=1)
def get_elsymbolslist() -> list[str]:
    """Return a list of element symbols.

    Example:
    -------
    elsymbolslist()[26] = 'Fe'.

    """
    return ["n", *pl.read_csv(get_path("datadir") / "elements.csv", has_header=True, separator=",")["symbol"].to_list()]


def get_elsymbols_df() -> pl.LazyFrame:
    """Return a polars LazyFrame of atomic number and element symbols."""
    return (
        pl
        .scan_csv(
            get_path("datadir") / "elements.csv", separator=",", has_header=True, schema_overrides={"Z": pl.Int32}
        )
        .drop("name")
        .rename({"symbol": "elsymbol", "Z": "atomic_number"})
    )


def get_atomic_number(elsymbol: str) -> int:
    """Return the atomic number of an element symbol."""
    assert elsymbol is not None
    elsymbol = elsymbol.removeprefix("X_")
    elsymbol = elsymbol.split("_")[0].split("-")[0].rstrip(string.digits)

    if elsymbol.title() in get_elsymbolslist():
        return get_elsymbolslist().index(elsymbol.title())

    return -1


def decode_roman_numeral(strin: str) -> int:
    """Return the integer corresponding to a Roman numeral."""
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def get_ion_stage_roman_numeral_df() -> pl.DataFrame:
    """Return a polars DataFrame of ionisation stage and roman numerals."""
    return pl.DataFrame({"ion_stage_roman": roman_numerals[1:]}, schema={"ion_stage_roman": pl.String}).with_row_index(
        "ion_stage", offset=1
    )


def get_elsymbol(atomic_number: int | np.int64) -> str:
    """Return the element symbol of an atomic number."""
    return get_elsymbolslist()[atomic_number]


def get_ion_tuple(ionstr: str) -> tuple[int, int] | int:
    """Return a tuple of the atomic number and ionisation stage such as (26,2) for an ion string like 'FeII', 'Fe II', or '26_2'.

    Return the atomic number for a string like 'Fe' or '26'.
    """
    ionstr = ionstr.removeprefix("X_").removeprefix("nnelement_").removeprefix("nnion_")

    if ionstr.isdigit():
        return int(ionstr)

    if ionstr in get_elsymbolslist():
        return get_atomic_number(ionstr)

    elem = "?"
    strion_stage = "?"
    if " " in ionstr:
        elem, strion_stage = ionstr.split(" ")
    elif "_" in ionstr:
        elem, strion_stage = ionstr.split("_")
    else:
        for elsym in get_elsymbolslist():
            if ionstr.startswith(elsym):
                elem = elsym
                strion_stage = ionstr.removeprefix(elsym)
                break

    if not elem:
        msg = f"Could not parse ionstr {ionstr}"
        raise ValueError(msg)

    atomic_number = int(elem) if elem.isdigit() else get_atomic_number(elem)
    ion_stage = int(strion_stage) if strion_stage.isdigit() else decode_roman_numeral(strion_stage)

    return (atomic_number, ion_stage)


@lru_cache(maxsize=16)
def get_ionstring(
    atomic_number: int | np.int64,
    ion_stage: int | np.int64 | t.Literal["ALL"] | None,
    style: t.Literal["spectral", "chargelatex", "charge"] = "spectral",
    sep: str = " ",
) -> str:
    """Return a string with the element symbol and ionisation stage."""
    if ion_stage is None or ion_stage == "ALL":
        return get_elsymbol(atomic_number)

    if isinstance(ion_stage, str) and ion_stage.startswith(get_elsymbol(atomic_number)):
        # nuclides like Sr89 get passed in as atomic_number=38, ion_stage='Sr89'
        return ion_stage

    assert not isinstance(ion_stage, str)

    if style == "spectral":
        return f"{get_elsymbol(atomic_number)}{sep}{roman_numerals[ion_stage]}"

    strcharge = ""
    if style == "chargelatex":
        # ion notion e.g. Co+, Fe2+
        if ion_stage > 2:
            strcharge = r"$^{" + str(ion_stage - 1) + r"{+}}$"
        elif ion_stage == 2:
            strcharge = r"$^{+}$"
        elif ion_stage == 1:
            strcharge = r"$^{0}$"
    elif ion_stage > 2:
        strcharge = f"{ion_stage - 1}+"
    elif ion_stage == 2:
        strcharge = "+"
    elif ion_stage == 1:
        strcharge = "0"

    return f"{get_elsymbol(atomic_number)}{strcharge}"


def get_nuclides(modelpath: Path | str) -> pl.LazyFrame:
    """Return LazyFrame with columns: pellet_nucindex, atomic_number, A, nucname from nuclides.out file and the -1 initial energy special case."""
    filepath = Path(modelpath, "nuclides.out")
    if not filepath.is_file():
        msg = f"File {filepath} not found"
        raise FileNotFoundError(msg)

    dfnuclides = (
        pl
        .scan_csv(filepath, separator=" ", has_header=True)
        .rename({"#nucindex": "pellet_nucindex", "Z": "atomic_number"})
        .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
        .with_columns(nucname=pl.col("elsymbol") + pl.col("A").cast(pl.String))
    ).with_columns(pl.col(pl.Int64).cast(pl.Int32))

    return pl.concat(
        [
            pl.LazyFrame(
                {
                    "pellet_nucindex": [-1],
                    "atomic_number": [-1],
                    "A": [-1],
                    "elsymbol": ["initial energy"],
                    "nucname": ["initial energy"],
                },
                schema=dfnuclides.collect_schema(),
            ),
            dfnuclides,
        ],
        how="vertical",
    ).lazy()


def get_bflist(modelpath: Path | str, get_ion_str: bool = False) -> pl.LazyFrame:
    """Return a LazyFrame of bound-free transitions from bflist.out."""
    compositiondata = get_composition_data(modelpath)
    bflistpath = firstexisting(["bflist.out", "bflist.dat"], folder=modelpath, tryzipped=True)
    print(f"Reading {bflistpath}")
    schema = {
        "bfindex": pl.Int32,
        "elementindex": pl.Int32,
        "ionindex": pl.Int32,
        "lowerlevel": pl.Int32,
        "upperionlevel": pl.Int32,
    }
    try:
        dfboundfree = pl.scan_csv(
            bflistpath,
            skip_rows=1,
            has_header=False,
            separator=" ",
            new_columns=["bfindex", "elementindex", "ionindex", "lowerlevel", "upperionlevel"],
            schema_overrides=schema,
        )
    except pl.exceptions.NoDataError:
        dfboundfree = pl.DataFrame(schema=schema).lazy()

    dfboundfree = dfboundfree.with_columns(
        atomic_number=pl.col("elementindex").map_elements(compositiondata["Z"].item, return_dtype=pl.Int32),
        ion_stage=(
            pl.col("ionindex")
            + pl.col("elementindex").map_elements(compositiondata["lowermost_ion_stage"].item, return_dtype=pl.Int32)
        ),
    )

    dfboundfree = dfboundfree.drop(["elementindex", "ionindex"])

    if get_ion_str:
        dfboundfree = (
            dfboundfree
            .join(get_ion_stage_roman_numeral_df().lazy(), on="ion_stage", how="left")
            .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
            .with_columns(ion_str=pl.col("elsymbol") + " " + pl.col("ion_stage_roman"))
        )

    return dfboundfree


class LineTuple(t.NamedTuple):
    """Named tuple for a line in linestat.out."""

    lambda_angstroms: float
    atomic_number: int
    ion_stage: int
    upperlevelindex: int
    lowerlevelindex: int


def read_linestatfile(
    filepath: Path | str,
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    """Load linestat.out containing transitions wavelength, element, ion, upper and lower levels."""
    if Path(filepath).is_dir():
        filepath = firstexisting("linestat.out", folder=filepath, tryzipped=True)

    print(f"Reading {filepath}")

    data = np.loadtxt(zopen(filepath))
    lambda_angstroms = data[0] * 1e8
    nlines = len(lambda_angstroms)

    atomic_numbers = data[1].astype(np.int32)
    assert len(atomic_numbers) == nlines

    ion_stages = data[2].astype(np.int32)
    assert len(ion_stages) == nlines

    # the file adds one to the levelindex, i.e. lowest level is 1
    upper_levels = data[3].astype(np.int32)
    assert len(upper_levels) == nlines

    lower_levels = data[4].astype(np.int32)
    assert len(lower_levels) == nlines

    return lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels


def get_linelist_pldf(modelpath: Path | str, get_ion_str: bool = False) -> pl.LazyFrame:
    textfile = firstexisting("linestat.out", folder=modelpath)
    parquetfile = Path(modelpath, "linelist.out.parquet")
    if not parquetfile.is_file() or parquetfile.stat().st_mtime < textfile.stat().st_mtime:
        lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels = read_linestatfile(textfile)

        pldf = (
            pl
            .DataFrame({
                "lambda_angstroms": lambda_angstroms,
                "atomic_number": atomic_numbers,
                "ion_stage": ion_stages,
                "upper_level": upper_levels,
                "lower_level": lower_levels,
            })
            .with_row_index(name="lineindex")
            .with_columns(cs.integer().cast(pl.Int32), cs.float().cast(pl.Float32))
        )
        write_parquet_atomic(pldf, parquetfile, compression_level=8)
        print(f"Wrote {parquetfile}")
    else:
        print(f"Reading {parquetfile}")

    linelist_lazy = (
        pl
        .scan_parquet(parquetfile)
        .with_columns(
            pl
            .when(pl.col("lambda_angstroms").is_between(2000, 20000))
            .then(pl.col("lambda_angstroms") / 1.0003)
            .otherwise(pl.col("lambda_angstroms"))
            .alias("lambda_angstroms_air"),
            pl.col(pl.UInt32).cast(pl.Int32),
            pl.col(pl.Int64).cast(pl.Int32),
            pl.col(pl.Float64).cast(pl.Float32),
        )
        .with_columns(upperlevelindex=pl.col("upper_level") - 1, lowerlevelindex=pl.col("lower_level") - 1)
        .drop(["upper_level", "lower_level"])
        .with_columns(pl.col(pl.Int64).cast(pl.Int32))
    )

    if "ionstage" in linelist_lazy.collect_schema().names():
        linelist_lazy = linelist_lazy.rename({"ionstage": "ion_stage"})

    if get_ion_str:
        linelist_lazy = (
            linelist_lazy
            .join(get_ion_stage_roman_numeral_df().lazy(), on="ion_stage", how="left")
            .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
            .with_columns(ion_str=pl.col("elsymbol") + " " + pl.col("ion_stage_roman"))
        )

    return linelist_lazy
