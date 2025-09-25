import io
import time
import typing as t
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at


def parse_adata(
    fadata: io.TextIOBase,
    phixsdict: dict[tuple[int, int, int], tuple[npt.NDArray[np.floating[t.Any]], npt.NDArray[np.floating[t.Any]]]],
    ionlist: Collection[tuple[int, int]] | None,
) -> Generator[tuple[int, int, int, float, pl.DataFrame]]:
    """Generate ions and their level lists from adata.txt."""
    firstlevelnumber = 1

    for line in fadata:
        if not line.strip():
            continue

        ionheader = line.split()
        Z = int(ionheader[0])
        ion_stage = int(ionheader[1])
        level_count = int(ionheader[2])

        if not ionlist or (Z, ion_stage) in ionlist:
            level_list: list[
                tuple[
                    float,
                    float,
                    int,
                    str | None,
                    npt.NDArray[np.floating[t.Any]] | None,
                    npt.NDArray[np.floating[t.Any]] | None,
                ]
            ] = []
            for levelindex in range(level_count):
                row = fadata.readline().split(maxsplit=4)

                levelname = (row[4]).strip("'") if len(row) >= 5 else None
                inputlevelnumber = int(row[0])
                assert levelindex == inputlevelnumber - firstlevelnumber
                phixstargetlist, phixstable = phixsdict.get((Z, ion_stage, inputlevelnumber), (None, None))

                level_list.append((float(row[1]), float(row[2]), int(row[3]), levelname, phixstargetlist, phixstable))

            dflevels = (
                pl.DataFrame(
                    level_list,
                    schema=[
                        ("energy_ev", pl.Float64),
                        ("g", pl.Float32),
                        ("transition_count", pl.Int32),
                        ("levelname", pl.Utf8),
                        ("phixstargetlist", pl.Object),
                        ("phixstable", pl.Object),
                    ],
                    orient="row",
                )
                .with_row_index("levelindex")
                .with_columns(pl.col("levelindex").cast(pl.Int32))
            )

            ionisation_energy_ev = float(ionheader[3])
            yield Z, ion_stage, level_count, ionisation_energy_ev, dflevels

        else:
            for _ in range(level_count):
                fadata.readline()


def parse_phixsdata(
    phixs_filename: Path | str, ionlist: Collection[tuple[int, int]] | None = None
) -> dict[tuple[int, int, int], tuple[npt.NDArray[np.floating[t.Any]], npt.NDArray[np.floating[t.Any]]]]:
    firstlevelnumber = 1
    phixsdict = {}
    with at.zopen(phixs_filename) as fphixs:
        nphixspoints = int(fphixs.readline())
        phixsnuincrement = float(fphixs.readline())
        xgrid = np.linspace(
            1.0, 1.0 + phixsnuincrement * nphixspoints, num=nphixspoints, endpoint=False, dtype=np.float64
        )
        for line in fphixs:
            if not line.strip():
                continue

            ionheader = line.split()
            Z = int(ionheader[0])
            upperion_stage = int(ionheader[1])
            upperionlevel = int(ionheader[2]) - firstlevelnumber
            lowerion_stage = int(ionheader[3])
            lowerionlevel = int(ionheader[4]) - firstlevelnumber
            # threshold_ev = float(ionheader[5])

            assert upperion_stage == lowerion_stage + 1

            if upperionlevel >= 0:
                nptargetlist = np.array([(upperionlevel, 1.0)], dtype=[("level", np.int32), ("fraction", np.float32)])
            else:
                ntargets = int(fphixs.readline())
                nptargetlist = np.empty((ntargets, 2), dtype=[("level", np.int32), ("fraction", np.float32)])
                # targetlist = [(-1, 0.0) for _ in range(ntargets)]
                for phixstargetindex in range(ntargets):
                    level, fraction = fphixs.readline().split()
                    nptargetlist[phixstargetindex, :] = (int(level) - firstlevelnumber, float(fraction))

            if not ionlist or (Z, lowerion_stage) in ionlist:
                phixslist = [float(fphixs.readline()) * 1e-18 for _ in range(nphixspoints)]
                phixstable = np.array(
                    list(zip(xgrid, phixslist, strict=True)), dtype=[("x", np.float64), ("sigma_cm2", np.float32)]
                )

                phixsdict[Z, lowerion_stage, lowerionlevel] = (nptargetlist, phixstable)

            else:
                for _ in range(nphixspoints):
                    fphixs.readline()

    return phixsdict


def add_transition_columns(
    dftransitions: pl.LazyFrame | pl.DataFrame, dflevels: pl.DataFrame | pl.LazyFrame, columns: Sequence[str]
) -> pl.LazyFrame:
    """Add columns to a polars DataFrame of transitions."""
    dftransitions = dftransitions.lazy()
    columns_before = dftransitions.collect_schema().names()

    dflevels = dflevels.select(["g", "energy_ev", "levelname", "levelindex"]).lazy()

    dftransitions = (
        dftransitions.join(
            dflevels.select(
                lower="levelindex",
                lower_g=pl.col("g"),
                lower_energy_ev=pl.col("energy_ev"),
                lower_level=pl.col("levelname"),
            ),
            how="left",
            on="lower",
            coalesce=True,
        )
        .join(
            dflevels.select(
                upper="levelindex",
                upper_g=pl.col("g"),
                upper_energy_ev=pl.col("energy_ev"),
                upper_level=pl.col("levelname"),
            ),
            how="left",
            on="upper",
            coalesce=True,
        )
        .with_columns(epsilon_trans_ev=(pl.col("upper_energy_ev") - pl.col("lower_energy_ev")))
    )

    hc = 12398.419843320025  # h * c in eV * Angstrom
    dftransitions = dftransitions.with_columns(lambda_angstroms=hc / pl.col("epsilon_trans_ev"))

    # clean up any columns used for intermediate calculations
    dftransitions.drop(
        col
        for col in dftransitions.collect_schema().names()
        if col not in columns_before and col not in columns and col != "levelindex"
    )

    for col in columns:
        assert col in dftransitions.collect_schema().names(), f"Invalid column name {col}"

    return dftransitions


def get_transitiondata(
    modelpath: str | Path, ionlist: Collection[tuple[int, int]] | None = None, quiet: bool = False
) -> dict[tuple[int, int], pl.LazyFrame]:
    """Return a dictionary of transitions."""
    ionlist = set(ionlist) if ionlist else None
    transition_filename = at.firstexisting("transitiondata.txt", folder=modelpath)

    time_start = time.perf_counter()
    if not quiet:
        print(f"Reading {transition_filename.relative_to(Path(modelpath).parent)}...")

    transitionsdict = {
        k: pl.DataFrame(v).lazy()
        for k, v in at.rustext.read_transitiondata(str(transition_filename), ionlist=ionlist).items()
    }

    if not quiet:
        print(f"  took {time.perf_counter() - time_start:.2f} seconds")

    return transitionsdict


def get_levels(
    modelpath: str | Path,
    ionlist: Collection[tuple[int, int]] | None = None,
    get_transitions: bool = False,
    get_photoionisations: bool = False,
    quiet: bool = False,
    derived_transitions_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return a polars DataFrame of energy levels."""
    adatafilename = Path(modelpath, "adata.txt")

    transitionsdict = get_transitiondata(modelpath, ionlist=ionlist, quiet=quiet) if get_transitions else {}

    phixsdict = {}
    if get_photoionisations:
        phixs_filename = Path(modelpath, "phixsdata_v2.txt")

        if not quiet:
            print(f"Reading {phixs_filename.relative_to(Path(modelpath).parent)}")

        phixsdict = parse_phixsdata(phixs_filename, ionlist)

    level_lists = []

    class IonTuple(t.NamedTuple):
        Z: int
        ion_stage: int
        level_count: int
        ion_pot: float
        levels: pl.DataFrame
        transitions: pl.LazyFrame

    with at.zopen(adatafilename) as fadata:
        if not quiet:
            print(f"Reading {adatafilename.relative_to(Path(modelpath).parent)}")

        for Z, ion_stage, level_count, ionisation_energy_ev, dflevels in parse_adata(fadata, phixsdict, ionlist):
            if (Z, ion_stage) in transitionsdict:
                dftransitions = transitionsdict[Z, ion_stage]
                if derived_transitions_columns is not None:
                    dftransitions = add_transition_columns(dftransitions, dflevels, derived_transitions_columns)
            else:
                dftransitions = pl.LazyFrame()

            level_lists.append(IonTuple(Z, ion_stage, level_count, ionisation_energy_ev, dflevels, dftransitions))

    return pl.DataFrame(level_lists)


def parse_recombratefile(frecomb: io.TextIOBase) -> Generator[tuple[int, int, pl.DataFrame]]:
    """Parse recombrates.txt file."""
    for line in frecomb:
        Z, upper_ion_stage, t_count = (int(x) for x in line.split())
        arr_log10t = []
        arr_rrc_low_n = []
        arr_rrc_total = []
        for _ in range(t_count):
            log10t, rrc_low_n, rrc_total = (float(x) for x in frecomb.readline().split())

            arr_log10t.append(log10t)
            arr_rrc_low_n.append(rrc_low_n)
            arr_rrc_total.append(rrc_total)

        recombdata_thision = pl.DataFrame({
            "log10T_e": arr_log10t,
            "rrc_low_n": arr_rrc_low_n,
            "rrc_total": arr_rrc_total,
        })

        recombdata_thision = recombdata_thision.with_columns(T_e=10 ** pl.col("log10T_e"))

        yield Z, upper_ion_stage, recombdata_thision


@lru_cache(maxsize=4)
def get_ionrecombratecalibration(modelpath: str | Path) -> dict[tuple[int, int], pl.DataFrame]:
    """Read recombrates.txt file."""
    recombdata = {}
    with Path(modelpath, "recombrates.txt").open("r", encoding="utf-8") as frecomb:
        for Z, upper_ion_stage, dfrrc in parse_recombratefile(frecomb):
            recombdata[Z, upper_ion_stage] = dfrrc

    return recombdata
