"""Artistools - NLTE population related functions."""

import re
import string
from collections.abc import Sequence
from pathlib import Path

import polars as pl

import artistools as at
from artistools.constants import K_B_ev_per_K


def texifyterm(strterm: str) -> str:
    """Replace a term string with TeX notation equivalent."""
    strtermtex = ""
    passed_term_Lchar = False

    for termpiece in re.split(r"([_A-Za-z])", strterm):
        if re.match(r"[0-9]", termpiece) is not None and not passed_term_Lchar:
            # 2S + 1 number
            strtermtex += r"$^{" + termpiece + r"}$"
        elif re.match(r"[A-Z]", termpiece) is not None:
            # L character - SPDFGH...
            strtermtex += termpiece
            passed_term_Lchar = True
        elif re.match(r"[eo]", termpiece) is not None and passed_term_Lchar:
            # odd flag, but don't want to confuse it with the energy index (e.g. o4Fo[2])
            if termpiece != "e":  # even is assumed by default (and looks neater with all the 'e's)
                strtermtex += r"$^{\rm " + termpiece + r"}$"
        elif re.match(r"[0-9]?.*\]", termpiece) is not None:
            # J value
            strtermtex += termpiece.split("[")[0] + r"$_{" + termpiece.lstrip(string.digits).strip("[]") + r"}$"
        elif re.match(r"[0-9]", termpiece) is not None and passed_term_Lchar:
            # extra number after S char
            strtermtex += termpiece

    return strtermtex.replace("$$", "")


def texifyconfiguration(levelname: str) -> str:
    """Replace a level configuration with the formatted LaTeX equivalent."""
    # the underscore gets confused with LaTeX subscript operator, so switch it to the hash symbol
    levelname = levelname.strip()
    strout = "#".join(levelname.split("_")[:-1]) + "#"
    for strorbitalocc in re.findall(r"[0-9][a-z][0-9]?[#(]", strout):
        n, lchar, occ = re.split(r"([a-z])", strorbitalocc)
        lastchar = "(" if occ.endswith("(") else "#"
        occ = occ.rstrip("#(")
        strorbitalocctex = n + lchar + (r"$^{" + occ + r"}$" if occ else "") + lastchar
        strout = strout.replace(strorbitalocc, strorbitalocctex)

    for parentterm in re.findall(r"\([0-9][A-Z][^)]?\)", strout):
        parentermtex = f"({texifyterm(parentterm.strip('()'))})"
        strout = strout.replace(parentterm, parentermtex)
    strterm = levelname.split("_")[-1]
    strout += " " + texifyterm(strterm)

    return strout.replace("#", "").replace("$$", "")


def add_lte_pops(
    dfpop: pl.DataFrame,
    adata: pl.DataFrame,
    columntemperature_tuples: Sequence[tuple[str, float | int]],
    noprint: bool = False,
    maxlevel: int = -1,
) -> pl.DataFrame:
    """Add columns to dfpop with LTE populations.

    columntemperature_tuples is a sequence of tuples of column name and temperature, e.g., ('mycolumn', 3000)
    """
    ionlevels_of_ion = {
        (Z, ion_stage): adata.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["levels"].item(0)
        for Z, ion_stage in dfpop.select("Z", "ion_stage").unique(maintain_order=True).iter_rows()
    }

    def ltepop_exprs(ionlevels: pl.DataFrame) -> list[pl.Expr]:
        """Return one expression per output column giving each level's LTE population relative to the ground state."""
        gs_g = float(ionlevels["g"].item(0))
        gs_energy = float(ionlevels["energy_ev"].item(0))
        return [
            (pl.col("g") / gs_g * (-(pl.col("energy_ev") - gs_energy) / K_B_ev_per_K / T_exc).exp()).alias(columnname)
            for columnname, T_exc in columntemperature_tuples
        ]

    # the LTE populations depend only on the ion and level index, so build one lookup frame and join it on.
    # Superlevel rows (level -1) and levels beyond the atomic data are left null, as before
    dflte = pl.concat([
        ionlevels.select(
            pl.lit(Z, dtype=dfpop.schema["Z"]).alias("Z"),
            pl.lit(ion_stage, dtype=dfpop.schema["ion_stage"]).alias("ion_stage"),
            pl.int_range(pl.len()).cast(dfpop.schema["level"]).alias("level"),
            *ltepop_exprs(ionlevels),
        )
        for (Z, ion_stage), ionlevels in ionlevels_of_ion.items()
    ])
    dfpop = dfpop.join(dflte, on=["Z", "ion_stage", "level"], how="left", maintain_order="left")

    for modelgridindex, timestep, Z, ion_stage in (
        dfpop
        .filter(pl.col("level") == -1)
        .select("modelgridindex", "timestep", "Z", "ion_stage")
        .unique(maintain_order=True)
        .iter_rows()
    ):
        maskion = (
            (pl.col("modelgridindex") == modelgridindex)
            & (pl.col("timestep") == timestep)
            & (pl.col("Z") == Z)
            & (pl.col("ion_stage") == ion_stage)
        )

        maxlevel_ion = dfpop.filter(maskion)["level"].max()
        assert isinstance(maxlevel_ion, int)
        levelnumber_sl = maxlevel_ion + 1

        if maxlevel < 0 or levelnumber_sl <= maxlevel:
            if not noprint:
                print(f"{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]} has a superlevel at level {levelnumber_sl}")

            ionlevels = ionlevels_of_ion[Z, ion_stage]
            superlevelpops = ionlevels[levelnumber_sl:].select(ltepop_exprs(ionlevels)).sum()
            dfpop = dfpop.with_columns(
                pl
                .when(maskion & (pl.col("level") == -1))
                .then(superlevelpops[columnname].item())
                .otherwise(pl.col(columnname))
                .alias(columnname)
                for columnname, _ in columntemperature_tuples
            )

        dfpop = dfpop.with_columns(
            pl
            .when(maskion & (pl.col("level") == -1))
            .then(levelnumber_sl + 2)
            .otherwise(pl.col("level"))
            .alias("level")
        )

    return dfpop


def read_files(
    modelpath: str | Path, timestep: int | None = None, modelgridindex: int | Sequence[int] | None = None
) -> pl.DataFrame:
    """Read in NLTE populations from a model for a particular timestep and one or more grid cells."""
    return at.read_rank_outputfiles(
        modelpath, "nlte_{mpirank:04d}.out", timestep=timestep, modelgridindex=modelgridindex
    )
