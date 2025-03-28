"""Artistools - NLTE population related functions."""

import math
import re
import string
import typing as t
from functools import lru_cache
from functools import partial
from pathlib import Path

import pandas as pd
import polars as pl

import artistools as at


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


def add_lte_pops(dfpop, adata, columntemperature_tuples, noprint=False, maxlevel=-1):
    """Add columns to dfpop with LTE populations.

    columntemperature_tuples is a sequence of tuples of column name and temperature, e.g., ('mycolumn', 3000)
    """
    K_B = 8.617333262145179e-05  # eV / K

    for _, row in dfpop.drop_duplicates(["modelgridindex", "timestep", "Z", "ion_stage"]).iterrows():
        modelgridindex = int(row.modelgridindex)
        timestep = int(row.timestep)
        Z = int(row.Z)
        ion_stage = int(row.ion_stage)

        ionlevels = adata.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["levels"].item(0)

        gs_g = ionlevels["g"].item(0)
        gs_energy = ionlevels["energy_ev"].item(0)

        # gs_pop = dfpop.query(
        #     "modelgridindex == @modelgridindex and timestep == @timestep "
        #     "and Z == @Z and ion_stage == @ion_stage and level == 0"
        # ).iloc[0]["n_NLTE"]

        masksuperlevel = (
            (dfpop["modelgridindex"] == modelgridindex)
            & (dfpop["timestep"] == timestep)
            & (dfpop["Z"] == Z)
            & (dfpop["ion_stage"] == ion_stage)
            & (dfpop["level"] == -1)
        )

        masknotsuperlevel = (
            (dfpop["modelgridindex"] == modelgridindex)
            & (dfpop["timestep"] == timestep)
            & (dfpop["Z"] == Z)
            & (dfpop["ion_stage"] == ion_stage)
            & (dfpop["level"] != -1)
        )

        def f_ltepop(x, T_exc: float, gsg: float, gse: float, ionlevels) -> float:
            ltepop = (
                ionlevels["g"].item(int(x.level))
                / gsg
                * math.exp(-(ionlevels["energy_ev"].item(int(x.level)) - gse) / K_B / T_exc)
            )
            assert isinstance(ltepop, float)
            return ltepop

        for columnname, T_exc in columntemperature_tuples:
            dfpop.loc[masknotsuperlevel, columnname] = dfpop.loc[masknotsuperlevel].apply(
                f_ltepop, args=(T_exc, gs_g, gs_energy, ionlevels), axis=1
            )

        if not dfpop[masksuperlevel].empty:
            levelnumber_sl = (
                dfpop.query(
                    "modelgridindex == @modelgridindex and timestep == @timestep "
                    "and Z == @Z and ion_stage == @ion_stage"
                ).level.max()
                + 1
            )

            if maxlevel < 0 or levelnumber_sl <= maxlevel:
                if not noprint:
                    print(
                        f"{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]} "
                        f"has a superlevel at level {levelnumber_sl}"
                    )

                for columnname, T_exc in columntemperature_tuples:
                    superlevelpop = (
                        ionlevels[levelnumber_sl:]
                        .select(pl.col("g") / gs_g * (-(pl.col("energy_ev") - gs_energy) / K_B / T_exc).exp())
                        .sum()
                        .item()
                    )
                    dfpop.loc[masksuperlevel, columnname] = superlevelpop

            dfpop.loc[masksuperlevel, "level"] = levelnumber_sl + 2

    return dfpop


def read_file(nltefilepath: str | Path) -> pd.DataFrame:
    """Read NLTE populations from one file."""
    try:
        nltefilepath = at.firstexisting(nltefilepath, tryzipped=True)
    except FileNotFoundError:
        # print(f"Warning: Could not find {nltefilepath}")
        return pd.DataFrame()

    filesize = Path(nltefilepath).stat().st_size / 1024 / 1024
    print(f"Reading {nltefilepath} ({filesize:.2f} MiB)")

    try:
        dfpop = pd.read_csv(nltefilepath, sep=r"\s+")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return dfpop.rename(columns={"ionstage": "ion_stage"}, errors="ignore")


def read_file_filtered(nltefilepath, strquery=None, dfqueryvars=None):
    dfpopfile = read_file(nltefilepath)

    if strquery and not dfpopfile.empty:
        dfpopfile = dfpopfile.query(strquery, local_dict=dfqueryvars)

    return dfpopfile


@lru_cache(maxsize=2)
def read_files(
    modelpath: str | Path,
    timestep: int = -1,
    modelgridindex: int = -1,
    dfquery: str | None = None,
    dfqueryvars: dict[str, t.Any] | None = None,
) -> pd.DataFrame:
    """Read in NLTE populations from a model for a particular timestep and grid cell."""
    if dfqueryvars is None:
        dfqueryvars = {}

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)

    nltefilepaths = [
        Path(folderpath, f"nlte_{mpirank:04d}.out")
        for folderpath in at.get_runfolders(modelpath, timestep=timestep)
        for mpirank in mpiranklist
    ]

    dfqueryvars["modelgridindex"] = modelgridindex
    dfqueryvars["timestep"] = timestep

    dfquery_full = "timestep==@timestep" if timestep >= 0 else ""
    if modelgridindex >= 0:
        if dfquery_full:
            dfquery_full += " and "
        dfquery_full += "modelgridindex==@modelgridindex"

    if dfquery:
        if dfquery_full:
            dfquery_full = f"({dfquery_full}) and "
        dfquery_full += f"({dfquery})"

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            arr_dfnltepop = pool.map(
                partial(read_file_filtered, strquery=dfquery_full, dfqueryvars=dfqueryvars), nltefilepaths
            )
            pool.close()
            pool.join()
    else:
        arr_dfnltepop = [read_file_filtered(f, strquery=dfquery_full, dfqueryvars=dfqueryvars) for f in nltefilepaths]
    dfconcat = pd.concat(arr_dfnltepop)
    assert isinstance(dfconcat, pd.DataFrame)
    return dfconcat
