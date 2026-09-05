# PYTHON_ARGCOMPLETE_OK
"""Summarise the mass, velocity structure, and abundances of an ARTIS input model."""

import argparse
import math
import os
import string
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.constants import MH_g
from artistools.constants import Msun_to_g


def calculate_model_electron_frac(dfmodel: pl.LazyFrame) -> float:
    """Calculate the electron fraction of the model from the isotopic composition."""
    exprs_protons = []
    exprs_nucleons = []
    for column in dfmodel.select(cs.matches("X_[A-z]+[0-9]")).collect_schema().names():
        species = column.removeprefix("X_")
        elsymb = species.rstrip(string.digits)
        atomic_number = at.get_atomic_number(elsymb)
        if atomic_number == 0:
            continue
        mass_number = float(species.removeprefix(elsymb))
        exprs_protons.append(atomic_number / (mass_number * MH_g) * pl.col(column) * pl.col("mass_g"))
        exprs_nucleons.append(1 / MH_g * pl.col(column) * pl.col("mass_g"))

    globalelectronfrac = (
        dfmodel
        .with_columns(protons=pl.sum_horizontal(exprs_protons), nucleons=pl.sum_horizontal(exprs_nucleons))
        .select(pl.col("protons").sum() / pl.col("nucleons").sum())
        .collect()
        .item()
    )

    assert globalelectronfrac is not None
    assert 0.0 <= globalelectronfrac <= 1.0
    assert isinstance(globalelectronfrac, float)
    return globalelectronfrac


def describe_model(modelpath: Path | str, args: argparse.Namespace) -> None:
    """Describe the ARTIS input model, such as the mass, velocity structure, and abundances."""
    at.print_heading(str(modelpath))
    dfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath,
        get_elemabundances=not args.noabund,
        printwarningsonly=False,
        derived_cols=["mass_g", "vel_r_mid", "kinetic_en_erg", "rho"],
    )

    # don't confuse neutrons (lowercase 'n') with Nitrogen (N)
    dfmodel = dfmodel.filter(pl.col("rho") > 0.0).drop(cs.starts_with("X_n"), strict=False)

    if args.noabund:
        dfmodel = dfmodel.drop(cs.starts_with("X_"), strict=False)

    elif not args.isotopes:
        dfmodel = dfmodel.drop(cs.matches("X_[A-z]+[0-9]"), strict=False)

    dfmodel = dfmodel.collect().lazy()

    t_model_init_days, vmax = modelmeta["t_model_init_days"], modelmeta["vmax_cmps"]

    t_model_init_seconds = t_model_init_days * day_to_s
    print(f"Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)")

    if modelmeta["dimensions"] == 1:
        vmax_kmps = dfmodel.select(pl.col("vel_r_max_kmps").max()).collect().item()
        vmax = vmax_kmps * km_to_cm
        print(
            f"Model contains {modelmeta['npts_model']} 1D spherical shells with vmax = {vmax / km_to_cm} km/s"
            f" ({vmax / C_cm_per_s:.2f} * c)"
        )
    else:
        nonemptycells = dfmodel.select(pl.len()).collect().item()
        print(
            f"Model contains {modelmeta['npts_model']} grid cells ({nonemptycells} nonempty) with "
            f"vmax = {vmax} cm/s ({vmax / C_cm_per_s:.2f} * c)"
        )
        vmax_corner_3d = vmax * math.sqrt(3)
        print(f"  3D corner vmax: {vmax_corner_3d:.2e} cm/s ({vmax_corner_3d / C_cm_per_s:.2f} * c)")
        if modelmeta["dimensions"] == 2:
            vmax_corner_2d = vmax * math.sqrt(2)
            print(f"  2D corner vmax: {vmax_corner_2d:.2e} cm/s ({vmax_corner_2d / C_cm_per_s:.2f} * c)")

    minrho, maxrho = dfmodel.select(minrho=pl.col("rho").min(), maxrho=pl.col("rho").max()).collect().row(0)
    for minmaxlabel, rho in (("min", minrho), ("max", maxrho)):
        cellcount = dfmodel.filter(pl.col("rho") == rho).select(pl.len()).collect().item()
        print(f"  {minmaxlabel} density: {rho:.2e} g/cm³. Cells with this density: {cellcount}")

    if args.cell is not None:
        mgi = int(args.cell)
        if mgi >= 0:
            print(f"Selected single cell mgi {mgi}:")
            dfmodel = dfmodel.filter(pl.col("inputcellid") == (mgi + 1))

            print(dfmodel.collect())

    try:
        assoc_cells, mgi_of_propcells, direct_model_propgrid_map = at.get_grid_mapping(modelpath)
        print(f"  {len(assoc_cells)} model cells have associated prop cells")
    except FileNotFoundError:
        print("  no cell mapping file found")
        assoc_cells, mgi_of_propcells, direct_model_propgrid_map = None, None, True

    if "Ye" in dfmodel.collect_schema().names():
        electronfrac = dfmodel.select(pl.col("Ye").dot(pl.col("mass_g")) / pl.col("mass_g").sum()).collect().item()
        assert electronfrac is not None
        print(f"  {'electron frac Ye':19s} {electronfrac:.3f}")
        if args.isotopes:
            # currently assumes that all isotopes are specified (i.e. not for Type Ia models)
            calcelectronfrac = calculate_model_electron_frac(dfmodel)
            assert calcelectronfrac is not None
            print(f"  {'snapshot Ye':19s} {calcelectronfrac:.3f}")

    if "q" in dfmodel.collect_schema().names():
        initial_energy = dfmodel.select(pl.col("q").dot(pl.col("mass_g"))).collect().item()
        assert initial_energy is not None
        print(f"  {'initial energy':19s} {initial_energy:.3e} erg")
    else:
        initial_energy = 0.0

    ejecta_ke_erg: float | int = dfmodel.select("kinetic_en_erg").sum().collect().item()

    print(f"  {'kinetic energy':19s} {ejecta_ke_erg:.2e} erg")

    mass_g_rho = dfmodel.select(pl.col("mass_g").sum()).collect().item()

    # velocity derived from ejecta kinetic energy to match Barnes et al. (2016) Section 2.1
    ejecta_v = np.sqrt(2 * ejecta_ke_erg / mass_g_rho)
    print(f"  {'v_ej=√(2KE/m)':19s} {ejecta_v / C_cm_per_s:.2f}c")

    mass_msun_rho = mass_g_rho / Msun_to_g

    if assoc_cells is not None and mgi_of_propcells is not None:
        if direct_model_propgrid_map:
            print("  detected direct mapping of model cells to propagation grid")
        else:
            print_mapped_masses(dfmodel, assoc_cells, mgi_of_propcells, modelmeta, vmax, initial_energy, mass_msun_rho)

    print(f"  {'M_tot_rho':19s} {mass_msun_rho:7.5f} MSun (density * volume)")

    if modelmeta["dimensions"] > 1:
        corner_mass = (
            dfmodel
            .select(["vel_r_mid", "mass_g"])
            .filter(pl.col("vel_r_mid") > vmax)
            .select(pl.col("mass_g").sum())
            .collect()
            .item()
        ) / Msun_to_g
        print(
            f"  {'M_corners':19s} {corner_mass:7.5f} MSun ("
            f" {100 * corner_mass / mass_msun_rho:.2f}% of M_tot in cells with v_r_mid > vmax)"
        )

    if not args.noabund:
        print_species_masses(dfmodel, args, mass_msun_rho)


def print_mapped_masses(
    dfmodel: pl.LazyFrame,
    assoc_cells: dict[int, list[int]],
    mgi_of_propcells: dict[int, int],
    modelmeta: dict[str, t.Any],
    vmax: float,
    initial_energy: float,
    mass_msun_rho: float,
) -> None:
    """Print the mass and the initial energy of the model after ARTIS maps it to the cubic propagation grid."""
    ncoordgridx = math.ceil(np.cbrt(max(mgi_of_propcells.keys())))
    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    wid_init = 2 * vmax * t_model_init_seconds / ncoordgridx

    dfpropcellcounts = pl.LazyFrame(
        {"modelgridindex": list(assoc_cells.keys()), "n_assoc_cells": [len(cells) for cells in assoc_cells.values()]},
        schema={"modelgridindex": pl.Int32, "n_assoc_cells": pl.Int64},
    )
    dfmapped = (
        dfmodel
        .with_columns(pl.col("modelgridindex").cast(pl.Int32))
        .join(dfpropcellcounts, on="modelgridindex", how="left", maintain_order="left")
        .with_columns(mass_g_mapped=pl.col("n_assoc_cells").fill_null(0) * wid_init**3 * pl.col("rho"))
    )

    if "q" in dfmodel.collect_schema().names():
        initial_energy_mapped = dfmapped.select(pl.col("q").dot(pl.col("mass_g_mapped"))).collect().item()
        print(
            f"  {'initial energy':19s} {initial_energy_mapped:.3e} erg (when mapped to"
            f" {ncoordgridx}^3 cubic grid, error"
            f" {100 * (initial_energy_mapped / initial_energy - 1):.2f}%)"
        )

    mtot_mapped_msun = dfmapped.select(pl.col("mass_g_mapped").sum()).collect().item() / Msun_to_g
    print(
        f"  {'M_tot_rho_map':19s} {mtot_mapped_msun:7.5f} MSun (density * volume when mapped to {ncoordgridx}^3"
        f" cubic grid, error {100 * (mtot_mapped_msun / mass_msun_rho - 1):.2f}%)"
    )


def print_species_masses(dfmodel: pl.LazyFrame, args: argparse.Namespace, mass_msun_rho: float) -> None:
    """Print the total mass of each element and isotope, and the lanthanide, actinide, and iron-group totals."""
    mass_msun_isotopes = 0.0
    mass_msun_elem = 0.0
    mass_msun_lanthanides = 0.0
    mass_msun_actinides = 0.0
    speciesmasses: dict[str, float] = {}

    abundcolumns = dfmodel.select(cs.starts_with("X_") - cs.by_name("X_Fegroup")).collect_schema().names()

    # one collect for every species, rather than re-running the whole scan once per column
    speciesabund_g_of_column: dict[str, t.Any] = (
        dfmodel
        .select(pl.col(col).dot(pl.col("mass_g")).alias(col) for col in abundcolumns)
        .collect()
        .row(0, named=True)
        if abundcolumns
        else {}
    )

    for column in abundcolumns:
        species = column.removeprefix("X_")

        speciesabund_g = speciesabund_g_of_column[column]

        assert isinstance(speciesabund_g, float)

        species_mass_msun = speciesabund_g / Msun_to_g

        atomic_number = at.get_atomic_number(species.rstrip(string.digits))

        if species[-1].isdigit():
            # isotopic species

            elname = species.rstrip(string.digits)
            strtotiso = f"{elname}_isosum"
            speciesmasses[strtotiso] = speciesmasses.get(strtotiso, 0.0) + speciesabund_g
            mass_msun_isotopes += species_mass_msun

            if args.isotopes and speciesabund_g > 0.0:
                speciesmasses[species] = speciesabund_g

        else:
            # elemental species
            mass_msun_elem += species_mass_msun

            if speciesabund_g > 0.0:
                speciesmasses[species] = speciesabund_g

            if 57 <= atomic_number <= 71:
                mass_msun_lanthanides += species_mass_msun
            elif 89 <= atomic_number <= 103:
                mass_msun_actinides += species_mass_msun

    print(
        f"  {'M_tot_elem':19s} {mass_msun_elem:7.5f} MSun ({mass_msun_elem / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    if args.isotopes:
        print(
            f"  {'M_tot_iso':19s} {mass_msun_isotopes:7.5f} MSun ({mass_msun_isotopes / mass_msun_rho * 100:6.2f}% "
            "of M_tot_rho, but can be < 100% if stable isotopes not tracked)"
        )

    mass_msun_fegroup = dfmodel.select(pl.col("X_Fegroup").dot(pl.col("mass_g"))).collect().item() / Msun_to_g
    print(
        f"  {'M_Fegroup':19s} {mass_msun_fegroup:7.5f} MSun"
        f" ({mass_msun_fegroup / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    print(
        f"  {'M_lanthanide_isosum':19s} {mass_msun_lanthanides:7.5f} MSun"
        f" ({mass_msun_lanthanides / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    print(
        f"  {'M_actinide_isosum':19s} {mass_msun_actinides:7.5f} MSun"
        f" ({mass_msun_actinides / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    def sortkey(tup_species_mass_g: tuple[str, float]) -> tuple[int, int, str] | tuple[float, str]:
        species, mass_g = tup_species_mass_g
        if args.sort in {"z", "a"}:
            # for a species like C_isosum, strmassnumber is "", so use -1 to sort it first
            strmassnumber = species.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz").rstrip(
                "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            )
            massnumber = int(strmassnumber) if strmassnumber else -1
            if args.sort == "z":
                return (at.get_atomic_number(species), massnumber, species)

            if args.sort == "a":
                return (massnumber, at.get_atomic_number(species), species)

        return (-mass_g, species)

    mass_g_min = min(speciesmasses.values())
    mass_g_max = max(speciesmasses.values())
    try:
        maxbarchars = os.get_terminal_size()[0] - 57
    except OSError:
        maxbarchars = 20
    for species, mass_g in sorted(speciesmasses.items(), key=sortkey):
        species_mass_msun = mass_g / Msun_to_g
        massfrac = species_mass_msun / mass_msun_rho
        strcomment = ""
        atomic_number = at.get_atomic_number(species)
        if species.endswith("_isosum"):
            elsymb = species.removesuffix("_isosum")
            elem_mass = speciesmasses.get(elsymb, 0.0)
            if mass_g == 0.0 or np.isclose(mass_g, elem_mass, rtol=1e-4):
                # iso sum is zero or matches the element mass, so don't show it
                continue
            strcomment += f"({mass_g / elem_mass * 100 if elem_mass > 0 else math.nan:6.2f}% of {elsymb} element mass from abundances.txt)"

            if mass_g > elem_mass * (1.0 + 1e-5):
                strcomment += " ERROR! isotope sum is greater than element abundance"

        zstr = str(atomic_number)
        mass_g_min_lim = max(mass_g_min, mass_g_max * 1e-4)
        barsize = int(maxbarchars * (math.log(mass_g / mass_g_min_lim)) / (math.log(mass_g_max / mass_g_min_lim)))
        barstr = "-" * barsize
        print(f"{zstr:>5} {species:7s} massfrac {massfrac:.3e}   {species_mass_msun:.3e} Msun  {barstr}")
        if strcomment:
            print(f"    {strcomment}")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        default=[],
        helptext="Path of input file or folder containing model.txt",
    )

    parser.add_argument("-cell", "-mgi", default=None, help="Focus on particular cell number (0-indexed)")

    parser.add_argument(
        "--noabund", action="store_true", help="Give total masses only, no nuclear or elemental abundances"
    )

    parser.add_argument("--isotopes", action="store_true", help="Show full set of isotopic abundances")

    parser.add_argument(
        "-sort",
        default="z",
        choices=["z", "a", "mass"],
        help="Sort order for abundances (z = atomic number, a = mass number, mass = total global mass)",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Describe an ARTIS input model, such as the mass, velocity structure, and abundances."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    assert args is not None
    args.modelpath = at.normalize_path_list(args.modelpath)

    for modelpath in args.modelpath:
        describe_model(modelpath, args)
        print()


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
