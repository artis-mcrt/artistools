#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import math
import os
import string
import typing as t
from pathlib import Path

import argcomplete
import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputfile", "-i", default=Path(), help="Path of input file or folder containing model.txt")

    parser.add_argument("-cell", "-mgi", default=None, help="Focus on particular cell number (0-indexed)")

    parser.add_argument(
        "--noabund", action="store_true", help="Give total masses only, no nuclear or elemental abundances"
    )

    parser.add_argument(
        "--isotopes",
        action="store_true",
        help="Show full set of isotopic abundances",
    )

    parser.add_argument(
        "-sort",
        default="z",
        choices=["z", "mass"],
        nargs="+",
        help="Sort order for abundances (z = atomic number, mass = mass fraction)",
    )


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Describe an ARTIS input model, such as the mass, velocity structure, and abundances."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    assert args is not None

    dfmodel, modelmeta = at.inputmodel.get_modeldata_polars(
        args.inputfile,
        get_elemabundances=not args.noabund,
        printwarningsonly=False,
        derived_cols=["mass_g", "vel_r_mid", "rho"],
    )

    dfmodel = dfmodel.filter(pl.col("rho") > 0.0).drop(
        cs.starts_with("X_n")
    )  # don't confuse neutrons (lowercase 'n') with Nitrogen (N)

    if args.noabund:
        dfmodel = dfmodel.drop(cs.starts_with("X_"))

    if not args.isotopes:
        dfmodel = dfmodel.drop(cs.matches("X_[A-z]+[0-9]"))

    dfmodel = dfmodel.collect().lazy()

    t_model_init_days, vmax = modelmeta["t_model_init_days"], modelmeta["vmax_cmps"]

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    msun_g = 1.989e33
    print(f"Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)")

    if modelmeta["dimensions"] == 1:
        vmax_kmps = dfmodel.select(pl.col("vel_r_max_kmps").max()).collect().item()
        vmax = vmax_kmps * 1e5
        print(
            f"Model contains {modelmeta['npts_model']} 1D spherical shells with vmax = {vmax / 1e5} km/s"
            f" ({vmax / 29979245800:.2f} * c)"
        )
    else:
        nonemptycells = dfmodel.filter(pl.col("rho") > 0.0).select(pl.len()).collect().item()
        print(
            f"Model contains {modelmeta['npts_model']} grid cells ({nonemptycells} nonempty) with "
            f"vmax = {vmax} cm/s ({vmax * 1e-5 / 299792.458:.2f} * c)"
        )
        vmax_corner_3d = math.sqrt(3 * vmax**2)
        print(f"  3D corner vmax: {vmax_corner_3d:.2e} cm/s ({vmax_corner_3d * 1e-5 / 299792.458:.2f} * c)")
        if modelmeta["dimensions"] == 2:
            vmax_corner_2d = math.sqrt(2 * vmax**2)
            print(f"  2D corner vmax: {vmax_corner_2d:.2e} cm/s ({vmax_corner_2d * 1e-5 / 299792.458:.2f} * c)")

    minrho = dfmodel.select(pl.col("rho").min()).collect().item()
    minrho_cellcount = (
        dfmodel.filter(pl.col("rho") == minrho)
        .group_by("rho", maintain_order=False)
        .agg(pl.len().alias("minrho_cellcount"))
        .select("minrho_cellcount")
        .collect()
        .item()
    )
    print(f"  min density: {minrho:.2e} g/cm³. Cells with this density: {minrho_cellcount}")

    if args.cell is not None:
        mgi = int(args.cell)
        if mgi >= 0:
            print(f"Selected single cell mgi {mgi}:")
            dfmodel = dfmodel.filter(pl.col("inputcellid") == (mgi + 1))

            print(dfmodel)

    try:
        assoc_cells, mgi_of_propcells = at.get_grid_mapping(args.inputfile)
        print(f"  {len(assoc_cells)} model cells have associated prop cells")
    except FileNotFoundError:
        print("  no cell mapping file found")
        assoc_cells, mgi_of_propcells = None, None

    if "q" in dfmodel.columns:
        initial_energy = dfmodel.select(pl.col("q").dot(pl.col("mass_g"))).collect().item()
        assert initial_energy is not None
        print(f'  {"initial energy":19s} {initial_energy:.3e} erg')
    else:
        initial_energy = 0.0

    ejecta_ke_erg: float
    if "vel_r_max_kmps" in dfmodel.columns:
        # vel_r_min_kmps is in km/s
        ejecta_ke_erg = (
            dfmodel.select((0.5 * (pl.col("mass_g") / 1000.0) * (1000 * pl.col("vel_r_max_kmps")) ** 2).sum())
            .collect()
            .item()
        ) * 1e7
    else:
        # vel_r_mid is in cm/s
        ejecta_ke_erg = (
            dfmodel.select((0.5 * (pl.col("mass_g") / 1000.0) * (pl.col("vel_r_mid") / 100.0) ** 2).sum())
            .collect()
            .item()
        ) * 1e7

    print(f"  kinetic energy: {ejecta_ke_erg:.2e} [erg]")

    mass_msun_rho = dfmodel.select(pl.col("mass_g").sum() / msun_g).collect().item()

    if assoc_cells is not None and mgi_of_propcells is not None:
        direct_model_propgrid_map = all(
            len(propcells) == 1 and mgi == propcells[0] for mgi, propcells in assoc_cells.items()
        )
        if direct_model_propgrid_map:
            print("  detected direct mapping of model cells to propagation grid")
        else:
            ncoordgridx = math.ceil(np.cbrt(max(mgi_of_propcells.keys())))
            wid_init = 2 * vmax * t_model_init_seconds / ncoordgridx
            wid_init3 = wid_init**3
            initial_energy_mapped = 0.0
            cellmass_mapped = [
                float(len(assoc_cells.get(modelgridindex, [])) * wid_init3 * rho)
                for modelgridindex, rho in dfmodel.select(["modelgridindex", "rho"]).collect().iter_rows()
            ]

            if "q" in dfmodel.columns:
                initial_energy_mapped = sum(
                    mass * float(q[0])
                    for mass, q in zip(cellmass_mapped, dfmodel.select(["q"]).collect().iter_rows(), strict=False)
                )

                print(
                    f'  {"initial energy":19s} {initial_energy_mapped:.3e} erg (when mapped to'
                    f" {ncoordgridx}^3 cubic grid, error"
                    f" {100 * (initial_energy_mapped / initial_energy - 1):.2f}%)"
                )

            mtot_mapped_msun = sum(cellmass_mapped) / msun_g
            print(
                f'  {"M_tot_rho_map":19s} {mtot_mapped_msun:8.5f} MSun (density * volume when mapped to {ncoordgridx}^3'
                f" cubic grid, error {100 * (mtot_mapped_msun / mass_msun_rho - 1):.2f}%)"
            )

    print(f'  {"M_tot_rho":19s} {mass_msun_rho:8.5f} MSun (density * volume)')

    if modelmeta["dimensions"] > 1:
        corner_mass = (
            dfmodel.select(["vel_r_mid", "mass_g"])
            .filter(pl.col("vel_r_mid") > vmax)
            .select(pl.col("mass_g").sum())
            .collect()
            .item()
        ) / msun_g
        print(
            f'  {"M_corners":19s} {corner_mass:8.5f} MSun ('
            f" {100 * corner_mass / mass_msun_rho:.2f}% of M_tot in cells with v_r_mid > vmax)"
        )

    if args.noabund:
        return

    mass_msun_isotopes = 0.0
    mass_msun_elem = 0.0
    mass_msun_lanthanides = 0.0
    mass_msun_actinides = 0.0
    speciesmasses: dict[str, float] = {}

    for column in dfmodel.select(cs.starts_with("X_") - cs.by_name("X_Fegroup")).columns:
        species = column.replace("X_", "")

        speciesabund_g = dfmodel.select(pl.col(column).dot(pl.col("mass_g"))).collect().item()

        assert isinstance(speciesabund_g, float)

        species_mass_msun = speciesabund_g / msun_g

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
        f'  {"M_tot_elem":19s} {mass_msun_elem:8.5f} MSun ({mass_msun_elem / mass_msun_rho * 100:6.2f}% of M_tot_rho)'
    )

    if args.isotopes:
        print(
            f'  {"M_tot_iso":19s} {mass_msun_isotopes:8.5f} MSun ({mass_msun_isotopes / mass_msun_rho * 100:6.2f}% '
            "of M_tot_rho, but can be < 100% if stable isotopes not tracked)"
        )

    mass_msun_fegroup = dfmodel.select(pl.col("X_Fegroup").dot(pl.col("mass_g"))).collect().item() / msun_g
    print(
        f'  {"M_Fegroup":19s} {mass_msun_fegroup:8.5f} MSun'
        f" ({mass_msun_fegroup / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    print(
        f'  {"M_lanthanide_isosum":19s} {mass_msun_lanthanides:8.5f} MSun'
        f" ({mass_msun_lanthanides / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    print(
        f'  {"M_actinide_isosum":19s} {mass_msun_actinides:8.5f} MSun'
        f" ({mass_msun_actinides / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    def sortkey(tup_species_mass_g):
        species, mass_g = tup_species_mass_g
        if args.sort == "z":
            # for a species like C_isosum, strmassnumber is "", so use -1 to sort it first
            strmassnumber = species.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz").rstrip(
                "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            )
            massnumber = int(strmassnumber) if strmassnumber else -1
            return (at.get_atomic_number(species), massnumber, species)

        return (-mass_g, species)

    mass_g_min = min(speciesmasses.values())
    mass_g_max = max(speciesmasses.values())
    try:
        maxbarchars = os.get_terminal_size()[0] - 65
    except OSError:
        maxbarchars = 20
    for species, mass_g in sorted(speciesmasses.items(), key=sortkey):
        species_mass_msun = mass_g / msun_g
        massfrac = species_mass_msun / mass_msun_rho
        strcomment = ""
        atomic_number = at.get_atomic_number(species)
        if species.endswith("_isosum"):
            elsymb = species.replace("_isosum", "")
            elem_mass = speciesmasses.get(elsymb, 0.0)
            if np.isclose(mass_g, elem_mass, rtol=1e-4):
                # iso sum matches the element mass, so don't show it
                continue
            strcomment += f"({mass_g / elem_mass * 100 if elem_mass > 0 else math.nan:6.2f}% of {elsymb} element mass from abundances.txt)"

            if mass_g > elem_mass * (1.0 + 1e-5):
                strcomment += " ERROR! isotope sum is greater than element abundance"

        zstr = str(atomic_number)
        barstr = "-" * int(maxbarchars * (mass_g - mass_g_min) / (mass_g_max - mass_g_min))
        print(f"{zstr:>5} {species:11s} massfrac {massfrac:.3e}   {species_mass_msun:.3e} Msun  {barstr}")
        if strcomment:
            print(f"    {strcomment}")


if __name__ == "__main__":
    main()
