#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import math
import typing as t
from pathlib import Path

import argcomplete
import numpy as np

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputfile", "-i", default=Path(), help="Path of input file or folder containing model.txt")

    parser.add_argument("-cell", "-mgi", default=None, help="Focus on particular cell number (0-indexed)")

    parser.add_argument(
        "--noabund", action="store_true", help="Give total masses only, no nuclear or elemental abundances"
    )

    parser.add_argument(
        "--noisotopes",
        action="store_true",
        help="Give element abundances only, no isotope abundances",
    )


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Describe an ARTIS input model, such as the mass, velocity structure, and abundances."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    dfmodel, modelmeta = at.inputmodel.get_modeldata(
        args.inputfile,
        get_elemabundances=True,
        printwarningsonly=False,
        dtype_backend="pyarrow",
        use_polars=False,
        derived_cols=["cellmass_grams", "vel_r_mid"],
    )
    t_model_init_days, vmax = modelmeta["t_model_init_days"], modelmeta["vmax_cmps"]

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    print(f"Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)")

    if modelmeta["dimensions"] == 1:
        vmax = dfmodel["velocity_outer"].max() * 1e5
        print(
            f"Model contains {len(dfmodel)} 1D spherical shells with vmax = {vmax/1e5} km/s"
            f" ({vmax / 29979245800:.2f} * c)"
        )
        dfmodel["rho"] = 10 ** dfmodel["logrho"]
    else:
        nonemptycells = sum(dfmodel["rho"] > 0.0)
        print(
            f"Model contains {len(dfmodel)} grid cells ({nonemptycells} nonempty) with "
            f"vmax = {vmax} cm/s ({vmax * 1e-5 / 299792.458:.2f} * c)"
        )
        vmax_corner_3d = math.sqrt(3 * vmax**2)
        print(f"  3D corner vmax: {vmax_corner_3d:.2e} cm/s ({vmax_corner_3d * 1e-5 / 299792.458:.2f} * c)")
        if modelmeta["dimensions"] == 2:
            vmax_corner_2d = math.sqrt(2 * vmax**2)
            print(f"  2D corner vmax: {vmax_corner_2d:.2e} cm/s ({vmax_corner_2d * 1e-5 / 299792.458:.2f} * c)")

    if args.cell is not None:
        mgi = int(args.cell)
        if mgi >= 0:
            print(f"Selected single cell mgi {mgi}:")
            dfmodel = dfmodel.query("inputcellid == (@mgi + 1)")

            print(dfmodel.iloc[0])

    try:
        assoc_cells, mgi_of_propcells = at.get_grid_mapping(args.inputfile)
        print(f"  {len(assoc_cells)} model cells have >0 associated prop cells")
    except FileNotFoundError:
        print("  no cell mapping file found")
        assoc_cells, mgi_of_propcells = None, None

    if "q" in dfmodel.columns:
        initial_energy = sum(
            mass * q for mass, q in dfmodel[["cellmass_grams", "q"]].itertuples(index=False, name=None)
        )
        print(f'  {"initial energy":19s} {initial_energy:.3e} erg')

    mass_msun_rho = dfmodel["cellmass_grams"].sum() / 1.989e33

    if assoc_cells is not None and mgi_of_propcells is not None:
        ncoordgridx = math.ceil(math.cbrt(max(mgi_of_propcells.keys())))
        wid_init = 2 * vmax * t_model_init_seconds / ncoordgridx
        wid_init3 = wid_init**3
        initial_energy_mapped = 0.0
        cellmass_mapped = [len(assoc_cells.get(mgi, [])) * wid_init3 * rho for mgi, rho in enumerate(dfmodel["rho"])]
        if "q" in dfmodel.columns:
            initial_energy_mapped = sum(mass * q for mass, q in zip(cellmass_mapped, dfmodel["q"]))
            print(
                f'  {"initial energy":19s} {initial_energy_mapped:.3e} erg (when mapped to'
                f" {ncoordgridx}^3 cubic grid, error"
                f" {100 * (initial_energy_mapped / initial_energy - 1):.2f}%)"
            )

        mtot_mapped_msun = sum(cellmass_mapped) / 1.989e33
        print(
            f'  {"M_tot_rho_map":19s} {mtot_mapped_msun:8.5f} MSun (density * volume when mapped to {ncoordgridx}^3'
            f" cubic grid, error {100 * (mtot_mapped_msun / mass_msun_rho - 1):.2f}%)"
        )

    print(f'  {"M_tot_rho":19s} {mass_msun_rho:8.5f} MSun (density * volume)')

    if modelmeta["dimensions"] > 1:
        corner_mass = dfmodel[dfmodel["vel_r_mid"] > vmax]["cellmass_grams"].sum() / 1.989e33
        print(
            f'  {"M_corners":19s} {corner_mass / 1.989e33:8.5f} MSun ('
            f" {100 * corner_mass / mass_msun_rho:.2f}% of M_tot in cells with v_r_mid > vmax)"
        )

    mass_msun_isotopes = 0.0
    mass_msun_elem = 0.0
    mass_lanthanides_isosum = 0.0
    mass_actinides_isosum = 0.0
    speciesmasses: dict[str, float] = {}
    for column in dfmodel.columns:
        if column.startswith("X_"):
            species = column.replace("X_", "")
            speciesabund_g = np.dot(dfmodel[column], dfmodel["cellmass_grams"])

            species_mass_msun = speciesabund_g / 1.989e33

            atomic_number = at.get_atomic_number(species.rstrip("0123456789"))

            if species[-1].isdigit():
                # isotopic species

                if atomic_number >= 57 and atomic_number <= 71:
                    mass_lanthanides_isosum += species_mass_msun
                elif atomic_number >= 89 and atomic_number <= 103:
                    mass_actinides_isosum += species_mass_msun

                elname = species.rstrip("0123456789")
                strtotiso = f"{elname}_isosum"
                speciesmasses[strtotiso] = speciesmasses.get(strtotiso, 0.0) + speciesabund_g
                mass_msun_isotopes += species_mass_msun

                if not args.noisotopes and speciesabund_g > 0.0:
                    speciesmasses[species] = speciesabund_g

            elif species.lower() != "fegroup":  # ignore special group abundance
                # elemental species
                mass_msun_elem += species_mass_msun
                if speciesabund_g > 0.0:
                    speciesmasses[species] = speciesabund_g

    print(
        f'  {"M_tot_elem":19s} {mass_msun_elem:8.5f} MSun ({mass_msun_elem / mass_msun_rho * 100:6.2f}% of M_tot_rho)'
    )

    print(
        f'  {"M_tot_iso":19s} {mass_msun_isotopes:8.5f} MSun ({mass_msun_isotopes / mass_msun_rho * 100:6.2f}% '
        "of M_tot_rho, but can be < 100% if stable isotopes not tracked)"
    )

    print(
        f'  {"M_lanthanide_isosum":19s} {mass_lanthanides_isosum:8.5f} MSun'
        f" ({mass_lanthanides_isosum / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    print(
        f'  {"M_actinide_isosum":19s} {mass_actinides_isosum:8.5f} MSun'
        f" ({mass_actinides_isosum / mass_msun_rho * 100:6.2f}% of M_tot_rho)"
    )

    if not args.noabund:

        def sortkey(tup_species_mass_g):
            species, mass_g = tup_species_mass_g
            # return -mass_g
            # return (-speciesmasses.get(species.rstrip("0123456789"), 0.0), species)
            return (at.get_atomic_number(species), species)

        for species, mass_g in sorted(speciesmasses.items(), key=sortkey):
            species_mass_msun = mass_g / 1.989e33
            massfrac = species_mass_msun / mass_msun_rho
            strcomment = ""
            atomic_number = at.get_atomic_number(species)
            if species.endswith("_isosum"):
                elsymb = species.replace("_isosum", "")
                elem_mass = speciesmasses.get(elsymb, 0.0)
                if np.isclose(mass_g, elem_mass, rtol=1e-5):
                    # iso sum matches the element mass, so don't show it
                    continue
                if elem_mass > 0.0:
                    strcomment += f" ({mass_g / elem_mass * 100:6.2f}% of {elsymb} element mass)"
                if mass_g > elem_mass * (1.0 + 1e-5):
                    strcomment += " ERROR! isotope sum is greater than element abundance"
            zstr = f"Z={atomic_number}"
            print(f"{zstr:>5} {species:11s} massfrac {massfrac:.3e}   {species_mass_msun:.3e} Msun{strcomment}")


if __name__ == "__main__":
    main()
