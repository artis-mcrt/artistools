"""Convert a CMFGEN SN_HYDRO_DATA snapshot to ARTIS model.txt and abundances.txt."""

import argparse
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

import artistools as at
from artistools.constants import Msun_to_g
from artistools.inputmodel.fromcmfgen.rd_cmfgen import rd_sn_hydro_data

# CMFGEN abbreviates species rather than using element symbols, so get_atomic_number cannot read these
CMFGEN_SPECIES_ATOMIC_NUMBER: t.Final[Mapping[str, int]] = MappingProxyType({
    "HYD": 1,
    "HE": 2,
    "CARB": 6,
    "NIT": 7,
    "OXY": 8,
    "FLU": 9,
    "NEON": 10,
    "SOD": 11,
    "MAG": 12,
    "ALUM": 13,
    "SIL": 14,
    "PHOS": 15,
    "SUL": 16,
    "CHL": 17,
    "ARG": 18,
    "POT": 19,
    "CAL": 20,
    "SCAN": 21,
    "TIT": 22,
    "VAN": 23,
    "CHRO": 24,
    "MAN": 25,
    "IRON": 26,
    "COB": 27,
    "NICK": 28,
    "BAR": 56,
})


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-snapshot", default="SN_HYDRO_DATA_1.300d", help="CMFGEN SN_HYDRO_DATA snapshot file")
    at.add_outputpath_arg(parser, default=Path(), astype=Path, helptext="Folder to write model.txt/abundances.txt to")


def get_cmfgen_atomic_numbers(specnames: Sequence[str]) -> list[int]:
    """Return the atomic number of each CMFGEN species name, in the snapshot's own order.

    CMFGEN abbreviates species rather than using element symbols (CARB, IRON, NICK), so the names cannot
    go through get_atomic_number directly. Deriving the mapping from the file's own species list, instead
    of a positional table, means a snapshot with a different species set converts correctly or fails
    loudly rather than writing every mass fraction into the wrong column.
    """
    unknown = [name for name in specnames if name not in CMFGEN_SPECIES_ATOMIC_NUMBER]
    if unknown:
        msg = f"Unknown CMFGEN species names {unknown}. Add them to CMFGEN_SPECIES_ATOMIC_NUMBER."
        raise ValueError(msg)

    return [CMFGEN_SPECIES_ATOMIC_NUMBER[name] for name in specnames]


def get_isotope_massfracs(a: dict[str, t.Any], nuclides: Sequence[tuple[str, int]]) -> dict[str, t.Any]:
    """Return the per-cell mass fraction column of each (CMFGEN species name, mass number) nuclide.

    The isotope table is looked up by name and mass number rather than by a hardcoded column index, so the
    result does not depend on the isotope ordering of one particular snapshot.
    """
    isonames = list(a["iso"])
    massnumbers = list(a["aiso"])

    massfracs = {}
    for specname, massnumber in nuclides:
        matches = [
            i
            for i, (name, A) in enumerate(zip(isonames, massnumbers, strict=True))
            if name == specname and massnumber == A
        ]
        if len(matches) != 1:
            msg = f"Expected exactly one {specname}{massnumber} isotope column, found {len(matches)}"
            raise ValueError(msg)

        elsymbol = at.get_elsymbol(CMFGEN_SPECIES_ATOMIC_NUMBER[specname])
        massfracs[f"X_{elsymbol}{massnumber}"] = a["isofrac"][:, matches[0]]

    return massfracs


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS model.txt and abundances.txt from a CMFGEN SN_HYDRO_DATA snapshot.

    The output describes the same time as the input snapshot: abundances are written as they appear in the
    file, with no decay or reverse evolution applied.
    """
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    outputpath = Path(args.outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)

    a: dict[str, t.Any] = rd_sn_hydro_data(args.snapshot, reverse=True)

    atomic_numbers = get_cmfgen_atomic_numbers(a["spec"])

    # Ba stands in for everything heavier than Ni in these models, so it counts towards the iron-group mass
    # fraction without getting an element column of its own.
    ige_index = np.array(atomic_numbers) > 20

    # The radii/velocity in the CMFGEN files are zone centered, while in ARTIS they represent
    # the outer radius of a given zone. So we need to do a transformation
    r = a["rad"] * 1e10
    rmax = 0.5 * (r[:-1] + r[1:])
    rout = rmax
    rout = np.append(rout, r[-1])  # cmfgen uses the radius of the outermost zone as the outer boundary
    rin = rmax
    rin = np.insert(rin, 0, 0)  # for artis we use 0 as inner radius for the innermost shell, cmfgen uses
    # the innermost radius r[0], this gives a slight discrepancy (<1%) in the total mass
    dm = 4 / 3 * np.pi * (rout**3 - rin**3) * a["dens"] / Msun_to_g
    print(f"total mass {dm.sum():.4f} Msun ({dm.sum() / (a['dmass'].sum() / Msun_to_g):.4f} of the snapshot's)")

    # ARTIS reads model.txt by column position, so build the frame by name and let save_modeldata place the
    # columns. The previous hand-rolled writer emitted the right values, but under local names that said Cr48
    # and V48 where the isotope table actually holds Fe52 and Cr48 — which repeatedly read as a column-order
    # bug on review.
    dfmodel = pl.DataFrame(
        {
            "inputcellid": np.arange(1, a["nd"] + 1),
            "vel_r_max_kmps": rout / a["time"] / 3600 / 24 / 1e5,
            "logrho": np.log10(a["dens"]),
            "X_Fegroup": a["specfrac"][:, ige_index].sum(axis=1),
        }
        | get_isotope_massfracs(a, [("NICK", 56), ("COB", 56), ("IRON", 52), ("CHRO", 48)])
    )

    dfelabundances = pl.DataFrame({
        "inputcellid": np.arange(1, a["nd"] + 1),
        **{
            f"X_{at.get_elsymbol(Z)}": a["specfrac"][:, i]
            for i, Z in enumerate(atomic_numbers)
            # Ba is folded into X_Fegroup above and has no element column in an ARTIS abundances.txt
            if Z <= 30
        },
    })

    at.inputmodel.save_modeldata(dfmodel, outpath=outputpath, t_model_init_days=a["time"], dimensions=1)
    at.inputmodel.save_initelemabundances(dfelabundances, outpath=outputpath)


if __name__ == "__main__":
    main()
