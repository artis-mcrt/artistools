import gzip
import itertools
import typing as t
from pathlib import Path

import pandas as pd

import artistools as at


def get_atomic_composition(modelpath: Path) -> dict[int, int]:
    """Read ion list from output file."""
    atomic_composition = {}

    with (modelpath / "output_0-0.txt").open(encoding="utf-8") as foutput:
        ioncount = 0
        Z = None
        for row in foutput:
            if row.split()[0] == "[input.c]":
                split_row = row.split()
                if split_row[1] == "element":
                    Z = int(split_row[4])
                    ioncount = 0
                elif split_row[1] == "ion":
                    ioncount += 1
                    assert Z is not None, "Z should be set before ioncount"
                    atomic_composition[Z] = ioncount
    return atomic_composition


def parse_ion_row_classic(row: list[str], outdict: dict[str, t.Any], atomic_composition: dict[int, int]) -> None:
    elements = atomic_composition.keys()

    i = 6  # skip first 6 numbers in est file. These are n, TR, Te, W, TJ, grey_depth.
    # Numbers after these 6 are populations
    for atomic_number in elements:
        for ion_stage in range(1, atomic_composition[atomic_number] + 1):
            value_thision = float(row[i])
            ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_")
            outdict[f"nnion_{ionstr}"] = value_thision
            i += 1

            elpop = outdict.get(f"nnelement_{atomic_number}", 0)
            outdict[f"nnelement_{atomic_number}"] = elpop + value_thision

            totalpop = outdict.get("nntot", 0)
            outdict["nntot"] = totalpop + value_thision


def get_first_ts_in_run_directory(modelpath: str | Path) -> dict[str, int]:
    folderlist_all = (*sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]), Path(modelpath))

    first_timesteps_in_dir = {}

    for folder in folderlist_all:
        if (folder / "output_0-0.txt").is_file():
            with (folder / "output_0-0.txt").open(encoding="utf-8") as output_0:
                timesteps_in_dir = [
                    line.strip(".\n").split(" ")[-1]
                    for line in output_0
                    if "[debug] update_packets: updating packet 0 for timestep" in line
                ]
            first_ts = timesteps_in_dir[0]
            first_timesteps_in_dir[str(folder)] = int(first_ts)

    return first_timesteps_in_dir


def read_classic_estimators(
    modelpath: Path,
    modeldata: pd.DataFrame,
    readonly_mgi: list[int] | None = None,
    readonly_timestep: list[int] | None = None,
) -> dict[tuple[int, int], t.Any] | None:
    estimfiles = list(
        itertools.chain(Path(modelpath).glob("estimators_????.out"), Path(modelpath).glob("*/estimators_????.out"))
    )
    if not estimfiles:
        print("No estimator files found")
        return None
    print(f"Reading {len(estimfiles)} estimator files...")

    first_timesteps_in_dir = get_first_ts_in_run_directory(modelpath)
    atomic_composition = get_atomic_composition(modelpath)

    inputparams = at.get_inputparams(modelpath)
    ndimensions = inputparams["n_dimensions"]

    estimators: dict[tuple[int, int], t.Any] = {}
    for estfilepath in estimfiles:
        opener: t.Any = gzip.open if str(estfilepath).endswith(".gz") else open

        # If classic plots break it's probably getting first timestep here
        # Try either of the next two lines
        timestep = first_timesteps_in_dir[str(estfilepath).split("/")[0]]  # get the starting timestep for the estfile
        # timestep = first_timesteps_in_dir[str(estfile[:-20])]
        # timestep = 0  # if the first timestep in the file is 0 then this is fine
        with opener(estfilepath) as estfile:
            modelgridindex = -1
            for line in estfile:
                row = line.split()
                if int(row[0]) <= modelgridindex:
                    timestep += 1
                modelgridindex = int(row[0])

                if (not readonly_mgi or modelgridindex in readonly_mgi) and (
                    not readonly_timestep or timestep in readonly_timestep
                ):
                    estimators[timestep, modelgridindex] = {}

                    if ndimensions == 1:
                        estimators[timestep, modelgridindex]["vel_r_max_kmps"] = modeldata["vel_r_max_kmps"][
                            modelgridindex
                        ]

                    estimators[timestep, modelgridindex]["TR"] = float(row[1])
                    estimators[timestep, modelgridindex]["Te"] = float(row[2])
                    estimators[timestep, modelgridindex]["W"] = float(row[3])
                    estimators[timestep, modelgridindex]["TJ"] = float(row[4])

                    parse_ion_row_classic(row, estimators[timestep, modelgridindex], atomic_composition)

                    # heatingrates[tid].ff, heatingrates[tid].bf, heatingrates[tid].collisional, heatingrates[tid].gamma,
                    # coolingrates[tid].ff, coolingrates[tid].fb, coolingrates[tid].collisional, coolingrates[tid].adiabatic)

                    estimators[timestep, modelgridindex]["heating_ff"] = float(row[-9])
                    estimators[timestep, modelgridindex]["heating_bf"] = float(row[-8])
                    estimators[timestep, modelgridindex]["heating_coll"] = float(row[-7])
                    estimators[timestep, modelgridindex]["heating_dep"] = float(row[-6])

                    estimators[timestep, modelgridindex]["cooling_ff"] = float(row[-5])
                    estimators[timestep, modelgridindex]["cooling_fb"] = float(row[-4])
                    estimators[timestep, modelgridindex]["cooling_coll"] = float(row[-3])
                    estimators[timestep, modelgridindex]["cooling_adiabatic"] = float(row[-2])

                    # estimators[(timestep, modelgridindex)]['cooling_coll - heating_coll'] = \
                    #     estimators[(timestep, modelgridindex)]['cooling_coll'] - estimators[(timestep, modelgridindex)]['heating_coll']
                    #
                    # estimators[(timestep, modelgridindex)]['cooling_fb - heating_bf'] = \
                    #     estimators[(timestep, modelgridindex)]['cooling_fb'] - estimators[(timestep, modelgridindex)]['heating_bf']
                    #
                    # estimators[(timestep, modelgridindex)]['cooling_ff - heating_ff'] = \
                    #     estimators[(timestep, modelgridindex)]['cooling_ff'] - estimators[(timestep, modelgridindex)]['heating_ff']
                    #
                    # estimators[(timestep, modelgridindex)]['cooling_adiabatic - heating_dep'] = \
                    #     estimators[(timestep, modelgridindex)]['cooling_adiabatic'] - estimators[(timestep, modelgridindex)]['heating_dep']

                    estimators[timestep, modelgridindex]["energy_deposition"] = float(row[-1])

    return estimators
