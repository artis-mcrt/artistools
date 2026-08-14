"""Read estimator files written by classic (pre-2020) versions of ARTIS."""

import itertools
import typing as t
from pathlib import Path

import artistools as at


def parse_ion_row_classic(row: list[str], outdict: dict[str, t.Any], atomic_composition: dict[int, int]) -> None:
    """Parse the per-ion populations of one estimator row into outdict."""
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
    """Return the first timestep contained in each run folder, since classic estimator files restart their numbering."""
    folderlist_all = (*sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]), Path(modelpath))

    first_timesteps_in_dir = {}

    for folder in folderlist_all:
        outputfile = at.firstexisting_or_none("output_0-0.txt", folder=folder, tryzipped=True)
        if outputfile is not None:
            with at.zopen(outputfile, encoding="utf-8") as output_0:
                timesteps_in_dir = [
                    line.strip(".\n").split(" ")[-1]
                    for line in output_0
                    if "[debug] update_packets: updating packet 0 for timestep" in line
                ]
            first_ts = timesteps_in_dir[0]
            first_timesteps_in_dir[str(folder)] = int(first_ts)

    return first_timesteps_in_dir


def read_classic_estimators(modelpath: Path) -> dict[tuple[int, int], t.Any] | None:
    """Return the classic estimators keyed by (timestep, modelgridindex), or None when no estimator files are found."""
    modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
    estimfiles = list(
        itertools.chain(Path(modelpath).glob("estimators_????.out"), Path(modelpath).glob("*/estimators_????.out"))
    )
    if not estimfiles:
        print("No estimator files found")
        return None
    print(f"Reading {len(estimfiles)} estimator files...")

    first_timesteps_in_dir = get_first_ts_in_run_directory(modelpath)
    dfcomposition = at.get_composition_data_from_outputfile(modelpath)
    atomic_composition = dict(zip(dfcomposition["Z"], dfcomposition["nions"], strict=True))

    inputparams = at.get_inputparams(modelpath)
    ndimensions = inputparams["n_dimensions"]

    estimators: dict[tuple[int, int], t.Any] = {}
    for estfilepath in estimfiles:
        # If classic plots break it's probably getting first timestep here
        # Try either of the next two lines
        timestep = first_timesteps_in_dir[str(estfilepath.parent)]  # get the starting timestep for the estfile
        # timestep = first_timesteps_in_dir[str(estfile[:-20])]
        # timestep = 0  # if the first timestep in the file is 0 then this is fine
        with at.zopen(estfilepath) as estfile:
            modelgridindex = -1
            for line in estfile:
                row = line.split()
                if int(row[0]) <= modelgridindex:
                    timestep += 1
                modelgridindex = int(row[0])

                estimcell: dict[str, t.Any] = {}
                estimators[timestep, modelgridindex] = estimcell

                if ndimensions == 1:
                    estimcell["vel_r_max_kmps"] = modeldata["vel_r_max_kmps"][modelgridindex]

                estimcell["TR"] = float(row[1])
                estimcell["Te"] = float(row[2])
                estimcell["W"] = float(row[3])
                estimcell["TJ"] = float(row[4])

                parse_ion_row_classic(row, estimcell, atomic_composition)

                # heatingrates[tid].ff, heatingrates[tid].bf, heatingrates[tid].collisional, heatingrates[tid].gamma,
                # coolingrates[tid].ff, coolingrates[tid].fb, coolingrates[tid].collisional, coolingrates[tid].adiabatic)

                estimcell["heating_ff"] = float(row[-9])
                estimcell["heating_bf"] = float(row[-8])
                estimcell["heating_coll"] = float(row[-7])
                estimcell["heating_dep"] = float(row[-6])

                estimcell["cooling_ff"] = float(row[-5])
                estimcell["cooling_fb"] = float(row[-4])
                estimcell["cooling_coll"] = float(row[-3])
                estimcell["cooling_adiabatic"] = float(row[-2])

                # estimcell['cooling_coll - heating_coll'] = estimcell['cooling_coll'] - estimcell['heating_coll']
                # estimcell['cooling_fb - heating_bf'] = estimcell['cooling_fb'] - estimcell['heating_bf']
                # estimcell['cooling_ff - heating_ff'] = estimcell['cooling_ff'] - estimcell['heating_ff']
                # estimcell['cooling_adiabatic - heating_dep'] = estimcell['cooling_adiabatic'] - estimcell['heating_dep']

                estimcell["energy_deposition"] = float(row[-1])

    return estimators
