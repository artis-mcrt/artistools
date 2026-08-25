"""Read estimator files written by classic (pre-2020) versions of ARTIS."""

import itertools
import typing as t
from pathlib import Path

import artistools as at


def get_atomic_composition(modelpath: Path) -> dict[int, int]:
    """Return the number of ions of each element, counted from the [input.c] lines of output_0-0.txt.

    This counts ion lines rather than reusing get_composition_data_from_outputfile, which returns
    uppermost - lowermost + 1 and yields a null count for an element with no ion lines at all. The
    estimator rows are sliced by these counts, so a null or a gap-inflated count misaligns every
    element after it.
    """
    atomic_composition = {}

    with at.zopen(Path(modelpath, "output_0-0.txt"), encoding="utf-8") as foutput:
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

            elsymbol = at.get_elsymbol(atomic_number)
            elpop = outdict.get(f"nnelement_{elsymbol}", 0)
            outdict[f"nnelement_{elsymbol}"] = elpop + value_thision


def get_first_ts_in_run_directory(modelpath: str | Path) -> dict[str, int]:
    """Return the first timestep contained in each run folder, since classic estimator files restart their numbering."""
    folderlist_all = (*sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]), Path(modelpath))

    first_timesteps_in_dir = {}

    for folder in folderlist_all:
        outputfile = at.firstexisting_or_none("output_0-0.txt", folder=folder, tryzipped=True, search_subfolders=False)
        if outputfile is not None:
            with at.zopen(outputfile, encoding="utf-8") as output_0:
                timesteps_in_dir = [
                    line.strip(".\n").split(" ")[-1]
                    for line in output_0
                    if "[debug] update_packets: updating packet 0 for timestep" in line
                ]
            # a log that records no packet update gives no first timestep, thus the folder is left out and
            # the caller starts it at zero
            if timesteps_in_dir:
                first_timesteps_in_dir[str(folder)] = int(timesteps_in_dir[0])

    return first_timesteps_in_dir


def read_classic_estimators(modelpath: Path) -> dict[tuple[int, int], t.Any] | None:
    """Return the classic estimators keyed by (timestep, modelgridindex), or None when no estimator files are found."""
    modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
    # the trailing wildcard accepts a compressed file, which at.zopen below reads. The macroatom reader
    # globs the same way.
    estimfiles = sorted(
        itertools.chain(Path(modelpath).glob("estimators_????.out*"), Path(modelpath).glob("*/estimators_????.out*"))
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
        # If classic plots break it's probably getting first timestep here
        # Try either of the next two lines
        # the run log gives the first timestep of the folder. A folder with no log starts at zero, which is
        # correct when the run was not restarted.
        if str(estfilepath.parent) in first_timesteps_in_dir:
            timestep = first_timesteps_in_dir[str(estfilepath.parent)]
        else:
            print(f"WARNING: no first timestep found for {estfilepath.parent}, assuming the run starts at timestep 0")
            timestep = 0
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

                estimcell["energy_deposition"] = float(row[-1])

    return estimators
