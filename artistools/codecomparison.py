"""
File readers for Blondin et al. code comparison file formats
The model paths are not real file system paths, but take a form like this:
codecomparison/[modelname]/[codename]

e.g., codecomparison/DDC10/artisnebular
"""
import math

import artistools as at
import numpy as np
import pandas as pd
from pathlib import Path


def get_timestep_times_float(modelpath, loc='mid'):
    modelpath = Path(modelpath)
    _, modelname, codename = modelpath.parts

    filepath = Path(at.config['codecomparisondata1path'], modelname, f"phys_{modelname}_{codename}.txt")

    with open(filepath, "r") as fphys:
        ntimes = int(fphys.readline().replace('#NTIMES:', ''))
        tmids = np.array([float(x) for x in fphys.readline().replace('#TIMES[d]:', '').split()])

    tstarts = np.zeros_like(tmids)
    tstarts[1:] = (tmids[1:] + tmids[:-1]) / 2.
    tstarts[0] = tmids[0] - (tstarts[1] - tmids[0])

    tends = np.zeros_like(tmids)
    tends[:-1] = (tmids[:-1] + tmids[1:]) / 2.
    tends[-1] = tmids[-1] + (tmids[-1] - tstarts[-1])

    if loc == 'mid':
        return tmids
    elif loc == 'start':
        return tstarts
    elif loc == 'end':
        return tends
    elif loc == 'delta':
        tdeltas = tends - tstarts
        return tdeltas
    else:
        raise ValueError("loc must be one of 'mid', 'start', 'end', or 'delta'")


def read_reference_estimators(modelpath, modelgridindex=None, timestep=None):
    """Read estimators from code comparison workshop file.

    """

    virtualfolder, inputmodel, codename = modelpath.parts
    assert virtualfolder == 'codecomparison'

    inputmodelfolder = Path(at.config['codecomparisondata1path'], inputmodel)

    physfilepath = Path(inputmodelfolder, f"phys_{inputmodel}_{codename}.txt")

    estimators = {}
    cell_vel = {}
    cur_timestep = -1
    cur_modelgridindex = -1
    with open(physfilepath, "r") as fphys:
        ntimes = int(fphys.readline().replace('#NTIMES:', ''))
        arr_timedays = np.array([float(x) for x in fphys.readline().replace('#TIMES[d]:', '').split()])
        assert len(arr_timedays) == ntimes

        for line in fphys:
            row = line.split()

            if row[0] == '#TIME:':

                cur_timestep += 1
                cur_modelgridindex = -1
                timedays = float(row[1])
                assert np.isclose(timedays, arr_timedays[cur_timestep], rtol=0.01)

            elif row[0] == '#NVEL:':

                nvel = int(row[1])

            elif not line.lstrip().startswith('#'):

                cur_modelgridindex += 1

                key = (cur_timestep, cur_modelgridindex)
                if key not in estimators:
                    estimators[key] = {'emptycell': False}

                estimators[key]['vel_mid'] = float(row[0])
                estimators[key]['Te'] = float(row[1])
                estimators[key]['rho'] = float(row[2])
                estimators[key]['nne'] = float(row[3])
                estimators[key]['nntot'] = float(row[4])

                estimators[key]['velocity_outer'] = estimators[key]['vel_mid']

    ionfracfilepaths = inputmodelfolder.glob(f"ionfrac_*_{inputmodel}_{codename}.txt")
    for ionfracfilepath in ionfracfilepaths:
        _, element, _, _ = ionfracfilepath.stem.split('_')

        with open(ionfracfilepath, "r") as fions:
            print(ionfracfilepath)
            ntimes_2 = int(fions.readline().replace('#NTIMES:', ''))
            assert ntimes_2 == ntimes

            nstages = int(fions.readline().replace('#NSTAGES:', ''))

            arr_timedays_2 = np.array([float(x) for x in fions.readline().replace('#TIMES[d]:', '').split()])
            assert np.allclose(arr_timedays, arr_timedays_2, rtol=0.01)

            cur_timestep = -1
            cur_modelgridindex = -1
            for line in fions:
                row = line.split()

                if row[0] == '#TIME:':

                    cur_timestep += 1
                    cur_modelgridindex = -1
                    timedays = float(row[1])
                    assert np.isclose(timedays, arr_timedays[cur_timestep], rtol=0.01)

                elif row[0] == '#NVEL:':

                    nvel = int(row[1])

                elif row[0] == '#vel_mid[km/s]':
                    row = [s for s in line.split('  ') if s]  # need a double space because some ion columns have a space
                    iontuples = []
                    ion_startnumber = None
                    for ionstr in row[1:]:
                        atomic_number = at.get_atomic_number(ionstr.strip().rstrip(' 0123456789').title())
                        ion_number = int(ionstr.lstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '))

                        # there is unfortunately an inconsistency between codes for
                        # whether the neutral ion is called 0 or 1
                        if ion_startnumber is None:
                            ion_startnumber = ion_number

                        ion_stage = ion_number + 1 if ion_startnumber == 0 else ion_number

                        iontuples.append((atomic_number, ion_stage))

                elif not line.lstrip().startswith('#'):
                    cur_modelgridindex += 1

                    tsmgi = (cur_timestep, cur_modelgridindex)
                    if 'populations' not in estimators[tsmgi]:
                        estimators[tsmgi]['populations'] = {}

                    assert len(row) == nstages + 1
                    assert len(iontuples) == nstages
                    for (atomic_number, ion_stage), strionfrac in zip(iontuples, row[1:]):
                        try:
                            ionfrac = float(strionfrac)
                            ionpop = ionfrac * estimators[tsmgi]['nntot']
                            if ionpop > 1e-80:
                                estimators[tsmgi]['populations'][(atomic_number, ion_stage)] = ionpop
                                estimators[tsmgi]['populations'].setdefault(atomic_number, 0.)
                                estimators[tsmgi]['populations'][atomic_number] += ionpop

                        except ValueError:
                            estimators[tsmgi]['populations'][(atomic_number, ion_stage)] = float('NaN')

                    assert np.isclose(float(row[0]), estimators[tsmgi]['vel_mid'], rtol=0.01)
                    assert estimators[key]['vel_mid']

    return estimators


def get_spectra(modelpath):
    modelpath = Path(modelpath)
    virtualfolder, inputmodel, codename = modelpath.parts
    assert virtualfolder == 'codecomparison'

    inputmodelfolder = Path(at.config['codecomparisondata1path'], inputmodel)

    specfilepath = Path(inputmodelfolder, f"spectra_{inputmodel}_{codename}.txt")

    with open(specfilepath, "r") as fspec:
        ntimes = int(fspec.readline().replace('#NTIMES:', ''))
        nwave = int(fspec.readline().replace('#NWAVE:', ''))
        arr_timedays = np.array([float(x) for x in fspec.readline().split()[1:]])
        assert len(arr_timedays) == ntimes

        dfspectra = pd.read_csv(fspec, delim_whitespace=True, header=None, names=['lambda'] + list(arr_timedays),
                                comment='#')

    return dfspectra, arr_timedays


def plot_spectrum(modelpath, timedays, ax, **plotkwargs):
    dfspectra, arr_timedays = get_spectra(modelpath)
    # print(dfspectra)
    timeindex = (np.abs(arr_timedays - float(timedays))).argmin()
    timedays_found = dfspectra.columns[timeindex + 1]

    print(f"{modelpath}: requested spectrum at {timedays} days. Closest matching spectrum is at {timedays_found} days")
    assert np.isclose(arr_timedays[timeindex], float(timedays_found), rtol=0.01)  # check columns match
    assert np.isclose(float(timedays), float(timedays_found), rtol=0.1)  # found a detect match to requested time
    # print(dfspectra[['lambda', timedays_found]])
    label = str(modelpath).lstrip('_') + f" {timedays_found}d"

    megaparsec_to_cm = 3.085677581491367e+24
    arr_flux = dfspectra[dfspectra.columns[timeindex + 1]] / 4 / math.pi / (megaparsec_to_cm ** 2)

    ax.plot(dfspectra['lambda'], arr_flux, label=label, **plotkwargs)
