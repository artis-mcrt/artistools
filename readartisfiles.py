#!/usr/bin/env python3
import collections
import math
import os

# import scipy.signal
import pandas as pd
from astropy import constants as const

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                  'XVIII', 'XIX', 'XX')

refspectralabels = {
    '2010lp_20110928_fors2.txt':
        'SN2010lp +264d (Taubenberger et al. 2013)',

    'dop_dered_SN2013aa_20140208_fc_final.txt':
        'SN2013aa +360d (Maguire et al. in prep)',

    '2003du_20031213_3219_8822_00.txt':
        'SN2003du +221.3d (Stanishev et al. 2007)',

    'nero-nebspec.txt':
        'NERO +300d one-zone',

    'maurer2011_RTJ_W7_338d.txt':
        'RTJ W7 +338d (Maurer et al. 2011)'
}


def showtimesteptimes(specfilename, numberofcolumns=5):
    """
        Print a table showing the timeteps and their corresponding times
    """
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    print('Time steps and corresponding times in days:\n')

    times = specdata.columns
    indexendofcolumnone = math.ceil((len(times) - 1) / numberofcolumns)
    for rownum in range(0, indexendofcolumnone):
        strline = ""
        for colnum in range(numberofcolumns):
            if colnum > 0:
                strline += '\t'
            newindex = rownum + colnum * indexendofcolumnone
            if newindex < len(times):
                strline += f'{newindex:4d}: {float(times[newindex + 1]):.3f}'
        print(strline)


def get_composition_data(filename):
    """
        Return a pandas DataFrame containing details of included
        elements and ions
    """

    columns = ('Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,'
               'abundance,mass,startindex').split(',')

    compdf = pd.DataFrame()

    with open(filename, 'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        startindex = 0
        for _ in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            row_list = list(map(int, linesplit[:5])) + list(map(float, linesplit[5:])) + [startindex]

            rowdf = pd.DataFrame([row_list], columns=columns)
            compdf = compdf.append(rowdf, ignore_index=True)

            startindex += int(rowdf['nions'])

    return compdf


def get_modeldata(filename):
    """
        Return a list containing named tuples for all model grid cells
    """
    modeldata = []
    gridcelltuple = collections.namedtuple('gridcell', 'cellid velocity logrho ffe fni fco f52fe f48cr')
    modeldata.append(gridcelltuple._make([-1, 0., 0., 0., 0., 0., 0., 0.]))
    with open(filename, 'r') as fmodel:
        line = fmodel.readline()
        # gridcellcount = int(line)
        line = fmodel.readline()
        # t_model_init = float(line)
        for line in fmodel:
            row = line.split()
            modeldata.append(gridcelltuple._make([int(row[0])] + list(map(float, row[1:]))))

    return modeldata


def get_initialabundances1d(filename):
    """
        Returns a list of mass fractions
    """
    abundancedata = []
    abundancedata.append([])
    with open(filename, 'r') as fabund:
        for line in fabund:
            row = line.split()
            abundancedata.append([int(row[0])] + list(map(float, row[1:])))

    return abundancedata


def get_spectrum(specfilename, timesteplow, timestephigh=-1, normalised=False, fnufilterfunc=None):
    """
        Return a pandas DataFrame containing an ARTIS emergent spectrum
    """
    if timestephigh < 0:
        timestephigh = timesteplow

    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    arraynu = specdata['0']

    delta_t = (float(get_timestep_time(specfilename, timesteplow + 1)) -
               float(get_timestep_time(specfilename, timesteplow)))
    delta_t_alltimesteps = delta_t
    array_fnu = specdata[specdata.columns[timesteplow + 1]] * delta_t

    for timestep in range(timesteplow + 1, timestephigh + 1):
        delta_t = (float(get_timestep_time(specfilename, timestep + 1)) -
                   float(get_timestep_time(specfilename, timestep)))
        delta_t_alltimesteps += delta_t
        array_fnu += specdata[specdata.columns[timestep + 1]] * delta_t

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter")
        array_fnu = fnufilterfunc(array_fnu)

    array_fnu = array_fnu / delta_t_alltimesteps

    dfspectrum = pd.DataFrame({'nu': arraynu,
                               'f_nu': array_fnu})

    dfspectrum['lambda_angstroms'] = const.c.value / dfspectrum['nu'] * 1e10
    dfspectrum['f_lambda'] = dfspectrum['f_nu'] * dfspectrum['nu'] / dfspectrum['lambda_angstroms'] * math.pi * 4

    if normalised:
        dfspectrum['f_nu'] /= dfspectrum['f_nu'].max()
        dfspectrum['f_lambda'] /= dfspectrum['f_lambda'].max()

    return dfspectrum


def get_timestep_times(specfilename):
    """
        Return a list of the time in days of each timestep using a spec.out file
    """
    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[1:]


def get_timestep_time(specfilename, timestep):
    """
        Return the time in days of a timestep number using a spec.out file
    """
    if os.path.isfile(specfilename):
        return get_timestep_times(specfilename)[timestep]
    else:
        return -1


def get_levels(adatafilename):
    """
        Return a list of lists of levels
    """
    level_lists = []
    iontuple = collections.namedtuple('ion', 'Z ion_stage level_count ion_pot level_list')
    leveltuple = collections.namedtuple('level', 'number energy_ev g transition_count levelname')

    with open(adatafilename, 'r') as fadata:
        for line in fadata:
            if len(line.strip()) > 0:
                ionheader = line.split()
                level_count = int(ionheader[2])

                level_list = []
                for _ in range(level_count):
                    line = fadata.readline()
                    row = line.split()
                    levelname = row[4].strip('\'')
                    level_list.append(leveltuple(int(row[0]), float(row[1]), float(row[2]), int(row[3]), levelname))

                level_lists.append(iontuple(int(ionheader[0]), int(ionheader[1]), level_count,
                                            float(ionheader[3]), list(level_list)))

    return level_lists


def get_nlte_populations(nltefile, modelgridindex, timestep, atomic_number, temperature_exc):
    all_levels = get_levels('adata.txt')

    dfpop = pd.read_csv(nltefile, delim_whitespace=True)
    dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep) & (Z==@atomic_number)',
                inplace=True)

    k_b = const.k_B.to('eV / K').value
    list_indicies = []
    list_ltepopcustom = []
    list_parity = []
    gspop = {}
    for index, row in dfpop.iterrows():
        list_indicies.append(index)

        ion_stage = row.ion_stage
        if (row.Z, row.ion_stage) not in gspop:
            gspop[(row.Z, row.ion_stage)] = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage and level==0').iloc[0].n_NLTE

        levelnumber = int(row.level)
        if levelnumber == -1:  # superlevel
            levelnumber = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage').level.max()
            print(f'{elsymbols[atomic_number]} {roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
            dfpop.loc[index, 'level'] = levelnumber + 2
            ltepopcustom = 0.0
            parity = 0
        else:
            for _, ion_data in enumerate(all_levels):
                if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
                    level = ion_data.level_list[levelnumber]
                    gslevel = ion_data.level_list[0]

            ltepopcustom = gspop[(row.Z, row.ion_stage)] * level.g / gslevel.g * math.exp(
                - (level.energy_ev - gslevel.energy_ev) / k_b / temperature_exc)

            levelname = level.levelname.split('[')[0]
            parity = 1 if levelname[-1] == 'o' else 0

        list_ltepopcustom.append(ltepopcustom)
        list_parity.append(parity)

    dfpop['n_LTE_custom'] = pd.Series(list_ltepopcustom, index=list_indicies)
    dfpop['parity'] = pd.Series(list_parity, index=list_indicies)

    return dfpop


def get_nlte_populations_oldformat(nltefile, modelgridindex, timestep, atomic_number, temperature_exc):
    compositiondata = get_composition_data('compositiondata.txt')
    elementdata = compositiondata.query('Z==@atomic_number')

    if len(elementdata) < 1:
        print(f'Error: element Z={atomic_number} not in composition file')
        return None

    all_levels = get_levels('adata.txt')

    skip_block = False
    dfpop = pd.DataFrame().to_sparse()
    with open(nltefile, 'r') as nltefile:
        for line in nltefile:
            row = line.split()

            if row and row[0] == 'timestep':
                skip_block = int(row[1]) != timestep
                if row[2] == 'modelgridindex' and int(row[3]) != modelgridindex:
                    skip_block = True

            if skip_block:
                continue
            elif len(row) > 2 and row[0] == 'nlte_index' and row[1] != '-':  # level row
                matchedgroundstateline = False
            elif len(row) > 1 and row[1] == '-':  # ground state
                matchedgroundstateline = True
            else:
                continue

            dfrow = parse_nlte_row(row, dfpop, elementdata, all_levels, timestep,
                                   temperature_exc, matchedgroundstateline)

            if dfrow is not None:
                dfpop = dfpop.append(dfrow, ignore_index=True)

    return dfpop


def parse_nlte_row(row, dfpop, elementdata, all_levels, timestep, temperature_exc, matchedgroundstateline):
    """
        Read a line from the NLTE output file and return a Pandas DataFrame
    """
    levelpoptuple = collections.namedtuple(
        'ionpoptuple', 'timestep Z ion_stage level energy_ev parity n_LTE n_NLTE n_LTE_custom')

    elementindex = elementdata.index[0]
    atomic_number = int(elementdata.iloc[0].Z)
    element = int(row[row.index('element') + 1])
    if element != elementindex:
        return None
    ion = int(row[row.index('ion') + 1])
    ion_stage = int(elementdata.iloc[0].lowermost_ionstage) + ion

    if row[row.index('level') + 1] != 'SL':
        levelnumber = int(row[row.index('level') + 1])
        superlevel = False
    else:
        levelnumber = dfpop.query('timestep==@timestep and ion_stage==@ion_stage').level.max() + 3
        print(f'{elsymbols[atomic_number]} {roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
        superlevel = True

    for _, ion_data in enumerate(all_levels):
        if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
            level = ion_data.level_list[levelnumber]
            gslevel = ion_data.level_list[0]

    ltepop = float(row[row.index('nnlevel_LTE') + 1])

    if matchedgroundstateline:
        nltepop = ltepop_custom = ltepop

        levelname = gslevel.levelname.split('[')[0]
        energy_ev = gslevel.energy_ev
    else:
        nltepop = float(row[row.index('nnlevel_NLTE') + 1])

        k_b = const.k_B.to('eV / K').value
        gspop = dfpop.query('timestep==@timestep and ion_stage==@ion_stage and level==0').iloc[0].n_NLTE
        levelname = level.levelname.split('[')[0]
        energy_ev = (level.energy_ev - gslevel.energy_ev)

        ltepop_custom = gspop * level.g / gslevel.g * math.exp(
            -energy_ev / k_b / temperature_exc)

    parity = 1 if levelname[-1] == 'o' else 0
    if superlevel:
        parity = 0

    newrow = levelpoptuple(timestep=timestep, Z=int(elementdata.iloc[0].Z), ion_stage=ion_stage,
                           level=levelnumber, energy_ev=energy_ev, parity=parity,
                           n_LTE=ltepop, n_NLTE=nltepop, n_LTE_custom=ltepop_custom)

    return pd.DataFrame(data=[newrow], columns=levelpoptuple._fields)


def plot_reference_spectra(axis, args, flambdafilterfunc=None):
    """
        Plot reference spectra listed in args.refspecfiles
    """
    if args.refspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        colorlist = ['black', '0.4']
        refspectra = [(fn, refspectralabels.get(fn, fn), c) for fn, c in zip(args.refspecfiles, colorlist)]
        for (filename, serieslabel, linecolor) in refspectra:
            filepath = os.path.join(scriptdir, 'spectra', filename)
            specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                                   names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

            specdata.query('lambda_angstroms > @args.xmin and lambda_angstroms < @args.xmax', inplace=True)

            print(f"'{serieslabel}' has {len(specdata)} points")

            if args.normalised:
                specdata['f_lambda'] = (specdata['f_lambda'] / specdata['f_lambda'].max())

            if flambdafilterfunc:
                specdata['f_lambda'] = flambdafilterfunc(specdata['f_lambda'])

            if len(specdata) > 5000:
                # specdata = scipy.signal.resample(specdata, 10000)
                print(f"downsamping {filename}")
                specdata = specdata.iloc[::3, :]

            specdata.plot(x='lambda_angstroms', y='f_lambda', lw=1.5, ax=axis,
                          label=serieslabel, zorder=-1, color=linecolor)


def addargs_timesteps(parser):
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestepmin', type=int, default=70,
                        help='First or only included timestep')
    parser.add_argument('-timestepmax', type=int, default=80,
                        help='Last included timestep')


def addargs_spectrum(parser):
    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=7000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')


if __name__ == "__main__":
    print("this script is for inclusion only")
