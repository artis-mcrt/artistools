#!/usr/bin/env python3
"""Artistools - NLTE population related functions."""
import math
import multiprocessing
# import os
import re
# import sys
from functools import lru_cache
from functools import partial
from pathlib import Path
# from itertools import chain

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np
import pandas as pd
from astropy import constants as const
# import numpy as np
# import matplotlib as mpl

import artistools as at


def texifyterm(strterm):
    """Replace a term string with TeX notation equivalent."""
    strtermtex = ''
    passed_term_Lchar = False

    for termpiece in re.split('([_A-Za-z])', strterm):
        if re.match('[0-9]', termpiece) is not None and not passed_term_Lchar:
            # 2S + 1 number
            strtermtex += r'$^{' + termpiece + r'}$'
        elif re.match('[A-Z]', termpiece) is not None:
            # L character - SPDFGH...
            strtermtex += termpiece
            passed_term_Lchar = True
        elif re.match('[eo]', termpiece) is not None and passed_term_Lchar:
            # odd flag, but don't want to confuse it with the energy index (e.g. o4Fo[2])
            if termpiece != 'e':  # even is assumed by default (and looks neater with all the 'e's)
                strtermtex += r'$^{\rm ' + termpiece + r'}$'
        elif re.match(r'[0-9]?.*\]', termpiece) is not None:
            # J value
            strtermtex += termpiece.split('[')[0] + r'$_{' + termpiece.lstrip('0123456789').strip('[]') + r'}$'
        elif re.match('[0-9]', termpiece) is not None and passed_term_Lchar:
            # extra number after S char
            strtermtex += termpiece

    strtermtex = strtermtex.replace('$$', '')
    return strtermtex


def texifyconfiguration(levelname):
    """Replace a level configuration with the formatted LaTeX equivalent."""
    # the underscore gets confused with LaTeX subscript operator, so switch it to the hash symbol
    strout = '#'.join(levelname.split('_')[:-1]) + '#'
    for strorbitalocc in re.findall(r'[0-9][a-z][0-9]?[#(]', strout):
        n, lchar, occ = re.split('([a-z])', strorbitalocc)
        lastchar = '(' if occ.endswith('(') else '#'
        occ = occ.rstrip('#(')
        strorbitalocctex = n + lchar + (r'$^{' + occ + r'}$' if occ else '') + lastchar
        strout = strout.replace(strorbitalocc, strorbitalocctex)

    for parentterm in re.findall(r'\([0-9][A-Z][^)]?\)', strout):
        parentermtex = f'({texifyterm(parentterm.strip("()"))})'
        strout = strout.replace(parentterm, parentermtex)
    strterm = levelname.split('_')[-1]
    strout += ' ' + texifyterm(strterm)

    strout = strout.replace('#', '')
    strout = strout.replace('$$', '')
    # print(f"Replacing levelname '{levelname}' with '{strout}'")
    return strout


def add_lte_pops(modelpath, dfpop, columntemperature_tuples, noprint=False, maxlevel=-1):
    """Add columns to dfpop with LTE populations.

    columntemperature_tuples is a sequence of tuples of column name and temperature, e.g., ('mycolumn', 3000)
    """
    k_b = const.k_B.to('eV / K').value

    for _, row in dfpop.drop_duplicates(['modelgridindex', 'timestep', 'Z', 'ion_stage']).iterrows():
        modelgridindex = int(row.modelgridindex)
        timestep = int(row.timestep)
        Z = int(row.Z)
        ion_stage = int(row.ion_stage)

        ionlevels = at.atomic.get_levels(modelpath).query('Z == @Z and ion_stage == @ion_stage').iloc[0].levels

        gs_g = ionlevels.iloc[0].g
        gs_energy = ionlevels.iloc[0].energy_ev

        gs_pop = dfpop.query(
            'modelgridindex == @modelgridindex and timestep == @timestep '
            'and Z == @Z and ion_stage == @ion_stage and level == 0').iloc[0]['n_NLTE']

        masksuperlevel = (
            (dfpop['modelgridindex'] == modelgridindex) & (dfpop['timestep'] == timestep)
            & (dfpop['Z'] == Z) & (dfpop['ion_stage'] == ion_stage) & (dfpop['level'] == -1))

        masknotsuperlevel = (
            (dfpop['modelgridindex'] == modelgridindex) & (dfpop['timestep'] == timestep)
            & (dfpop['Z'] == Z) & (dfpop['ion_stage'] == ion_stage) & (dfpop['level'] != -1))

        def f_ltepop(x, T_exc, gsg, gse, ionlevels):
            return (ionlevels.iloc[int(x.level)].g / gsg
                    * math.exp(- (ionlevels.iloc[int(x.level)].energy_ev - gse) / k_b / T_exc))

        for columnname, T_exc in columntemperature_tuples:
            dfpop.loc[masknotsuperlevel, columnname] = dfpop.loc[masknotsuperlevel].apply(
                f_ltepop, args=(T_exc, gs_g, gs_energy, ionlevels), axis=1)

        if not dfpop[masksuperlevel].empty:
            levelnumber_sl = dfpop.query(
                'modelgridindex == @modelgridindex and timestep == @timestep '
                'and Z == @Z and ion_stage == @ion_stage').level.max() + 1

            if maxlevel < 0 or levelnumber_sl <= maxlevel:
                if not noprint:
                    print(f'{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]} '
                          f'has a superlevel at level {levelnumber_sl}')

                for columnname, T_exc in columntemperature_tuples:
                    dfpop.loc[masksuperlevel, columnname] = ionlevels.iloc[levelnumber_sl:].eval(
                        'g / @gs_g * exp(- (energy_ev - @gs_energy) / @k_b / @T_exc)').sum()

            dfpop.loc[masksuperlevel, 'level'] = levelnumber_sl + 2

    return dfpop


@at.diskcache(savezipped=True)
def read_file(nltefilepath):
    """Read NLTE populations from one file."""

    if not nltefilepath.is_file():
        nltefilepathgz = Path(str(nltefilepath) + '.gz')
        nltefilepathxz = Path(str(nltefilepath) + '.xz')
        if nltefilepathxz.is_file():
            nltefilepath = nltefilepathxz
        elif nltefilepathgz.is_file():
            nltefilepath = nltefilepathgz
        else:
            # print(f'Warning: Could not find {nltefilepath}')
            return pd.DataFrame()

    filesize = Path(nltefilepath).stat().st_size / 1024 / 1024
    print(f'Reading {nltefilepath} ({filesize:.2f} MiB)')

    try:
        dfpop = pd.read_csv(nltefilepath, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return dfpop


def read_file_filtered(nltefilepath, strquery=None, dfqueryvars=None):
    dfpopfile = read_file(nltefilepath)

    if strquery and not dfpopfile.empty:
        dfpopfile.query(strquery, local_dict=dfqueryvars, inplace=True)

    return dfpopfile


@lru_cache(maxsize=2)
@at.diskcache(savezipped=True, funcversion="2020-07-03.1327", saveonly=False)
def read_files(modelpath, timestep=-1, modelgridindex=-1, dfquery=None, dfqueryvars={}):
    """Read in NLTE populations from a model for a particular timestep and grid cell."""

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)

    dfpop = pd.DataFrame()

    nltefilepaths = [Path(folderpath, f'nlte_{mpirank:04d}.out')
                     for folderpath in at.get_runfolders(modelpath, timestep=timestep) for mpirank in mpiranklist]

    dfqueryvars['modelgridindex'] = modelgridindex
    dfqueryvars['timestep'] = timestep

    dfquery_full = ''
    if timestep >= 0:
        dfquery_full = 'timestep==@timestep'

    if modelgridindex >= 0:
        if dfquery_full:
            dfquery_full += ' and '
        dfquery_full += 'modelgridindex==@modelgridindex'

    if dfquery:
        if dfquery_full:
            dfquery_full = f'({dfquery_full}) and '
        dfquery_full += f'({dfquery})'

    if at.config['num_processes'] > 1:
        with multiprocessing.Pool(processes=at.config['num_processes']) as pool:
            arr_dfnltepop = pool.map(partial(read_file_filtered, strquery=dfquery_full, dfqueryvars=dfqueryvars),
                                     nltefilepaths)
            pool.close()
            pool.join()
            pool.terminate()
    else:
        arr_dfnltepop = [read_file_filtered(f, strquery=dfquery_full, dfqueryvars=dfqueryvars) for f in nltefilepaths]

    dfpop = pd.concat(arr_dfnltepop).copy()

    return dfpop
