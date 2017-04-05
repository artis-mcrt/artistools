#!/usr/bin/env python3

import math
# import os

# import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u

# from collections import namedtuple


def read_files(radfield_files, modelgridindex=None):
    radfielddata = None
    if not radfield_files:
        print("No radfield files")
    else:
        for index, radfield_file in enumerate(radfield_files):
            print(f'Loading {radfield_file}...')

            radfielddata_thisfile = pd.read_csv(radfield_file, delim_whitespace=True)
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)
            if modelgridindex:
                radfielddata_thisfile.query('modelgridindex==@modelgridindex', inplace=True)
            if radfielddata_thisfile is not None:
                if len(radfielddata_thisfile) > 0:
                    if radfielddata is None:
                        radfielddata = radfielddata_thisfile.copy()
                    else:
                        radfielddata = radfielddata.append(radfielddata_thisfile.copy(), ignore_index=True)

        if radfielddata is None or len(radfielddata) == 0:
            print("No radfield data found")

    return radfielddata


def plot_field_estimators(axis, radfielddata):
    """
        Plot the dJ/dlambda estimators for each bin
    """
    bindata = radfielddata.copy().query('bin_num >= 0')  # exclude the global fit parameters

    arr_lambda = const.c.to('angstrom/s').value / bindata['nu_upper'].values

    bindata['dlambda'] = bindata.apply(
        lambda row: const.c.to('angstrom/s').value * (1 / row['nu_lower'] - 1 / row['nu_upper']), axis=1)

    yvalues = bindata.apply(
        lambda row: row['J'] / row['dlambda'] if (
            not math.isnan(row['J'] / row['dlambda']) and row['T_R'] >= 0) else 0.0, axis=1).values

    # add the starting point
    arr_lambda = np.insert(arr_lambda, 0, const.c.to('angstrom/s').value / bindata['nu_lower'].iloc[0])
    yvalues = np.insert(yvalues, 0, 0.)

    axis.step(arr_lambda, yvalues, where='pre', linewidth=1.5, label='Field estimators', color='blue')
    return max(yvalues)


def plot_fitted_field(axis, radfielddata, xmin, xmax):
    """
        Plot the fitted diluted blackbody for each bin as well as the global fit
    """
    fittedxvalues = []
    fittedyvalues = []
    ymaxglobalfit = -1

    for _, row in radfielddata.iterrows():
        if row['bin_num'] == -1 or row['W'] >= 0:
            if row['bin_num'] == -1:
                # Full-spectrum fit
                nu_lower = const.c.to('angstrom/s').value / xmin
                nu_upper = const.c.to('angstrom/s').value / xmax
            else:
                nu_lower = row['nu_lower']
                nu_upper = row['nu_upper']

            arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])

            arr_lambda = const.c.to('angstrom/s').value / arr_nu_hz
            arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

            if row['bin_num'] == -1:
                ymaxglobalfit = max(arr_j_lambda)
                axis.plot(arr_lambda, arr_j_lambda, linewidth=1.5, color='purple', label='Full-spectrum fitted field')
            else:
                fittedxvalues += list(arr_lambda)
                fittedyvalues += list(arr_j_lambda)
        else:
            arr_nu_hz = (row['nu_lower'], row['nu_upper'])
            arr_j_lambda = [0., 0.]

            fittedxvalues += [const.c.to('angstrom/s').value / nu for nu in arr_nu_hz]
            fittedyvalues += arr_j_lambda

    if fittedxvalues:
        axis.plot(fittedxvalues, fittedyvalues, linewidth=1.5, color='green', label='Fitted field', alpha=0.8)

    return max(max(fittedyvalues), ymaxglobalfit)


def j_nu_dbb(arr_nu_hz, W, T):
    """# CGS units J_nu for diluted blackbody"""

    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        return [W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b)) for nu_hz in arr_nu_hz]
    else:
        return [0. for _ in arr_nu_hz]
