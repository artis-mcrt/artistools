from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad


DAY = 86400  # day in seconds
MSUN = 1.989e33  # solar mass in grams


def write_energydistribution_file(energydistdata, outputfilepath='.'):
    print('Writing energydistribution.txt')
    with open(Path(outputfilepath) / 'energydistribution.txt', 'w') as fmodel:
        fmodel.write(f'{len(energydistdata["cell_energy"])}\n')  # write number of points
        energydistdata.to_csv(fmodel, header=False, sep='\t', index=False, float_format='%g')


def write_energyrate_file(energy_rate_data, outputfilepath='.'):
    print('Writing energyrate.txt')
    with open(Path(outputfilepath) / 'energyrate.txt', 'w') as fmodel:
        fmodel.write(f'{len(energy_rate_data["times"])}\n')  # write number of points
        energy_rate_data.to_csv(fmodel, sep='\t', index=False, header=False, float_format='%.10f')


def rprocess_const_and_powerlaw():
    """Following eqn 4 Korobkin 2012"""

    def integrand(t_days, t0, epsilon0, sigma, alpha, thermalisation_factor):
        return (epsilon0 * ((1/2) - (1/np.pi * np.arctan((t_days-t0)/sigma)))**alpha) * (thermalisation_factor / 0.5)

    tmin = 0.01*DAY
    tmax = 50*DAY
    t0 = 1.3 #seconds
    epsilon0 = 2e18
    sigma = 0.11
    alpha = 1.3
    thermalisation_factor = 0.5

    E_tot = quad(integrand, tmin, tmax, args=(t0, epsilon0, sigma, alpha, thermalisation_factor))  # ergs/s/g
    print("Etot per gram", E_tot[0])
    E_tot = E_tot[0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=200)
    energy_per_gram_cumulative = [0]
    for time in times[1:]:
        cumulative_integral = (quad(integrand, tmin, time, args=(t0, epsilon0, sigma, alpha, thermalisation_factor)))  # ergs/s/g
        energy_per_gram_cumulative.append(cumulative_integral[0])

    energy_per_gram_cumulative = np.array(energy_per_gram_cumulative)

    rate = energy_per_gram_cumulative / E_tot

    times_and_rate = {'times': times/DAY, 'rate': rate}
    times_and_rate = pd.DataFrame(data=times_and_rate)

    return times_and_rate, E_tot


def make_energydistribution_weightedbyrho(rho, E_tot_per_gram, Mtot_grams):

    Etot = E_tot_per_gram * Mtot_grams
    print("Etot", Etot)
    numberofcells = len(rho)

    cellenergy = np.array([Etot] * numberofcells)
    cellenergy = cellenergy * (rho / sum(rho))

    energydistdata = {
        'cellid': np.arange(1, len(rho)+1),
        'cell_energy': cellenergy
    }

    print("sum energy cells", sum(energydistdata['cell_energy']))
    energydistdata = pd.DataFrame(data=energydistdata)

    return energydistdata


def make_energy_files(rho, Mtot_grams):
    times_and_rate, E_tot_per_gram = rprocess_const_and_powerlaw()
    energydistributiondata = make_energydistribution_weightedbyrho(rho, E_tot_per_gram, Mtot_grams)

    write_energydistribution_file(energydistributiondata)
    write_energyrate_file(times_and_rate)
