import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

import artistools as at
import artistools.inputmodel


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
    t0 = 1.3  # seconds
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

    nuclear_heating_power = []
    for time in times:
        nuclear_heating_power.append(integrand(time, t0, epsilon0, sigma, alpha, thermalisation_factor))

    # times_and_rate = {'times': times/DAY, 'rate': rate, 'nuclear_heating_power': nuclear_heating_power}
    times_and_rate = {'times': times/DAY, 'rate': rate}
    times_and_rate = pd.DataFrame(data=times_and_rate)

    return times_and_rate, E_tot


def energy_from_rprocess_calculation(energy_thermo_data, get_rate=True):

    index_time_greaterthan = energy_thermo_data[energy_thermo_data['time/s'] > 1e7].index  # 1e7 seconds = 116 days
    energy_thermo_data.drop(index_time_greaterthan, inplace=True)
    # print("Dropping times later than 116 days")

    skipfirstnrows = 0  # not sure first values look sensible -- check this
    times = energy_thermo_data['time/s'][skipfirstnrows:]
    qdot = energy_thermo_data['Qdot'][skipfirstnrows:]

    E_tot = np.trapz(y=qdot, x=times)  # erg / g

    if get_rate:
        print(f"E_tot {E_tot} erg/g")

        import scipy.integrate
        cumulative_integrated_energy = scipy.integrate.cumulative_trapezoid(y=qdot, x=times)
        cumulative_integrated_energy = np.insert(cumulative_integrated_energy, 0, 0)

        rate = cumulative_integrated_energy / E_tot

        times_and_rate = {'times': times/DAY, 'rate': rate}
        times_and_rate = pd.DataFrame(data=times_and_rate)

        return times_and_rate, E_tot

    else:
        return E_tot


def get_rprocess_calculation_files(path_to_rprocess_calculation, interpolate_trajectories=False, thermalisation=False):
    tarfiles = [file for file in os.listdir(path_to_rprocess_calculation) if file.endswith(".tar.xz")]

    trajectory_ids = []
    trajectory_E_tot = []

    if interpolate_trajectories:
        interpolated_trajectories = {'time/s': np.logspace(-1, 7, 300)}

    energy_thermo_filepath = "./Run_rprocess/energy_thermo.dat"
    for file in tarfiles:
        trajectory_id = file.split('.')[0]
        tar = tarfile.open(path_to_rprocess_calculation / file, mode='r:*')

        energythermo_file = tar.extractfile(member=energy_thermo_filepath)
        energy_thermo_data = pd.read_csv(energythermo_file, delim_whitespace=True)
        # print(energy_thermo_data['Qdot'])
        # print(energy_thermo_data['time/s'])

        if interpolate_trajectories:
            qdotinterp = np.interp(interpolated_trajectories['time/s'], energy_thermo_data['time/s'],
                                   energy_thermo_data['Qdot'])
            interpolated_trajectories[trajectory_id] = qdotinterp

        E_tot = energy_from_rprocess_calculation(energy_thermo_data, get_rate=False, thermalisation=thermalisation)

        trajectory_ids.append(float(trajectory_id))
        trajectory_E_tot.append(E_tot)

    if interpolate_trajectories:
        interpolated_trajectories = pd.DataFrame.from_dict(interpolated_trajectories)
        interpolated_trajectories['mean'] = interpolated_trajectories.iloc[:, 1:].mean(axis=1)

        index_time_lessthan = interpolated_trajectories[interpolated_trajectories['time/s'] < 1.1e-1].index
        interpolated_trajectories.drop(index_time_lessthan, inplace=True)

        interpolated_trajectories.to_csv(path_to_rprocess_calculation / 'interpolatedQdot.dat', sep=' ', index=False)
    print(f"sum etot {sum(trajectory_E_tot)}")
    trajectory_energy = {'id': trajectory_ids, 'E_tot': trajectory_E_tot}
    trajectory_energy = pd.DataFrame.from_dict(trajectory_energy)
    trajectory_energy = trajectory_energy.sort_values(by='id')
    print(trajectory_energy)
    trajectory_energy.to_csv(path_to_rprocess_calculation / 'trajectoryQ.dat', sep=' ', index=False)


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


def make_energy_files(rho, Mtot_grams, outputpath=None):
    times_and_rate, E_tot_per_gram = rprocess_const_and_powerlaw()
    energydistributiondata = make_energydistribution_weightedbyrho(rho, E_tot_per_gram, Mtot_grams)

    write_energydistribution_file(energydistributiondata, outputfilepath=outputpath)
    write_energyrate_file(times_and_rate, outputfilepath=outputpath)


def plot_energy_rate(modelpath):
    times_and_rate, E_tot = at.inputmodel.energyinputfiles.rprocess_const_and_powerlaw()
    model, _, _ = at.inputmodel.get_modeldata(modelpath)
    Mtot_grams = model['cellmass_grams'].sum()
    plt.plot(times_and_rate['times'], np.array(times_and_rate['nuclear_heating_power'])*Mtot_grams,
             color='k', zorder=10)
