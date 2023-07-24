import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

DAY = 86400
TWOHOVERCLIGHTSQUARED = 1.4745007e-47
HOVERKB = 4.799243681748932e-11
PARSEC = 3.086e18
c_cgs = const.c.to("cm/s").value
c_ang_s = const.c.to("angstrom/s").value

modelpath = Path()

xmin = 2500  # Angstroms
xmax = 30000
n_nu_bins = 500  # number of frequency bins
delta_lambda = xmax - xmin

nu_lower = const.c.to("angstrom/s").value / xmin
nu_upper = const.c.to("angstrom/s").value / xmax
delta_nu = nu_lower - nu_upper
# arr_nu_hz = np.linspace(nu_lower, nu_upper, num=n_nu_bins)
arr_nu_hz = np.linspace(nu_upper, nu_lower, num=n_nu_bins)
arr_min_nu_hz = arr_nu_hz[:-1]
arr_max_nu_hz = arr_nu_hz[1:]
arr_delta_nu_hz = arr_max_nu_hz - arr_min_nu_hz

arr_lambda = const.c.to("angstrom/s").value / arr_nu_hz

n_angle_bins = 100

types = {
    10: "TYPE_GAMMA",
    11: "TYPE_RPKT",
    20: "TYPE_NTLEPTON",
    32: "TYPE_ESCAPE",
}
type_ids = {v: k for k, v in types.items()}


def sample_planck(T, nu_max_r, nu_min_r):
    def planck(nu, T):
        return TWOHOVERCLIGHTSQUARED * pow(nu, 3) / (np.exp(HOVERKB * nu / T) - 1)

    nu_peak = 5.879e10 * T
    # if (nu_peak > nu_max_r or nu_peak < nu_min_r):
    #     print(f"[warning] sample_planck: intensity peaks outside frequency range:"
    #           f" T {T} nu peak {nu_peak:.2E} nu_max_r {nu_max_r:.2E} nu_min_r {nu_min_r:.2E}")
    B_peak = planck(nu_peak, T)

    i = 0
    while i < 100:
        i += 1
        zrand = random.uniform(0, 1)
        zrand2 = random.uniform(0, 1)
        nu = nu_min_r + zrand * (nu_max_r - nu_min_r)
        # print(nu, nu_peak, nu_min_r, nu_max_r)
        if zrand2 * B_peak <= planck(nu, T):
            return nu

    # print(f'failed to get nu. nu peak {nu_peak:.2E}')
    return 0.0


# with open(modelpath / 'specpol_res.out', 'r') as specpol_res_file:  # get timesteps
#     time_list = [float(x) for x in specpol_res_file.readline().split()]
#
# column_names = time_list[:int(len(time_list)/3)+1]
arr_tstart = at.get_timestep_times(modelpath, loc="start")
arr_tend = at.get_timestep_times(modelpath, loc="end")
column_names = np.append(arr_tstart, arr_tend[-1])
column_names = np.insert(column_names, 0, 0.0, axis=0)

timemin_seconds = column_names[1] * u.day.to("s")
timemax_seconds = arr_tend[-1] * u.day.to("s")

specpol_data_bb = {column_names[0]: arr_min_nu_hz}
packet_contribution_count = {}
for time in column_names[1:]:  # initialise empty arrays
    specpol_data_bb[time] = np.zeros_like(arr_min_nu_hz)
    packet_contribution_count[time] = 0

specpol_res_data_bb = [copy.deepcopy(specpol_data_bb) for _ in range(n_angle_bins)]
# need deep copy to make new empty array of same size


packetsfiles = at.packets.get_packetsfilepaths(modelpath)
nprocs = at.get_nprocs(modelpath)
# nprocs = 100
for npacketfile in range(nprocs):
    dfpackets = at.packets.readfile(packetsfiles[npacketfile])  # , type='TYPE_ESCAPE', escape_type='TYPE_RPKT')
    dfpackets = at.packets.bin_packet_directions(modelpath, dfpackets)
    dfpackets = dfpackets.query(f'type_id == {type_ids["TYPE_ESCAPE"]} and escape_type_id == {type_ids["TYPE_RPKT"]}')

    # print(max(dfpackets['t_arrive_d']))
    # print(dfpackets)

    for timestep, timedays in enumerate(arr_tstart):
        # print('ts', timestep, timedays, 'days')

        # get packets escaping within timestep
        timelow = column_names[timestep] * u.day.to("s")
        timehigh = arr_tend[timestep] * u.day.to("s")
        # timelow = float(arr_tstart[timestep])
        # timehigh = float(arr_tend[timestep])
        # print('ts', timestep, 'low', timelow, 'high', timehigh)

        dfpackets_timestep = dfpackets.query(
            "@timelow < escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs < @timehigh", inplace=False
        )
        # dfpackets_timestep = dfpackets.query('t_arrive_d > @timelow and t_arrive_d < @timehigh', inplace=False).copy()

        # if len(dfpackets_timestep) > 0:
        #     print('timestep:')
        #     print((dfpackets_timestep[['escape_time', 'escape_time_d', 't_arrive_d', 'em_TR']]))
        #     print('initial df:')
        #     print((dfpackets[['escape_time', 'escape_time_d', 't_arrive_d', 'em_TR']]))
        #     print('\n\n\n')
        #     # sys.exit(1)

        for _df_index, row in dfpackets_timestep.iterrows():
            TR = row["em_TR"]
            # if TR not in [100, 140000]:

            nu = sample_planck(TR, nu_lower, nu_upper)

            if nu > 0.0:
                angle = int(row["angle_bin"])
                if angle > 99:  # 100 is getting in there somehow??
                    continue
                e_rf = row["e_rf"]
                hist, _ = np.histogram([nu], bins=arr_nu_hz)  # get frequency bin - returns array with 1 in correct bin
                hist = hist * e_rf  # multiply by packet rf energy to get the energy in the right bin
                freq_bin_number = np.nonzero(hist)  # the frequency bin number is where hist is non zero

                # add to angle bin in this timestep
                specpol_res_data_bb[angle][timedays] += (
                    hist
                    / (timehigh - timelow)
                    / arr_delta_nu_hz[freq_bin_number[0][0]]
                    / 4.0e12
                    / np.pi
                    / PARSEC
                    / PARSEC
                    * n_angle_bins
                    / nprocs
                )
                # packet_contribution_count_res[angle][timedays] += 1

for angle in range(n_angle_bins):
    df = pd.DataFrame.from_dict(specpol_res_data_bb[angle])
    if angle == 0:
        df.to_csv(modelpath / "spec_res_bb.out", sep=" ", index=False)  # create file
    else:
        # append to file
        df.to_csv(modelpath / "spec_res_bb.out", mode="a", sep=" ", index=False)

print("Blackbody spectra written to spec_res_bb.out")
