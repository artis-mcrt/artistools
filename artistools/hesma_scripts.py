#!/usr/bin/env python3

import artistools as at
# import artistools.spectra
# import artistools.lightcurve.writebollightcurvedata

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_hesma_spectrum(timeavg, axes):

    hesma_file = Path("/Users/ccollins/Downloads/hesma_files/M2a/hesma_specseq.dat")
    hesma_spec = pd.read_csv(hesma_file, comment="#", delim_whitespace=True, dtype=float)
    # print(hesma_spec)

    def match_closest_time(reftime):
        return str("{}".format(min([float(x) for x in hesma_spec.keys()[1:]], key=lambda x: abs(x - reftime))))

    closest_time = (match_closest_time(timeavg))
    closest_time = f'{closest_time:.2f}'
    print(closest_time)

    # Scale distance to 1 Mpc
    dist_mpc = 1e-5  # HESMA specta at 10 pc
    hesma_spec[closest_time] = hesma_spec[closest_time] * (1e-5) ** 2  # refspecditance Mpc / 1 Mpc ** 2

    for ax in axes:
        ax.plot(hesma_spec['0.00'], hesma_spec[closest_time], label='HESMA model')


def plothesmaresspec(fig, ax):

    # specfiles = ["/Users/ccollins/Downloads/hesma_files/M2a_i55/hesma_specseq_theta.dat"]
    specfiles = ["/Users/ccollins/Downloads/hesma_files/M2a/hesma_virtualspecseq_theta.dat"]

    for specfilename in specfiles:
        specdata = pd.read_csv(specfilename, delim_whitespace=True, header=None, dtype=float)

        # index_to_split = specdata.index[specdata.iloc[:, 1] == specdata.iloc[0, 1]]
        # res_specdata = []
        # for i, index_value in enumerate(index_to_split):
        #     if index_value != index_to_split[-1]:
        #         chunk = specdata.iloc[index_to_split[i]:index_to_split[i + 1], :]
        #     else:
        #         chunk = specdata.iloc[index_to_split[i]:, :]
        #     res_specdata.append(chunk)

        res_specdata = at.gather_res_data(specdata)

        column_names = res_specdata[0].iloc[0]
        column_names[0] = 'lambda'
        print(column_names)

        for i, res_spec in enumerate(res_specdata):
            res_specdata[i] = res_specdata[i].rename(columns=column_names).drop(res_specdata[i].index[0])

        ax.plot(res_specdata[0]['lambda'], res_specdata[0][11.7935] * (1e-5) ** 2, label="hesma 0")
        ax.plot(res_specdata[1]['lambda'], res_specdata[1][11.7935] * (1e-5) ** 2, label="hesma 1")
        ax.plot(res_specdata[2]['lambda'], res_specdata[2][11.7935] * (1e-5) ** 2, label="hesma 2")
        ax.plot(res_specdata[3]['lambda'], res_specdata[3][11.7935] * (1e-5) ** 2, label="hesma 3")
        ax.plot(res_specdata[4]['lambda'], res_specdata[4][11.7935] * (1e-5) ** 2, label="hesma 4")

    fig.legend()
    # plt.show()


def make_hesma_vspecfiles(modelpath):
    angles = [0, 1, 2, 3, 4]

    for angle in angles:
        vspecdata_all = at.spectra.get_specpol_data(angle=angle, modelpath=modelpath)
        vspecdata = vspecdata_all['I']

        timearray = vspecdata.columns.values[1:]
        vspecdata.sort_values(by='nu', ascending=False, inplace=True)
        vspecdata.eval('lambda_angstroms = 2.99792458e+18 / nu', inplace=True)
        for time in timearray:
            vspecdata[time] = vspecdata[time] * vspecdata['nu'] / vspecdata['lambda_angstroms']
            vspecdata[time] = vspecdata[time] * (1e5) ** 2  # Scale to 10 pc (1 Mpc/10 pc) ** 2

        vspecdata = vspecdata.set_index('lambda_angstroms').reset_index()
        vspecdata = vspecdata.drop(['nu'], axis=1)

        vspecdata = vspecdata.rename(columns={'lambda_angstroms': '0'})

        print(vspecdata)

        if angle == 0:
            vspecdata.to_csv(modelpath / 'hesma_virtualspecseq_theta.dat', sep=' ', index=False)  # create file
        else:
            # append to file
            vspecdata.to_csv(modelpath / 'hesma_virtualspecseq_theta.dat', mode='a', sep=' ', index=False)


def make_hesma_bol_lightcurve(modelpath, outpath, timemin, timemax):
    """UVOIR bolometric light curve (angle-averaged)"""

    lightcurvedataframe = at.lightcurve.writebollightcurvedata.get_bol_lc_from_lightcurveout(modelpath)
    print(lightcurvedataframe)
    lightcurvedataframe = lightcurvedataframe[lightcurvedataframe.time > timemin]
    lightcurvedataframe = lightcurvedataframe[lightcurvedataframe.time < timemax]

    modelname = at.get_model_name(modelpath)
    outfilename = f'doubledet_2021_{modelname}.dat'

    lightcurvedataframe.to_csv(outpath / outfilename, sep=' ', index=False, header=False)


def make_hesma_peakmag_dm15_dm40(pathtofiles, modelname, outpath):

    dm15filename = f"bolband_{modelname}_viewing_angle_data.txt"
    dm40filename = f"bolband_{modelname}_viewing_angle_data_deltam40.txt"

    dm15data = pd.read_csv(pathtofiles / dm15filename, delim_whitespace=True, header=None, names=['peakmag', 'risetime', 'dm15'], skiprows=1)
    dm40data = pd.read_csv(pathtofiles / dm40filename, delim_whitespace=True, header=None, names=['peakmag', 'risetime', 'dm40'], skiprows=1)

    outdata = {}
    outdata['peakmag'] = dm15data['peakmag']  # dm15 peak mag probably more accurate - shorter time window
    outdata['dm15'] = dm15data['dm15']
    outdata['dm40'] = dm40data['dm40']

    outdataframe = pd.DataFrame(outdata)
    outdataframe = outdataframe.round(decimals=4)
    outdataframe.to_csv(outpath / f"doubledet_2021_bol_dm_{modelname}.dat", sep=' ', index=False, header=True)


def read_hesma_peakmag_dm15_dm40(pathtofiles):
    data = []
    for filename in os.listdir(pathtofiles):
        print(filename)
        data.append(pd.read_csv(pathtofiles / filename, delim_whitespace=True))
    print(data[0])

    for df in data:
        print(df)
        plt.scatter(df['dm15'], df['peakmag'])
    plt.gca().invert_yaxis()
    plt.show()


# def main():
#     # pathtomodel = Path("/home/localadmin_ccollins/harddrive4TB/parameterstudy/")
#     # modelnames = ['M08_03', 'M08_05', 'M08_10', 'M09_03', 'M09_05', 'M09_10',
#     #               'M10_02_end55', 'M10_03', 'M10_05', 'M10_10', 'M11_05_1']
#     # outpath = Path("/home/localadmin_ccollins/harddrive4TB/parameterstudy/hesma_lc")
#     # timemin = 5
#     # timemax = 70
#     # for modelname in modelnames:
#     #     modelpath = pathtomodel / modelname
#     #     make_hesma_bol_lightcurve(modelpath, outpath, timemin, timemax)
#
#     # pathtofiles = Path("/home/localadmin_ccollins/harddrive4TB/parameterstudy/declinerate")
#     # read_hesma_peakmag_dm15_dm40(pathtofiles)
#
#
# if __name__ == '__main__':
#     main()
