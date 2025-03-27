from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import artistools as at


def plot_hesma_spectrum(timeavg, axes) -> None:
    hesma_file = Path("/Users/ccollins/Downloads/hesma_files/M2a/hesma_specseq.dat")
    hesma_spec = pd.read_csv(hesma_file, comment="#", sep=r"\s+", dtype=float)
    # print(hesma_spec)

    def match_closest_time(reftime):
        return str(min((float(x) for x in hesma_spec.keys()[1:]), key=lambda x: abs(x - reftime)))

    closest_time = match_closest_time(timeavg)
    closest_time = f"{closest_time:.2f}"
    print(closest_time)

    # Scale distance to 1 Mpc
    dist_mpc = 1e-5  # HESMA specta at 10 pc
    hesma_spec[closest_time] *= dist_mpc**2  # refspecditance Mpc / 1 Mpc ** 2

    for ax in axes:
        ax.plot(hesma_spec["0.00"], hesma_spec[closest_time], label="HESMA model")


def plothesmaresspec(fig, ax) -> None:
    # specfiles = ["/Users/ccollins/Downloads/hesma_files/M2a_i55/hesma_specseq_theta.dat"]
    specfiles = ["/Users/ccollins/Downloads/hesma_files/M2a/hesma_virtualspecseq_theta.dat"]

    for specfilename in specfiles:
        specdata = pd.read_csv(specfilename, sep=r"\s+", header=None, dtype=float)

        # index_to_split = specdata.index[specdata.iloc[:, 1] == specdata.iloc[0, 1]]
        # res_specdata = []
        # for i, index_value in enumerate(index_to_split):
        #     if index_value != index_to_split[-1]:
        #         chunk = specdata.iloc[index_to_split[i]:index_to_split[i + 1], :]
        #     else:
        #         chunk = specdata.iloc[index_to_split[i]:, :]
        #     res_specdata.append(chunk)

        res_specdata = {
            dirbin: pldf.to_pandas(use_pyarrow_extension_array=True)
            for dirbin, pldf in at.split_multitable_dataframe(specdata).items()
        }

        new_column_names = res_specdata[0].iloc[0]
        new_column_names[0] = "lambda"
        print(new_column_names)

        for i in range(len(res_specdata)):
            res_specdata[i] = (
                res_specdata[i]
                .rename(columns=dict(zip(res_specdata[i].columns, new_column_names, strict=True)))
                .drop(res_specdata[i].index[0])
            )

        ax.plot(res_specdata[0]["lambda"], res_specdata[0][11.7935] * (1e-5) ** 2, label="hesma 0")
        ax.plot(res_specdata[1]["lambda"], res_specdata[1][11.7935] * (1e-5) ** 2, label="hesma 1")
        ax.plot(res_specdata[2]["lambda"], res_specdata[2][11.7935] * (1e-5) ** 2, label="hesma 2")
        ax.plot(res_specdata[3]["lambda"], res_specdata[3][11.7935] * (1e-5) ** 2, label="hesma 3")
        ax.plot(res_specdata[4]["lambda"], res_specdata[4][11.7935] * (1e-5) ** 2, label="hesma 4")

    fig.legend()
    # plt.show()


def make_hesma_vspecfiles(modelpath: Path, outpath: Path | None = None) -> None:
    if not outpath:
        outpath = modelpath
    modelname = at.get_model_name(modelpath)
    angles = [0, 1, 2, 3, 4]
    vpkt_config = at.get_vpkt_config(modelpath)
    angle_names = []

    for angle in angles:
        angle_names.append(rf"cos(theta) = {vpkt_config['cos_theta'][angle]}")
        print(rf"cos(theta) = {vpkt_config['cos_theta'][angle]}")
        vspecdata = at.spectra.get_specpol_data(angle=angle, modelpath=modelpath)["I"].to_pandas(
            use_pyarrow_extension_array=True
        )

        timearray = vspecdata.columns.to_numpy()[1:]
        vspecdata = vspecdata.sort_values(by="nu", ascending=False)
        vspecdata["lambda_angstroms"] = 2.99792458e18 / vspecdata["nu"]
        for time in timearray:
            vspecdata[time] = vspecdata[time] * vspecdata["nu"] / vspecdata["lambda_angstroms"]
            vspecdata[time] *= 100000.0**2  # Scale to 10 pc (1 Mpc/10 pc) ** 2

        vspecdata = vspecdata.set_index("lambda_angstroms").reset_index()
        vspecdata = vspecdata.drop(["nu"], axis=1)

        vspecdata = vspecdata.rename(columns={"lambda_angstroms": "0"})

        outfilename = f"{modelname}_vspec_res.dat"
        if angle == 0:
            vspecdata.to_csv(outpath / outfilename, sep=" ", index=False)  # create file
        else:
            # append to file
            vspecdata.to_csv(outpath / outfilename, mode="a", sep=" ", index=False)

    with (outpath / outfilename).open("r+") as f:  # add comment to start of file
        content = f.read()
        f.seek(0, 0)
        f.write(
            f"# File contains spectra at observer angles {angle_names} for Model {modelname}.\n# A header line"
            " containing spectra time is repeated at the beginning of each observer angle. Column 0 gives wavelength."
            " \n# Spectra are at a distance of 10 pc."
            "\n" + content
        )


def make_hesma_bol_lightcurve(modelpath: Path, outpath: Path, timemin: float, timemax: float) -> None:  # noqa: ARG001
    """UVOIR bolometric light curve (angle-averaged)."""
    lightcurvedataframe = at.lightcurve.get_bol_lc_from_lightcurveout(modelpath)
    print(lightcurvedataframe)
    lightcurvedataframe = lightcurvedataframe.query("time > @timemin and time < @timemax")

    modelname = at.get_model_name(modelpath)
    outfilename = f"doubledet_2021_{modelname}.dat"

    lightcurvedataframe.to_csv(outpath / outfilename, sep=" ", index=False, header=False)


def make_hesma_peakmag_dm15_dm40(
    band: str, pathtofiles: Path, modelname: str, outpath: Path, dm40: bool = False
) -> None:
    dm15filename = f"{band}band_{modelname}_viewing_angle_data.txt"
    dm15data = pd.read_csv(
        pathtofiles / dm15filename, sep=r"\s+", header=None, names=["peakmag", "risetime", "dm15"], skiprows=1
    )

    if dm40:
        dm40filename = f"{band}band_{modelname}_viewing_angle_data_deltam40.txt"
        dm40data = pd.read_csv(
            pathtofiles / dm40filename, sep=r"\s+", header=None, names=["peakmag", "risetime", "dm40"], skiprows=1
        )

    outdata = {
        "peakmag": dm15data["peakmag"],  # dm15 peak mag probably more accurate - shorter time window
        "dm15": dm15data["dm15"],
        "angle_bin": np.arange(0, 100),
    }
    if dm40:
        outdata["dm40"] = dm40data["dm40"]

    outdataframe = pd.DataFrame(outdata).round(decimals=4)
    outdataframe.to_csv(outpath / f"{modelname}_width-luminosity.dat", sep=" ", index=False, header=True)


def read_hesma_peakmag_dm15_dm40(pathtofiles) -> None:
    data = []
    for filepath in Path(pathtofiles).iterdir():
        print(filepath)
        data.append(pd.read_csv(filepath, sep=r"\s+"))
    print(data[0])

    for df in data:
        print(df)
        plt.scatter(df["dm15"], df["peakmag"])
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
