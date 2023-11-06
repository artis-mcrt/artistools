import hashlib
import shutil
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
modelpath_3d = at.get_config()["path_testartismodel"].parent / "testmodel_3d_10^3"
outputpath = at.get_config()["path_testoutput"]
testdatapath = at.get_config()["path_testdata"]


def clear_modelfiles() -> None:
    (outputpath / "model.txt").unlink(missing_ok=True)
    (outputpath / "model.parquet").unlink(missing_ok=True)
    (outputpath / "abundances.txt").unlink(missing_ok=True)
    (outputpath / "abundances.parquet").unlink(missing_ok=True)


def test_describeinputmodel() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath, get_elemabundances=True)


def test_describeinputmodel_3d() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath_3d, get_elemabundances=True)


def test_get_modeldata_1d() -> None:
    for getheadersonly in [False, True]:
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 800000000.0)
        assert modelmeta["dimensions"] == 1
        assert modelmeta["npts_model"] == 1

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, derived_cols=["mass_g"])
    assert np.isclose(dfmodel.mass_g.sum(), 1.416963e33)


def test_get_modeldata_3d() -> None:
    for getheadersonly in [False, True]:
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 2892020000.0)
        assert modelmeta["dimensions"] == 3
        assert modelmeta["npts_model"] == 1000
        assert modelmeta["ncoordgridx"] == 10

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, derived_cols=["mass_g"])
    assert np.isclose(dfmodel.mass_g.sum(), 2.7861855e33)


def test_get_cell_angle() -> None:
    modeldata, modelmeta = at.inputmodel.get_modeldata(
        modelpath=modelpath_3d, derived_cols=["pos_x_mid", "pos_y_mid", "pos_z_mid"]
    )
    at.inputmodel.inputmodel_misc.get_cell_angle(modeldata, modelpath=modelpath_3d)
    assert "cos_bin" in modeldata


def test_downscale_3dmodel() -> None:
    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, get_elemabundances=True, derived_cols=["mass_g"])
    modelpath_3d_small = at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(
        modelpath_3d, outputgridsize=2, outputfolder=outputpath
    )
    dfmodel_small, modelmeta_small = at.get_modeldata(
        modelpath_3d_small, get_elemabundances=True, derived_cols=["mass_g"]
    )
    assert np.isclose(dfmodel["mass_g"].sum(), dfmodel_small["mass_g"].sum())
    assert np.isclose(modelmeta["vmax_cmps"], modelmeta_small["vmax_cmps"])
    assert np.isclose(modelmeta["t_model_init_days"], modelmeta_small["t_model_init_days"])

    abundcols = (x for x in dfmodel.columns if x.startswith("X_"))
    for abundcol in abundcols:
        assert np.isclose(
            (dfmodel[abundcol] * dfmodel["mass_g"]).sum(),
            (dfmodel_small[abundcol] * dfmodel_small["mass_g"]).sum(),
        )


def test_get_modeldata_tuple() -> None:
    dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata_tuple(modelpath, get_elemabundances=True)
    assert np.isclose(t_model_init_days, 0.00115740740741, rtol=0.0001)
    assert np.isclose(vmax_cmps, 800000000.0, rtol=0.0001)


def verify_file_checksums(
    checksums_expected: dict[Path | str, str], digest: str = "sha256", folder: Path | str = Path()
) -> None:
    checksums_actual = {}

    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        m = hashlib.new(digest)
        with Path(fullpath).open("rb") as f:
            for chunk in f:
                m.update(chunk)

        checksums_actual[fullpath] = str(m.hexdigest())
        print(f"{filename}: {checksums_actual[fullpath]} expected {checksum_expected}")

    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        assert (
            checksums_actual[fullpath] == checksum_expected
        ), f"{filename} checksum mismatch. Expecting {checksum_expected} but calculated {checksums_actual[fullpath]}"


def test_maptogrid() -> None:
    outpath_kn = outputpath / "kilonova"
    shutil.copytree(
        testdatapath / "kilonova", outpath_kn, dirs_exist_ok=True, ignore=shutil.ignore_patterns("trajectories")
    )
    at.inputmodel.maptogrid.main(argsraw=[], inputpath=outpath_kn, outputpath=outpath_kn, ncoordgrid=16)

    verify_file_checksums(
        {
            "ejectapartanalysis.dat": "e8694a679515c54c2b4867122122263a375d9ffa144a77310873ea053bb5a8b4",
            "grid.dat": "ea930d0decca79d2e65ac1df1aaaa1eb427fdf45af965a623ed38240dce89954",
            "gridcontributions.txt": "a2c09b96d32608db2376f9df61980c2ad1423066b579fbbe744f07e536f2891e",
        },
        digest="sha256",
        folder=outpath_kn,
    )


def test_makeartismodelfromparticlegridmap() -> None:
    outpath_kn = outputpath / "kilonova"
    at.inputmodel.modelfromhydro.main(
        argsraw=[],
        gridfolderpath=outpath_kn,
        trajectoryroot=testdatapath / "kilonova" / "trajectories",
        outputpath=outpath_kn,
        dimensions=3,
        targetmodeltime_days=0.1,
    )

    verify_file_checksums(
        {
            "abundances.txt": "3e7ad41548eedcc3b3a042208fd6ad6d7b6dd35c474783dc2abbbc5036f306aa",
            "model.txt": "7a3eee92f9653eb478a01080d16b711773031bedd38a90ec167c7fda98c15ef9",
            "gridcontributions.txt": "12f006c43c0c8d1f84c3927b3c80959c1b2cecc01598be92c2f24a130892bc60",
        },
        digest="sha256",
        folder=outpath_kn,
    )


def test_make1dmodelfromcone() -> None:
    at.inputmodel.slice1dfromconein3dmodel.main(argsraw=[], modelpath=[modelpath_3d], outputpath=outputpath)


def test_makemodel_botyanski2017() -> None:
    clear_modelfiles()
    at.inputmodel.botyanski2017.main(argsraw=[], outputpath=outputpath)


def test_makemodel() -> None:
    clear_modelfiles()
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, outputpath=outputpath)


def test_makemodel_energyfiles() -> None:
    clear_modelfiles()
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=modelpath, makeenergyinputfiles=True, modeldim=1, outputpath=outputpath
    )


def test_maketardismodel() -> None:
    clear_modelfiles()
    at.inputmodel.to_tardis.main(argsraw=[], inputpath=modelpath, outputpath=outputpath)


def test_make_empty_abundance_file() -> None:
    clear_modelfiles()
    at.inputmodel.save_empty_abundance_file(ngrid=50, outputfilepath=outputpath)


def test_opacity_by_Ye_file() -> None:
    griddata = {
        "cellYe": [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5],
        "rho": [0, 99, 99, 99, 99, 99, 99, 99],
        "inputcellid": range(1, 9),
    }
    at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, griddata=griddata)


def test_plotdensity() -> None:
    at.inputmodel.plotdensity.main(argsraw=[], modelpath=[modelpath], outputpath=outputpath)


def test_save_load_3d_model() -> None:
    clear_modelfiles()
    dfmodel_pl, modelmeta = at.inputmodel.get_empty_3d_model(ncoordgrid=50, vmax=1000, t_model_init_days=1)
    dfmodel = dfmodel_pl.collect().to_pandas(use_pyarrow_extension_array=True)
    dfmodel.loc[75000, ["rho"]] = 1
    dfmodel.loc[75001, ["rho"]] = 2
    dfmodel.loc[95200, ["rho"]] = 3
    dfmodel.loc[75001, ["rho"]] = 0.5

    at.inputmodel.save_modeldata(outpath=outputpath, dfmodel=dfmodel, modelmeta=modelmeta)
    dfmodel2, modelmeta2 = at.inputmodel.get_modeldata(modelpath=outputpath)
    assert dfmodel.equals(dfmodel2.drop("modelgridindex", axis=1))
    assert modelmeta == modelmeta2

    # next load will use the parquet file
    dfmodel3, modelmeta3 = at.inputmodel.get_modeldata(modelpath=outputpath)
    assert dfmodel.equals(dfmodel3.drop("modelgridindex", axis=1))
    assert modelmeta == modelmeta3


def test_dimension_reduce_3d_model() -> None:
    clear_modelfiles()
    dfmodel3d_pl_lazy, modelmeta_3d = at.inputmodel.get_empty_3d_model(ncoordgrid=50, vmax=1000, t_model_init_days=1)
    dfmodel3d_pl = dfmodel3d_pl_lazy.collect()
    mgi1 = 26 * 26 * 26 + 26 * 26 + 26
    dfmodel3d_pl[mgi1, "rho"] = 2
    dfmodel3d_pl[mgi1, "X_Ni56"] = 0.5
    mgi2 = 25 * 25 * 25 + 25 * 25 + 25
    dfmodel3d_pl[mgi2, "rho"] = 1
    dfmodel3d_pl[mgi1, "X_Ni56"] = 0.75
    dfmodel3d = dfmodel3d_pl.to_pandas(use_pyarrow_extension_array=True)
    dfmodel3d = (
        at.inputmodel.add_derived_cols_to_modeldata(
            dfmodel=pl.DataFrame(dfmodel3d), modelmeta=modelmeta_3d, derived_cols=["mass_g"]
        )
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
    )
    for outputdimensions in [1, 2]:
        (
            dfmodel_lowerd,
            dfabundances_lowerd,
            dfgridcontributions_lowerd,
            modelmeta_lowerd,
        ) = at.inputmodel.dimension_reduce_3d_model(
            dfmodel=dfmodel3d, modelmeta=modelmeta_3d, outputdimensions=outputdimensions
        )
        dfmodel_lowerd = (
            at.inputmodel.add_derived_cols_to_modeldata(
                dfmodel=pl.DataFrame(dfmodel_lowerd), modelmeta=modelmeta_lowerd, derived_cols=["mass_g"]
            )
            .collect()
            .to_pandas(use_pyarrow_extension_array=True)
        )

        # check that the total mass is conserved
        assert np.isclose(dfmodel_lowerd["mass_g"].sum(), dfmodel3d["mass_g"].sum())

        # check that the total mass of each species is conserved
        for col in dfmodel3d.columns:
            if col.startswith("X_"):
                assert np.isclose(
                    (dfmodel_lowerd["mass_g"] * dfmodel_lowerd[col]).sum(),
                    (dfmodel3d["mass_g"] * dfmodel3d[col]).sum(),
                )
