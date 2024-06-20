import hashlib
import shutil
import typing as t
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_3d = at.get_config()["path_testdata"] / "testmodel_3d_10^3"
outputpath = at.get_config()["path_testoutput"]
testdatapath = at.get_config()["path_testdata"]


def test_describeinputmodel() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath, isotopes=True)


@pytest.mark.benchmark()
def test_describeinputmodel_3d() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath_3d, isotopes=True)


def test_get_modeldata_1d() -> None:
    for getheadersonly in (False, True):
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 800000000.0)
        assert modelmeta["dimensions"] == 1
        assert modelmeta["npts_model"] == 1

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, derived_cols=["mass_g"])
    assert np.isclose(dfmodel.mass_g.sum(), 1.416963e33)


@pytest.mark.benchmark()
def test_get_modeldata_3d() -> None:
    for getheadersonly in (False, True):
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 2892020000.0)
        assert modelmeta["dimensions"] == 3
        assert modelmeta["npts_model"] == 1000
        assert modelmeta["ncoordgridx"] == 10

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, derived_cols=["mass_g"])
    assert np.isclose(dfmodel.mass_g.sum(), 2.7861855e33)


def test_get_cell_angle() -> None:
    modeldata, _ = at.inputmodel.get_modeldata(
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
    _, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata_tuple(modelpath, get_elemabundances=True)
    assert np.isclose(t_model_init_days, 0.00115740740741, rtol=0.0001)
    assert np.isclose(vmax_cmps, 800000000.0, rtol=0.0001)


def verify_file_checksums(checksums_expected: dict, digest: str = "sha256", folder: Path | str = Path()) -> None:
    checksums_actual = {}

    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        m = hashlib.new(digest)
        with Path(fullpath).open("rb") as f:
            for chunk in f:
                m.update(chunk)

        checksums_actual[fullpath] = m.hexdigest()
        strpassfail = "pass" if checksums_actual[fullpath] == checksum_expected else "FAILED"
        print(f"{filename}: {strpassfail} if actual {checksums_actual[fullpath]} expected {checksum_expected}")

    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        assert (
            checksums_actual[fullpath] == checksum_expected
        ), f"{folder}/{filename} checksum mismatch. Expecting {checksum_expected} but calculated {checksums_actual[fullpath]}"


def test_makeartismodelfrom_sph_particles() -> None:
    gridfolderpath = outputpath / "kilonova"

    config_checksums_3d: list[dict[str, dict[str, t.Any]]] = [
        {
            "maptogridargs": {"ncoordgrid": 16},
            "maptogrid_sums": {
                "ejectapartanalysis.dat": "e8694a679515c54c2b4867122122263a375d9ffa144a77310873ea053bb5a8b4",
                "grid.dat": "b179427dc76e3b465d83fb303c866812fa9cb775114d1b8c45411dd36bf295b2",
                "gridcontributions.txt": "63e6331666c4928bdc6b7d0f59165e96d6555736243ea8998a779519052a425f",
            },
            "makeartismodel_sums": {
                "gridcontributions.txt": "6327d196b4800eedb18faee15097f76af352ecbaa9ee59055161b81378bd4af7",
                "abundances.txt": "b84fb2542b1872291e1f45385b43fad2e5249f7fccbe7e4cab59b9c3b6c63916",
                "model.txt": "c268277b78d9053b447396519c183b8f8ad38404b40ed4a820670987a4d2bba2",
            },
        },
        {
            "maptogridargs": {"ncoordgrid": 16, "shinglesetal23hbug": True},
            "maptogrid_sums": {
                "ejectapartanalysis.dat": "e8694a679515c54c2b4867122122263a375d9ffa144a77310873ea053bb5a8b4",
                "grid.dat": "ea930d0decca79d2e65ac1df1aaaa1eb427fdf45af965a623ed38240dce89954",
                "gridcontributions.txt": "a2c09b96d32608db2376f9df61980c2ad1423066b579fbbe744f07e536f2891e",
            },
            "makeartismodel_sums": {
                "gridcontributions.txt": "c06b4cbbe7f3bf423ed636afd63e3d8e30cc3ffa928d3275ffc3ce13f2e4dbef",
                "abundances.txt": "b84fb2542b1872291e1f45385b43fad2e5249f7fccbe7e4cab59b9c3b6c63916",
                "model.txt": "6bca370bf85e759b95707b5819d9acb717840ede8168f9d3d70007d74c8afc23",
            },
        },
    ]

    for tag, config in zip(["", "_shinglesetal23hbug"], config_checksums_3d, strict=False):
        shutil.copytree(
            testdatapath / "kilonova", gridfolderpath, dirs_exist_ok=True, ignore=shutil.ignore_patterns("trajectories")
        )

        at.inputmodel.maptogrid.main(
            argsraw=[], inputpath=gridfolderpath, outputpath=gridfolderpath, **config["maptogridargs"]
        )

        verify_file_checksums(
            config["maptogrid_sums"],
            digest="sha256",
            folder=gridfolderpath,
        )

        dfcontribs = {}
        for dimensions in (3, 2, 1, 0):
            outpath_kn = outputpath / f"kilonova_{dimensions:d}d{tag}"
            outpath_kn.mkdir(exist_ok=True, parents=True)

            shutil.copyfile(gridfolderpath / "gridcontributions.txt", outpath_kn / "gridcontributions.txt")

            at.inputmodel.modelfromhydro.main(
                argsraw=[],
                gridfolderpath=gridfolderpath,
                trajectoryroot=testdatapath / "kilonova" / "trajectories",
                outputpath=outpath_kn,
                dimensions=dimensions,
                targetmodeltime_days=0.1,
            )

            dfcontribs[dimensions] = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(outpath_kn)

            if dimensions == 3:
                verify_file_checksums(
                    config["makeartismodel_sums"],
                    digest="sha256",
                    folder=outpath_kn,
                )
                dfcontrib_source = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(gridfolderpath)

                pltest.assert_frame_equal(
                    dfcontrib_source,
                    dfcontribs[3]
                    .drop("frac_of_cellmass")
                    .rename({"frac_of_cellmass_includemissing": "frac_of_cellmass"}),
                    rtol=1e-4,
                    atol=1e-4,
                )
            else:
                dfmodel3lz, _ = at.inputmodel.get_modeldata_polars(
                    modelpath=outputpath / f"kilonova_{3:d}d", derived_cols=["mass_g"]
                )
                dfmodel3 = dfmodel3lz.collect()
                dfmodel_lowerdlz, _ = at.inputmodel.get_modeldata_polars(
                    modelpath=outputpath / f"kilonova_{dimensions:d}d", derived_cols=["mass_g"]
                )
                dfmodel_lowerd = dfmodel_lowerdlz.collect()

                # check that the total mass is conserved
                assert np.isclose(dfmodel_lowerd["mass_g"].sum(), dfmodel3["mass_g"].sum(), rtol=5e-2)
                assert np.isclose(dfmodel_lowerd["tracercount"].sum(), dfmodel3["tracercount"].sum(), rtol=1e-1)

                # check that the total mass of each species is conserved
                for col in dfmodel3.columns:
                    if col.startswith("X_"):
                        lowerd_mass = (dfmodel_lowerd["mass_g"] * dfmodel_lowerd[col]).sum()
                        model3_mass = (dfmodel3["mass_g"] * dfmodel3[col]).sum()
                        assert np.isclose(lowerd_mass, model3_mass, rtol=5e-2)


@pytest.mark.benchmark()
def test_makeartismodelfrom_fortrangriddat() -> None:
    gridfolderpath = testdatapath / "kilonova"
    outpath_kn = outputpath / "kilonova"
    at.inputmodel.modelfromhydro.main(
        argsraw=[],
        gridfolderpath=gridfolderpath,
        outputpath=outpath_kn,
        dimensions=3,
        targetmodeltime_days=0.1,
    )


def test_make1dmodelfromcone() -> None:
    at.inputmodel.slice1dfromconein3dmodel.main(argsraw=[], modelpath=[modelpath_3d], outputpath=outputpath, axis="-z")


def test_makemodel_botyanski2017() -> None:
    outpath = outputpath / "test_makemodel_botyanski2017"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.botyanski2017.main(argsraw=[], outputpath=outpath)


def test_makemodel() -> None:
    outpath = outputpath / "test_makemodel"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, outputpath=outpath)


def test_makemodel_energyfiles() -> None:
    outpath = outputpath / "test_makemodel_energyfiles"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, makeenergyinputfiles=True, outputpath=outpath)


def test_maketardismodel() -> None:
    outpath = outputpath / "test_maketardismodel"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.to_tardis.main(argsraw=[], inputpath=modelpath, outputpath=outpath)


def test_make_empty_abundance_file() -> None:
    outpath = outputpath / "test_make_empty_abundance_file"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.save_empty_abundance_file(npts_model=50, outputfilepath=outpath)


def test_opacity_by_Ye_file() -> None:
    griddata = {
        "cellYe": [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5],
        "rho": [0, 99, 99, 99, 99, 99, 99, 99],
        "inputcellid": range(1, 9),
    }
    at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, griddata=griddata)


def test_plotdensity() -> None:
    at.inputmodel.plotdensity.main(argsraw=[], modelpath=[modelpath], outputpath=outputpath)


@pytest.mark.benchmark()
def test_plotinitialcomposition() -> None:
    at.inputmodel.plotinitialcomposition.main(
        argsraw=["-modelpath", str(modelpath_3d), "-o", str(outputpath), "rho", "Fe"]
    )


@pytest.mark.benchmark()
def test_save_load_3d_model() -> None:
    lzdfmodel, modelmeta = at.inputmodel.get_empty_3d_model(ncoordgrid=50, vmax=1000, t_model_init_days=1)
    dfmodel = lzdfmodel.collect()

    dfmodel[75000, "rho"] = 1
    dfmodel[75001, "rho"] = 2
    dfmodel[95200, "rho"] = 3
    dfmodel[75001, "rho"] = 0.5
    rng = np.random.default_rng()

    # give a random rho to half of the cells
    dfmodel[rng.integers(0, dfmodel.height, dfmodel.height // 2), "rho"] = 10.0 * rng.random(dfmodel.height // 2)

    # give random abundances to the cells with rho > 0
    dfmodel = dfmodel.with_columns([
        pl.Series(iso, rng.random(dfmodel.height), dtype=pl.Float32)
        for iso in (
            "X_La154 X_La155 X_La156 X_La157 X_La158 X_La159 X_La160 X_La161 X_La162 X_La163 X_La164 X_La165 X_La166 X_La167 X_La168 X_Ce130 X_Ce132 X_Ce133 X_Ce134 X_Ce135 X_Ce136 X_Ce137 X_Ce138 "
            "X_Ce139 X_Ce140 X_Ce141 X_Ce142 X_Ce143 X_Ce144 X_Ce145 X_Ce146 X_Ce147 X_Ce148 X_Ce149 X_Ce150 X_Ce151 X_Ce152 X_Ce153 X_Ce154 X_Ce155 X_Ce156 X_Ce157 X_Ce158 X_Ce159 X_Ce160 X_Ce161 "
            "X_Ce162 X_Ce163 X_Ce164 X_Ce165 X_Ce166 X_Ce167 X_Ce168 X_Ce169 X_Pr134 X_Pr135 X_Pr136 X_Pr137 X_Pr138 X_Pr139 X_Pr140 X_Pr141 X_Pr142 X_Pr143 X_Pr144 X_Pr145 X_Pr146 X_Pr147 X_Pr148 "
            "X_Pr149 X_Pr150 X_Pr151 X_Pr152 X_Pr153 X_Pr154 X_Pr155 X_Pr156 X_Pr157 X_Pr158 X_Pr159 X_Pr160 X_Pr161 X_Pr162 X_Pr163 X_Pr164 X_Pr165 X_Pr166 X_Pr167 X_Pr168 X_Pr169 X_Nd128 X_Nd136 "
            "X_Nd137 X_Nd138 X_Nd139 X_Nd140 X_Nd141 X_Nd142 X_Nd143 X_Nd144 X_Nd145 X_Nd146 X_Nd147 X_Nd148 X_Nd149 X_Nd150 X_Nd151 X_Nd152 X_Nd153 X_Nd154 X_Nd155 X_Nd156 X_Nd157 X_Nd158 X_Nd159 "
            "X_Nd160 X_Nd161 X_Nd162 X_Nd163 X_Nd164 X_Nd165 X_Nd166 X_Nd167 X_Nd168 X_Nd169 X_Pm137 X_Pm141 X_Pm142 X_Pm143 X_Pm144 X_Pm145 X_Pm147 X_Pm148 X_Pm149 X_Pm150 X_Pm151 X_Pm152 X_Pm153 "
            "X_Pm154 X_Pm155 X_Pm156 X_Pm157 X_Pm158 X_Pm159 X_Pm160 X_Pm161 X_Pm162 X_Pm163 X_Pm164 X_Pm165 X_Pm166 X_Pm167 X_Pm168 X_Pm169 X_Sm140 X_Sm142 X_Sm144 X_Sm145 X_Sm146 X_Sm147 X_Sm148 "
            "X_Sm149 X_Sm150 X_Sm151 X_Sm152 X_Sm153 X_Sm154 X_Sm155 X_Sm156 X_Sm157 X_Sm158 X_Sm159 X_Sm160 X_Sm161 X_Sm162 X_Sm163 X_Sm164 X_Sm165 X_Sm166 X_Sm167 X_Sm168 X_Sm169 X_Eu145 X_Eu146 "
            "X_Eu147 X_Eu148 X_Eu149 X_Eu151 X_Eu152 X_Eu153 X_Eu154 X_Eu155 X_Eu156 X_Eu157 X_Eu158 X_Eu159 X_Eu160 X_Eu161 X_Eu162 X_Eu163 X_Eu164 X_Eu165 X_Eu166 X_Eu167 X_Eu168 X_Eu169 X_Gd145 "
            "X_Gd146 X_Gd147 X_Gd148 X_Gd149 X_Gd150 X_Gd152 X_Gd153 X_Gd154 X_Gd155 X_Gd156 X_Gd157 X_Gd158 X_Gd159 X_Gd160 X_Gd161 X_Gd162 X_Gd163 X_Gd164 X_Gd165 X_Gd166 X_Gd167 X_Gd168 X_Gd169 "
            "X_Gd170 X_Tb147 X_Tb148 X_Tb151 X_Tb152 X_Tb153 X_Tb154 X_Tb155 X_Tb156 X_Tb157 X_Tb158 X_Tb159 X_Tb160 X_Tb161 X_Tb162 X_Tb163 X_Tb164 X_Tb165 X_Tb166 X_Tb167 X_Tb168 X_Tb169 X_Tb170 "
            "X_Tb171 X_Dy160 X_Dy161 X_Dy162 X_Dy163 X_Dy164 X_Dy165 X_Dy166 X_Dy167 X_Dy168 X_Dy169 X_Dy170 X_Dy171 X_Dy172 X_Dy173 X_Dy174 X_Ho165 X_Ho166 X_Ho167 X_Ho168 X_Ho169 X_Ho170 X_Ho171 "
            "X_Ho172 X_Ho173 X_Ho174 X_Ho175 X_Er166 X_Er167 X_Er168 X_Er169 X_Er170 X_Er171 X_Er172 X_Er173 X_Er174 X_Er175 X_Er176 X_Er177 X_Tm169 X_Tm170 X_Tm171 X_Tm172 X_Tm173 X_Tm174 X_Tm175 "
            "X_Tm176 X_Tm177 X_Tm178 X_Tm179 X_Yb170 X_Yb171 X_Yb172 X_Yb173 X_Yb174 X_Yb175 X_Yb176 X_Yb177 X_Yb178 X_Yb179 X_Yb180 X_Yb181 X_Yb182 X_Lu175 X_Lu176 X_Lu177 X_Lu178 X_Lu179 X_Lu180 "
            "X_Lu181 X_Lu182 X_Lu183 X_Hf168 X_Hf170 X_Hf174 X_Hf176 X_Hf177 X_Hf178 X_Hf179 X_Hf180 X_Hf181 X_Hf182 X_Hf183 X_Hf184 X_Hf185 X_Hf186 X_Hf187 X_Ta174 X_Ta181 X_Ta182 X_Ta183 X_Ta184 "
            "X_Ta185 X_Ta186 X_Ta187 X_Ta188 X_Ta189 X_W174 X_W176 X_W178 X_W180 X_W182 X_W183 X_W184 X_W185 X_W186 X_W187 X_W188 X_W189 X_W190 X_W191 X_W192 X_Re185 X_Re186 X_Re187 X_Re188 X_Re189 "
            "X_Re190 X_Re191 X_Re192 X_Re193 X_Re194 X_Re195 X_Os180 X_Os181 X_Os182 X_Os186 X_Os187 X_Os188 X_Os189 X_Os190 X_Os191 X_Os192 X_Os193 X_Os194 X_Os195 X_Os196 X_Os197 X_Os198 X_Os199".split()
        )
    ])

    # abundances don't matter if rho is zero, but we'll set them to zero to match the resulting dataframe
    dfmodel = dfmodel.with_columns(
        pl.when(dfmodel["rho"] > 0).then(pl.col(col)).otherwise(0) for col in dfmodel.columns if col.startswith("X_")
    )

    outpath = outputpath / "test_save_load_3d_model"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.save_modeldata(outpath=outpath, dfmodel=dfmodel, modelmeta=modelmeta)
    dfmodel2, modelmeta2 = at.inputmodel.get_modeldata_polars(modelpath=outpath)
    pltest.assert_frame_equal(
        dfmodel,
        dfmodel2.collect(),
        check_column_order=False,
        check_dtypes=False,
        rtol=1e-4,
        atol=1e-4,
    )
    assert modelmeta == modelmeta2

    # next load will use the parquet file
    dfmodel3, modelmeta3 = at.inputmodel.get_modeldata_polars(modelpath=outpath)
    pltest.assert_frame_equal(
        dfmodel,
        dfmodel3.collect(),
        check_column_order=False,
        check_dtypes=False,
        rtol=1e-4,
        atol=1e-4,
    )
    assert modelmeta == modelmeta3


def lower_dim_and_check_mass_conservation(outputdimensions: int) -> None:
    dfmodel3d_pl_lazy, modelmeta_3d = at.inputmodel.get_empty_3d_model(ncoordgrid=50, vmax=100000, t_model_init_days=1)
    dfmodel3d_pl = dfmodel3d_pl_lazy.collect()
    mgi1 = 26 * 26 * 26 + 26 * 26 + 26
    dfmodel3d_pl[mgi1, "rho"] = 2
    dfmodel3d_pl[mgi1, "X_Ni56"] = 0.5
    mgi2 = 25 * 25 * 25 + 25 * 25 + 25
    dfmodel3d_pl[mgi2, "rho"] = 1
    dfmodel3d_pl[mgi1, "X_Ni56"] = 0.75

    dfmodel3d_pl = at.inputmodel.add_derived_cols_to_modeldata(
        dfmodel=dfmodel3d_pl, modelmeta=modelmeta_3d, derived_cols=["mass_g"]
    ).collect()

    outpath = outputpath / f"test_dimension_reduce_3d_{outputdimensions:d}d"
    outpath.mkdir(exist_ok=True, parents=True)
    (
        dfmodel_lowerd,
        _,
        _,
        modelmeta_lowerd,
    ) = at.inputmodel.dimension_reduce_model(
        dfmodel=dfmodel3d_pl, modelmeta=modelmeta_3d, outputdimensions=outputdimensions
    )

    at.inputmodel.save_modeldata(outpath=outpath, dfmodel=dfmodel_lowerd, modelmeta=modelmeta_lowerd)

    dfmodel_lowerd_lz, modelmeta_lowerd = at.inputmodel.get_modeldata_polars(modelpath=outpath, derived_cols=["mass_g"])
    dfmodel_lowerd = dfmodel_lowerd_lz.collect()

    # check that the total mass is conserved
    assert np.isclose(dfmodel_lowerd["mass_g"].sum(), dfmodel3d_pl["mass_g"].sum())

    # check that the total mass of each species is conserved
    for col in dfmodel3d_pl.columns:
        if col.startswith("X_"):
            assert np.isclose(
                (dfmodel_lowerd["mass_g"] * dfmodel_lowerd[col]).sum(),
                (dfmodel3d_pl["mass_g"] * dfmodel3d_pl[col]).sum(),
            )


@pytest.mark.benchmark()
def test_dimension_reduce_3d_2d() -> None:
    lower_dim_and_check_mass_conservation(outputdimensions=2)


@pytest.mark.benchmark()
def test_dimension_reduce_3d_1d() -> None:
    lower_dim_and_check_mass_conservation(outputdimensions=1)


@pytest.mark.benchmark()
def test_dimension_reduce_3d_0d() -> None:
    lower_dim_and_check_mass_conservation(outputdimensions=0)
