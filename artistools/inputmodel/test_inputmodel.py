import hashlib
import shutil
import typing as t
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
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
                "abundances.txt": "1dffae45ff2abe72df3055305853e61ffe471913c5c8cc39cc6f67e38b5f5f45",
                "model.txt": "6a98aba2b541af5c24497362b053dbe7749bba74a49778f494b440a453c735fc",
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
                "abundances.txt": "1dffae45ff2abe72df3055305853e61ffe471913c5c8cc39cc6f67e38b5f5f45",
                "model.txt": "f9adee21771c18aa6e10c58c252f5f37e3892d3d6007685d915bc1b9cc961a74",
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
    lzdfmodel, modelmeta = at.inputmodel.get_empty_3d_model(ncoordgrid=10, vmax=1000, t_model_init_days=1)
    dfmodel = lzdfmodel.collect()

    # dfmodel[75000, "rho"] = 1
    # dfmodel[75001, "rho"] = 2
    # dfmodel[95200, "rho"] = 3
    # dfmodel[75001, "rho"] = 0.5
    rng = np.random.default_rng()

    # give a random rho to half of the cells
    dfmodel[rng.integers(0, dfmodel.height, dfmodel.height // 2), "rho"] = 10.0 * rng.random(
        dfmodel.height // 2, dtype=np.float32
    )

    dfelements = (
        at.get_elsymbols_df()
        .filter(pl.col("atomic_number").is_between(1, 110))
        .sort(by="atomic_number")
        .with_columns(
            elemcolname="X_" + pl.col("elsymbol"), mass_number_example=(pl.col("atomic_number") * 2).cast(pl.Int32)
        )
    )

    # for elemental abundance file, we must start from hydrogen and not skip any atomic numbers
    assert dfelements["elemcolname"].unique().len() == dfelements["atomic_number"].max()
    elcolnames = dfelements["elemcolname"].to_list()

    # this give us several isotopes for each element (doesn't matter if they are real or not)
    dfisocolnames = pl.Series(
        "isocolname",
        pl.concat(
            dfelements.select(pl.col("elemcolname") + (pl.col("mass_number_example") + i).cast(pl.Utf8))
            for i in range(5)
        ),
    )
    isocolnames = dfisocolnames.to_list()

    dfelements = (
        at.get_elsymbols_df()
        .filter(pl.col("atomic_number").is_between(1, 80))
        .sort(by="atomic_number")
        .with_columns(
            elemcolname="X_" + pl.col("elsymbol"), mass_number_example=(pl.col("atomic_number") * 2).cast(pl.Int32)
        )
    )

    # for elemental abundance file, we must start from hydrogen and not skip any atomic numbers
    assert dfelements["elemcolname"].unique().len() == dfelements["atomic_number"].max()
    elcolnames = dfelements["elemcolname"].to_list()

    # this give us several isotopes for each element (doesn't matter if they are real or not)
    dfisocolnames = pl.Series(
        "isocolname",
        pl.concat(
            dfelements.select(pl.col("elemcolname") + (pl.col("mass_number_example") + i).cast(pl.Utf8))
            for i in range(5)
        ),
    )
    isocolnames = dfisocolnames.to_list()

    # give random abundances to the cells with rho > 0
    dfmodel = dfmodel.with_columns([
        pl.Series(isocol, list(rng.random(dfmodel.height, dtype=np.float32)), dtype=pl.Float32)
        for isocol in isocolnames
    ])

    # abundances don't matter if rho is zero, so we'll set them to zero to match the resulting dataframe that will be loaded
    dfmodel = dfmodel.with_columns(
        pl.when(dfmodel["rho"] > 0).then(pl.col(col)).otherwise(0) for col in dfmodel.columns if col.startswith("X_")
    )

    # sum isotopic abundances to get elemental abundances
    dfelemabundances = dfmodel.select([
        "inputcellid",
        *[pl.sum_horizontal(cs.starts_with(elcol)).cast(pl.Float32).alias(elcol) for elcol in elcolnames],
    ])

    outpath = outputpath / "test_save_load_3d_model"
    outpath.mkdir(exist_ok=True, parents=True)
    at.inputmodel.save_modeldata(outpath=outpath, dfmodel=dfmodel, modelmeta=modelmeta)
    at.inputmodel.save_initelemabundances(outpath=outpath, dfelabundances=dfelemabundances)

    # first load will be from text, second from parquet
    for _ in (0, 1):
        dfmodel_loaded, modelmeta_loaded = at.inputmodel.get_modeldata_polars(modelpath=outpath)
        pltest.assert_frame_equal(
            dfmodel,
            dfmodel_loaded.collect(),
            check_column_order=False,
            check_dtypes=False,
            rtol=1e-4,
            atol=1e-4,
        )
        assert modelmeta == modelmeta_loaded

        dfelemabundances_loaded = at.inputmodel.get_initelemabundances_polars(outpath)
        pltest.assert_frame_equal(
            dfelemabundances,
            dfelemabundances_loaded.collect(),
            check_column_order=False,
            check_dtypes=False,
            rtol=1e-3,
            atol=1e-3,
        )


def lower_dim_and_check_mass_conservation(outputdimensions: int) -> None:
    dfmodel3d_pl_lazy, modelmeta_3d = at.inputmodel.get_empty_3d_model(ncoordgrid=50, vmax=100000, t_model_init_days=1)
    dfmodel3d_pl = dfmodel3d_pl_lazy.collect()

    # it's important that we don't fill cells in the cube corners, as they will be lost when reducing dimensions
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
    assert np.isclose(dfmodel_lowerd["mass_g"].sum(), dfmodel3d_pl["mass_g"].sum(), rtol=1e-3)

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
