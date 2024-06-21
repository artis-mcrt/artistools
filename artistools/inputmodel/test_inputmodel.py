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
            "X_Re190 X_Re191 X_Re192 X_Re193 X_Re194 X_Re195 X_Os180 X_Os181 X_Os182 X_Os186 X_Os187 X_Os188 X_Os189 X_Os190 X_Os191 X_Os192 X_Os193 X_Os194 X_Os195 X_Os196 X_Os197 X_Os198 X_Os199 "
            "X_Fr232 X_Fr233 X_Fr234 X_Fr235 X_Fr236 X_Fr237 X_Fr238 X_Fr239 X_Fr240 X_Fr241 X_Fr242 X_Fr243 X_Fr244 X_Fr245 X_Fr246 X_Fr247 X_Fr248 X_Fr249 X_Fr250 X_Fr251 X_Fr252 X_Fr253 X_Fr254 "
            "X_Fr255 X_Fr256 X_Fr257 X_Fr258 X_Fr259 X_Ra207 X_Ra208 X_Ra209 X_Ra210 X_Ra211 X_Ra212 X_Ra213 X_Ra214 X_Ra221 X_Ra222 X_Ra223 X_Ra224 X_Ra225 X_Ra226 X_Ra227 X_Ra228 X_Ra229 X_Ra230 "
            "X_Ra231 X_Ra232 X_Ra233 X_Ra234 X_Ra235 X_Ra236 X_Ra237 X_Ra238 X_Ra239 X_Ra240 X_Ra241 X_Ra242 X_Ra243 X_Ra244 X_Ra245 X_Ra246 X_Ra247 X_Ra248 X_Ra249 X_Ra250 X_Ra251 X_Ra252 X_Ra253 "
            "X_Ra254 X_Ra255 X_Ra256 X_Ra257 X_Ra258 X_Ra259 X_Ra260 X_Ac224 X_Ac225 X_Ac226 X_Ac227 X_Ac228 X_Ac229 X_Ac230 X_Ac231 X_Ac232 X_Ac233 X_Ac234 X_Ac235 X_Ac236 X_Ac237 X_Ac238 X_Ac239 "
            "X_Ac240 X_Ac241 X_Ac242 X_Ac243 X_Ac244 X_Ac245 X_Ac246 X_Ac247 X_Ac248 X_Ac249 X_Ac250 X_Ac251 X_Ac252 X_Ac253 X_Ac254 X_Ac255 X_Ac256 X_Ac257 X_Ac258 X_Ac259 X_Ac260 X_Ac261 X_Ac262 "
            "X_Ac265 X_Th226 X_Th227 X_Th228 X_Th229 X_Th230 X_Th231 X_Th232 X_Th233 X_Th234 X_Th235 X_Th236 X_Th237 X_Th238 X_Th239 X_Th240 X_Th241 X_Th242 X_Th243 X_Th244 X_Th245 X_Th246 X_Th247 "
            "X_Th248 X_Th249 X_Th250 X_Th251 X_Th252 X_Th253 X_Th254 X_Th255 X_Th256 X_Th257 X_Th258 X_Th259 X_Th260 X_Th261 X_Th262 X_Th263 X_Th264 X_Th265 X_Th266 X_Th270 X_Pa229 X_Pa231 X_Pa232 "
            "X_Pa233 X_Pa234 X_Pa235 X_Pa236 X_Pa237 X_Pa238 X_Pa239 X_Pa240 X_Pa241 X_Pa242 X_Pa243 X_Pa244 X_Pa245 X_Pa246 X_Pa247 X_Pa248 X_Pa249 X_Pa250 X_Pa251 X_Pa252 X_Pa253 X_Pa254 X_Pa255 "
            "X_Pa256 X_Pa257 X_Pa258 X_Pa259 X_Pa260 X_Pa261 X_Pa262 X_Pa263 X_Pa264 X_Pa265 X_Pa266 X_Pa267 X_Pa268 X_Pa269 X_Pa270 X_U229 X_U232 X_U233 X_U234 X_U235 X_U236 X_U237 X_U238 X_U239 "
            "X_U240 X_U241 X_U242 X_U243 X_U244 X_U245 X_U246 X_U247 X_U248 X_U249 X_U250 X_U251 X_U252 X_U253 X_U254 X_U255 X_U256 X_U257 X_U258 X_U259 X_U260 X_U261 X_U262 X_U263 X_U264 X_U265 "
            "X_U266 X_U267 X_U268 X_U269 X_U270 X_U271 X_U272 X_U273 X_U274 X_U275 X_Np229 X_Np232 X_Np237 X_Np238 X_Np239 X_Np240 X_Np241 X_Np242 X_Np243 X_Np244 X_Np245 X_Np246 X_Np247 X_Np248 "
            "X_Np249 X_Np250 X_Np251 X_Np252 X_Np253 X_Np254 X_Np255 X_Np256 X_Np257 X_Np258 X_Np259 X_Np260 X_Np261 X_Np262 X_Np263 X_Np264 X_Np265 X_Np266 X_Np267 X_Np268 X_Np269 X_Np270 X_Np271 "
            "X_Np272 X_Np273 X_Np274 X_Np275 X_Np276 X_Np277 X_Np279 X_Np280 X_Np281 X_Np283 X_Np284 X_Np285 X_Np287 X_Pu232 X_Pu238 X_Pu239 X_Pu240 X_Pu241 X_Pu242 X_Pu243 X_Pu244 X_Pu245 X_Pu246 "
            "X_Pu247 X_Pu248 X_Pu249 X_Pu250 X_Pu251 X_Pu252 X_Pu253 X_Pu254 X_Pu255 X_Pu256 X_Pu257 X_Pu258 X_Pu259 X_Pu260 X_Pu261 X_Pu262 X_Pu263 X_Pu264 X_Pu265 X_Pu266 X_Pu267 X_Pu268 X_Pu269 "
            "X_Pu270 X_Pu271 X_Pu272 X_Pu273 X_Pu274 X_Pu275 X_Pu276 X_Pu277 X_Pu278 X_Pu279 X_Pu280 X_Pu281 X_Pu282 X_Pu283 X_Pu284 X_Pu285 X_Pu286 X_Pu287 X_Pu288 X_Am233 X_Am241 X_Am242 X_Am243 "
            "X_Am244 X_Am245 X_Am246 X_Am247 X_Am248 X_Am249 X_Am250 X_Am251 X_Am252 X_Am253 X_Am254 X_Am255 X_Am256 X_Am257 X_Am258 X_Am259 X_Am260 X_Am261 X_Am262 X_Am263 X_Am264 X_Am265 X_Am266 "
            "X_Am267 X_Am268 X_Am269 X_Am270 X_Am271 X_Am272 X_Am273 X_Am274 X_Am275 X_Am276 X_Am277 X_Am278 X_Am279 X_Am280 X_Am281 X_Am282 X_Am283 X_Am284 X_Am285 X_Am286 X_Cm241 X_Cm242 X_Cm243 "
            "X_Cm244 X_Cm245 X_Cm246 X_Cm247 X_Cm248 X_Cm249 X_Cm250 X_Cm251 X_Cm252 X_Cm253 X_Cm254 X_Cm255 X_Cm256 X_Cm257 X_Cm258 X_Cm259 X_Cm260 X_Cm261 X_Cm262 X_Cm263 X_Cm264 X_Cm265 X_Cm266 "
            "X_Cm267 X_Cm268 X_Cm269 X_Cm270 X_Cm271 X_Cm272 X_Cm273 X_Cm274 X_Cm275 X_Cm276 X_Cm277 X_Cm278 X_Cm279 X_Cm280 X_Cm281 X_Cm282 X_Bk241 X_Bk245 X_Bk249 X_Bk250 X_Bk251 X_Bk252 X_Bk253 "
            "X_Bk254 X_Bk255 X_Bk256 X_Bk257 X_Bk258 X_Bk259 X_Bk260 X_Bk261 X_Bk262 X_Bk263 X_Bk264 X_Bk265 X_Bk266 X_Bk268 X_Bk269 X_Bk270 X_Bk271 X_Bk272 X_Bk273 X_Bk274 X_Bk275 X_Bk276 X_Bk277 "
            "X_Bk278 X_Bk279 X_Bk280 X_Bk281 X_Cf245 X_Cf246 X_Cf247 X_Cf248 X_Cf249 X_Cf250 X_Cf251 X_Cf252 X_Cf253 X_Cf254 X_Cf255 X_Cf256 X_Cf257 X_Cf258 X_Cf259 X_Cf260 X_Cf261 X_Cf262 X_Cf263 "
            "X_Cf271 X_Cf272 X_Cf273 X_Cf274 X_Cf275 X_Cf276 X_Cf277 X_Cf278 X_Cf279 X_Cf280 X_Es245 X_Es246 X_Es247 X_Es248 X_Es251 X_Es253 X_Es254 X_Es255 X_Es256 X_Es257 X_Es258 X_Es259 X_Es260 "
            "X_Es261 X_Es262 X_Es263 X_Es274 X_Es275 X_Es276 X_Es277 X_Es278 X_Fm240 X_Fm250 X_Fm251 X_Fm252 X_Fm254 X_Fm255 X_Fm256 X_Fm257 X_Fm258 X_Fm259 X_Fm260 X_Fm261 X_Fm262 X_Fm263 X_Fm264 "
            "X_Fm277 X_Md241 X_Md242 X_Md243 X_Md249 X_Md250 X_Md251 X_Md252 X_Md254 X_Md261 X_Md262 X_Md263 X_No254 X_No261 X_No263 X_Lr254 X_Lr255 X_Lr256 X_Db258 X_Db260 X_Db262 X_Bh263 X_Bh264 "
            "X_Mt277 X_Mt278 X_Mt279 X_Mt280 X_Mt281 X_Mt283 X_Ds280 X_Ds281 X_Ds282 X_Ds291 X_Ds293".split()
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
