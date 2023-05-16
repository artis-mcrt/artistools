import numpy as np
import pandas as pd

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
modelpath_3d = at.get_config()["path_testartismodel"].parent / "testmodel_3d_10^3"
outputpath = at.get_config()["path_testoutput"]


def test_describeinputmodel() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath, get_elemabundances=True)


def test_describeinputmodel_3d() -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath_3d, get_elemabundances=True)


def test_get_modeldata_1d() -> None:
    for getheadersonly in [False, True]:
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 800000000.0)
        assert modelmeta["dimensions"] == 1
        assert modelmeta["modelcellcount"] == 1

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath)
    assert np.isclose(dfmodel.cellmass_grams.sum(), 1.416963e33)


def test_get_modeldata_3d() -> None:
    for getheadersonly in [False, True]:
        dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, getheadersonly=getheadersonly)
        assert np.isclose(modelmeta["vmax_cmps"], 2892020000.0)
        assert modelmeta["dimensions"] == 3
        assert modelmeta["modelcellcount"] == 1000
        assert modelmeta["ncoordgrid"] == 10

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d)
    assert np.isclose(dfmodel.cellmass_grams.sum(), 2.7861855e33)


def test_downscale_3dmodel() -> None:
    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath_3d, get_elemabundances=True)
    modelpath_3d_small = at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(modelpath_3d, outputgridsize=2)
    dfmodel_small, modelmeta_small = at.get_modeldata(modelpath_3d_small, get_elemabundances=True)
    assert np.isclose(dfmodel["cellmass_grams"].sum(), dfmodel_small["cellmass_grams"].sum())
    assert np.isclose(modelmeta["vmax_cmps"], modelmeta_small["vmax_cmps"])
    assert np.isclose(modelmeta["t_model_init_days"], modelmeta_small["t_model_init_days"])

    abundcols = (x for x in dfmodel.columns if x.startswith("X_"))
    for abundcol in abundcols:
        assert np.isclose(
            (dfmodel[abundcol] * dfmodel["cellmass_grams"]).sum(),
            (dfmodel_small[abundcol] * dfmodel_small["cellmass_grams"]).sum(),
        )


def test_makemodel_botyanski2017() -> None:
    at.inputmodel.botyanski2017.main(argsraw=[], outputpath=outputpath)


def test_makemodel() -> None:
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, outputpath=outputpath)


def test_makemodel_energyfiles() -> None:
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=modelpath, makeenergyinputfiles=True, modeldim=1, outputpath=outputpath
    )


def test_maketardismodel() -> None:
    at.inputmodel.maketardismodelfromartis.main(argsraw=[], inputpath=modelpath, outputpath=outputpath)


def test_make_empty_abundance_file() -> None:
    at.inputmodel.save_empty_abundance_file(ngrid=50, outputfilepath=outputpath)


def test_opacity_by_Ye_file() -> None:
    griddata = {
        "cellYe": [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5],
        "rho": [0, 99, 99, 99, 99, 99, 99, 99],
        "inputcellid": range(1, 9),
    }
    at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, griddata=griddata)


def test_save3Dmodel() -> None:
    dfmodel = pd.DataFrame(
        {
            "inputcellid": [1, 2, 3, 4, 5, 6, 7, 8],
            "pos_x_min": [-1, 1, -1, 1, -1, 1, -1, 1],
            "pos_y_min": [-1, -1, 1, 1, -1, -1, 1, 1],
            "pos_z_min": [-1, -1, -1, -1, 1, 1, 1, 1],
            "rho": [0, 2, 3, 2, 5, 7, 8, 2],
            "cellYe": [0, 0.1, 0.2, 0.1, 0.5, 0.1, 0.3, 3],
        }
    )
    tmodel = 100
    vmax = 1000
    at.inputmodel.save_modeldata(
        modelpath=outputpath, dfmodel=dfmodel, t_model_init_days=tmodel, vmax=vmax, dimensions=3
    )
