#!/usr/bin/env python3

# import hashlib
# import math
# import numpy as np
# import os
# import os.path
import pandas as pd
# import pytest
from pathlib import Path

import artistools as at
import artistools.inputmodel

modelpath = at.config['path_testartismodel']
outputpath = Path(at.config['path_testoutput'])
# outputpath.mkdir(parents=True, exist_ok=True)
at.enable_diskcache = False


def test_describeinputmodel():
    at.inputmodel.describeinputmodel.main(argsraw=[], inputfile=modelpath, get_abundances=True)


def test_makemodel_botyanski2017():
    at.inputmodel.botyanski2017.main(argsraw=[], outputpath=outputpath)


def test_makemodel():
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, outputpath=outputpath)


def test_makemodel_energyfiles():
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=modelpath, makeenergyinputfiles=True, modeldim=1, outputpath=outputpath)


def test_maketardismodel():
    at.inputmodel.maketardismodelfromartis.main(
        argsraw=[], inputpath=modelpath, outputpath=outputpath)


def test_make_empty_abundance_file():
    at.inputmodel.save_empty_abundance_file(ngrid=50, outputfilepath=outputpath)


def test_opacity_by_Ye_file():
    griddata = {'cellYe': [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5],
                'rho': [0, 99, 99, 99, 99, 99, 99, 99],
                'inputcellid': range(1, 9)}
    at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, griddata=griddata)


def test_save3Dmodel():
    dfmodel = pd.DataFrame(
        {'inputcellid': [1, 2, 3, 4, 5, 6, 7, 8],
         'pos_x_min': [-1, 1, -1, 1, -1, 1, -1, 1],
         'pos_y_min': [-1, -1, 1, 1, -1, -1, 1, 1],
         'pos_z_min': [-1, -1, -1, -1, 1, 1, 1, 1],
         'rho': [0, 2, 3, 2, 5, 7, 8, 2],
         'cellYe': [0, 0.1, 0.2, .1, .5, .1, .3, 3]})
    tmodel = 100
    vmax = 1000
    at.inputmodel.save_modeldata(
        modelpath=outputpath, dfmodel=dfmodel, t_model_init_days=tmodel,
        vmax=vmax, dimensions=3)
