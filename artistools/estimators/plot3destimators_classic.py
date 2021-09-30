import matplotlib.pyplot as plt

import artistools as at
import artistools.estimators.estimators_classic
import artistools.inputmodel.slice1Dfromconein3dmodel

from pathlib import Path
from collections import namedtuple


def read_selected_mgi(modelpath, readonly_mgi):
    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
    estimators = at.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata,
                                                                          readonly_mgi=readonly_mgi)
    return estimators


def get_modelgridcells_along_axis(modelpath):
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)

    ArgsTuple = namedtuple('args', 'modelpath sliceaxis other_axis1 other_axis2 positive_axis')
    args = ArgsTuple(
        modelpath=modelpath,
        sliceaxis='x',
        other_axis1='z',
        other_axis2='y',
        positive_axis=True
    )

    profile1d = at.inputmodel.slice1Dfromconein3dmodel.get_profile_along_axis(args=args)
    readonly_mgi = []
    for index, row in profile1d.iterrows():
        print(row['inputcellid'], row['rho'])
        if row['rho'] > 0:
            mgi = mgi_of_propcells[int(row['inputcellid'])-1]
            readonly_mgi.append(mgi)

    return readonly_mgi


