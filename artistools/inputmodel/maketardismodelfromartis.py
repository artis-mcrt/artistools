#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argcomplete
import argparse
from yaml import dump as yamldump

from pathlib import Path
from artistools import CustomArgHelpFormatter, get_model_name
from artistools.inputmodel import get_modeldata


def addargs(parser):

    parser.add_argument('-inputpath', '-i', default='model.txt',
                        help='Path of input ARTIS model')

    parser.add_argument('-temperature', '-T', default=10000,
                        help='Temperature to use in TARDIS file')

    parser.add_argument('-dilution_factor', '-W', default=1.,
                        help='Dilution factor to use in TARDIS file')

    parser.add_argument('-outputpath', '-o', default='.',
                        help='Path of output TARDIS model file')


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=CustomArgHelpFormatter,
            description='Convert an ARTIS format model to TARDIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    temperature = args.temperature
    dilution_factor = args.dilution_factor

    modelpath = Path(args.inputpath)

    dfmodel, t_model_init_days, _ = get_modeldata(modelpath)
    dfmodel.eval('rho = 10 ** logrho', inplace=True)

    listnuclei = [
        col[2:] for col in dfmodel.columns
        if col.startswith('X_') and col.upper() != 'X_FEGROUP' and col[-1].isdigit()]

    # listnuclei = listnuclei[:10]  # for testing, avoid huge number of isotopes

    modelname = get_model_name(modelpath)
    outputfilepath = Path(args.outputpath, f'{modelname}.csvy')
    dictmeta = {
        'name': modelname,
        'description': 'This model was converted from ARTIS format by artistools',
        'model_density_time_0': f'{t_model_init_days} day',
        'model_isotope_time_0': f'{t_model_init_days} day',
        'tardis_model_config_version': 'v1.0',
        'datatype': {
            'fields': [
                {'name': 'velocity',
                 'unit': 'km/s',
                 'desc': 'velocities of shell outer boundaries'},
                {'name': 'density',
                 'unit': 'g/cm^3',
                 'desc': 'density of shell'},
                {'name': 't_rad',
                 'unit': 'K',
                 'desc': 'radiative temperature'},
                {'name': 'dilution_factor',
                 'desc': 'dilution factor of shell'},
                *[{'name': strnuc, 'desc': f'fractional {strnuc} abundance'} for strnuc in listnuclei]
            ]
        }
    }
    with open(outputfilepath, 'w') as fileout:
        fileout.write('---\n')
        yamldump(dictmeta, fileout, sort_keys=False)
        fileout.write('---\n')
        fileout.write(','.join(['velocity', 'density', 't_rad', 'dilution_factor', *listnuclei]))
        fileout.write('\n')

        fileout.write(f'{0.},{0.:.4e},{0.},{0.},{",".join([f"{0.:.4e}" for _ in listnuclei])}\n')

        for cell in dfmodel.itertuples(index=False):
            abundlist = [f'{getattr(cell, "X_" + strnuc):.4e}' for strnuc in listnuclei]
            fileout.write(f'{cell.velocity_outer},{cell.rho:.4e},{temperature},{dilution_factor},{",".join(abundlist)}\n')

    print(f'Saved {outputfilepath}')


if __name__ == "__main__":
    main()
