#!/usr/bin/env python3

import numpy as np

from pathlib import Path
import artistools as at
import artistools.estimators


def main():
    modelpath = Path('.')
    timestep = 14
    elmass = {el.Z: el.mass for _, el in at.get_composition_data(modelpath).iterrows()}
    outfilename = 'massfracs.txt'
    with open(outfilename, 'wt') as fout:
        modelgridindexlist = range(10)
        estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindexlist)
        for modelgridindex in modelgridindexlist:
            tdays = estimators[(timestep, modelgridindex)]['tdays']
            popdict = estimators[(timestep, modelgridindex)]['populations']

            numberdens = {}
            totaldens = 0.  # number density times atomic mass summed over all elements
            for key in popdict.keys():
                try:
                    atomic_number = int(key)
                    numberdens[atomic_number] = popdict[atomic_number]
                    totaldens += numberdens[atomic_number] * elmass[atomic_number]
                except ValueError:
                    pass
                except TypeError:
                    pass

            massfracs = {
                atomic_number: numberdens[atomic_number] * elmass[atomic_number] / totaldens
                for atomic_number in numberdens.keys()
            }

            fout.write(f'{tdays}d shell {modelgridindex}\n')
            massfracsum = 0.
            for atomic_number in massfracs.keys():
                massfracsum += massfracs[atomic_number]
                fout.write(f'{atomic_number} {at.elsymbols[atomic_number]} {massfracs[atomic_number]}\n')

            assert np.isclose(massfracsum, 1.0)

    print(f'Saved {outfilename}')


if __name__ == "__main__":
    main()
