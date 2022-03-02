'''various functions to read/write CMFGEN input/output files:

rd_nuc_decay_data
rd_sn_hydro_data
'''

# import os
import sys

# import re
import numpy as np

# from pdb import set_trace as stop

# constants
DAY2SEC = 86400.0
MEV2ERG = 1.60217733e-6


def rd_nuc_decay_data(file, quiet=False):
    """read in NUC_DECAY_DATA file and return namespace
    set quiet=True to disable verbose output
    """

    # open file
    with open(file, 'r') as f:

        # read in header
        while True:
            line = f.readline()
            linearr = line.split()
            if 'Format date' in line:
                date = linearr[0]
            elif 'Number of species' in line:
                nspec = int(linearr[0])
            elif 'Total number of isotopes' in line:
                niso = int(linearr[0])
            elif 'Maximum number of isotopes/species' in line:
                maxisospec = int(linearr[0])
            elif 'Number of reactions' in line:
                nreac = int(linearr[0])
                break
        if not quiet:
            print('*** INPUT FILE: ' + file)
            print('Format date: ' + str(date))
            print('Number of species: ' + str(nspec))
            print('Total number of isotopes: ' + str(niso))
            print('Maximum number of isotopes/species: ' + str(maxisospec))
            print('Number of reactions: ' + str(nreac))

        # isotopes
        isospec = []
        amu = np.zeros(niso)
        aiso = np.zeros(niso, dtype='int32')
        stable = []  # True or False
        for i in range(niso):
            line = ''
            while len(line.strip()) == 0 or line[0] == '!':
                line = f.readline()
            linearr = line.split()
            isospec.append(linearr[0])
            amu[i] = float(linearr[1])
            aiso[i] = np.rint(amu[i])
            stable.append(True) if linearr[2] == 's' else stable.append(False)
        if not quiet:
            print("INFO - Read in isotope information")

        # decay chains
        isospec_parent = []
        amu_parent = np.zeros(nreac)
        aiso_parent = np.zeros(nreac, dtype='int32')
        thalf = np.zeros(nreac)
        decay_const = np.zeros(nreac)
        isospec_daughter = []
        amu_daughter = np.zeros(nreac)
        aiso_daughter = np.zeros(nreac, dtype='int32')
        edec = np.zeros(nreac)
        seqnum = []
        nlines = np.zeros(nreac, dtype='int32')

        nchains = 0
        for i in range(nreac):
            line = ''
            while len(line.strip()) == 0 or line[0] == '!':
                line = f.readline()
            linearr = line.split()
            isospec_parent.append(linearr[0])
            amu_parent[i] = float(linearr[1])
            aiso_parent[i] = np.rint(amu_parent[i])
            thalf[i] = float(linearr[2]) * DAY2SEC  # convert to seconds
            decay_const[i] = np.log(2) / thalf[i]
            isospec_daughter.append(linearr[3])
            amu_daughter[i] = float(linearr[4])
            aiso_daughter[i] = np.rint(amu_daughter[i])
            edec[i] = float(linearr[5]) * MEV2ERG  # convert to ergs
            seqnum.append(linearr[6])
            if seqnum[-1] in ['F', 'E']:
                nchains = nchains + 1
            nlines[i] = int(linearr[7])
        if not quiet:
            print("INFO - Read in decay chains")

    # output
    out = {}
    out['date'] = date
    out['nspec'] = nspec
    out['niso'] = niso
    out['maxisospec'] = maxisospec
    out['nreac'] = nreac
    out['nchains'] = nchains
    out['isospec'] = isospec
    out['amu'] = amu
    out['aiso'] = aiso
    out['stable'] = stable
    out['isospec_parent'] = isospec_parent
    out['amu_parent'] = amu_parent
    out['aiso_parent'] = aiso_parent
    out['thalf'] = thalf
    out['decay_const'] = decay_const
    out['isospec_daughter'] = isospec_daughter
    out['amu_daughter'] = amu_daughter
    out['aiso_daughter'] = aiso_daughter
    out['edec'] = edec
    out['seqnum'] = seqnum
    out['nlines'] = nlines

    # end
    return out

###############################################################################


def rd_sn_hydro_data(file, ncol=8, reverse=False,
                     quiet=False):
    """read in SN_HYDRO_DATA or SN_HYDRO_FOR_NEXT_MODEL files and return namespace
    set reverse=True to output vectors from vmin to vmax (CMFGEN's grid moves inward from vmax to vmin)
    set quiet=True to disable verbose output
    """

    MAX_POP_DIFF = 1e-5  # maximum absolute difference between sum(isofrac) and corresponding specfrac

    # open file
    with open(file, 'r') as f:

        # read in header
        okhdr = 0
        nd, nspec, niso = 0, 0, 0
        time = 0.0
        while okhdr == 0:
            line = f.readline()
            if 'Number of data points:' in line:
                nd = int(line.split()[4])
                nrow = int(np.ceil(nd / float(ncol)))
            elif 'Number of mass fractions:' in line:
                nspec = int(line.split()[4])
            elif 'Number of isotopes:' in line:
                niso = int(line.split()[3])
            elif 'Time(days) since explosion:' in line:
                time = float(line.split()[3])
            elif 'Radius grid' in line:
                if nd == 0 or nspec == 0 or niso == 0 or time == 0.0:
                    sys.exit('nd, nspec, niso or model time undefined')
                else:
                    okhdr = 1
        if not quiet:
            print(' *** INPUT FILE: ' + file)
            print(' Number of data points:          ' + str(nd))
            print(' Number of mass fractions:       ' + str(nspec))
            print(' Number of isotopes:             ' + str(niso))
            print(' Time(days) since explosion:     ' + str(time))

        # read in hydro grid vectors
        rad = np.zeros(nd)       # radius grid (10^10 cm)
        vel = np.zeros(nd)       # velocity (km/s)
        sigma = np.zeros(nd)     # sigma = dlnV/dlnR-1 (=0 for pure hubble flow)
        temp = np.zeros(nd)      # temperature (10^4 K)
        dens = np.zeros(nd)      # mass density (g/cm^3)
        atomdens = np.zeros(nd)  # atom density (/cm^3)
        ed = np.zeros(nd)        # electron density (/cm^3)
        rossopac = np.zeros(nd)  # Rossland mean opacity (10^-10 cm^-1)
        kappa = np.zeros(nd)     # mass absorption coefficient (cm^2/g)
        okhydro = 0
        while okhydro == 0:
            while line == '':
                line = f.readline()
            if 'Radius grid' in line:
                rad = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Velocity' in line:
                vel = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Sigma' in line:
                sigma = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Temperature' in line:
                temp = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Density' in line:
                dens = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Atom density' in line:
                atomdens = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Electron density' in line:
                ed = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Rosseland mean' in line:
                rossopac = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'Kappa' in line:
                kappa = np.fromfile(f, count=nd, sep=" ", dtype=float)
            elif 'mass fraction' in line:
                if rad[0] == 0.0 or temp[0] == 0.0 or atomdens[0] == 0.0 or ed[0] == 0.0:
                    sys.exit('Error reading SN hydro data: R or T is zero')
                else:
                    okhydro = 1
            if okhydro == 0:
                line = ''
        if not quiet:
            print(' INFO - Read in hydro grid vectors')

        # compute shell volumes and masses
        dvol = np.zeros(nd)   # cm^3
        dmass = np.zeros(nd)  # g
        rad_cgs = rad * 1e10  # rad in cm
        dvol[0] = rad_cgs[0]**3 - (0.5 * (rad_cgs[1] + rad_cgs[0]))**3
        dvol[nd - 1] = (0.5 * (rad_cgs[nd - 1] + rad_cgs[nd - 2]))**3 - rad_cgs[nd - 1]**3
        for i in range(1, nd - 1):
            rmin = 0.5 * (rad_cgs[i + 1] + rad_cgs[i])
            rmax = 0.5 * (rad_cgs[i] + rad_cgs[i - 1])
            dvol[i] = rmax**3 - rmin**3
        dvol = dvol * 4.0 / 3.0 * np.pi
        dmass = dens * dvol
        if not quiet:
            print(' INFO - Computed shell volumes and masses')

        # read in mass fractions
        spec = []
        specfrac = np.zeros((nd, nspec))
        for ispec in range(nspec):
            while 'mass fraction' not in line:
                line = f.readline()
            spec.append(line.split()[0])
            for ii in range(nrow):
                line = f.readline()
                if '*0.' in line:
                    specfrac[:, ispec] = 0.0
                    if not quiet:
                        print(' INFO - set mass fraction = 0.0 everywhere for ' + spec[ispec])
                    break
                else:
                    entries = line.split()
                    # set mass fractions to 0.0 if < 1D-99 (written e.g. 1.0000000-100)
                    entries = [entries[k] if 'E' in entries[k] else '0.0' for k in range(len(entries))]
                    idx0 = ii * ncol
                    idx1 = idx0 + len(entries)
                    specfrac[idx0:idx1, ispec] = np.array([float(xx) for xx in entries])
            line = ''
        if not quiet:
            print(' INFO - Read in species mass fractions')

        # read in isotope mass fractions
        iso = []
        aiso = np.zeros(niso, dtype='int32')
        isofrac = np.zeros((nd, niso))
        for iiso in range(niso):
            while 'mass fraction' not in line:
                line = f.readline()
            iso.append(line.split()[0])
            aiso[iiso] = int(line.split()[1])
            for ii in range(nrow):
                line = f.readline()
                if '*0.' in line:
                    isofrac[:, iiso] = 0.0
                    if not quiet:
                        print(' INFO - set mass fraction = 0.0 everywhere for ' + iso[iiso] + ' ' + str(aiso[iiso]))
                    break
                else:
                    entries = line.split()
                    # set mass fractions to 0.0 if < 1D-99 (written e.g. 1.0000000-100)
                    entries = [entries[k] if 'E' in entries[k] else '0.0' for k in range(len(entries))]
                    idx0 = ii * ncol
                    idx1 = idx0 + len(entries)
                    isofrac[idx0:idx1, iiso] = np.array([float(xx) for xx in entries])
            line = ''
        if not quiet:
            print(' INFO - Read in isotope mass fractions')

    # check sum isotope mass fractions = species mass fractions
    for s in list(set(iso)):
        # find all indices of species name in iso list
        idx = [i for i, x in enumerate(iso) if x == s]
        # sum isotope mass fractions
        sumisofrac = np.zeros(nd)
        for iiso in range(len(idx)):
            sumisofrac += isofrac[:, idx[iiso]]
        # compare to corresponding species mass fraction
        absdiff = np.abs(specfrac[:, spec.index(s)] - sumisofrac)
        relabsdiff = absdiff / sumisofrac
        if np.max(relabsdiff) > MAX_POP_DIFF:
            sys.exit("ERROR - Maximum absolute difference > MAX_POP_DIFF for species {0:s}".format(s))

    # reversed vectors if reverse=True
    if reverse:
        rad = rad[::-1]
        vel = vel[::-1]
        sigma = sigma[::-1]
        temp = temp[::-1]
        dens = dens[::-1]
        atomdens = atomdens[::-1]
        ed = ed[::-1]
        rossopac = rossopac[::-1]
        kappa = kappa[::-1]
        for i in range(nspec):
            specfrac[:, i] = specfrac[::-1, i]
        for i in range(niso):
            isofrac[:, i] = isofrac[::-1, i]
        dvol = dvol[::-1]
        dmass = dmass[::-1]

    # output
    out = {}
    out['nd'] = nd
    out['nspec'] = nspec
    out['niso'] = niso
    out['time'] = time
    out['rad'] = rad
    out['vel'] = vel
    out['sigma'] = sigma
    out['temp'] = temp
    out['dens'] = dens
    out['atomdens'] = atomdens
    out['ed'] = ed
    out['rossopac'] = rossopac
    out['kappa'] = kappa
    out['spec'] = spec
    out['specfrac'] = specfrac
    out['iso'] = iso
    out['aiso'] = aiso
    out['isofrac'] = isofrac
    out['dvol'] = dvol
    out['dmass'] = dmass

    # end
    return out

###############################################################################


if __name__ == '__main__':

    a = rd_sn_hydro_data('DDC10/SN_HYDRO_DATA_0.976d')
    b = rd_nuc_decay_data('NUC_DECAY_DATA')
    print(a['vel'])
    print(b['amu_parent'])
