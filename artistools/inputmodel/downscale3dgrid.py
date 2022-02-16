import numpy as np


def make_downscaled_3d_grid(modelpath, inputgridsize=200, outputgridsize=50):
    """Should be same as downscale_3d_grid.pro
    Expects a 3D model with grid^3 cells and outputs 3D model with smallgrid^3 cells"""

    grid = int(inputgridsize)
    smallgrid = int(outputgridsize)

    merge = grid / smallgrid
    merge = int(merge)

    modelfile = modelpath / "model.txt"
    abundancefile = modelpath / "abundances.txt"
    smallmodelfile = modelpath / "model_small.txt"
    smallabundancefile = modelpath / "abundances_small.txt"

    rho = np.zeros((grid, grid, grid))
    ffe = np.zeros((grid, grid, grid))
    fni = np.zeros((grid, grid, grid))
    fco = np.zeros((grid, grid, grid))
    ffe52 = np.zeros((grid, grid, grid))
    fcr48 = np.zeros((grid, grid, grid))
    abund = np.zeros((grid, grid, grid, 31))
    abread = np.zeros(31)

    print("reading abundance file")
    with open(abundancefile, 'r') as sourceabundancefile:
        for x in range(0, grid):
            for y in range(0, grid):
                for z in range(0, grid):
                    abread = sourceabundancefile.readline().split()
                    abund[x, y, z] = abread

    print("reading model file")
    with open(modelfile, 'r') as sourcemodelfile:
        x = sourcemodelfile.readline()
        t_model = sourcemodelfile.readline()
        vmax = sourcemodelfile.readline()

        for x in range(0, grid):
            for y in range(0, grid):
                for z in range(0, grid):
                    dum1, dum2, dum3, dum4, rhoread = sourcemodelfile.readline().split()
                    rho[x, y, z] = rhoread
                    fferead, fniread, fcoread, ffe52read, fcr48read = sourcemodelfile.readline().split()
                    ffe[x, y, z] = fferead
                    fni[x, y, z] = fniread
                    fco[x, y, z] = fcoread
                    ffe52[x, y, z] = ffe52read
                    fcr48[x, y, z] = fcr48read

    rho_small = np.zeros((smallgrid, smallgrid, smallgrid))
    ffe_small = np.zeros((smallgrid, smallgrid, smallgrid))
    fni_small = np.zeros((smallgrid, smallgrid, smallgrid))
    fco_small = np.zeros((smallgrid, smallgrid, smallgrid))
    ffe52_small = np.zeros((smallgrid, smallgrid, smallgrid))
    fcr48_small = np.zeros((smallgrid, smallgrid, smallgrid))
    abund_small = np.zeros((smallgrid, smallgrid, smallgrid, 31))

    for x in range(0, smallgrid):
        for y in range(0, smallgrid):
            for z in range(0, smallgrid):
                for xx in range(0, merge):
                    for yy in range(0, merge):
                        for zz in range(0, merge):
                            rho_small[x, y, z] += rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            ffe_small[x, y, z] += ffe[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            fni_small[x, y, z] += fni[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            fco_small[x, y, z] += fco[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            ffe52_small[x, y, z] += ffe52[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            fcr48_small[x, y, z] += fcr48[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
                            abund_small[x, y, z, :] += abund[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]

    for x in range(0, smallgrid):
        for y in range(0, smallgrid):
            for z in range(0, smallgrid):
                if (rho_small[x, y, z] > 0):
                    ffe_small[x, y, z] /= rho_small[x, y, z]
                    fni_small[x, y, z] /= rho_small[x, y, z]
                    fco_small[x, y, z] /= rho_small[x, y, z]
                    ffe52_small[x, y, z] /= rho_small[x, y, z]
                    fcr48_small[x, y, z] /= rho_small[x, y, z]
                    for i in range(1, 31):  # check this
                        abund_small[x, y, z, i] /= rho_small[x, y, z]
                    rho_small[x, y, z] /= merge ** 3

    print("writing abundance file")
    i = 0
    with open(modelpath / smallabundancefile, 'w') as newabundancefile:
        for x in range(0, smallgrid):
            for y in range(0, smallgrid):
                for z in range(0, smallgrid):
                    line = abund_small[x, y, z, :].tolist()  # index 1:30 are abundances
                    line[0] = int(i+1)  # replace index 0 with index id
                    i += 1
                    newabundancefile.writelines("%s " % item for item in line)
                    newabundancefile.writelines("\n")

    print("writing model file")
    xmax = float(vmax)*float(t_model)*3600*24
    i = 0
    with open(modelpath / smallmodelfile, 'w') as newmodelfile:
        gridsize = int(smallgrid**3)
        newmodelfile.write(f'{gridsize}\n')
        newmodelfile.write(f'{t_model}')
        newmodelfile.write(f'{vmax}')

        for x in range(0, smallgrid):
            for y in range(0, smallgrid):
                for z in range(0, smallgrid):
                    line1 = [int(i+1), -xmax + 2 * x * xmax / smallgrid, -xmax + 2 * y * xmax / smallgrid,  -xmax + 2 * z * xmax / smallgrid, rho_small[x, y, z]]
                    line2 = [ffe_small[x, y, z], fni_small[x, y, z], fco_small[x, y, z], ffe52_small[x, y, z], fcr48_small[x, y, z]]
                    i += 1
                    newmodelfile.writelines("%s " % item for item in line1)
                    newmodelfile.writelines("\n")
                    newmodelfile.writelines("%s " % item for item in line2)
                    newmodelfile.writelines("\n")
