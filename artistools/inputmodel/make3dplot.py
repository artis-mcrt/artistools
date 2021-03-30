import pyvista as pv
import numpy as np
from pathlib import Path
import artistools as at


pv.set_plot_theme("document")

# modelpath = Path("/home/localadmin_ccollins/artis/artisinput/")
modelpath = Path("/home/localadmin_ccollins/Documents/kilonova_inputfiles/test3d/opacityYedependent-axisfixed")

model, t_model, vmax = at.inputmodel.get_modeldata(modelpath, dimensions=3, get_abundances=False)

# choose what surface will be coloured by - eg rho
coloursurfaceby = np.array(model['rho'])

# generate grid from data
grid = round(len(model['rho']) ** (1./3.))
surfacecolorscale = np.zeros((grid, grid, grid))  # needs 3D array
xgrid = np.zeros(grid)

i = 0
for z in range(0, grid):
    for y in range(0, grid):
        for x in range(0, grid):
            surfacecolorscale[x, y, z] = coloursurfaceby[i]
            xgrid[x] = -vmax + 2 * x * vmax / grid
            i += 1

x,y,z = np.meshgrid(xgrid,xgrid,xgrid)

mesh = pv.StructuredGrid(x,y,z)
print(mesh) # tells you the properties of the mesh

mesh['surfacecolorscale'] = surfacecolorscale.ravel(order='F') # add data to the mesh
# mesh.plot()

surfacepositions = [1, 50, 100, 300, 500, 800, 1000, 1100, 1200, 1300, 1400, 1450, 1500] # choose these

surf = mesh.contour(surfacepositions,scalars='surfacecolorscale') # create isosurfaces

surf.plot(opacity='linear', screenshot='bla.png')    # plot surfaces and save screenshot


# clip planes
# clipped = surf.clip('x') # normal = x direction, plane =yz
# clipped.plot(opacity='linear',screenshot='bla.png')

# # now let's try do create a nice plot
#
# lset = np.load('r10d1/lset.npy')[8]
# mesh['lset'] = lset.ravel(order='F')
#
# flame = mesh.contour([0],scalars='lset') # zero contour of lset is flame surface
#
# p = pv.Plotter()
# p.add_mesh(flame, color='red') # red flame surface
# p.show()
#
# p = pv.Plotter()
# p.add_mesh(flame,scalars=np.log10(flame['rho']),cmap='inferno') # color flame surface with other scalar value, here rho
# p.add_axes() #axis triangel
# p.add_scalar_bar() # colorbar
# p.show()
#
# # with subplots
# p = pv.Plotter(shape=(1,2))
# #p.set_background(color='white')
#
# p.subplot(0,0)
# p.add_mesh(flame,scalars=np.log10(flame['rho']),cmap='inferno') # color flame surface with other scalar value, here rho
# p.add_axes() #axis triangel
# p.add_scalar_bar() # colorbar
#
# p.subplot(0,1)
#
# p.add_mesh(flame,color='silver')
# p.add_mesh(clipped,scalars=np.log10(clipped['rho']),cmap='inferno',opacity='linear')
#
# p.add_scalar_bar() # colorbar
# p.show(screenshot='haha.png') # you can also specity camera position via cpos= [(x,y,z),(focal point), (upview)]
#
# # save your mesh to file
#
# evtk.hl.gridToVTK('grid',x,x,x,cellData={'rho':rho, 'lset': lset}) # also pointData and fieldData possible
#
# mesh = pv.read('grid.vtr').cell_data_to_point_data() # create point data (other way around also possible)
#
# #..dd_mesh(clipped,scalars=np.log10(clipped['rho']),cmap='inferno',opacity='linear')measure area
# cont = mesh.contour([0],'lset') # flame surface
# sizes = cont.compute_cell_sizes()
# size['Area'] #surface area of the contour


