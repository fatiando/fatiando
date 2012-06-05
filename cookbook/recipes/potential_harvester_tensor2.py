"""
Example of 3D inversion of synthetic gravity gradient data using Harvester
"""
import numpy
import fatiando as ft
from fatiando.mesher.ddd import Prism, PrismMesh

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, -500, 2000]
model = [Prism(600, 1200, 200, 4200, 400, 900, {'density':1500}),
         Prism(3000, 4000, 1000, 2000, 200, 800, {'density':1000}),
         Prism(2700, 3200, 3700, 4200, 0, 900, {'density':800})]
# show it
ft.vis.figure3d()
ft.vis.prisms(model, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds],
              fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
# and use it to generate some tensor data
shape = (25, 25)
area = bounds[0:4]
x, y, z = ft.grd.regular(area, shape, z=-650)
gxy = ft.utils.contaminate(ft.pot.prism.gxy(x, y, z, model), 1)
gzz = ft.utils.contaminate(ft.pot.prism.gzz(x, y, z, model), 1)
# Create a prism mesh
mesh = PrismMesh(bounds, (20, 50, 50))
# Make the data modules
datamods = ft.pot.harvester.wrapdata(mesh, x, y, z, gxy=gxy, gzz=gzz)
# and the seeds
points =[(901, 701, 750),
         (901, 1201, 750),
         (901, 1701, 750),
         (901, 2201, 750),
         (901, 2701, 750),
         (901, 3201, 750),
         (901, 3701, 750),
         (3701, 1201, 501),
         (3201, 1201, 501),
         (3701, 1701, 501),
         (3201, 1701, 501),
         (2951, 3951, 301),
         (2951, 3951, 701)]         
densities = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000,
             800, 800]
seeds = ft.pot.harvester.sow(points, {'density':densities},
    mesh, mu=10, delta=0.0001)
# Run the inversion and collect the results
estimate, goals, misfits = ft.pot.harvester.harvest(datamods, seeds)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = ft.msh.ddd.vfilter(1, 2000, 'density', mesh)
# Get the predicted data from the data modules
tensor = (gxy, gzz)
predicted = [dm.get_predicted() for dm in datamods]
# Plot the results
for true, pred in zip(tensor, predicted):
    ft.vis.figure()
    ft.vis.title("True: color | Inversion: contour")
    ft.vis.axis('scaled')
    levels = ft.vis.contourf(y*0.001, x*0.001, true, shape, 12)
    ft.vis.colorbar()
    ft.vis.contour(y*0.001, x*0.001, pred, shape, levels, color='k')
    ft.vis.xlabel('Horizontal coordinate y (km)')
    ft.vis.ylabel('Horizontal coordinate x (km)')
ft.vis.show()
ft.vis.figure3d()
ft.vis.prisms(model, 'density', style='wireframe')
ft.vis.prisms(density_model, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds),
              ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
