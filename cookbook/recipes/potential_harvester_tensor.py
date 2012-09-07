"""
Example of 3D inversion of synthetic gravity gradient data using Harvester
"""
import numpy
import fatiando as ft
from fatiando.msh.ddd import Prism, PrismMesh

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, 0, 1500]
model = [Prism(500, 4500, 3000, 3500, 200, 700, {'density':1200}),
         Prism(3000, 4500, 1800, 2300, 200, 700, {'density':1200}),
         Prism(500, 1500, 500, 1500, 0, 800, {'density':600}),
         Prism(0, 800, 1800, 2300, 0, 200, {'density':600}),
         Prism(4000, 4800, 100, 900, 0, 300, {'density':600}),
         Prism(0, 2000, 4500, 5000, 0, 200, {'density':600}),
         Prism(3000, 4200, 2500, 2800, 200, 700, {'density':-1000}),
         Prism(300, 2500, 1800, 2700, 500, 1000, {'density':-1000}),
         Prism(4000, 4500, 500, 1500, 400, 1000, {'density':-1000}),
         Prism(1800, 3700, 500, 1500, 300, 1300, {'density':-1000}),
         Prism(500, 4500, 4000, 4500, 400, 1300, {'density':-1000})]
# show it
ft.vis.figure3d()
ft.vis.prisms(model, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds],
              fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
# and use it to generate some tensor data
shape = (51, 51)
area = bounds[0:4]
noise = 2
x, y, z = ft.grd.regular(area, shape, z=-150)
gyy = ft.utils.contaminate(ft.pot.prism.gyy(x, y, z, model), noise)
gyz = ft.utils.contaminate(ft.pot.prism.gyz(x, y, z, model), noise)
gzz = ft.utils.contaminate(ft.pot.prism.gzz(x, y, z, model), noise)
# Create a prism mesh
mesh = PrismMesh(bounds, (15, 50, 50))
# Make the data modules
datamods = ft.pot.harvester.wrapdata(mesh, x, y, z, gyy=gyy, gyz=gyz, gzz=gzz)
# and the seeds
seeds = ft.pot.harvester.sow(
    [( 800, 3250, 600, {'density':1200}),
     (1200, 3250, 600, {'density':1200}),
     (1700, 3250, 600, {'density':1200}),
     (2100, 3250, 600, {'density':1200}),
     (2500, 3250, 600, {'density':1200}),
     (2900, 3250, 600, {'density':1200}),
     (3300, 3250, 600, {'density':1200}),
     (3700, 3250, 600, {'density':1200}),
     (4200, 3250, 600, {'density':1200}),
     (3300, 2050, 600, {'density':1200}),
     (3600, 2050, 600, {'density':1200}),
     (4000, 2050, 600, {'density':1200}),
     (4300, 2050, 600, {'density':1200})], mesh, mu=0.1, delta=0.0001)
# Run the inversion and collect the results
estimate, goals, misfits = ft.pot.harvester.harvest(datamods, seeds)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = ft.msh.ddd.vfilter(1100, 1300, 'density', mesh)
# Get the predicted data from the data modules
tensor = (gyy, gyz, gzz)
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
ft.vis.prisms(density_model, 'density', vmin=0)
ft.vis.axes3d(ft.vis.outline3d(bounds),
              ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
