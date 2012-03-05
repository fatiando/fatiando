"""
Example of 3D inversion of synthetic gravity gradient data using Harvester
"""
import numpy
from matplotlib import pyplot
from fatiando.mesher.ddd import Prism, PrismMesh, extract, vfilter
from fatiando.potential import harvester
from fatiando import potential, logger, gridder, utils, vis

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
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

vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model))
vis.vtk.add_axes(vis.vtk.add_outline(bounds), ranges=[i*0.001 for i in bounds],
    fmt='%.1f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
#vis.vtk.mlab.show()

shape = (51, 51)
area = bounds[0:4]
noise = 2
x, y, z = gridder.regular(area, shape, z=-150)
gyy = utils.contaminate(potential.prism.gyy(x, y, z, model), noise)
gyz = utils.contaminate(potential.prism.gyz(x, y, z, model), noise)
gzz = utils.contaminate(potential.prism.gzz(x, y, z, model), noise)

log.info("Setting up the inversion")
mesh = PrismMesh(bounds, (15, 50, 50))
datamods = harvester.wrapdata(mesh, x, y, z, gyy=gyy, gyz=gyz, gzz=gzz)
points =[(800, 3250, 600),
         (1200, 3250, 600),
         (1700, 3250, 600),
         (2100, 3250, 600),
         (2500, 3250, 600),
         (2900, 3250, 600),
         (3300, 3250, 600),
         (3700, 3250, 600),
         (4200, 3250, 600),
         (3300, 2050, 600),
         (3600, 2050, 600),
         (4000, 2050, 600),
         (4300, 2050, 600)]
seeds = harvester.sow_prisms(points, {'density':[1200]*len(points)}, mesh,
    mu=0.1, delta=0.0001)

log.info("Run the inversion and collect the results")
estimate, goals, misfits = harvester.harvest(datamods, seeds)
mesh.addprop('density', estimate['density'])
density_model = vfilter(1100, 1300, 'density', mesh)
tensor = (gyy, gyz, gzz)
predicted = [dm.predicted for dm in datamods]

log.info("Plotting")
for true, pred in zip(tensor, predicted):
    pyplot.figure()
    pyplot.title("True: color | Inversion: contour")
    pyplot.axis('scaled')
    levels = vis.map.contourf(y*0.001, x*0.001, true, shape, 12)
    pyplot.colorbar()
    vis.map.contour(y*0.001, x*0.001, pred, shape, levels, color='k')
    pyplot.xlabel('Horizontal coordinate y (km)')
    pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()

vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model), style='wireframe')
vis.vtk.prisms(density_model, extract('density', density_model), vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
