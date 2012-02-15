"""
Example of gravity gradiometry 3D inversion using harvester.
"""
import numpy
from matplotlib import pyplot
from fatiando import potential, logger, gridder, utils, vis
from fatiando.mesher.volume import Prism3D, PrismMesh3D, extract, vfilter
from fatiando.inversion import harvester

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
bounds = [0, 5000, 0, 5000, 0, 1500]
model = [Prism3D(500, 4500, 3000, 3500, 200, 700, {'density':1200}),
         Prism3D(3000, 4500, 1800, 2300, 200, 700, {'density':1200}),
         Prism3D(500, 1500, 500, 1500, 0, 800, {'density':600}),
         Prism3D(0, 800, 1800, 2300, 0, 200, {'density':600}),
         Prism3D(4000, 4800, 100, 900, 0, 300, {'density':600}),
         Prism3D(0, 2000, 4500, 5000, 0, 200, {'density':600}),              
         Prism3D(3000, 4200, 2500, 2800, 200, 700, {'density':-1000}),
         Prism3D(300, 2500, 1800, 2700, 500, 1000, {'density':-1000}),
         Prism3D(4000, 4500, 500, 1500, 400, 1000, {'density':-1000}),
         Prism3D(1800, 3700, 500, 1500, 300, 1300, {'density':-1000}),
         Prism3D(500, 4500, 4000, 4500, 400, 1300, {'density':-1000})]
shape = (51, 51)
area = bounds[0:4]
noise = 2
x, y, z = gridder.regular(area, shape, z=-150)
gyy = utils.contaminate(potential.prism.gyy(x, y, z, model), noise)
gyz = utils.contaminate(potential.prism.gyz(x, y, z, model), noise)
gzz = utils.contaminate(potential.prism.gzz(x, y, z, model), noise)

log.info("Setting up the inversion")
datamods = [harvester.PrismGyyModule(x, y, z, gyy, norm=1),
            harvester.PrismGyzModule(x, y, z, gyz, norm=1),
            harvester.PrismGzzModule(x, y, z, gzz, norm=1)]
mesh = PrismMesh3D(bounds, (15, 50, 50))
dens = {'density':1200}
seeds = harvester.sow(mesh, [((800, 3250, 600), dens),
                             ((1200, 3250, 600), dens),
                             ((1700, 3250, 600), dens),
                             ((2100, 3250, 600), dens),
                             ((2500, 3250, 600), dens),
                             ((2900, 3250, 600), dens),
                             ((3300, 3250, 600), dens),
                             ((3700, 3250, 600), dens),
                             ((4200, 3250, 600), dens),
                             ((3300, 2050, 600), dens),
                             ((3600, 2050, 600), dens),
                             ((4000, 2050, 600), dens),
                             ((4300, 2050, 600), dens)])
regul = harvester.ConcentrationRegularizer(seeds, mesh, 0.1, 1)
jury = harvester.standard_jury(regul, thresh=0.0001)

log.info("Run the inversion and collect the results")
results, goals = harvester.harvest(seeds, mesh, datamods, jury)
for prop in results['estimate']:
    mesh.addprop(prop, results['estimate'][prop])
density_model = vfilter(1100, 1200, 'density', mesh)
seeds = [mesh[s['index']] for s in seeds]
tensor = (gyy, gyz, gzz)
predicted = [dm.predicted for dm in datamods]

log.info("Plotting")
shape = (51, 51)
for true, pred in zip(tensor, predicted):
    pyplot.figure()
    pyplot.title("True: color | Inversion: contour")
    pyplot.axis('scaled')
    levels = vis.contourf(y*0.001, x*0.001, true, shape, 12)
    pyplot.colorbar()
    vis.contour(y*0.001, x*0.001, pred, shape, levels, color='k')
    pyplot.xlabel('Horizontal coordinate y (km)')
    pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model))
vis.add_axes3d(vis.add_outline3d(bounds), ranges=[i*0.001 for i in bounds],
    fmt='%.1f', nlabels=6)
vis.wall_bottom(bounds)
vis.wall_north(bounds)

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), style='wireframe')
vis.prisms3D(seeds, extract('density', seeds), vmin=0)
vis.prisms3D(density_model, extract('density', density_model), vmin=0)
vis.add_axes3d(vis.add_outline3d(bounds), ranges=[i*0.001 for i in bounds],
    fmt='%.1f', nlabels=6)
vis.wall_bottom(bounds)
vis.wall_north(bounds)
vis.mlab.show()
