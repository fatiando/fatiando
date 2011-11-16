"""
Example of inverting synthetic gz data from a single prism using harvester
"""
try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab
from matplotlib import pyplot
from fatiando import potential, vis, logger, utils, gridder
from fatiando.mesher.volume import Prism3D, PrismMesh3D, extract
from fatiando.inversion import harvester

vis.mlab = mlab

log = logger.get()
log.info(__doc__)

log.info("First make the synthetic model:\n")
extent = (0, 10000, 0, 10000, 0, 6000)
model = [Prism3D(4000, 6000, 2000, 8000, 2000, 4000, props={'density':800})]

mlab.figure(bgcolor=(1,1,1))
vis.prisms3D(model, extract('density', model))
outline = mlab.outline(color=(0,0,0), extent=extent)
vis.add_axes3d(outline)
vis.wall_bottom(extent)
vis.wall_north(extent)
mlab.show()

log.info("\nSecond calculate the synthetic data:")
shape = (100,100)
x, y, z = gridder.scatter(extent[0:4], 200, z=-1)
gz = utils.contaminate(potential.prism.gz(x, y, z, model), 0.1)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title('Synthetic gz data')
levels = vis.contourf(y, x, gz, shape, 10, interpolate=True)
vis.contour(y, x, gz, shape, levels, interpolate=True)
pyplot.plot(y, x, 'xk')
pyplot.xlabel('East (km)')
pyplot.ylabel('North (km)')
pyplot.show()

log.info("\nThird make a prism mesh:")
mesh = PrismMesh3D(extent, (50, 50, 30))

mlab.figure(bgcolor=(1,1,1))
vis.prisms3D(mesh, (0 for i in xrange(mesh.size)))
outline = mlab.outline(color=(0,0,0), extent=extent)
vis.add_axes3d(outline)
mlab.show()

log.info("\nFourth sow the seeds:")
seeds = [harvester.sow(mesh, (5000, 5000, 3000), {'density':800})]

log.info("\nFith harvest the results:")
datamods = [harvester.GzModule(x, y, z, gz)]
mesh = harvester.harvest(seeds, mesh, datamods)

log.info("\nSixth plot things:")
