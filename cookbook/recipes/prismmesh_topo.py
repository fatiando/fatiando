"""
Make and plot a 3D prism mesh with topography.
"""
from matplotlib import pyplot
from fatiando import utils, gridder, logger, vis
from fatiando.mesher.ddd import PrismMesh

log = logger.get()
log.info(logger.header())
log.info(__doc__)

x1, x2 = -100, 100
y1, y2 = -200, 200
bounds = (x1, x2, y1, y2, -200, 0)

log.info("Generating synthetic topography")
x, y = gridder.regular((x1, x2, y1, y2), (50,50))
height = (100 +
          -50*utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the prism mesh")
mesh = PrismMesh(bounds, (20,40,20))
mesh.carvetopo(x, y, height)

log.info("Plotting")
vis.vtk.figure()
vis.vtk.prisms(mesh, (0 for i in xrange(mesh.size)))
vis.vtk.add_axes(vis.vtk.add_outline(bounds), fmt='%.0f')
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
