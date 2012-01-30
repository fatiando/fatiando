"""
Example of generating a 3D prism model of the topography
"""
import numpy
from matplotlib import pyplot
from fatiando import utils, gridder, logger, vis
from fatiando.mesher.ddd import PrismRelief

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (100, 50)
x, y = gridder.regular(area, shape)
height = (-80*utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
nodes = (x, y, -1*height) # -1 is to convert height to z coordinate
reference = 0 # z coordinate of the reference surface
relief = PrismRelief(reference, gridder.spacing(area, shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

log.info("Plotting")
vis.vtk.figure()
vis.vtk.prisms(relief, relief.props['density'], edges=False)
axes = vis.vtk.add_axes(vis.vtk.add_outline())
vis.vtk.wall_bottom(axes.axes.bounds, opacity=0.2)
vis.vtk.wall_north(axes.axes.bounds)
vis.vtk.mlab.show()
