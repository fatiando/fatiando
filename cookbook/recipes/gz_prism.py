"""
Generate synthetic gz data from a prism model.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, gridder, vis, logger
from fatiando.mesher.ddd import Prism, extract

log = logger.get()
log.info(logger.header())
log.info("Example of direct modelling using right rectangular prisms")

prisms = [Prism(-4000,-3000,-4000,-3000,0,2000,{'density':1000}),
          Prism(-1000,1000,-1000,1000,0,2000,{'density':-1000}),
          Prism(2000,4000,3000,4000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-100)
gz = potential.prism.gz(xp, yp, zp, prisms)

pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()

vis.mayavi_figure()
plot = vis.prisms3D(prisms, extract('density', prisms))
vis.add_outline3d()
axes = vis.add_axes3d(plot)
vis.wall_bottom(axes.axes.bounds, opacity=0.2)
vis.wall_north(axes.axes.bounds)
vis.mlab.show()
