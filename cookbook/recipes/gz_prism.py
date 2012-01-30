"""
Generate synthetic gz data from a 3D prism model.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, gridder, vis, logger
from fatiando.mesher.ddd import Prism, extract

log = logger.get()
log.info(logger.header())
log.info(__doc__)

prisms = [Prism(-4000,-3000,-4000,-3000,0,2000,{'density':1000}),
          Prism(-1000,1000,-1000,1000,0,2000,{'density':-1000}),
          Prism(2000,4000,3000,4000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-100)
gz = potential.prism.gz(xp, yp, zp, prisms)

pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.map.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()

vis.vtk.figure()
vis.vtk.prisms(prisms, extract('density', prisms))
axes = vis.vtk.add_axes(vis.vtk.add_outline())
vis.vtk.wall_bottom(axes.axes.bounds, opacity=0.2)
vis.vtk.wall_north(axes.axes.bounds)
vis.vtk.mlab.show()
