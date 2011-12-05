"""
Calculating gz cause by a topographic model using prisms
"""
from matplotlib import pyplot
from fatiando import utils, gridder, logger, vis, potential
from fatiando.mesher.volume import PrismRelief3D

log = logger.get()
log.info(logger.header())

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (30, 15)
x, y = gridder.regular(area, shape)
height = (-80*utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          200*utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
nodes = (x, y, -1*height)
relief = PrismRelief3D(0, gridder.spacing(area,shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

log.info("Calculating gz effect")
gridarea = (-80, 80, -220, 220)
xp, yp, zp = gridder.regular(gridarea, shape, z=-200)
gz = potential.prism.gz(xp, yp, zp, relief)

log.info("Plotting")
pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.pcolor(x, y, height, shape)
pyplot.colorbar()
vis.square(gridarea, label='Computation grid')
pyplot.legend()
pyplot.subplot(1,2,2)
pyplot.title("Topographic gz effect")
pyplot.axis('scaled')
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()

vis.mayavi_figure()
plot = vis.prisms3D(relief, relief.props['density'])
vis.add_outline3d()
axes = vis.add_axes3d(plot)
vis.wall_bottom(axes.axes.bounds, opacity=0.2)
vis.wall_north(axes.axes.bounds)
vis.mlab.show()
