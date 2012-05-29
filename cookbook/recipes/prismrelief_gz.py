"""
Calculate the gravity anomaly caused by a topographic model using prisms
"""
from matplotlib import pyplot
from fatiando import utils, gridder, logger, vis, potential
from fatiando.mesher.ddd import PrismRelief

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (30, 15)
x, y = gridder.regular(area, shape)
height = (-80*utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          200*utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
nodes = (x, y, -1*height)
relief = PrismRelief(0, gridder.spacing(area,shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

log.info("Calculating gz effect")
gridarea = (-80, 80, -220, 220)
gridshape = (100, 100)
xp, yp, zp = gridder.regular(gridarea, gridshape, z=-200)
gz = potential.prism.gz(xp, yp, zp, relief)

log.info("Plotting")
pyplot.figure(figsize=(10,7))
pyplot.subplot(1, 2, 1)
pyplot.title("Synthetic topography")
pyplot.axis('scaled')
vis.map.pcolor(x, y, height, shape)
cb = pyplot.colorbar()
cb.set_label("meters")
vis.map.square(gridarea, label='Computation grid')
pyplot.legend()
pyplot.subplot(1, 2, 2)
pyplot.title("Topographic effect")
pyplot.axis('scaled')
vis.map.pcolor(xp, yp, gz, gridshape)
cb = pyplot.colorbar()
cb.set_label("mGal")
pyplot.show()

vis.vtk.figure()
vis.vtk.prisms(relief, prop='density')
axes = vis.vtk.add_axes(vis.vtk.add_outline())
vis.vtk.wall_bottom(axes.axes.bounds, opacity=0.2)
vis.vtk.wall_north(axes.axes.bounds)
vis.vtk.mlab.show()
