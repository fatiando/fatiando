"""
Create synthetic data from a prism with polygonal crossection.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, mesher, gridder, vis, logger
from fatiando.mesher.prism import PolygonalPrism3D, draw_polygon
log = logger.get()
log.info(logger.header())
log.info("Example of direct modelling using right rectangular prisms")

area = (-5000, 5000, -5000, 5000)
# Set up the figure for picking
fig = pyplot.figure()
axes = fig.add_subplot(1,1,1)
pyplot.axis('scaled')
axes.set_xlim(area[0], area[1])
axes.set_ylim(area[2], area[3])
prisms = [PolygonalPrism3D(draw_polygon(area, axes),0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = gridder.regular(area, shape, z=-100)
gz = potential.polyprism.gz(xp, yp, zp, prisms)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
vis.polyprism_contours(prisms, ['.-k', '.-b'], ['z=0', 'z=2km'])
pyplot.legend()
pyplot.show()
