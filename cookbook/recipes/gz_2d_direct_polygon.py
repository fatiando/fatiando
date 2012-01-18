"""
2D direct gravity modeling with polygons
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, vis, logger
from fatiando.mesher.ddd import draw_polygon
from fatiando.mesher.dd import Polygon

log = logger.get()
log.info(logger.header())
log.info("Example of 2D direct modelling using polygons")

# Notice that the last two number are switched.
# This way, the z axis in the plots points down.
area = (-5000, 5000, 5000, 0)
axes = pyplot.figure().add_subplot(1,1,1)
pyplot.xlabel("x (m)")
pyplot.ylabel("z (m)")
polygons = [Polygon(draw_polygon(area, axes), {'density':500})]
xp = numpy.arange(-4500, 4500, 100)
zp = numpy.zeros_like(xp)
gz = potential.talwani.gz(xp, zp, polygons)

pyplot.figure()
pyplot.axis('scaled')
pyplot.subplot(2,1,1)
pyplot.title(r"$g_z$ produced by model", fontsize=18)
pyplot.plot(xp, gz, '.-k')
pyplot.ylabel(r"$g_z$ (mGal)")
pyplot.xlim(-5000, 5000)
pyplot.subplot(2,1,2)
pyplot.title("Synthetic model", fontsize=18)
vis.polyprism_contours(polygons, ['.-k'])
pyplot.xlabel("x (m)")
pyplot.ylabel("z (m)")
pyplot.xlim(-5000, 5000)
pyplot.ylim(5000, 0)
pyplot.show()
