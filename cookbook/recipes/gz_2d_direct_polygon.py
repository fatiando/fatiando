"""
2D direct gravity modeling with polygons
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, vis, logger
from fatiando.ui import picker
from fatiando.mesher.dd import Polygon

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Notice that the last two number are switched.
# This way, the z axis in the plots points down.
area = (-5000, 5000, 5000, 0)
axes = pyplot.figure().add_subplot(1,1,1)
pyplot.xlabel("X")
pyplot.ylabel("Z")
pyplot.axis('scaled')
polygons = [Polygon(picker.draw_polygon(area, axes, width=2), {'density':500})]
xp = numpy.arange(-4500, 4500, 100)
zp = numpy.zeros_like(xp)
gz = potential.talwani.gz(xp, zp, polygons)

pyplot.figure()
pyplot.axis('scaled')
pyplot.subplot(2,1,1)
pyplot.title(r"Gravity anomaly produced by the model")
pyplot.plot(xp, gz, '-k', linewidth=2)
pyplot.ylabel("mGal")
pyplot.xlim(-5000, 5000)
pyplot.subplot(2,1,2)
vis.map.polygon(polygons[0], 'o-k', linewidth=2)
pyplot.xlabel("X")
pyplot.ylabel("Z")
vis.map.set_area(area)
pyplot.show()
