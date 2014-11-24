"""
GravMag: 2D forward modeling with polygons
"""
import numpy
from fatiando import utils, mesher
from fatiando.gravmag import talwani
from fatiando.vis import mpl

# Notice that the last two number are switched.
# This way, the z axis in the plots points down.
area = (-5000, 5000, 5000, 0)
axes = mpl.figure().gca()
mpl.xlabel("X")
mpl.ylabel("Z")
mpl.axis('scaled')
polygons = [mesher.Polygon(mpl.draw_polygon(area, axes),
                           {'density': 500})]
xp = numpy.arange(-4500, 4500, 100)
zp = numpy.zeros_like(xp)
gz = talwani.gz(xp, zp, polygons)

mpl.figure()
mpl.axis('scaled')
mpl.subplot(2, 1, 1)
mpl.title(r"Gravity anomaly produced by the model")
mpl.plot(xp, gz, '-k', linewidth=2)
mpl.ylabel("mGal")
mpl.xlim(-5000, 5000)
mpl.subplot(2, 1, 2)
mpl.polygon(polygons[0], 'o-k', linewidth=2, fill='k', alpha=0.5)
mpl.xlabel("X")
mpl.ylabel("Z")
mpl.set_area(area)
mpl.show()
