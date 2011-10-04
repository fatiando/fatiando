"""
Create synthetic data from a prism with polygonal crossection.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, mesher, gridder, vis, logger
from fatiando.mesher.prism import PolygonalPrism3D, draw_polygon

log = logger.get()
log.info(logger.header())
log.info("Example of direct modelling using 3D polygonal prisms")
log.info("\nDRAW 3 POLYGONS, ONE AT A TIME.")
log.info("DEPTHS TO THE TOP: 0, 1 and 2km")
log.info("DENSITY CONTRASTS: 0.5, 1 and 1.5 g/cm3\n")

area = (-5000, 5000, -5000, 5000)

axes = pyplot.figure().add_subplot(1,1,1)
prism1 = PolygonalPrism3D(draw_polygon(area, axes),0,1000,{'density':500})

axes = pyplot.figure().add_subplot(1,1,1)
vis.polyprism_contours([prism1])
prism2 = PolygonalPrism3D(draw_polygon(area, axes),1000,2000,{'density':1000})

axes = pyplot.figure().add_subplot(1,1,1)
vis.polyprism_contours([prism1, prism2])
prism3 = PolygonalPrism3D(draw_polygon(area, axes),2000,3000,{'density':1500})

prisms = [prism1, prism2, prism3]
shape = (100,100)
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = potential.polyprism.gz(xp, yp, zp, prisms)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
vis.polyprism_contours(prisms, ['.-k', '.-b', '.-g'], ['0', '1km', '2km'])
pyplot.legend()
pyplot.show()
