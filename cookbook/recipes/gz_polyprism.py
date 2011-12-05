"""
Create synthetic data from a prism with polygonal crossection.
"""
from matplotlib import pyplot
from fatiando import potential, mesher, gridder, vis, logger
from fatiando.mesher.volume import PolygonalPrism3D, draw_polygon

log = logger.get()
log.info(logger.header())
log.info("Example of direct modelling using 3D polygonal prisms")

area = (-5000, 5000, -5000, 5000)

axes = pyplot.figure().add_subplot(1,1,1)
prism1 = PolygonalPrism3D(draw_polygon(area, axes),0,1000,{'density':500})

prisms = [prism1]
shape = (100,100)
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = potential.polyprism.gz(xp, yp, zp, prisms)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
vis.polyprism_contours(prisms, ['.-k'], ['z1=0km'])
pyplot.legend()
pyplot.show()
