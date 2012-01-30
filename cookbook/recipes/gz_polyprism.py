"""
Create synthetic data from a 3D prism with polygonal horizontal crossection.
"""
from matplotlib import pyplot
from fatiando import potential, mesher, gridder, vis, logger
from fatiando.mesher.ddd import PolygonalPrism, draw_polygon

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (-5000, 5000, -5000, 5000)

axes = pyplot.figure().add_subplot(1,1,1)
prisms = [PolygonalPrism(draw_polygon(area, axes),0,1000,{'density':500})]
shape = (100,100)
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = potential.polyprism.gz(xp, yp, zp, prisms)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.map.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
vis.map.polygon(prisms[0], '.-k', label='z1=0km')
pyplot.legend()
pyplot.show()
