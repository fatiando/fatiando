"""
Create synthetic data from a prism with polygonal crossection.
"""
from matplotlib import pyplot
import numpy
from fatiando import potential, mesher, gridder, vis, logger

log = logger.get()
log.info(logger.header())
log.info("Example of direct modelling using right rectangular prisms")

prisms = [{'density':1000,'top':0,'bottom':2000,'x':[-1000,-1000,1000,1000],'y':[-1000,1000,1000,-1000]}]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-100)
gz = potential.polyprism.gz(xp, yp, zp, prisms)

pyplot.axis('scaled')
pyplot.title("gz produced by prism model (mGal)")
vis.pcolor(xp, yp, gz, shape)
pyplot.colorbar()
pyplot.show()
