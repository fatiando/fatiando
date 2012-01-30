"""
Gravity inversion for the relief of a 2D triangular basin
"""
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import Polygon
from fatiando.potential import talwani, basin2d
from fatiando.inversion.gradient import levmarq
from fatiando import logger, vis, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
verts = [(10000, 1.), (90000, 1.), (80000, 5000)]
left, middle, right = verts
model = Polygon(verts, {'density':-100})
xp = numpy.arange(0., 100000., 1000.)
zp = numpy.zeros_like(xp)
gz = utils.contaminate(talwani.gz(xp, zp, [model]), 1)

log.info("Preparing for the inversion")
dm = basin2d.TriangularGzDM(xp, zp, gz, prop=-100, verts=[left, middle])
solver = levmarq(initial=(10000, 1000))
p, residuals = basin2d.triangular([dm], solver)
estimate = Polygon([left, middle, p])

pyplot.figure()
pyplot.subplot(2, 1, 1)
pyplot.title("Gravity anomaly")
pyplot.plot(xp, gz, 'ok', label='Observed')
pyplot.plot(xp, gz - residuals[0], '--r', linewidth=2, label='Predicted')
pyplot.legend(loc='lower left')
pyplot.ylabel("mGal")
pyplot.xlim(0, 100000)
pyplot.subplot(2, 1, 2)
vis.map.polygon(estimate, 'o-r', linewidth=2, label='Estimated')
vis.map.polygon(model, '--k', linewidth=2, label='True')
pyplot.legend(loc='lower left', numpoints=1)
pyplot.xlabel("X")
pyplot.ylabel("Z")
vis.map.set_area((0, 100000, 10000, -500))
pyplot.show()
