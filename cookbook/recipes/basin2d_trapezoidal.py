"""
Gravity inversion for the relief of a 2D trapezoidal basin
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
verts = [(10000, 1.), (90000, 1.), (90000, 7000), (10000, 3330)]
model = Polygon(verts, {'density':-100})
xp = numpy.arange(0., 100000., 1000.)
zp = numpy.zeros_like(xp)
gz = utils.contaminate(talwani.gz(xp, zp, [model]), 0.5)

log.info("Preparing for the inversion")
dm = basin2d.TrapezoidalGzDM(xp, zp, gz, prop=-100, verts=verts[0:2])
solver = levmarq(initial=(9000, 500))
p, residuals = basin2d.trapezoidal([dm], solver)
estimate = Polygon([(10000, 1.), (90000, 1.), (90000, p[0]), (10000, p[1])])

pyplot.figure()
pyplot.subplot(2, 1, 1)
pyplot.title("Gravity anomaly")
pyplot.plot(xp, gz, 'ok', label='Observed')
pyplot.plot(xp, gz - residuals[0], '-r', linewidth=2, label='Predicted')
pyplot.legend(loc='lower left', numpoints=1)
pyplot.ylabel("mGal")
pyplot.xlim(0, 100000)
pyplot.subplot(2, 1, 2)
vis.map.polygon(estimate, 'o-r', linewidth=2, fill='r', alpha=0.3,
                label='Estimated')
vis.map.polygon(model, '--k', linewidth=2, label='True')
pyplot.legend(loc='lower left', numpoints=1)
pyplot.xlabel("X")
pyplot.ylabel("Z")
vis.map.set_area((0, 100000, 10000, -500))
pyplot.show()
