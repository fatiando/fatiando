"""
GravMag: Simple gravity inversion for the relief of a 2D trapezoidal basin
"""
import numpy
from fatiando import logger, utils, mesher, gravmag, inversion
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
verts = [(10000, 1.), (90000, 1.), (90000, 7000), (10000, 3330)]
model = mesher.Polygon(verts, {'density':-100})
xp = numpy.arange(0., 100000., 1000.)
zp = numpy.zeros_like(xp)
gz = utils.contaminate(gravmag.talwani.gz(xp, zp, [model]), 0.5)

log.info("Preparing for the inversion")
solver = inversion.gradient.levmarq(initial=(9000, 500))
estimate, residuals = gravmag.basin2d.trapezoidal(xp, zp, gz, verts[0:2], -100,
    solver)

mpl.figure()
mpl.subplot(2, 1, 1)
mpl.title("Gravity anomaly")
mpl.plot(xp, gz, 'ok', label='Observed')
mpl.plot(xp, gz - residuals, '-r', linewidth=2, label='Predicted')
mpl.legend(loc='lower left', numpoints=1)
mpl.ylabel("mGal")
mpl.xlim(0, 100000)
mpl.subplot(2, 1, 2)
mpl.polygon(estimate, 'o-r', linewidth=2, fill='r', alpha=0.3,
                label='Estimated')
mpl.polygon(model, '--k', linewidth=2, label='True')
mpl.legend(loc='lower left', numpoints=1)
mpl.xlabel("X")
mpl.ylabel("Z")
mpl.set_area((0, 100000, 10000, -500))
mpl.show()
