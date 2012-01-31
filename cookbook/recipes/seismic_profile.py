"""
Simulate the vertical seismic profile and invert for the velocity of the layers.
"""
from matplotlib import pyplot
import numpy
from fatiando.seismic import profile
from fatiando.inversion import linear
from fatiando import logger, vis, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
thickness = [10, 20, 5, 30, 8]
velocity = [2000, 1000, 10000, 4000, 15000]
zp = numpy.arange(1, sum(thickness), 1)
tts, error = utils.contaminate(profile.vertical(thickness, velocity, zp), 0.02,
                               percent=True, return_stddev=True)

log.info("Preparing for the inversion")
solver = linear.overdet(len(thickness))
p, residuals = profile.invert_vertical(tts, zp, thickness, solver, damping=0.01)

print 1./p

pyplot.figure(figsize=(12,5))
#pyplot.subplot(1, 2, 1)
pyplot.title("Climate signal (abrupt)")
pyplot.plot(tts, zp, 'ok', label='Observed')
pyplot.plot(tts - residuals, zp, '--r', linewidth=3, label='Predicted')
pyplot.legend(loc='upper right', numpoints=1)
pyplot.xlabel("Travel-time (s)")
pyplot.ylabel("Z")
pyplot.ylim(sum(thickness), 0)
pyplot.show()
