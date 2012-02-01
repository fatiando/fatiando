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
thickness = [10, 20, 5, 30, 60]
velocity = [2000, 1000, 4000, 1500, 6000]
zp = numpy.arange(0.5, sum(thickness), 0.5)
tts, error = utils.contaminate(profile.vertical(thickness, velocity, zp), 0.02,
                               percent=True, return_stddev=True)

log.info("Preparing for the inversion")
solver = linear.overdet(len(thickness))
damping = 1000.
estimates = []
for i in xrange(30):    
    p, r = profile.invert_vertical(utils.contaminate(tts, error), zp, thickness,
                                   solver, damping=damping)
    estimates.append(1./p)
estimate = utils.vecmean(estimates)
predicted = profile.vertical(thickness, estimate, zp)

log.info("Plotting results...")
pyplot.figure(figsize=(12,5))
pyplot.subplot(1, 2, 1)
pyplot.grid()
pyplot.title("Vertical seismic profile")
pyplot.plot(tts, zp, 'ok', label='Observed')
pyplot.plot(predicted, zp, '-r', linewidth=3, label='Predicted')
pyplot.legend(loc='upper right', numpoints=1)
pyplot.xlabel("Travel-time (s)")
pyplot.ylabel("Z (m)")
pyplot.ylim(sum(thickness), 0)
pyplot.subplot(1, 2, 2)
pyplot.grid()
pyplot.title("Velocity profile")
for p in estimates:    
    vis.map.layers(thickness, p, '-r', linewidth=2, alpha=0.2)
vis.map.layers(thickness, estimate, 'o-r', linewidth=2, label='Mean estimate')
vis.map.layers(thickness, velocity, '--b', linewidth=2, label='True')
pyplot.ylim(sum(thickness), 0)
pyplot.xlim(0, 10000)
pyplot.legend(loc='upper right', numpoints=1)
pyplot.xlabel("Velocity (m/s)")
pyplot.ylabel("Z (m)")
pyplot.show()
