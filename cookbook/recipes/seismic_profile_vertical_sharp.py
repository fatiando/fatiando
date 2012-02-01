"""
Simulate the vertical seismic profile and invert for the velocity of a series
of layers with thicknesses that don't match the true ones. Regularizes the
problem using sharpness (total variation).
"""
from matplotlib import pyplot
import numpy
from fatiando.seismic import profile
from fatiando.inversion import gradient
from fatiando import logger, vis, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
thickness = [10, 20, 10, 30, 40, 60]
velocity = [2000, 1000, 5000, 1000, 3000, 6000]
zp = numpy.arange(1, sum(thickness), 1, dtype='f')
tts, error = utils.contaminate(profile.vertical(thickness, velocity, zp), 0.02,
                               percent=True, return_stddev=True)

log.info("Preparing for the inversion using 5 m thick layers")
thick = 10.
mesh = [thick]*int(sum(thickness)/thick)
sharp = 0.00015
beta = 10.**(-12)
estimates = []
for i in xrange(30):    
    solver = gradient.levmarq(initial=0.00001*numpy.ones(len(mesh)))
    p, r = profile.invert_vertical(utils.contaminate(tts, error), zp, mesh,
           solver, sharp=sharp, beta=beta)
    estimates.append(1./p)
estimate = utils.vecmean(estimates)
predicted = profile.vertical(mesh, estimate, zp)

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
pyplot.title("True velocity + sharp estimate")
for p in estimates:    
    vis.map.layers(mesh, p, '-r', linewidth=2, alpha=0.2)
vis.map.layers(mesh, estimate, '.-k', linewidth=2, label='Mean estimate')
vis.map.layers(thickness, velocity, '--b', linewidth=2, label='True')
pyplot.ylim(sum(thickness), 0)
pyplot.xlim(0, 10000)
pyplot.legend(loc='upper right', numpoints=1)
pyplot.xlabel("Velocity (m/s)")
pyplot.ylabel("Z (m)")
pyplot.show()
