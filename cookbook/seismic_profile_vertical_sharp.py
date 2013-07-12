"""
Seismic: Invert vertical seismic profile (VSP) traveltimes using sharpness
regularization
"""
import numpy
from fatiando import utils, seismic, vis

thickness = [10, 20, 10, 30, 40, 60]
velocity = [2000, 1000, 5000, 1000, 3000, 6000]
zp = numpy.arange(1, sum(thickness), 1, dtype='f')
tts, error = utils.contaminate(
    seismic.profile.vertical(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)

thick = 10.
mesh = [thick]*int(sum(thickness)/thick)
sharp = 0.00015
beta = 10.**(-12)
estimates = []
for i in xrange(30):
    p, r = seismic.profile.ivertical(utils.contaminate(tts, error),
        zp, mesh, sharp=sharp, beta=beta)
    estimates.append(1./p)
estimate = utils.vecmean(estimates)
predicted = seismic.profile.vertical(mesh, estimate, zp)

vis.mpl.figure(figsize=(12,5))
vis.mpl.subplot(1, 2, 1)
vis.mpl.grid()
vis.mpl.title("Vertical seismic profile")
vis.mpl.plot(tts, zp, 'ok', label='Observed')
vis.mpl.plot(predicted, zp, '-r', linewidth=3, label='Predicted')
vis.mpl.legend(loc='upper right', numpoints=1)
vis.mpl.xlabel("Travel-time (s)")
vis.mpl.ylabel("Z (m)")
vis.mpl.ylim(sum(thickness), 0)
vis.mpl.subplot(1, 2, 2)
vis.mpl.grid()
vis.mpl.title("True velocity + sharp estimate")
for p in estimates:
    vis.mpl.layers(mesh, p, '-r', linewidth=2, alpha=0.2)
vis.mpl.layers(mesh, estimate, '.-k', linewidth=2, label='Mean estimate')
vis.mpl.layers(thickness, velocity, '--b', linewidth=2, label='True')
vis.mpl.ylim(sum(thickness), 0)
vis.mpl.xlim(0, 10000)
vis.mpl.legend(loc='upper right', numpoints=1)
vis.mpl.xlabel("Velocity (m/s)")
vis.mpl.ylabel("Z (m)")
vis.mpl.show()
