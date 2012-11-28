"""
seismic: Invert vertical seismic profile (VSP) traveltimes using sharpness
regularization
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
thickness = [10, 20, 10, 30, 40, 60]
velocity = [2000, 1000, 5000, 1000, 3000, 6000]
zp = numpy.arange(1, sum(thickness), 1, dtype='f')
tts, error = ft.utils.contaminate(
    ft.seismic.profile.vertical(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)

log.info("Preparing for the inversion using 5 m thick layers")
thick = 10.
mesh = [thick]*int(sum(thickness)/thick)
sharp = 0.00015
beta = 10.**(-12)
estimates = []
for i in xrange(30):
    p, r = ft.seismic.profile.ivertical(ft.utils.contaminate(tts, error),
        zp, mesh, sharp=sharp, beta=beta)
    estimates.append(1./p)
estimate = ft.utils.vecmean(estimates)
predicted = ft.seismic.profile.vertical(mesh, estimate, zp)

log.info("Plotting results...")
ft.vis.figure(figsize=(12,5))
ft.vis.subplot(1, 2, 1)
ft.vis.grid()
ft.vis.title("Vertical seismic profile")
ft.vis.plot(tts, zp, 'ok', label='Observed')
ft.vis.plot(predicted, zp, '-r', linewidth=3, label='Predicted')
ft.vis.legend(loc='upper right', numpoints=1)
ft.vis.xlabel("Travel-time (s)")
ft.vis.ylabel("Z (m)")
ft.vis.ylim(sum(thickness), 0)
ft.vis.subplot(1, 2, 2)
ft.vis.grid()
ft.vis.title("True velocity + sharp estimate")
for p in estimates:
    ft.vis.layers(mesh, p, '-r', linewidth=2, alpha=0.2)
ft.vis.layers(mesh, estimate, '.-k', linewidth=2, label='Mean estimate')
ft.vis.layers(thickness, velocity, '--b', linewidth=2, label='True')
ft.vis.ylim(sum(thickness), 0)
ft.vis.xlim(0, 10000)
ft.vis.legend(loc='upper right', numpoints=1)
ft.vis.xlabel("Velocity (m/s)")
ft.vis.ylabel("Z (m)")
ft.vis.show()
