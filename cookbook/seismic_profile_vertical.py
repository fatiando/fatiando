"""
Seismic: Invert vertical seismic profile (VSP) traveltimes for the velocity of the
layers
"""
import numpy
from fatiando import logger, utils, seismic, vis

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Draw the layered model")
# The limits in velocity and depths
area = (0, 10000, 0, 100)
vmin, vmax, zmin, zmax = area
figure = vis.mpl.figure()
vis.mpl.xlabel("Velocity (m/s)")
vis.mpl.ylabel("Depth (m)")
thickness, velocity = vis.mpl.draw_layers(area, figure.gca())

log.info("Generating synthetic data")
zp = numpy.arange(zmin + 0.5, zmax, 0.5)
tts, error = utils.contaminate(
    seismic.profile.vertical(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)

log.info("Preparing for the inversion")
damping = 100.
estimates = []
for i in xrange(30):
    p, r = seismic.profile.ivertical(utils.contaminate(tts, error), zp,
        thickness, damping=damping)
    estimates.append(1./p)
estimate = utils.vecmean(estimates)
predicted = seismic.profile.vertical(thickness, estimate, zp)

log.info("Plotting results...")
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
vis.mpl.title("Velocity profile")
for p in estimates:
    vis.mpl.layers(thickness, p, '-r', linewidth=2, alpha=0.2)
vis.mpl.layers(thickness, estimate, 'o-k', linewidth=2, label='Mean estimate')
vis.mpl.layers(thickness, velocity, '--b', linewidth=2, label='True')
vis.mpl.ylim(zmax, zmin)
vis.mpl.xlim(vmin, vmax)
leg = vis.mpl.legend(loc='upper right', numpoints=1)
leg.get_frame().set_alpha(0.5)
vis.mpl.xlabel("Velocity (m/s)")
vis.mpl.ylabel("Z (m)")
vis.mpl.show()
