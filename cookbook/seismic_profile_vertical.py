"""
Seis: Invert vertical seismic profile (VSP) traveltimes for the velocity of the
layers
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Draw the layered model")
# The limits in velocity and depths
area = (0, 10000, 0, 100)
vmin, vmax, zmin, zmax = area
figure = ft.vis.figure()
ft.vis.xlabel("Velocity (m/s)")
ft.vis.ylabel("Depth (m)")
thickness, velocity = ft.ui.picker.draw_layers(area, figure.gca())

log.info("Generating synthetic data")
zp = numpy.arange(zmin + 0.5, zmax, 0.5)
tts, error = ft.utils.contaminate(
    ft.seis.profile.vertical(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)

log.info("Preparing for the inversion")
damping = 100.
estimates = []
for i in xrange(30):
    p, r = ft.seis.profile.ivertical(ft.utils.contaminate(tts, error), zp,
        thickness, damping=damping)
    estimates.append(1./p)
estimate = ft.utils.vecmean(estimates)
predicted = ft.seis.profile.vertical(thickness, estimate, zp)

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
ft.vis.title("Velocity profile")
for p in estimates:
    ft.vis.layers(thickness, p, '-r', linewidth=2, alpha=0.2)
ft.vis.layers(thickness, estimate, 'o-k', linewidth=2, label='Mean estimate')
ft.vis.layers(thickness, velocity, '--b', linewidth=2, label='True')
ft.vis.ylim(zmax, zmin)
ft.vis.xlim(vmin, vmax)
leg = ft.vis.legend(loc='upper right', numpoints=1)
leg.get_frame().set_alpha(0.5)
ft.vis.xlabel("Velocity (m/s)")
ft.vis.ylabel("Z (m)")
ft.vis.show()
