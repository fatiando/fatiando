"""
Seismic: Invert vertical seismic profile (VSP) traveltimes for the velocity
of a layered model.
"""
import numpy
from fatiando import utils
from fatiando.seismic.profile import layered_straight_ray, LayeredStraight
from fatiando.inversion.regularization import Damping
from fatiando.vis import mpl

# The limits in velocity and depths, respectively
area = (0, 10000, 0, 100)
vmin, vmax, zmin, zmax = area
# Use the interactive functions of mpl to draw a layered model
figure = mpl.figure()
mpl.xlabel("Velocity (m/s)")
mpl.ylabel("Depth (m)")
thickness, velocity = mpl.draw_layers(area, figure.gca())
# Make some synthetic noise-corrupted travel-time data
zp = numpy.arange(zmin + 0.5, zmax, 0.5)
tts, error = utils.contaminate(
    layered_straight_ray(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)
# Make the solver and run the inversion using damping regularization
# (assumes known thicknesses of the layers)
solver = (LayeredStraight(tts, zp, thickness) +
          0.1 * Damping(len(thickness))).fit()
velocity_ = solver.estimate_

# Plot the results
mpl.figure(figsize=(12, 5))
mpl.subplot(1, 2, 1)
mpl.grid()
mpl.title("Vertical seismic profile")
mpl.plot(tts, zp, 'ok', label='Observed')
mpl.plot(solver[0].predicted(), zp, '-r', linewidth=3, label='Predicted')
mpl.legend(loc='upper right', numpoints=1)
mpl.xlabel("Travel-time (s)")
mpl.ylabel("Z (m)")
mpl.ylim(sum(thickness), 0)
mpl.subplot(1, 2, 2)
mpl.grid()
mpl.title("Velocity profile")
mpl.layers(thickness, velocity_, 'o-k', linewidth=2, label='Estimated')
mpl.layers(thickness, velocity, '--b', linewidth=2, label='True')
mpl.ylim(zmax, zmin)
mpl.xlim(vmin, vmax)
leg = mpl.legend(loc='upper right', numpoints=1)
leg.get_frame().set_alpha(0.5)
mpl.xlabel("Velocity (m/s)")
mpl.ylabel("Z (m)")
mpl.show()
