"""
Seismic: Invert vertical seismic profile (VSP) traveltimes using smoothness
regularization and unknown layer thicknesses
"""
import numpy
from fatiando import utils
from fatiando.seismic.profile import layered_straight_ray, LayeredStraight
from fatiando.inversion import Smoothness1D
from fatiando.vis import mpl

# Make a layered model
thickness = [10, 20, 10, 30, 40, 60]
velocity = [2000, 1000, 5000, 1000, 3000, 6000]
zp = numpy.arange(1, sum(thickness), 1, dtype='f')
# Produce some noise-corrupted synthetic data
tts, error = utils.contaminate(
    layered_straight_ray(thickness, velocity, zp),
    0.02, percent=True, return_stddev=True)
# Assume that the thicknesses are unknown. In this case, use a mesh of many
# thin layers and invert for each slowness
thick = 10.
mesh = [thick] * int(sum(thickness) / thick)
solver = (LayeredStraight(tts, zp, mesh) +
          5 * Smoothness1D(len(mesh))).fit()
velocity_ = solver.estimate_

mpl.figure(figsize=(12, 5))
mpl.subplot(1, 2, 1)
mpl.grid()
mpl.title("Vertical seismic profile")
mpl.plot(tts, zp, 'ok', label='Observed')
mpl.plot(solver[0].predicted(), zp, '-r', linewidth=3, label='Predicted')
mpl.legend(loc='upper right', numpoints=1)
mpl.xlabel("Travel-time (s)")
mpl.ylabel("Z (m)")
mpl.ylim(sum(mesh), 0)
mpl.subplot(1, 2, 2)
mpl.grid()
mpl.title("True velocity + smooth estimate")
mpl.layers(mesh, velocity_, '.-k', linewidth=2, label='Estimated')
mpl.layers(thickness, velocity, '--b', linewidth=2, label='True')
mpl.ylim(sum(mesh), 0)
mpl.xlim(0, 10000)
mpl.legend(loc='upper right', numpoints=1)
mpl.xlabel("Velocity (m/s)")
mpl.ylabel("Z (m)")
mpl.show()
