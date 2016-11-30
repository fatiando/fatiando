"""
Forward modeling magnetic data using spheres in Cartesian coordinates
-----------------------------------------------------------------------

The :mod:`fatiando.gravmag` has many functions for forward modeling gravity and
magnetic data. Here we'll show how to build a model out of spheres and
calculate the total field magnetic anomaly and the 3 components of the magnetic
induction.
"""
from __future__ import division, print_function
from fatiando import mesher, gridder, utils
from fatiando.gravmag import sphere
import matplotlib.pyplot as plt
import numpy as np

# Create a model using geometric objects from fatiando.mesher
# Each model element has a dictionary with its physical properties.
# The spheres have different total magnetization vectors (total = induced +
# remanent + any other effects). Notice that the magnetization has to be a
# vector. Function utils.ang2vec converts intensity, inclination, and
# declination into a 3 component vector for easier handling.
model = [
    mesher.Sphere(x=10e3, y=10e3, z=2e3, radius=1.5e3,
                  props={'magnetization': utils.ang2vec(1, inc=50, dec=-30)}),
    mesher.Sphere(x=20e3, y=20e3, z=2e3, radius=1.5e3,
                  props={'magnetization': utils.ang2vec(1, inc=-70, dec=30)})]

# Set the inclination and declination of the geomagnetic field.
inc, dec = -10, 13
# Create a regular grid at a constant height
shape = (300, 300)
area = [0, 30e3, 0, 30e3]
x, y, z = gridder.regular(area, shape, z=-10)

fields = [
    ['Total field Anomaly (nt)', sphere.tf(x, y, z, model, inc, dec)],
    ['Bx (nT)', sphere.bx(x, y, z, model)],
    ['By (nT)', sphere.by(x, y, z, model)],
    ['Bz (nT)', sphere.bz(x, y, z, model)],
]

# Make maps of all fields calculated
fig = plt.figure(figsize=(7, 6))
plt.rcParams['font.size'] = 10
X, Y = x.reshape(shape)/1000, y.reshape(shape)/1000
for i, tmp in enumerate(fields):
    ax = plt.subplot(2, 2, i + 1)
    field, data = tmp
    scale = np.abs([data.min(), data.max()]).max()
    ax.set_title(field)
    plot = ax.pcolormesh(Y, X, data.reshape(shape), cmap='RdBu_r',
                         vmin=-scale, vmax=scale)
    plt.colorbar(plot, ax=ax, aspect=30, pad=0)
    ax.set_xlabel('y (km)')
    ax.set_ylabel('x (km)')
plt.tight_layout(pad=0.5)
plt.show()
