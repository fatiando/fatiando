"""
Forward modeling gravity data using spheres in Cartesian coordinates
--------------------------------------------------------------------

The :mod:`fatiando.gravmag` has many functions for forward modeling gravity and
magnetic data. Here we'll show how to build a model out of spheres and
calculate the gravitational attraction and it's gradients in Cartesian
coordinates.

"""
from __future__ import division, print_function
from fatiando import mesher, gridder, utils
from fatiando.gravmag import sphere
import matplotlib.pyplot as plt
import numpy as np

# Create a model using geometric objects from fatiando.mesher
# Each model element has a dictionary with its physical properties.
# We'll use two spheres with opposite density contrast values.
model = [mesher.Sphere(x=10e3, y=10e3, z=1.5e3, radius=1.5e3,
                       props={'density': 500}),
         mesher.Sphere(x=20e3, y=20e3, z=1.5e3, radius=1.5e3,
                       props={'density': -500})]

# Create a regular grid at a constant height
shape = (300, 300)
area = [0, 30e3, 0, 30e3]
x, y, z = gridder.regular(area, shape, z=-100)

fields = [
    ['Gravity (mGal)', sphere.gz(x, y, z, model)],
    ['gxx (Eotvos)', sphere.gxx(x, y, z, model)],
    ['gyy (Eotvos)', sphere.gyy(x, y, z, model)],
    ['gzz (Eotvos)', sphere.gzz(x, y, z, model)],
    ['gxy (Eotvos)', sphere.gxy(x, y, z, model)],
    ['gxz (Eotvos)', sphere.gxz(x, y, z, model)],
    ['gyz (Eotvos)', sphere.gyz(x, y, z, model)],
]

# Make maps of all fields calculated
fig = plt.figure(figsize=(10, 8))
plt.rcParams['font.size'] = 10
X, Y = x.reshape(shape)/1000, y.reshape(shape)/1000
for i, tmp in enumerate(fields):
    ax = plt.subplot(3, 3, i + 3)
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
