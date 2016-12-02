"""
Potential field imaging through the Generalized Inverse method
---------------------------------------------------------------

Module :mod:`fatiando.gravmag.imaging` has functions for imaging methods in
potential fields. These methods produce an image of the subsurface without
doing an inversion. However, there is a tradeoff with the quality of the result
being generally inferior to an inversion result.

Here we'll show how the Generalized Inverse imaging method can be used on some
synthetic data. We'll plot the final result as slices across the x, y, z axis.
"""
from __future__ import division
from fatiando import gridder, mesher
from fatiando.gravmag import prism, imaging
from fatiando.vis.mpl import square
import matplotlib.pyplot as plt
import numpy as np

# Make some synthetic gravity data from a simple prism model
model = [mesher.Prism(-1000, 1000, -3000, 3000, 0, 2000, {'density': 800})]
shape = (25, 25)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-10)
data = prism.gz(xp, yp, zp, model)

# Run the Generalized Inverse
mesh = imaging.geninv(xp, yp, zp, data, shape, zmin=0, zmax=5000, nlayers=25)

# Plot the results
fig = plt.figure()

X, Y = xp.reshape(shape)/1000, yp.reshape(shape)/1000
image = mesh.props['density'].reshape(mesh.shape)

# First plot the original gravity data
ax = plt.subplot(2, 2, 1)
ax.set_title('Gravity data (mGal)')
ax.set_aspect('equal')
scale = np.abs([data.min(), data.max()]).max()
tmp = ax.contourf(Y, X, data.reshape(shape), 30, cmap="RdBu_r", vmin=-scale,
                  vmax=scale)
plt.colorbar(tmp, ax=ax, pad=0)
ax.set_xlim(Y.min(), Y.max())
ax.set_ylim(X.min(), X.max())
ax.set_xlabel('y (km)')
ax.set_ylabel('x (km)')

# Then plot model slices in the x, y, z directions through the middle of the
# model. Also show the outline of the true model for comparison.
scale = 0.1*np.abs([image.min(), image.max()]).max()
x = mesh.get_xs()/1000
y = mesh.get_ys()/1000
z = mesh.get_zs()/1000
x1, x2, y1, y2, z1, z2 = np.array(model[0].get_bounds())/1000

ax = plt.subplot(2, 2, 2)
ax.set_title('Model slice at z={} km'.format(z[len(z)//2]))
ax.set_aspect('equal')
ax.pcolormesh(y, x, image[mesh.shape[0]//2, :, :], cmap="cubehelix",
              vmin=-scale, vmax=scale)
square([y1, y2, x1, x2])
ax.set_ylim(x.min(), x.max())
ax.set_xlim(y.min(), y.max())
ax.set_xlabel('y (km)')
ax.set_ylabel('x (km)')

ax = plt.subplot(2, 2, 3)
ax.set_title('Model slice at y={} km'.format(y[len(y)//2]))
ax.set_aspect('equal')
ax.pcolormesh(x, z, image[:, :, mesh.shape[1]//2], cmap="cubehelix",
              vmin=-scale, vmax=scale)
square([x1, x2, z1, z2])
ax.set_ylim(z.max(), z.min())
ax.set_xlim(x.min(), x.max())
ax.set_xlabel('x (km)')
ax.set_ylabel('z (km)')

ax = plt.subplot(2, 2, 4)
ax.set_title('Model slice at x={} km'.format(x[len(x)//2]))
ax.set_aspect('equal')
ax.pcolormesh(y, z, image[:, mesh.shape[2]//2, :], cmap="cubehelix",
              vmin=-scale, vmax=scale)
square([y1, y2, z1, z2])
ax.set_ylim(z.max(), z.min())
ax.set_xlim(y.min(), y.max())
ax.set_xlabel('y (km)')
ax.set_ylabel('z (km)')

plt.tight_layout()
plt.show()
