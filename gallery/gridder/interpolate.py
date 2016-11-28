"""
Interpolate irregular data
--------------------------

The functions :func:`fatiando.gridder.interp` and
:func:`fatiando.gridder.interp_at` offer convenient wrappers around
``scipy.interpolate.griddata``. The scipy function is more general and can
interpolate n-dimensional data. Our functions offer the convenience of
generating the regular grid points and optionally using nearest-neighbor
interpolation to extrapolate outside the convex hull of the data points.
"""
from fatiando import gridder
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data measured at random points
area = (0, 1, 0, 1)
x, y = gridder.scatter(area, n=500, seed=0)
data = x*(1 - x)*np.cos(4*np.pi*x)*np.sin(4*np.pi*y**2)**2

# Say we want to interpolate the data onto a regular grid with a given shape
shape = (100, 200)

# The gridder.interp function takes care of selecting the containing area of
# the data and generating the regular grid for us.
# Let's interpolate using the different options offered by gridddata and plot
# them all.

plt.figure(figsize=(10, 8))

xp, yp, nearest = gridder.interp(x, y, data, shape, algorithm='nearest')
plt.subplot(2, 2, 1)
plt.title('Nearest-neighbors')
plt.contourf(yp.reshape(shape), xp.reshape(shape), nearest.reshape(shape),
             30, cmap='RdBu_r')

xp, yp, linear = gridder.interp(x, y, data, shape, algorithm='linear')
plt.subplot(2, 2, 2)
plt.title('Linear')
plt.contourf(yp.reshape(shape), xp.reshape(shape), linear.reshape(shape),
             30, cmap='RdBu_r')

xp, yp, cubic = gridder.interp(x, y, data, shape, algorithm='cubic')
plt.subplot(2, 2, 3)
plt.title('Cubic')
plt.contourf(yp.reshape(shape), xp.reshape(shape), cubic.reshape(shape),
             30, cmap='RdBu_r')

# Notice that the cubic and linear interpolation leave empty the points that
# are outside the convex hull (bounding region) of the original scatter data.
# These data points will have NaN values or be masked in the data array, which
# can cause some problems for processing and inversion (any FFT operation in
# fatiando.gravmag will fail, for example). Use "extrapolate=True" to use
# nearest-neighbors to fill in those missing points.
xp, yp, cubic_ext = gridder.interp(x, y, data, shape, algorithm='cubic',
                                   extrapolate=True)
plt.subplot(2, 2, 4)
plt.title('Cubic with extrapolation')
plt.contourf(yp.reshape(shape), xp.reshape(shape), cubic_ext.reshape(shape),
             30, cmap='RdBu_r')

plt.tight_layout()
plt.show()
