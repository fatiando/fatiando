"""
Extracting a profile from spacial data
========================================

The function :func:`fatiando.gridder.profile` can be used to extract a profile
of data from a map. It interpolates the data onto the profile points so you can
specify the profile in any direction and use irregular point data as input.
"""
import matplotlib.pyplot as plt
import numpy as np
from fatiando import gridder, utils

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=1000, seed=1)
# And calculate 2D Gaussians on these points as sample data
data = 2*utils.gaussian2d(x, y, -0.6, -1) - utils.gaussian2d(x, y, 1.5, 1.5)

# Extract a profile between points 1 and 2
p1, p2 = [-1.5, -0.5], [1.5, 1.5]
xp, yp, distance, profile = gridder.profile(x, y, data, p1, p2, 100)

# Plot the profile and the original map data
plt.figure()
plt.subplot(2, 1, 1)
plt.title('Extracted profile points')
plt.plot(distance, profile, '.k')
plt.xlim(distance.min(), distance.max())
plt.grid()
plt.subplot(2, 1, 2)
plt.title("Original data")
plt.plot(xp, yp, '-k', label='Profile', linewidth=2)
scale = np.abs([data.min(), data.max()]).max()
plt.tricontourf(x, y, data, 50, cmap='RdBu_r', vmin=-scale, vmax=scale)
plt.colorbar(orientation='horizontal', aspect=50)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
