"""
Cutting a section from spacial data
-----------------------------------

The :func:`fatiando.gridder.cut` function extracts points from spatially
distributed  data that are inside a given area. It doesn't matter whether or
not the points are on a regular grid.
"""
from fatiando import gridder
import matplotlib.pyplot as plt
import numpy as np

# Generate some synthetic data
area = (-100, 100, -60, 60)
x, y = gridder.scatter(area, 1000, seed=0)
data = np.sin(0.1*x)*np.cos(0.1*y)
# Select the data that fall inside "section"
section = [-40, 40, -25, 25]
# Tip: you pass more than one data array as input. Use this to cut multiple
# data sources (e.g., gravity + height + topography).
x_sub, y_sub, [data_sub] = gridder.cut(x, y, [data], section)

# Plot the original data besides the cut section
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.axis('scaled')
plt.title("Whole data")
plt.tricontourf(y, x, data, 30, cmap='RdBu_r')
plt.plot(y, x, 'xk')
x1, x2, y1, y2 = section
plt.plot([y1, y2, y2, y1, y1], [x1, x1, x2, x2, x1], '-k', linewidth=3)
plt.xlim(area[2:])
plt.ylim(area[:2])

plt.subplot(1, 2, 2)
plt.axis('scaled')
plt.title("Subsection")
plt.plot(y_sub, x_sub, 'xk')
plt.tricontourf(y_sub, x_sub, data_sub, 30, cmap='RdBu_r')
plt.xlim(section[2:])
plt.ylim(section[:2])

plt.tight_layout()
plt.show()
