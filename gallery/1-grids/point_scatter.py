"""
Point scatters
--------------

You can create the (x, y, z) coordinate arrays for a random scatter of points
using :func:`fatiando.gridder.scatter`.

"""
from __future__ import print_function
from fatiando import gridder
import matplotlib.pyplot as plt

# Define the area where the points are generated in meters: [x1, x2, y1, y2]
area = [0, 10e3, -5e3, 5e3]

# The point scatter is generated from a uniform random distribution.
# So running the scatter function twice will produce different results.
# You can control this by passing a fixed seed value for the random number
# generator.
x, y = gridder.scatter(area, n=100, seed=42)

# x and y are 1d arrays with the coordinates of each point
print('x =', x)
print('y =', y)

# Optionally, you can generate a 3rd array with constant z values
# (remember that z is positive downward).
x, y, z = gridder.scatter(area, n=100, z=-150, seed=42)
print('z =', z)

plt.figure(figsize=(6, 5))
plt.title('Point scatter')
# In Fatiando, x is North and y is East.
# So we should plot x in the vertical axis and y in horizontal.
plt.plot(y, x, 'ok')
plt.xlabel('y (m)')
plt.ylabel('x (m)')
plt.xlim(-6e3, 6e3)
plt.ylim(-1e3, 11e3)
plt.grid(True)
plt.tight_layout()
plt.show()
