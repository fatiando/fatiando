"""
Regular grids
-------------

You can create the (x, y, z) coordinate arrays for regular grids using
:func:`fatiando.gridder.regular`.

"""
from __future__ import print_function
from fatiando import gridder
import matplotlib.pyplot as plt

# Define the area of the grid in meters: [x1, x2, y1, y2]
area = [0, 10e3, -5e3, 5e3]
# The shape is the number of points in the grid: (nx, ny)
shape = (5, 9)

x, y = gridder.regular(area, shape)

# x and y are 1d arrays with the coordinates of each point in the grid
print('x =', x)
print('y =', y)

# Optionally, you can generate a 3rd array with constant z values
# (remember that z is positive downward)
x, y, z = gridder.regular(area, shape, z=-150)
print('z =', z)

plt.figure(figsize=(6, 5))
plt.title('Regular grid')
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
