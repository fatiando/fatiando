"""
Vis: Plotting irregularly sampled map data
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=200)
# And calculate a 2D Gaussian on these points
z = utils.gaussian2d(x, y, 1, 1)

# Functions pcolor, contour and contourf take an interp argument
# If it is True, will interpolate the data before plotting using the specified
# grid shape
shape = (100, 100)
mpl.figure()
mpl.subplot(2, 2, 1)
mpl.axis('scaled')
mpl.title("contourf")
mpl.contourf(x, y, z, shape, 50, interp=True)
mpl.subplot(2, 2, 2)
mpl.axis('scaled')
mpl.title("contour")
mpl.contour(x, y, z, shape, 15, interp=True)
mpl.subplot(2, 2, 3)
mpl.axis('scaled')
mpl.title("pcolor")
mpl.pcolor(x, y, z, shape, interp=True)
# You can tell these functions to extrapolate the data to fill in the margins
mpl.subplot(2, 2, 4)
mpl.axis('scaled')
mpl.title("contourf extrapolate")
mpl.contourf(x, y, z, shape, 50, interp=True, extrapolate=True)
mpl.show()
