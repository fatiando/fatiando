"""
Gridding: Generate and plot irregular grids (scatter)
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=200)
# And calculate a 2D Gaussian on these points
z = utils.gaussian2d(x, y, 1, 1)

mpl.axis('scaled')
mpl.title("Irregular grid")
mpl.plot(x, y, '.k', label='Grid points')
# Make a filled contour plot and tell the function to automatically interpolate
# the data on a 100x100 grid
mpl.contourf(x, y, z, (100, 100), 50, interp=True)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)
mpl.show()
