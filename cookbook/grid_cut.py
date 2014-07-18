"""
Gridding: Cut a section from a grid
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate some synthetic data on a regular grid
x, y = gridder.regular((-10, 10, -10, 10), (100, 100))
# Using a 2D Gaussian
z = utils.gaussian2d(x, y, 1, 1)
subarea = [-2, 2, -3, 3]
subx, suby, subscalar = gridder.cut(x, y, [z], subarea)

mpl.figure(figsize=(12, 5))
mpl.subplot(1, 2, 1)
mpl.title("Whole grid")
mpl.axis('scaled')
mpl.pcolor(x, y, z, (100, 100))
mpl.square(subarea, 'k', linewidth=2, label='Cut this region')
mpl.legend(loc='lower left')
mpl.subplot(1, 2, 2)
mpl.title("Cut grid")
mpl.axis('scaled')
mpl.pcolor(subx, suby, subscalar[0], (40, 60), interp=True)
mpl.show()
