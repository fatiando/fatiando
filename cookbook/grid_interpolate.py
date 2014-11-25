"""
Gridding: Grid irregularly sampled data.
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
area = (-2, 2, -2, 2)
x, y = gridder.scatter(area, n=200, seed=0)
# And calculate 2D Gaussians on these points as sample data


def data(x, y):
    return (utils.gaussian2d(x, y, -0.6, -1)
            - utils.gaussian2d(x, y, 1.5, 1.5))
z = data(x, y)

shape = (100, 100)

# First, we need to know the real data at the grid points
grdx, grdy = gridder.regular(area, shape)
grdz = data(grdx, grdy)
mpl.figure()
mpl.subplot(2, 2, 1)
mpl.axis('scaled')
mpl.title("True grid data")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

# Use the default interpolation (cubic)
grdx, grdy, grdz = gridder.interp(x, y, z, shape)
mpl.subplot(2, 2, 2)
mpl.axis('scaled')
mpl.title("Interpolated using cubic minimum-curvature")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

# Use the nearest neighbors interpolation
grdx, grdy, grdz = gridder.interp(x, y, z, shape, algorithm='nearest')
mpl.subplot(2, 2, 3)
mpl.axis('scaled')
mpl.title("Interpolated using nearest neighbors")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

# interp doesn't extrapolates the data by default. Lets see what happens if we
# enable extrapolation
grdx, grdy, grdz = gridder.interp(x, y, z, shape, extrapolate=True)
mpl.subplot(2, 2, 4)
mpl.axis('scaled')
mpl.title("Interpolated with extrapolation")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

mpl.show()
