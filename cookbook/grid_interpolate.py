"""
Gridding: Grid irregularly sampled data.
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=200, seed=0)
# And calculate a 2D Gaussian on these points as sample data
z = utils.gaussian2d(x, y, 1, 1)

shape = (100, 100)
grdx, grdy, grdz = gridder.interp(x, y, z, shape)

mpl.figure()
mpl.axis('scaled')
mpl.title("Interpolated grid")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

# interp extrapolates the data by default. Lets see what happens if we disable
# extrapolation
grdx, grdy, grdz = gridder.interp(x, y, z, shape, extrapolate=False)
mpl.figure()
mpl.axis('scaled')
mpl.title("Interpolated grid (no extrapolation)")
mpl.plot(x, y, '.k', label='Data points')
mpl.contourf(grdx, grdy, grdz, shape, 50)
mpl.colorbar()
mpl.legend(loc='lower right', numpoints=1)

mpl.show()
