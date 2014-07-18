"""
Meshing: Making a grid of 3D point sources
"""
from fatiando import mesher, utils, gridder
from fatiando.gravmag import sphere
from fatiando.vis import mpl

grid = mesher.PointGrid([0, 1000, 0, 2000], 500, (50, 50))
# Add some density to the grid
density = 1000000000 * utils.gaussian2d(grid.x, grid.y, 100, 500,
                                        x0=500, y0=1000, angle=-60)
grid.addprop('density', density)
# and some magnetization
inc, dec = -45, 0
grid.addprop('magnetization', [d / 100. * utils.ang2vec(1, inc, dec)
                               for d in grid.props['density']])
# plot the layer
mpl.figure()
mpl.subplot(2, 1, 1)
mpl.axis('scaled')
mpl.title('Density (mass)')
mpl.pcolor(grid.y, grid.x, grid.props['density'], grid.shape)
mpl.colorbar()
mpl.subplot(2, 1, 2)
mpl.axis('scaled')
mpl.title('Magnetization intensity (dipole moment)')
mpl.pcolor(grid.y, grid.x, utils.vecnorm(grid.props['magnetization']),
           grid.shape)
mpl.colorbar()
mpl.show()

# Now do some calculations with the grid
shape = (100, 100)
x, y, z = gridder.regular(grid.area, shape, z=0)
gz = sphere.gz(x, y, z, grid)
tf = sphere.tf(x, y, z, grid, inc, dec)
mpl.figure()
mpl.subplot(2, 1, 1)
mpl.axis('scaled')
mpl.title('Gravity anomaly')
mpl.contourf(y, x, gz, shape, 30)
mpl.colorbar()
mpl.subplot(2, 1, 2)
mpl.axis('scaled')
mpl.title('Magnetic total field anomaly')
mpl.contourf(y, x, tf, shape, 30)
mpl.colorbar()
mpl.show()
