"""
Gridding: Generate and plot irregular grids (scatter)
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Generate random points
x, y = ft.gridder.scatter((-2, 2, -2, 2), n=200)
# And calculate a 2D Gaussian on these points
z = ft.utils.gaussian2d(x, y, 1, 1)

ft.vis.axis('scaled')
ft.vis.title("Irregular grid")
ft.vis.plot(x, y, '.k', label='Grid points')
# Make a filled contour plot and tell the function to automatically interpolate
# the data on a 100x100 grid
ft.vis.contourf(x, y, z, (100, 100) , 50, interp=True)
ft.vis.colorbar()
ft.vis.legend(loc='lower right', numpoints=1)
ft.vis.show()
