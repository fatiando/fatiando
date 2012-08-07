"""
Generate and plot irregular grids.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Calculating...")
x, y = ft.grd.scatter((-2, 2, -2, 2), n=200)
z = ft.utils.gaussian2d(x, y, 1, 1)

log.info("Plotting...")
shape = (100, 100)
ft.vis.axis('scaled')
ft.vis.title("Irregular grid")
ft.vis.plot(x, y, '.k', label='Grid points')
levels = ft.vis.contourf(x, y, z, shape, 12, interp=True)
ft.vis.contour(x, y, z, shape, levels, interp=True)
ft.vis.legend(loc='lower right', numpoints=1)
ft.vis.show()
