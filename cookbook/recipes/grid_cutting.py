"""
Cut a section from a grid.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

x, y = ft.grd.regular((-10, 10, -10, 10), (100,100))
z = ft.utils.gaussian2d(x, y, 1, 1)
subarea = [-2, 2, -3, 3]
subx, suby, subscalar = ft.grd.cut(x, y, [z], subarea)

ft.vis.figure(figsize=(12, 5))
ft.vis.subplot(1, 2, 1)
ft.vis.title("Whole grid")
ft.vis.axis('scaled')
ft.vis.pcolor(x, y, z, (100,100))
ft.vis.square(subarea, 'k', linewidth=2, label='Cut this region')
ft.vis.legend(loc='lower left')
ft.vis.subplot(1, 2, 2)
ft.vis.title("Cut grid")
ft.vis.axis('scaled')
ft.vis.pcolor(subx, suby, subscalar[0], (40,60), interp=True)
ft.vis.show()
