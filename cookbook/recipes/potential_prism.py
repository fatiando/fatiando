"""
Generate synthetic potential field data from a 3D prism model.
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

prisms = [ft.msh.ddd.Prism(-4000,-3000,-4000,-3000,0,2000,{'density':1000}),
          ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':-1000}),
          ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
fields = [ft.pot.prism.gx(xp, yp, zp, prisms),
          ft.pot.prism.gy(xp, yp, zp, prisms),
          ft.pot.prism.gz(xp, yp, zp, prisms)]
titles = ['gx', 'gy', 'gz']
ft.vis.figure()
ft.vis.suptitle("Potential fields produced by prism model")
for i, field in enumerate(fields):
    ft.vis.subplot(3, 1, i + 1)
    ft.vis.axis('scaled')
    ft.vis.title(titles[i])
    ft.vis.pcolor(yp, xp, field, shape)
    ft.vis.colorbar()
ft.vis.show()

ft.vis.figure3d()
ft.vis.prisms(prisms, prop='density')
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
