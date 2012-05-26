"""
Generate synthetic potential field data from a 3D prism model.
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

prisms = [ft.msh.ddd.Prism(-4000,-3000,-4000,-3000,0,2000,{'density':500}),
          ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':-200}),
          ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,{'density':300})]
shape = (100,100)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
fields = [ft.pot.prism.pot(xp, yp, zp, prisms),
          ft.pot.prism.gx(xp, yp, zp, prisms),
          ft.pot.prism.gy(xp, yp, zp, prisms),
          ft.pot.prism.gz(xp, yp, zp, prisms),
          ft.pot.prism.gxx(xp, yp, zp, prisms),
          ft.pot.prism.gxy(xp, yp, zp, prisms),
          ft.pot.prism.gxz(xp, yp, zp, prisms),
          ft.pot.prism.gyy(xp, yp, zp, prisms),
          ft.pot.prism.gyz(xp, yp, zp, prisms),
          ft.pot.prism.gzz(xp, yp, zp, prisms)
          ]
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
ft.vis.figure()
ft.vis.suptitle("Potential fields produced by prism model")
for i, field in enumerate(fields):
    ft.vis.subplot(2, 5, i + 1)
    ft.vis.axis('scaled')
    ft.vis.title(titles[i])
    levels = ft.vis.contourf(yp*0.001, xp*0.001, field, shape, 12)
    ft.vis.contour(yp*0.001, xp*0.001, field, shape, levels)
ft.vis.show()

ft.vis.figure3d()
ft.vis.prisms(prisms, prop='density')
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
