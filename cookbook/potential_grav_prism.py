"""
Potential: Forward modeling of the gravitational potential and its derivatives
using 3D prisms
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

prisms = [ft.msh.ddd.Prism(-4000,-3000,-4000,-3000,0,2000,{'density':1000}),
          ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':-900}),
          ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,{'density':1300})]
shape = (100,100)
xp, yp, zp = ft.gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
log.info("Calculating fileds...")
fields = [ft.pot.prism.pot(xp, yp, zp, prisms),
          ft.pot.prism.gx(xp, yp, zp, prisms),
          ft.pot.prism.gy(xp, yp, zp, prisms),
          ft.pot.prism.gz(xp, yp, zp, prisms),
          ft.pot.prism.gxx(xp, yp, zp, prisms),
          ft.pot.prism.gxy(xp, yp, zp, prisms),
          ft.pot.prism.gxz(xp, yp, zp, prisms),
          ft.pot.prism.gyy(xp, yp, zp, prisms),
          ft.pot.prism.gyz(xp, yp, zp, prisms),
          ft.pot.prism.gzz(xp, yp, zp, prisms)]
log.info("Plotting...")
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
ft.vis.figure(figsize=(8, 9))
ft.vis.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.92, hspace=0.3)
ft.vis.suptitle("Potential fields produced by a 3 prism model")
for i, field in enumerate(fields):
    ft.vis.subplot(4, 3, i + 3)
    ft.vis.axis('scaled')
    ft.vis.title(titles[i])
    levels = ft.vis.contourf(yp*0.001, xp*0.001, field, shape, 15)
    cb = ft.vis.colorbar()
    ft.vis.contour(yp*0.001, xp*0.001, field, shape, levels, clabel=False, linewidth=0.1)
ft.vis.show()

ft.vis.figure3d()
ft.vis.prisms(prisms, prop='density')
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
