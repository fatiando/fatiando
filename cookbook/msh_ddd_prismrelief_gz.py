"""
Meshing: Generate a topographic model using prisms and calculate its gravity
anomaly
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (30, 15)
x, y = ft.grd.regular(area, shape)
height = (-80*ft.utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          200*ft.utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
nodes = (x, y, -1*height)
relief = ft.msh.ddd.PrismRelief(0, ft.grd.spacing(area,shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

log.info("Calculating gz effect")
gridarea = (-80, 80, -220, 220)
gridshape = (100, 100)
xp, yp, zp = ft.grd.regular(gridarea, gridshape, z=-200)
gz = ft.pot.prism.gz(xp, yp, zp, relief)

log.info("Plotting")
ft.vis.figure(figsize=(10,7))
ft.vis.subplot(1, 2, 1)
ft.vis.title("Synthetic topography")
ft.vis.axis('scaled')
ft.vis.pcolor(x, y, height, shape)
cb = ft.vis.colorbar()
cb.set_label("meters")
ft.vis.square(gridarea, label='Computation grid')
ft.vis.legend()
ft.vis.subplot(1, 2, 2)
ft.vis.title("Topographic effect")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, gz, gridshape)
cb = ft.vis.colorbar()
cb.set_label("mGal")
ft.vis.show()

ft.vis.figure3d()
ft.vis.prisms(relief, prop='density')
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
