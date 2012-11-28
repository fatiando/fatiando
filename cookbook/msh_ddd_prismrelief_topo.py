"""
Meshing: Generate a 3D prism model of the topography
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Generating synthetic topography")
area = (-150, 150, -300, 300)
shape = (100, 50)
x, y = ft.gridder.regular(area, shape)
height = (-80*ft.utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*ft.utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the 3D relief")
nodes = (x, y, -1*height) # -1 is to convert height to z coordinate
reference = 0 # z coordinate of the reference surface
relief = ft.msh.ddd.PrismRelief(reference, ft.gridder.spacing(area, shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

log.info("Plotting")
ft.vis.figure3d()
ft.vis.prisms(relief, prop='density', edges=False)
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
