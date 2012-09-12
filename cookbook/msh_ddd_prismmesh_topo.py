"""
Make and plot a 3D prism mesh with topography.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

x1, x2 = -100, 100
y1, y2 = -200, 200
bounds = (x1, x2, y1, y2, -200, 0)

log.info("Generating synthetic topography")
x, y = ft.grd.regular((x1, x2, y1, y2), (50,50))
height = (100 +
          -50*ft.utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          100*ft.utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

log.info("Generating the prism mesh")
mesh = ft.msh.ddd.PrismMesh(bounds, (20,40,20))
mesh.carvetopo(x, y, height)

log.info("Plotting")
ft.vis.figure3d()
ft.vis.prisms(mesh)
ft.vis.axes3d(ft.vis.outline3d(bounds), fmt='%.0f')
ft.vis.wall_north(bounds)
ft.vis.show3d()
