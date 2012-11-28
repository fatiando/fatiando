"""
Potential: 3D imaging using the sandwich model method on synthetic gravity data
(more complex model)
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Make some synthetic gravity data from a simple prism model
prisms = [ft.mesher.Prism(-4000,0,-4000,-2000,2000,5000,{'density':1200}),
          ft.mesher.Prism(-1000,1000,-1000,1000,1000,7000,{'density':-300}),
          ft.mesher.Prism(2000,4000,3000,4000,0,2000,{'density':600})]
shape = (25, 25)
xp, yp, zp = ft.gridder.regular((-10000, 10000, -10000, 10000), shape, z=-10)
gz = ft.pot.prism.gz(xp, yp, zp, prisms)

# Plot the data
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, gz, shape, 30)
ft.vis.colorbar()
ft.vis.xlabel('East (km)')
ft.vis.ylabel('North (km)')
ft.vis.m2km()
ft.vis.show()

mesh = ft.pot.imaging.sandwich(xp, yp, zp, gz, shape, 0, 10000, 25)

# Plot the results
ft.vis.figure3d()
ft.vis.prisms(prisms, 'density', style='wireframe', linewidth=3)
ft.vis.prisms(mesh, 'density', edges=False)
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
