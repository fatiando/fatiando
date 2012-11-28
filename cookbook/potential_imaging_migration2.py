"""
Potential: 3D imaging using the migration method on synthetic gravity data
(more complex model + noisy data)
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Make some synthetic gravity data from a simple prism model
prisms = [ft.msh.ddd.Prism(-4000,0,-4000,-2000,2000,5000,{'density':1200}),
          ft.msh.ddd.Prism(-1000,1000,-1000,1000,1000,7000,{'density':-800}),
          ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,{'density':600})]
# Calculate on a scatter of points to show that migration doesn't need gridded
# data
xp, yp, zp = ft.gridder.scatter((-6000, 6000, -6000, 6000), 1000, z=-10)
gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.1)

# Plot the data
shape = (50, 50)
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, gz, shape, 30, interp=True)
ft.vis.colorbar()
ft.vis.plot(yp, xp, '.k')
ft.vis.xlabel('East (km)')
ft.vis.ylabel('North (km)')
ft.vis.m2km()
ft.vis.show()

mesh = ft.pot.imaging.migrate(xp, yp, zp, gz, 0, 10000, (30, 30, 30), power=0.8)

# Plot the results
ft.vis.figure3d()
ft.vis.prisms(prisms, 'density', style='wireframe', linewidth=2)
ft.vis.prisms(mesh, 'density', edges=False)
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
