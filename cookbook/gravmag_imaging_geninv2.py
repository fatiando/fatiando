"""
GravMag: 3D imaging using the Generalized Inverse method on synthetic gravity
data (more complex model + noisy data)
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Make some synthetic gravity data from a simple prism model
prisms = [ft.mesher.Prism(-4000,-1000,-4000,-2000,2000,5000,{'density':800}),
          ft.mesher.Prism(-1000,1000,-1000,1000,1000,6000,{'density':-800}),
          ft.mesher.Prism(2000,4000,3000,4000,0,4000,{'density':600})]
shape = (25, 25)
xp, yp, zp = ft.gridder.regular((-5000, 5000, -5000, 5000), shape, z=-10)
gz = ft.utils.contaminate(ft.gravmag.prism.gz(xp, yp, zp, prisms), 0.1)

# Plot the data
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, gz, shape, 30)
ft.vis.colorbar()
ft.vis.xlabel('East (km)')
ft.vis.ylabel('North (km)')
ft.vis.m2km()
ft.vis.show()

# Run the Generalized Inverse
mesh = ft.gravmag.imaging.geninv(xp, yp, zp, gz, shape,
    0, 10000, 25)

# Plot the results
ft.vis.figure3d()
ft.vis.prisms(prisms, 'density', style='wireframe')
ft.vis.prisms(mesh, 'density', edges=False, linewidth=5)
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
