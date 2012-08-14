"""
Run the Generalized Inverse imaging method on synthetic gravity data to get a 3D
density distribution estimate.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Make some synthetic gravity data from a simple prism model
prisms = [ft.msh.ddd.Prism(-1000,1000,-3000,3000,2000,5000,{'density':1000})]
shape = (25, 25)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-10)
gz = ft.pot.prism.gz(xp, yp, zp, prisms)/ft.constants.SI2MGAL

# Plot the data
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, gz*ft.constants.SI2MGAL, shape, 30)
ft.vis.colorbar()
ft.vis.xlabel('East (km)')
ft.vis.ylabel('North (km)')
ft.vis.m2km()
ft.vis.show()

# Run the Generalized Inverse
mesh = ft.pot.imaging.geninv(xp, yp, zp, gz, shape, 0, 10000, 25)

# Plot the results
ft.vis.figure3d()
ft.vis.prisms(prisms, 'density', style='wireframe', linewidth=5)
ft.vis.prisms(mesh, 'density', edges=False)
axes = ft.vis.axes3d(ft.vis.outline3d())
ft.vis.wall_bottom(axes.axes.bounds)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
