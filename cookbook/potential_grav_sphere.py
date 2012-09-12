"""
Potential: Forward modeling of the gravity anomaly using spheres (calculate on
random points)
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

spheres = [ft.msh.ddd.Sphere(0, 0, -2000, 1000, {'density':1000})]
# Create a set of points at 100m height
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = ft.grd.scatter(area, 500, z=-100)
# Calculate the anomaly
gz = ft.utils.contaminate(ft.pot.sphere.gz(xp, yp, zp, spheres), 0.1)
# Plot
shape = (100, 100)
ft.vis.figure()
ft.vis.title("gz (mGal)")
ft.vis.axis('scaled')
ft.vis.plot(yp*0.001, xp*0.001, '.k')
ft.vis.contourf(yp*0.001, xp*0.001, gz, shape, 15, interp=True)
ft.vis.colorbar()
ft.vis.xlabel('East y (km)')
ft.vis.ylabel('North x (km)')
ft.vis.show()
