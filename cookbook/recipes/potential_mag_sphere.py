"""
Calculate the total-field anomaly of sphere model with induced magnetization
only.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

spheres = [ft.msh.ddd.Sphere(0, 0, 3000, 1000, {'magnetization':1})]
# Create a regular grid at 100m height
shape = (100, 100)
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = ft.grd.regular(area, shape, z=-100)
# Calculate the anomaly for a given regional field
tf = ft.pot.sphere.tf(xp, yp, zp, spheres, 90, 0)
# Plot
ft.vis.figure()
ft.vis.title("Total-field anomaly (nT)")
ft.vis.axis('scaled')
ft.vis.contourf(yp*0.001, xp*0.001, tf, shape, 15)
ft.vis.colorbar()
ft.vis.xlabel('East y (km)')
ft.vis.ylabel('North x (km)')
ft.vis.show()
