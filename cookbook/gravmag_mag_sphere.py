"""
GravMag: 3D forward modeling of total-field magnetic anomaly using spheres
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Create a sphere model
# Considering only induced magnetization, so I don't give the 'inclination' and
# 'declination' properties
spheres = [ft.mesher.Sphere(0, 0, 3000, 1000, {'magnetization':1})]
# Create a regular grid at 100m height
shape = (100, 100)
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = ft.gridder.regular(area, shape, z=-100)
# Calculate the anomaly for a given regional field
tf = ft.gravmag.sphere.tf(xp, yp, zp, spheres, 30, 0)
# Plot
ft.vis.figure()
ft.vis.title("Total-field anomaly (nT)")
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, tf, shape, 15)
ft.vis.colorbar()
ft.vis.xlabel('East y (km)')
ft.vis.ylabel('North x (km)')
ft.vis.m2km()
ft.vis.show()
