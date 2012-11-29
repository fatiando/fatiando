"""
GravMag: 3D forward modeling of total-field magnetic anomaly using spheres
"""
from fatiando import logger, mesher, gridder, gravmag
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Create a sphere model
# Considering only induced magnetization, so I don't give the 'inclination' and
# 'declination' properties
spheres = [mesher.Sphere(0, 0, 3000, 1000, {'magnetization':1})]
# Create a regular grid at 100m height
shape = (100, 100)
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = gridder.regular(area, shape, z=-100)
# Calculate the anomaly for a given regional field
tf = gravmag.sphere.tf(xp, yp, zp, spheres, 30, 0)
# Plot
mpl.figure()
mpl.title("Total-field anomaly (nT)")
mpl.axis('scaled')
mpl.contourf(yp, xp, tf, shape, 15)
mpl.colorbar()
mpl.xlabel('East y (km)')
mpl.ylabel('North x (km)')
mpl.m2km()
mpl.show()
