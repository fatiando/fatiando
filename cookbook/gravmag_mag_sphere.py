"""
GravMag: 3D forward modeling of total-field magnetic anomaly using spheres
"""
from fatiando import logger, mesher, gridder, gravmag, utils
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Set the inclination and declination of the regional field
inc, dec = -30, 45
mag = utils.ang2vec(10, inc, dec)
# Create a sphere model considering only induced magnetization
spheres = [mesher.Sphere(0, 0, 3000, 1000, {'magnetization':mag})]
# Create a regular grid at 100m height
shape = (100, 100)
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = gridder.regular(area, shape, z=-100)
# Calculate the anomaly for a given regional field
tf = gravmag.sphere.tf(xp, yp, zp, spheres, inc, dec)
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
