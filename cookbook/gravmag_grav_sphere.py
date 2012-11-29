"""
GravMag: Forward modeling of the gravity anomaly using spheres (calculate on
random points)
"""
from fatiando import logger, mesher, gridder, utils, gravmag
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

spheres = [mesher.Sphere(0, 0, -2000, 1000, {'density':1000})]
# Create a set of points at 100m height
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = gridder.scatter(area, 500, z=-100)
# Calculate the anomaly
gz = utils.contaminate(gravmag.sphere.gz(xp, yp, zp, spheres), 0.1)
# Plot
shape = (100, 100)
mpl.figure()
mpl.title("gz (mGal)")
mpl.axis('scaled')
mpl.plot(yp*0.001, xp*0.001, '.k')
mpl.contourf(yp*0.001, xp*0.001, gz, shape, 15, interp=True)
mpl.colorbar()
mpl.xlabel('East y (km)')
mpl.ylabel('North x (km)')
mpl.show()
