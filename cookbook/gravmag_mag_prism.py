"""
GravMag: 3D forward modeling of total-field magnetic anomaly using rectangular
prisms (model with induced and remanent magnetization)
"""
from fatiando import logger, mesher, gridder, gravmag
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())
log.info(__doc__)

bounds = [-5000, 5000, -5000, 5000, 0, 5000]
prisms = [
    mesher.Prism(-4000,-3000,-4000,-3000,0,2000,
        {'magnetization':2}),
    mesher.Prism(-1000,1000,-1000,1000,0,2000,
        {'magnetization':1}),
    # This prism has remanent magnetization because it's physical property
    # dict has inclination and declination
    mesher.Prism(2000,4000,3000,4000,0,2000,
        {'magnetization':3, 'inclination':-10, 'declination':45})]
# Create a regular grid at 100m height
shape = (200, 200)
area = bounds[:4]
xp, yp, zp = gridder.regular(area, shape, z=-500)
# Calculate the anomaly for a given regional field
tf = gravmag.prism.tf(xp, yp, zp, prisms, 30, -15)
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
# Show the prisms
myv.figure()
myv.prisms(prisms, 'magnetization')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds])
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.show()
