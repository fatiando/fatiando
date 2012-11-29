"""
GravMag: Upward continuation of noisy gz data using the analytical formula
"""
from fatiando import logger, mesher, gridder, utils, gravmag
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
prisms = [mesher.Prism(-3000,-2000,-3000,-2000,500,2000,{'density':1000}),
          mesher.Prism(-1000,1000,-1000,1000,0,2000,{'density':-800}),
          mesher.Prism(1000,3000,2000,3000,0,1000,{'density':500})]
area = (-5000, 5000, -5000, 5000)
shape = (50, 50)
z0 = -100
xp, yp, zp = gridder.regular(area, shape, z=z0)
gz = utils.contaminate(gravmag.prism.gz(xp, yp, zp, prisms), 0.5)

# Now do the upward continuation using the analytical formula
height = 2000
dims = gridder.spacing(area, shape)
gzcont = gravmag.transform.upcontinue(gz, height, xp, yp, dims)

log.info("Computing true values at new height")
gztrue = gravmag.prism.gz(xp, yp, zp - height, prisms)

log.info("Plotting")
mpl.figure(figsize=(14,6))
mpl.subplot(1, 2, 1)
mpl.title("Original")
mpl.axis('scaled')
mpl.contourf(xp, yp, gz, shape, 15)
mpl.contour(xp, yp, gz, shape, 15)
mpl.subplot(1, 2, 2)
mpl.title("Continued + true")
mpl.axis('scaled')
levels = mpl.contour(xp, yp, gzcont, shape, 12, color='b',
    label='Continued', style='dashed')
mpl.contour(xp, yp, gztrue, shape, levels, color='r', label='True',
    style='solid')
mpl.legend()
mpl.show()
