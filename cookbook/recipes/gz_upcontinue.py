"""
Upcontinue noisy gz data using the analytical formula
"""
from matplotlib import pyplot
from fatiando import gridder, potential, vis, logger, utils
from fatiando.mesher.ddd import Prism

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
prisms = [Prism(-3000,-2000,-3000,-2000,500,2000,{'density':1000}),
          Prism(-1000,1000,-1000,1000,0,2000,{'density':-800}),
          Prism(1000,3000,2000,3000,0,1000,{'density':500})]
area = (-5000, 5000, -5000, 5000)
shape = (25, 25)
z0 = -100
xp, yp, zp = gridder.regular(area, shape, z=z0)
gz = utils.contaminate(potential.prism.gz(xp, yp, zp, prisms), 0.5)

# Now do the upward continuation using the analytical formula
height = 2000
dims = gridder.spacing(area, shape)
gzcont = potential.trans.upcontinue(gz, z0, height, xp, yp, dims)

log.info("Computing true values at new height")
gztrue = potential.prism.gz(xp, yp, zp - height, prisms)

log.info("Plotting")
pyplot.figure(figsize=(14,6))
pyplot.subplot(1, 2, 1)
pyplot.title("Original")
pyplot.axis('scaled')
vis.map.contourf(xp, yp, gz, shape, 15)
vis.map.contour(xp, yp, gz, shape, 15)
pyplot.subplot(1, 2, 2)
pyplot.title("Continued + true")
pyplot.axis('scaled')
levels = vis.map.contour(xp, yp, gzcont, shape, 12, color='b',
    label='Continued', style='dashed')
vis.map.contour(xp, yp, gztrue, shape, levels, color='r', label='True',
    style='solid')
pyplot.legend()
pyplot.show()
