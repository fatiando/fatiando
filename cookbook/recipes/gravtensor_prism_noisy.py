"""
Generate noise-corrupted gravity gradient tensor data from a prism model.
"""
from matplotlib import pyplot
from fatiando import potential, gridder, vis, logger, utils
from fatiando.mesher.ddd import Prism

log = logger.get()
log.info(logger.header())
log.info("Example of generating noise-corrupted synthetic FTG data")

log.info("Calculating...")                
prisms = [Prism(-1000,1000,-1000,1000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-200)
components = [potential.prism.gxx, potential.prism.gxy, potential.prism.gxz,
              potential.prism.gyy, potential.prism.gyz, potential.prism.gzz]
ftg = [utils.contaminate(comp(xp, yp, zp, prisms),5.0) for comp in components]

log.info("Plotting...")
pyplot.figure()
pyplot.suptitle("Contaminated FTG data")
names = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for i, data in enumerate(ftg):
    pyplot.subplot(2,3,i+1)
    pyplot.title(names[i])
    pyplot.axis('scaled')
    levels = vis.contourf(xp*0.001, yp*0.001, data, (100,100), 12)
    pyplot.colorbar()
    vis.contour(xp*0.001, yp*0.001, data, shape, levels, clabel=False)
pyplot.show()
