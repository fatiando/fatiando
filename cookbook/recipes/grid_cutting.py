"""
Cut a section from a grid.
"""
from matplotlib import pyplot
from fatiando import gridder, utils, vis, logger

log = logger.get()
log.info(logger.header())
log.info(__doc__)

x, y = gridder.regular((-10, 10, -10, 10), (100,100))
z = utils.gaussian2d(x, y, 1, 1)
subarea = [-2, 2, -3, 3]
subx, suby, subscalar = gridder.cut(x, y, [z], subarea)

pyplot.figure(figsize=(12, 5))
pyplot.subplot(1, 2, 1)
pyplot.title("Whole grid")
pyplot.axis('scaled')
vis.map.pcolor(x, y, z, (100,100))
vis.map.square(subarea, 'k', linewidth=2, label='Cut this region')
pyplot.legend(loc='lower left')
pyplot.subplot(1, 2, 2)
pyplot.title("Cut grid")
pyplot.axis('scaled')
vis.map.pcolor(subx, suby, subscalar[0], (40,60), interpolate=True)
pyplot.show()
