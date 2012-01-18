"""
Cut a section from a grid.
"""
from matplotlib import pyplot
from fatiando import gridder, utils, vis

x, y = gridder.regular((-10, 10, -10, 10), (100,100))
z = utils.gaussian2d(x, y, 1, 1)
subarea = [-2, 2, -3, 3]
subx, suby, subscalar = gridder.cut(x, y, [z], subarea)

pyplot.figure()
pyplot.title("Whole grid")
pyplot.axis('scaled')
vis.pcolor(x, y, z, (100,100))
vis.square(subarea, 'k')
pyplot.figure()
pyplot.title("Cut grid")
pyplot.axis('scaled')
vis.pcolor(subx, suby, subscalar[0], (40,60), interpolate=True)
pyplot.show()
