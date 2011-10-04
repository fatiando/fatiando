"""
Example of cutting a grid into a smaller grid
"""
from matplotlib import pyplot
from fatiando import gridder, stats, vis

x, y = gridder.regular((-10, 10, -10, 10), (100,100))
z = stats.gaussian2d(x, y, 1, 1)
subx, suby, subscalar = gridder.cut(x, y, [z], -2, 2, -3, 3)

pyplot.figure()
pyplot.title("Whole grid")
pyplot.axis('scaled')
vis.pcolor(x, y, z, (100,100))
pyplot.figure()
pyplot.title("Cut grid")
pyplot.axis('scaled')
vis.pcolor(subx, suby, subscalar[0], (40,60), interpolate=True)
pyplot.show()
