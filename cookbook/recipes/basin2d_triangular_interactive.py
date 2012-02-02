"""
2D interactive direct gravity modeling of a triangular 2D basin
"""
import numpy
from fatiando.ui.gui import BasinTri

area = (0, 100000, 0, 5000)
xp = numpy.arange(0, 100000, 1000)
zp = numpy.zeros_like(xp)
nodes = [[20000, 1], [80000, 1]]
app = BasinTri(area, nodes, xp, zp)
app.run()
