"""
GravMag: Interactive 2D forward gravity modeling of a triangular basin
"""
import numpy
from fatiando.gui.simple import BasinTri

area = (0, 100000, 0, 5000)
xp = numpy.arange(0, 100000, 1000)
zp = numpy.zeros_like(xp)
nodes = [[20000, 1], [80000, 1]]
app = BasinTri(area, nodes, xp, zp)
app.run()
