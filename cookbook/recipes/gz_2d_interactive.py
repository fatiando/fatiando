"""
2D interactive direct gravity modeling with polygons
"""
import numpy
from fatiando.ui.gui import Potential2DModeler

area = (0, 100000, 0, 5000)
xp = numpy.arange(0, 100000, 1000)
zp = numpy.zeros_like(xp)
app = Potential2DModeler(area, xp, zp)
