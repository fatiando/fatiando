"""
Seismic: Interactive forward modeling of 1D vertical seismic profile (VSP) data in
layered media
"""
import numpy
from fatiando.gui.simple import Lasagne

thickness = [10, 20, 5, 10, 45, 80]
zp = numpy.arange(0.5, sum(thickness), 0.5)
vmin, vmax = 500, 10000
app = Lasagne(thickness, zp, vmin, vmax)
app.run()
