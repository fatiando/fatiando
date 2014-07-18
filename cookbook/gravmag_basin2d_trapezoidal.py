"""
GravMag: Simple gravity inversion for the relief of a 2D trapezoidal basin
"""
import numpy
from fatiando import utils
from fatiando.mesher import Polygon
from fatiando.gravmag import talwani, basin2d
from fatiando.vis import mpl

verts = [(10000, 1.), (90000, 1.), (90000, 7000), (10000, 3330)]
model = Polygon(verts, {'density': -100})
x = numpy.arange(0., 100000., 1000.)
z = numpy.zeros_like(x)
gz = utils.contaminate(talwani.gz(x, z, [model]), 0.5)

solver = basin2d.Trapezoidal(x, z, gz, verts[0:2], density=-100).config(
    'levmarq', initial=[9000, 500]).fit()
estimate = solver.estimate_

mpl.figure()
mpl.subplot(2, 1, 1)
mpl.title("Gravity anomaly")
mpl.plot(x, gz, 'ok', label='Observed')
mpl.plot(x, solver.predicted(), '-r', linewidth=2, label='Predicted')
mpl.legend(loc='lower left', numpoints=1)
mpl.ylabel("mGal")
mpl.xlim(0, 100000)
mpl.subplot(2, 1, 2)
mpl.polygon(estimate, 'o-r', linewidth=2, fill='r',
            alpha=0.3, label='Estimated')
mpl.polygon(model, '--k', linewidth=2, label='True')
mpl.legend(loc='lower left', numpoints=1)
mpl.xlabel("X")
mpl.ylabel("Z")
mpl.set_area((0, 100000, 10000, -500))
mpl.show()
