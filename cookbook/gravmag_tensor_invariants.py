"""
GravMag: Calculate the gravity gradient tensor invariants
"""
import numpy
from fatiando import gridder
from fatiando.mesher import Prism
from fatiando.gravmag import prism, tensor
from fatiando.vis import mpl

area = [-5000, 5000, -5000, 5000]
model = [Prism(-3000, 3000, -1000, 1000, 0, 1000, {'density': 1000})]
shape = (100, 100)
xp, yp, zp = gridder.regular(area, shape, z=-500)
data = [prism.gxx(xp, yp, zp, model),
        prism.gxy(xp, yp, zp, model),
        prism.gxz(xp, yp, zp, model),
        prism.gyy(xp, yp, zp, model),
        prism.gyz(xp, yp, zp, model),
        prism.gzz(xp, yp, zp, model)]
# Calculate the 3 invariants
invariants = tensor.invariants(data)
data = data + invariants
# and plot it
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Tensor and invariants (Eotvos)")
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'I1', 'I2', 'I']
for i in xrange(len(data)):
    mpl.subplot(3, 3, i + 1)
    mpl.title(titles[i])
    levels = 20
    if i == 8:
        levels = numpy.linspace(0, 1, levels)
    mpl.contourf(yp, xp, data[i], shape, levels, cmap=mpl.cm.RdBu_r)
    mpl.colorbar()
    mpl.m2km()
mpl.show()
