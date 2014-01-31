"""
GravMag: Calculate the gravity gradient tensor invariants
"""
import numpy
from fatiando import mesher, gridder
from fatiando.gravmag import polyprism, tensor
from fatiando.vis import mpl

print "Draw the polygons one by one"
area = [-10000, 10000, -10000, 10000]
dataarea = [-5000, 5000, -5000, 5000]
model = []
for depth in [5000, 5000]:
    fig = mpl.figure()
    mpl.axis('scaled')
    mpl.square(dataarea)
    for p in model:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(area)
    model.append(
        mesher.PolygonalPrism(
            mpl.draw_polygon(area, fig.gca(), xy2ne=True),
            0, depth, {'density':500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(dataarea, shape, z=-500)
data = [
    polyprism.gxx(xp, yp, zp, model),
    polyprism.gxy(xp, yp, zp, model),
    polyprism.gxz(xp, yp, zp, model),
    polyprism.gyy(xp, yp, zp, model),
    polyprism.gyz(xp, yp, zp, model),
    polyprism.gzz(xp, yp, zp, model)]
# Calculate the 3 invariants
invariants = tensor.invariants(data)
data = data + invariants
# and plot it
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Tensor and invariants produced by prism model (Eotvos)")
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'I1', 'I2', 'I']
for i in xrange(len(data)):
    mpl.subplot(3, 3, i + 1)
    mpl.title(titles[i])
    levels = 20
    if i == 8:
        levels = numpy.linspace(0, 1, levels)
    mpl.contourf(yp, xp, data[i], shape, levels)
    mpl.colorbar()
    for p in model:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(dataarea)
    mpl.m2km()
mpl.show()
