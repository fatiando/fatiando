"""
GravMag: Calculate the gravity gradient tensor invariants
"""
import numpy
from fatiando import mesher, gridder, gravmag
from fatiando.vis import mpl

print("Draw the polygons one by one")
area = [-10000, 10000, -10000, 10000]
dataarea = [-5000, 5000, -5000, 5000]
prisms = []
for depth in [5000, 5000]:
    fig = mpl.figure()
    mpl.axis('scaled')
    mpl.square(dataarea)
    for p in prisms:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(area)
    prisms.append(
        mesher.PolygonalPrism(
            mpl.draw_polygon(area, fig.gca(), xy2ne=True),
            0, depth, {'density':500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(dataarea, shape, z=-500)
tensor = [
    gravmag.polyprism.gxx(xp, yp, zp, prisms),
    gravmag.polyprism.gxy(xp, yp, zp, prisms),
    gravmag.polyprism.gxz(xp, yp, zp, prisms),
    gravmag.polyprism.gyy(xp, yp, zp, prisms),
    gravmag.polyprism.gyz(xp, yp, zp, prisms),
    gravmag.polyprism.gzz(xp, yp, zp, prisms)]
# Calculate the 3 invariants
invariants = gravmag.tensor.invariants(tensor)
data = tensor + invariants
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
    for p in prisms:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(dataarea)
    mpl.m2km()
mpl.show()
