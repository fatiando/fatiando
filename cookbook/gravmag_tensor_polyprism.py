"""
GravMag: Generate synthetic gradient tensor data from polygonal prisms
"""
from fatiando import mesher, gridder
from fatiando.gravmag import polyprism
from fatiando.vis import mpl, myv

print "Draw the polygons one by one"
bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
axis = mpl.figure().gca()
mpl.axis('scaled')
model = [
    mesher.PolygonalPrism(
        mpl.draw_polygon(area, axis, xy2ne=True),
        0, 1000, {'density': 500})]
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(area, shape, z=-500)
data = [
    polyprism.gxx(xp, yp, zp, model),
    polyprism.gxy(xp, yp, zp, model),
    polyprism.gxz(xp, yp, zp, model),
    polyprism.gyy(xp, yp, zp, model),
    polyprism.gyz(xp, yp, zp, model),
    polyprism.gzz(xp, yp, zp, model)]
# and plot it
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Gravity tensor produced by prism model (Eotvos)")
for i in xrange(len(data)):
    mpl.subplot(3, 2, i + 1)
    mpl.title(titles[i])
    mpl.contourf(yp, xp, data[i], shape, 20)
    mpl.colorbar()
    for p in model:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(area)
    mpl.m2km()
mpl.show()
# Show the prisms
myv.figure()
myv.polyprisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds])
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.show()
