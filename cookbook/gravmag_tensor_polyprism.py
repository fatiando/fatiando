"""
GravMag: Generate synthetic gradient tensor data from polygonal prisms
"""
from fatiando import mesher, gridder, gravmag
from fatiando.vis import mpl, myv

print("Draw the polygons one by one")
bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
axis = mpl.figure().gca()
mpl.axis('scaled')
prisms = [
    mesher.PolygonalPrism(
        mpl.draw_polygon(area, axis, xy2ne=True),
        0, 1000, {'density':500})]
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(area, shape, z=-500)
tensor = [
    gravmag.polyprism.gxx(xp, yp, zp, prisms),
    gravmag.polyprism.gxy(xp, yp, zp, prisms),
    gravmag.polyprism.gxz(xp, yp, zp, prisms),
    gravmag.polyprism.gyy(xp, yp, zp, prisms),
    gravmag.polyprism.gyz(xp, yp, zp, prisms),
    gravmag.polyprism.gzz(xp, yp, zp, prisms)]
# and plot it
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Gravity tensor produced by prism model (Eotvos)")
for i in xrange(len(tensor)):
    mpl.subplot(3, 2, i + 1)
    mpl.title(titles[i])
    mpl.contourf(yp, xp, tensor[i], shape, 20)
    mpl.colorbar()
    for p in prisms:
        mpl.polygon(p, '.-k', xy2ne=True)
    mpl.set_area(area)
    mpl.m2km()
mpl.show()
# Show the prisms
myv.figure()
myv.polyprisms(prisms, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds])
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.show()
