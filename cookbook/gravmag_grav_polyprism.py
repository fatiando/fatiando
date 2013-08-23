"""
GravMag: Forward gravity modeling using a stack of 3D polygonal prisms
"""
from fatiando import mesher, gridder, gravmag
from fatiando.vis import mpl, myv

bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
depths = [0, 1000, 2000, 3000, 4000]
prisms = []
for i in range(1, len(depths)):
    axes = mpl.figure().gca()
    mpl.axis('scaled')
    for p in prisms:
        mpl.polygon(p, '.-k', xy2ne=True)
    prisms.append(
        mesher.PolygonalPrism(
            mpl.draw_polygon(area, axes, xy2ne=True),
            depths[i - 1], depths[i], {'density':500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = gravmag.polyprism.gz(xp, yp, zp, prisms)
# and plot it
mpl.figure()
mpl.axis('scaled')
mpl.title("gz produced by prism model (mGal)")
mpl.contourf(yp, xp, gz, shape, 20)
mpl.colorbar()
for p in prisms:
    mpl.polygon(p, '.-k', xy2ne=True)
mpl.set_area(area)
mpl.show()
# Show the prisms
myv.figure()
myv.polyprisms(prisms, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds])
myv.show()
