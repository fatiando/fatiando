"""
GravMag: Forward gravity modeling using a stack of 3D polygonal model
"""
from fatiando import mesher, gridder
from fatiando.gravmag import polyprism
from fatiando.vis import mpl, myv

bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
depths = [0, 1000, 2000, 3000, 4000]
model = []
for i in range(1, len(depths)):
    axes = mpl.figure().gca()
    mpl.axis('scaled')
    for p in model:
        mpl.polygon(p, '.-k', xy2ne=True)
    model.append(
        mesher.PolygonalPrism(
            mpl.draw_polygon(area, axes, xy2ne=True),
            depths[i - 1], depths[i], {'density': 500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = polyprism.gz(xp, yp, zp, model)
# and plot it
mpl.figure()
mpl.axis('scaled')
mpl.title("gz produced by prism model (mGal)")
mpl.contourf(yp, xp, gz, shape, 20)
mpl.colorbar()
for p in model:
    mpl.polygon(p, '.-k', xy2ne=True)
mpl.set_area(area)
mpl.show()
# Show the model
myv.figure()
myv.polyprisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds])
myv.show()
