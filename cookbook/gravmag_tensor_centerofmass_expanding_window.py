"""
GravMag: Center of mass estimation using the first eigenvector of the gravity
gradient tensor (2 sources with expanding windows)
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import tensor, prism
from fatiando.vis import mpl, myv

# Generate some synthetic data
model = [mesher.Prism(-2500, -500, -1000, 1000, 500, 2500, {'density': 1000}),
         mesher.Prism(500, 2500, -1000, 1000, 500, 2500, {'density': 1000})]
shape = (100, 100)
area = (-5000, 5000, -5000, 5000)
xp, yp, zp = gridder.regular(area, shape, z=-150)
noise = 2
data = [utils.contaminate(prism.gxx(xp, yp, zp, model), noise),
        utils.contaminate(prism.gxy(xp, yp, zp, model), noise),
        utils.contaminate(prism.gxz(xp, yp, zp, model), noise),
        utils.contaminate(prism.gyy(xp, yp, zp, model), noise),
        utils.contaminate(prism.gyz(xp, yp, zp, model), noise),
        utils.contaminate(prism.gzz(xp, yp, zp, model), noise)]
# Get the eigenvectors from the tensor data
eigenvals, eigenvecs = tensor.eigen(data)
# Plot the data
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
for i, title in enumerate(titles):
    mpl.subplot(3, 2, i + 1)
    mpl.title(title)
    mpl.axis('scaled')
    levels = mpl.contourf(yp, xp, data[i], shape, 10)
    mpl.contour(yp, xp, data[i], shape, levels)
    mpl.m2km()
mpl.show()

# Pick the centers of the expanding windows
# The number of final solutions will be the number of points picked
mpl.figure()
mpl.suptitle('Pick the centers of the expanding windows')
mpl.axis('scaled')
mpl.contourf(yp, xp, data[-1], shape, 50)
mpl.colorbar()
centers = mpl.pick_points(area, mpl.gca(), xy2ne=True)
# Use the first eigenvector to estimate the center of mass for each expanding
# window group
cms = [tensor.center_of_mass(xp, yp, zp, eigenvecs[0], windows=100, wcenter=c)
       for c in centers]

# Plot the prism and the estimated center of mass
# It won't work well because we're using only a single window
myv.figure()
myv.points(cms, size=200.)
myv.prisms(model, prop='density', opacity=0.5)
axes = myv.axes(myv.outline(extent=[-5000, 5000, -5000, 5000, 0, 5000]))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
