"""
GravMag: Center of mass estimation using the first eigenvector of the gravity
gradient tensor (simple model)
"""
from fatiando.vis import mpl, myv
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, tensor

# Generate some synthetic data
model = [mesher.Prism(-1000, 1000, -1000, 1000, 1000, 3000, {'density': 1000})]
shape = (100, 100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
noise = 2
data = [utils.contaminate(prism.gxx(xp, yp, zp, model), noise),
        utils.contaminate(prism.gxy(xp, yp, zp, model), noise),
        utils.contaminate(prism.gxz(xp, yp, zp, model), noise),
        utils.contaminate(prism.gyy(xp, yp, zp, model), noise),
        utils.contaminate(prism.gyz(xp, yp, zp, model), noise),
        utils.contaminate(prism.gzz(xp, yp, zp, model), noise)]
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
# Get the eigenvectors from the tensor data
eigenvals, eigenvecs = tensor.eigen(data)
# Use the first eigenvector to estimate the center of mass
cm = tensor.center_of_mass(xp, yp, zp, eigenvecs[0])

# Plot the prism and the estimated center of mass
myv.figure()
myv.points([cm], size=200.)
myv.prisms(model, prop='density', opacity=0.5)
axes = myv.axes(
    myv.outline(extent=[-5000, 5000, -5000, 5000, 0, 5000]))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
