"""
GravMag: Center of mass estimation using the first eigenvector of the gravity
gradient tensor (inverted pyramid model)
"""
from fatiando.vis import mpl, myv
from fatiando import mesher, gridder, utils, gravmag

# Generate some synthetic data
prisms = [mesher.Prism(-2000,2000,-2000,2000,500,1000,{'density':1000}),
          mesher.Prism(-1000,1000,-1000,1000,1000,1500,{'density':1000}),
          mesher.Prism(-500,500,-500,500,1500,2000,{'density':1000})]
shape = (100, 100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
noise = 1
tensor = [utils.contaminate(gravmag.prism.gxx(xp, yp, zp, prisms), noise),
          utils.contaminate(gravmag.prism.gxy(xp, yp, zp, prisms), noise),
          utils.contaminate(gravmag.prism.gxz(xp, yp, zp, prisms), noise),
          utils.contaminate(gravmag.prism.gyy(xp, yp, zp, prisms), noise),
          utils.contaminate(gravmag.prism.gyz(xp, yp, zp, prisms), noise),
          utils.contaminate(gravmag.prism.gzz(xp, yp, zp, prisms), noise)]
# Plot the data
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
for i, title in enumerate(titles):
    mpl.subplot(3, 2, i + 1)
    mpl.title(title)
    mpl.axis('scaled')
    levels = mpl.contourf(yp, xp, tensor[i], shape, 10)
    mpl.contour(yp, xp, tensor[i], shape, levels)
    mpl.m2km()
mpl.show()
# Get the eigenvectors from the tensor data
eigenvals, eigenvecs = gravmag.tensor.eigen(tensor)
# Use the first eigenvector to estimate the center of mass
cm, sigma = gravmag.tensor.center_of_mass(xp, yp, zp, eigenvecs[0])
print "Sigma = %g" % (sigma)
# Plot the prism and the estimated center of mass
myv.figure()
myv.points([cm], size=300.)
myv.prisms(prisms, prop='density', opacity=0.5)
axes = myv.axes(
    myv.outline(extent=[-5000, 5000, -5000, 5000, 0, 5000]))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
