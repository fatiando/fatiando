"""
GravMag: Classic 3D Euler deconvolution of magnetic data using an
expanding window
"""
import numpy as np
from fatiando.mesher import Prism
from fatiando import gridder, utils
from fatiando.gravmag import prism, fourier
from fatiando.gravmag.euler import Classic, ExpandingWindow
from fatiando.vis import mpl, myv

# The regional field
inc, dec = -45, 0
# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    Prism(-1500, -500, -1500, -500, 1000, 2000, {'magnetization': 2}),
    Prism(500, 1500, 1000, 2000, 1000, 2000, {'magnetization': 2})]
# Generate some data from the model
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert from nanoTesla to Tesla because euler and derivatives require things
# in SI
tf = (utils.nt2si(prism.tf(xp, yp, zp, model, inc, dec))
      + baselevel)
# Calculate the derivatives using FFT
xderiv = fourier.derivx(xp, yp, tf, shape)
yderiv = fourier.derivy(xp, yp, tf, shape)
zderiv = fourier.derivz(xp, yp, tf, shape)

mpl.figure()
titles = ['Total field', 'x derivative', 'y derivative', 'z derivative']
for i, f in enumerate([tf, xderiv, yderiv, zderiv]):
    mpl.subplot(2, 2, i + 1)
    mpl.title(titles[i])
    mpl.axis('scaled')
    mpl.contourf(yp, xp, f, shape, 50)
    mpl.colorbar()
    mpl.m2km()
mpl.show()

# Pick the centers of the expanding windows
# The number of final solutions will be the number of points picked
mpl.figure()
mpl.suptitle('Pick the centers of the expanding windows')
mpl.axis('scaled')
mpl.contourf(yp, xp, tf, shape, 50)
mpl.colorbar()
centers = mpl.pick_points(area, mpl.gca(), xy2ne=True)

# Run the euler deconvolution on an expanding window
# Structural index is 3
euler = Classic(xp, yp, zp, tf, xderiv, yderiv, zderiv, 3)
sizes = np.linspace(500, 5000, 20)
results = [ExpandingWindow(euler, c, sizes).fit() for c in centers]

print "Base level used: %g" % (baselevel)
for i, res in enumerate(results):
    print "Center %d:" % (i + 1)
    print "  Base level: %g" % (res.baselevel_)
    print "  Source location: %s" % (str(res.estimate_))

myv.figure()
myv.points([r.estimate_ for r in results], size=100.)
myv.prisms(model, opacity=0.5)
axes = myv.axes(myv.outline(bounds), ranges=[b * 0.001 for b in bounds])
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
