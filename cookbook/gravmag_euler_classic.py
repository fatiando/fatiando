"""
GravMag: Classic 3D Euler deconvolution of magnetic data (single window)
"""
from fatiando.mesher import Prism
from fatiando import gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.gravmag.euler import Classic
from fatiando.vis import mpl, myv

# The regional field
inc, dec = -45, 0
# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    Prism(-1500, -500, -500, 500, 1000, 2000, {'magnetization': 2})]
# Generate some data from the model
shape = (200, 200)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert from nanoTesla to Tesla because euler and derivatives require things
# in SI
tf = (utils.nt2si(prism.tf(xp, yp, zp, model, inc, dec)) + baselevel)
# Calculate the derivatives using FFT
xderiv = transform.derivx(xp, yp, tf, shape)
yderiv = transform.derivy(xp, yp, tf, shape)
zderiv = transform.derivz(xp, yp, tf, shape)

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

# Run the Euler deconvolution on the whole dataset
euler = Classic(xp, yp, zp, tf, xderiv, yderiv, zderiv, 3).fit()
print "Base level used: %g" % (baselevel)
print "Estimated:"
print "  Base level:             %g" % (euler.baselevel_)
print "  Source location:        %s" % (str(euler.estimate_))

myv.figure()
myv.points([euler.estimate_], size=100.)
myv.prisms(model, 'magnetization', opacity=0.5)
axes = myv.axes(myv.outline(extent=bounds))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
