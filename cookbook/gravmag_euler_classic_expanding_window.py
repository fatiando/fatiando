"""
GravMag: Classic 3D Euler deconvolution of magnetic data using an
expanding window
"""
from fatiando import logger, mesher, gridder, utils, gravmag
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    mesher.Prism(-1500, -500, -1500, -500, 1000, 2000, {'magnetization':2}),
    mesher.Prism(500, 1500, 500, 2000, 1000, 2000, {'magnetization':2})]
# Generate some data from the model
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert from nanoTesla to Tesla because euler and derivatives require things
# in SI
tf = (utils.nt2si(gravmag.prism.tf(xp, yp, zp, model, inc=-45, dec=0))
      + baselevel)
# Calculate the derivatives using FFT
xderiv = gravmag.fourier.derivx(xp, yp, tf, shape)
yderiv = gravmag.fourier.derivy(xp, yp, tf, shape)
zderiv = gravmag.fourier.derivz(xp, yp, tf, shape)

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

# Pick the center of the expanding window
mpl.figure()
mpl.suptitle('Pick the center of the expanding window')
mpl.axis('scaled')
mpl.contourf(yp, xp, tf, shape, 50)
mpl.colorbar()
center = mpl.pick_points(area, mpl.gca(), xy2ne=True)[0]

# Run the euler deconvolution on an expanding window
# Structural index is 3
index = 3
results = gravmag.euler.expanding_window(xp, yp, zp, tf, xderiv, yderiv, zderiv,
    index, gravmag.euler.classic, center, 500, 5000)
print "Base level used: %g" % (baselevel)
print "Estimated base level: %g" % (results['baselevel'])
print "Estimated source location: %s" % (str(results['point']))

myv.figure()
myv.points([results['point']], size=300.)
myv.prisms(model, prop='magnetization', opacity=0.5)
axes = myv.axes(myv.outline(extent=bounds))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
