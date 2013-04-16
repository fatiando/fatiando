"""
GravMag: Classic 3D Euler deconvolution of noisy magnetic data (single window)
"""
from fatiando import logger, mesher, gridder, utils, gravmag
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# The regional field
inc, dec = 30, -15
# Use only induced magnetization with and amplitude of 2 A/m
mag = utils.ang2vec(2, inc, dec)
# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    mesher.Prism(-500, 500, -500, 500, 1000, 2000, {'magnetization':mag})]
# Generate some data from the model
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert from nanoTesla to Tesla because euler and derivatives require things
# in SI
tf = (utils.contaminate(
        utils.nt2si(gravmag.prism.tf(xp, yp, zp, model, inc, dec)),
        0.005, percent=True)
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

# Run the euler deconvolution on a single window
# Structural index is 3
results = gravmag.euler.classic(xp, yp, zp, tf, xderiv, yderiv, zderiv, 3)
print "Base level used: %g" % (baselevel)
print "Estimated base level: %g" % (results['baselevel'])

myv.figure()
myv.points([results['point']], size=300.)
myv.prisms(model, prop='magnetization', opacity=0.5)
axes = myv.axes(myv.outline(extent=bounds))
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
