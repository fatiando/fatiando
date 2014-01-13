"""
Seismic: 2D straight-ray tomography using sharpness (total variation)
regularization

Uses synthetic data and a model generated from an image file.
"""
import urllib
from os import path
import numpy
from fatiando.mesher import SquareMesh
from fatiando.seismic import ttime2d, srtomo
from fatiando.inversion.regularization import TotalVariation2D
from fatiando.vis import mpl
from fatiando import utils

area = (0, 100000, 0, 100000)
shape = (20, 20)
model = SquareMesh(area, shape)
# Fetch the image from the online docs
urllib.urlretrieve(
    'http://fatiando.readthedocs.org/en/latest/_static/logo.png', 'logo.png')
vmin, vmax = 4000, 10000
model.img2prop('logo.png', vmin, vmax, 'vp')

# Make some travel time data and add noise
seed = 0 # Set the random seed so that points are the same everythime
src_loc = utils.random_points(area, 80, seed=seed)
rec_loc = utils.circular_points(area, 30, random=True, seed=seed)
srcs, recs = utils.connect_points(src_loc, rec_loc)
tts = ttime2d.straight(model, 'vp', srcs, recs)
tts, error = utils.contaminate(tts, 0.01, percent=True, return_stddev=True)
# Make the mesh
mesh = SquareMesh(area, shape)
# and run the inversion
tomo = (srtomo.SRTomo(tts, srcs, recs, mesh) +
        1*TotalVariation2D(10**-8, mesh.shape))
# Since Total Variation is a non-linear function, then the tomography becomes
# non-linear. So we need to configure fit to use the Levemberg-Marquardt
# algorithm, a gradient descent method, that requires an initial estimate
tomo.config('levmarq', initial=0.0005*numpy.ones(mesh.size)).fit()
residuals = tomo.residuals()
mesh.addprop('vp', tomo.estimate_)

# Calculate and print the standard deviation of the residuals
# it should be close to the data error if the inversion was able to fit the data
print "Assumed error: %f" % (error)
print "Standard deviation of residuals: %f" % (numpy.std(residuals))

mpl.figure(figsize=(14, 5))
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('Vp synthetic model of the Earth')
mpl.squaremesh(model, prop='vp', vmin=vmin, vmax=vmax,
    cmap=mpl.cm.seismic)
cb = mpl.colorbar()
cb.set_label('Velocity')
mpl.points(src_loc, '*y', label="Sources")
mpl.points(rec_loc, '^r', label="Receivers")
mpl.legend(loc='lower left', shadow=True, numpoints=1, prop={'size':10})
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.axis('scaled')
mpl.title('Tomography result (sharp)')
mpl.squaremesh(mesh, prop='vp', vmin=vmin, vmax=vmax,
    cmap=mpl.cm.seismic)
cb = mpl.colorbar()
cb.set_label('Velocity')
mpl.m2km()
mpl.figure()
mpl.grid()
mpl.title('Residuals (data with %.4f s error)' % (error))
mpl.hist(residuals, color='gray', bins=10)
mpl.xlabel("seconds")
mpl.show()
