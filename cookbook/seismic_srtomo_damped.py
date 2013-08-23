"""
Seismic: 2D straight-ray tomography using damping regularization

Uses synthetic data and a model generated from an image file.
"""
import urllib
import time
import numpy
from fatiando import mesher, utils, seismic, vis

area = (0, 500000, 0, 500000)
shape = (30, 30)
model = mesher.SquareMesh(area, shape)
# Fetch the image from the online docs
urllib.urlretrieve(
    'http://fatiando.readthedocs.org/en/latest/_static/logo.png', 'logo.png')
model.img2prop('logo.png', 4000, 10000, 'vp')

# Make some travel time data and add noise
seed = 0 # Set the random seed so that points are the same everythime
src_loc = utils.random_points(area, 80, seed=seed)
rec_loc = utils.circular_points(area, 30, random=True, seed=seed)
srcs, recs = utils.connect_points(src_loc, rec_loc)
start = time.time()
tts = seismic.ttime2d.straight(model, 'vp', srcs, recs, par=True)
tts, error = utils.contaminate(tts, 0.01, percent=True, return_stddev=True)
# Make the mesh
mesh = mesher.SquareMesh(area, shape)
# and run the inversion
estimate, residuals = seismic.srtomo.run(tts, srcs, recs, mesh, damping=10**9)
# Convert the slowness estimate to velocities and add it the mesh
mesh.addprop('vp', seismic.srtomo.slowness2vel(estimate))

# Calculate and print the standard deviation of the residuals
# it should be close to the data error if the inversion was able to fit the data
print "Assumed error: %g" % (error)
print "Standard deviation of residuals: %g" % (numpy.std(residuals))

vis.mpl.figure(figsize=(14, 5))
vis.mpl.subplot(1, 2, 1)
vis.mpl.axis('scaled')
vis.mpl.title('Vp synthetic model of the Earth')
vis.mpl.squaremesh(model, prop='vp', cmap=vis.mpl.cm.seismic)
cb = vis.mpl.colorbar()
cb.set_label('Velocity')
vis.mpl.points(src_loc, '*y', label="Sources")
vis.mpl.points(rec_loc, '^r', label="Receivers")
vis.mpl.legend(loc='lower left', shadow=True, numpoints=1, prop={'size':10})
vis.mpl.m2km()
vis.mpl.subplot(1, 2, 2)
vis.mpl.axis('scaled')
vis.mpl.title('Tomography result (damped)')
vis.mpl.squaremesh(mesh, prop='vp', vmin=4000, vmax=10000,
    cmap=vis.mpl.cm.seismic)
cb = vis.mpl.colorbar()
cb.set_label('Velocity')
vis.mpl.m2km()
vis.mpl.figure()
vis.mpl.grid()
vis.mpl.title('Residuals (data with %.4f s error)' % (error))
vis.mpl.hist(residuals, color='gray', bins=10)
vis.mpl.xlabel("seconds")
vis.mpl.show()
