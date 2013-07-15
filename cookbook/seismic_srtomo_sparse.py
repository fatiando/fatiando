"""
Seismic: 2D straight-ray tomography of large data sets and models using
sparse matrices

Uses synthetic data and a model generated from an image file.

Since the image is big, use sparse matrices and a steepest descent solver
(it doesn't require Hessians).

WARNING: may take a long time to calculate.

"""
import urllib
from os import path
import numpy
from fatiando import mesher, utils, seismic, vis, inversion

area = (0, 100000, 0, 100000)
shape = (100, 100)
model = mesher.SquareMesh(area, shape)
# Fetch the image from the online docs
urllib.urlretrieve(
    'http://fatiando.readthedocs.org/en/latest/_static/logo.png', 'logo.png')
model.img2prop('logo.png', 4000, 10000, 'vp')

# Make some travel time data and add noise
src_loc = utils.random_points(area, 200)
rec_loc = utils.circular_points(area, 80, random=True)
srcs, recs = utils.connect_points(src_loc, rec_loc)
ttimes = seismic.ttime2d.straight(model, 'vp', srcs, recs, par=True)
ttimes, error = utils.contaminate(ttimes, 0.01, percent=True,
    return_stddev=True)
# Make the mesh
mesh = mesher.SquareMesh(area, shape)
# Since the matrices are big, use the Steepest Descent solver to avoid dealing
# with Hessian matrices. It needs a starting guess, so start with 1000
inversion.gradient.use_sparse()
solver = inversion.gradient.steepest(1000*numpy.ones(mesh.size))
# and run the inversion
estimate, residuals = seismic.srtomo.run(ttimes, srcs, recs, mesh, sparse=True,
    solver=solver, smooth=0.01)
# Convert the slowness estimate to velocities and add it the mesh
mesh.addprop('vp', seismic.srtomo.slowness2vel(estimate))

# Calculate and print the standard deviation of the residuals
# it should be close to the data error if the inversion was able to fit the data
print "Assumed error: %f" % (error)
print "Standard deviation of residuals: %f" % (numpy.std(residuals))

vis.mpl.figure(figsize=(14, 5))
vis.mpl.subplot(1, 2, 1)
vis.mpl.axis('scaled')
vis.mpl.title('Vp synthetic model of the Earth')
vis.mpl.squaremesh(model, prop='vp', vmin=4000, vmax=10000,
    cmap=vis.mpl.cm.seismic)
cb = vis.mpl.colorbar()
cb.set_label('Velocity')
vis.mpl.points(src_loc, '*y', label="Sources")
vis.mpl.points(rec_loc, '^r', label="Receivers")
vis.mpl.legend(loc='lower left', shadow=True, numpoints=1, prop={'size':10})
vis.mpl.subplot(1, 2, 2)
vis.mpl.axis('scaled')
vis.mpl.title('Tomography result')
vis.mpl.squaremesh(mesh, prop='vp', vmin=4000, vmax=10000,
    cmap=vis.mpl.cm.seismic)
cb = vis.mpl.colorbar()
cb.set_label('Velocity')
vis.mpl.figure()
vis.mpl.grid()
vis.mpl.title('Residuals (data with %.4f s error)' % (error))
vis.mpl.hist(residuals, color='gray', bins=15)
vis.mpl.xlabel("seconds")
vis.mpl.show()
vis.mpl.show()
