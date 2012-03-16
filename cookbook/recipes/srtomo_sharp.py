"""
Example of running a 2D straight-ray tomography on synthetic data generated
based on an image file. Uses sharpness (total variation) regularization.
"""
from os import path
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import SquareMesh
from fatiando.seismic import srtomo, traveltime
from fatiando import vis, logger, utils, inversion

log = logger.get()
log.info(logger.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
area = (0, 5, 0, 5)
shape = (30, 30)
model = SquareMesh(area, shape)
model.img2prop(imgfile, 4, 10, 'vp')

log.info("Generating synthetic travel-time data")
src_loc = utils.random_points(area, 80)
rec_loc = utils.circular_points(area, 30, random=True)
srcs, recs = utils.connect_points(src_loc, rec_loc)
tts, error = utils.contaminate(
    traveltime.straight_ray_2d(model, 'vp', srcs, recs), 0.01, percent=True,
    return_stddev=True)

mesh = SquareMesh(area, shape)
log.info("Since Total Variation is non-linear, need to use a gradient solver")
solver = inversion.gradient.steepest(numpy.zeros(mesh.size))
results = srtomo.run(tts, srcs, recs, mesh, solver, sharp=0.01, beta=10**(-5))
estimate, residuals = results
mesh.addprop('vp', estimate)

log.info("Assumed error: %g" % (error))
log.info("Standard deviation of residuals: %g" % (numpy.std(residuals)))

pyplot.figure(figsize=(14, 5))
pyplot.subplot(1, 2, 1)
pyplot.axis('scaled')
pyplot.title('Vp synthetic model of the Earth')
vis.map.squaremesh(model, prop='vp', cmap=pyplot.cm.seismic)
cb = pyplot.colorbar()
cb.set_label('Velocity')
vis.map.points(src_loc, '*y', label="Sources")
vis.map.points(rec_loc, '^r', label="Receivers")
pyplot.legend(loc='lower left', shadow=True, numpoints=1, prop={'size':10})
pyplot.subplot(1, 2, 2)
pyplot.axis('scaled')
pyplot.title('Tomography result (sharp)')
vis.map.squaremesh(mesh, prop='vp', vmin=0.1, vmax=0.25,
    cmap=pyplot.cm.seismic_r)
cb = pyplot.colorbar()
cb.set_label('Slowness')
pyplot.figure()
pyplot.grid()
pyplot.title('Residuals (data with %.4f s error)' % (error))
pyplot.hist(residuals, color='gray', bins=10)
pyplot.xlabel("seconds")
pyplot.show()
