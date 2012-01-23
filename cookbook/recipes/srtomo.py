"""
Example of running a straight ray tomography on synthetic data generated based
on an image file.
"""
from os import path
import numpy
from matplotlib import pyplot
from fatiando.mesher.dd import SquareMesh
from fatiando.seismic import srtomo, traveltime
from fatiando import vis, logger, utils
from fatiando.inversion import gradient, regularizer

log = logger.get()
log.info(logger.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
area = (0, 5000, 0, 5000)
shape = (50, 50)
model = SquareMesh(area, shape)
model.img2prop(imgfile, 5, 10, 'vp')

log.info("Generating synthetic travel-time data")
src_loc = utils.random_points(area, 20)
rec_loc = utils.circular_points(area, 20, random=False)
srcs, recs = utils.connect_points(src_loc, rec_loc)
ttimes = traveltime.straight_ray_2d(model, 'vp', srcs, recs)

pyplot.figure()
pyplot.title('Vp synthetic model of the Earth')
vis.squaremesh(model, model.props['vp'])
vis.paths(srcs, recs, '-k')
vis.points(src_loc, '*g', size=15)
vis.points(rec_loc, '^r')
pyplot.colorbar()
pyplot.show()

mesh = SquareMesh(area, (20, 20))
estimate, residuals = srtomo.smooth(ttimes, srcs, recs, mesh, damping=0.001)

pyplot.figure()
pyplot.title('Vp synthetic model of the Earth')
vis.squaremesh(mesh, estimate)
pyplot.colorbar()
pyplot.show()
