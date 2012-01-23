"""
Example of running a straight ray tomography on synthetic data generated based
on an image file.
"""
from os import path
from matplotlib import pyplot
from fatiando.mesher.dd import SquareMesh
from fatiando.seismic import srtomo, traveltime
from fatiando import vis, logger, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
area = (0, 5, 0, 5)
shape = (30, 30)
model = SquareMesh(area, shape)
model.img2prop(imgfile, 5, 10, 'vp')

log.info("Generating synthetic travel-time data")
src_loc = utils.random_points(area, 50)
rec_loc = utils.circular_points(area, 50)
srcs, recs = utils.connect_points(src_loc, rec_loc)
ttimes = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
#ttimes = utils.contaminate(traveltime.straight_ray_2d(model, 'vp', srcs, recs),
                           #0.01, percent=True)

mesh = SquareMesh(area, shape)
estimate, residuals = srtomo.smooth(ttimes, srcs, recs, mesh, damping=0.1)

pyplot.figure()
pyplot.title('Vp synthetic model of the Earth')
vis.squaremesh(model, model.props['vp'])
#vis.paths(srcs, recs, '-k')
vis.points(src_loc, '*g', size=15)
vis.points(rec_loc, '^r')
pyplot.colorbar()
pyplot.figure()
pyplot.title('Tomography result')
vis.squaremesh(mesh, estimate, vmin=5, vmax=10)
pyplot.colorbar()
pyplot.show()
