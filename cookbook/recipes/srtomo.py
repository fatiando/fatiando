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
shape = (20, 20)
model = SquareMesh(area, shape)
model.img2prop(imgfile, 5, 10, 'vp')

src_loc = utils.random_points(area, 20)
rec_loc = utils.circular_points(area, 20, random=False)
src, rec = utils.connect_points(src_loc, rec_loc)
ttimes = traveltime.straight_ray_2d(model, 'vp', src, rec)

mesh = SquareMesh(area, (20, 20))
regs = [regularizer.Damping(0.001)]
dms = [srtomo.TravelTimeStraightRay2D(ttimes, src, rec, mesh)]
for chset in gradient.newton(dms, numpy.zeros(mesh.size, dtype='f'), regs, tol=0.001):
    continue

pyplot.figure()
pyplot.title('Vp synthetic model of the Earth')
vis.squaremesh(model, model.props['vp'])
vis.paths(src, rec, '-k')
vis.points(src_loc, '*g', size=15)
vis.points(rec_loc, '^r')
pyplot.colorbar()

pyplot.figure()
pyplot.title('Vp synthetic model of the Earth')
vis.squaremesh(mesh, chset['estimate'])
pyplot.colorbar()

pyplot.show()
