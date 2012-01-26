"""
Example of epicenter estimation assuming a homogeneous and flat Earth.
"""
from matplotlib import pyplot
from fatiando.mesher.dd import Square
from fatiando.seismic import epicenter, traveltime
from fatiando import vis, logger, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp = 2
vs = 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Generating synthetic travel-time data")
src = (5, 5)
srcs, recs = utils.connect_points([src], utils.random_points(area, 3))
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
ttresiduals = stime - ptime

initial = (0, 0)
estimate, residuals = epicenter.flat_homogeneous(ttresiduals, recs, vp, vs,
                                                 initial)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title('Epicenter + recording stations')
vis.points([src], '*y', label="True")
vis.points(recs, '^r', label="Stations")
vis.points([estimate], '*b', label="Estimate")
vis.set_area(area)
pyplot.legend(loc='lower right', shadow=True)

pyplot.figure()
pyplot.title('Travel-time residuals')
pyplot.plot(ttresiduals, '-*b', label="Observed")
pyplot.plot(ttresiduals - residuals, '-*r', label="Predicted")
pyplot.xlim(-1, len(recs))
pyplot.legend(loc='lower right', shadow=True)
pyplot.show()
