"""
Example of epicenter estimation assuming a homogeneous and flat Earth.
"""
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import Square
from fatiando.seismic import epicenter, traveltime
from fatiando import vis, logger, utils, inversion

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Generating synthetic travel-time data")
src = (5, 5)
srcs, recs = utils.connect_points([src], utils.random_points(area, 4))
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
ttresiduals, error = utils.contaminate(stime - ptime, 0.10, percent=True,
                                       return_stddev=True)

log.info("Will solve the inverse problem using Newton's method")
solver = inversion.gradient.newton(initial=(0, 0), maxit=1000, tol=10**(-3))
result = epicenter.solve_flat(ttresiduals, recs, vp, vs, solver)
estimate, residuals = result
predicted = ttresiduals - residuals

log.info("Plotting")
pyplot.figure(figsize=(10,4))
pyplot.subplot(1, 2, 1)
pyplot.title('Epicenter + recording stations')
pyplot.axis('scaled')
vis.points([src], '*y', label="True")
vis.points(recs, '^r', label="Stations")
vis.points([estimate], '*g', label="Estimate")
vis.set_area(area)
pyplot.legend(loc='lower right', shadow=True, numpoints=1, prop={'size':12})
pyplot.xlabel("X")
pyplot.ylabel("Y")
ax = pyplot.subplot(1, 2, 2)
pyplot.title('Travel-time residuals + 10% error')
s = numpy.arange(len(ttresiduals)) + 1
width = 0.3
pyplot.bar(s - width, ttresiduals, width, color='g', label="Observed",
           yerr=error)
pyplot.bar(s, predicted, width, color='r', label="Predicted")
ax.set_xticks(s)
pyplot.legend(loc='upper right', shadow=True, prop={'size':12})
pyplot.xlabel("Station number")
pyplot.ylabel("Travel-time residual")
pyplot.show()
