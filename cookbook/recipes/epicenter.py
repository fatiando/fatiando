"""
Example of epicenter estimation assuming a homogeneous and flat Earth.
"""
import sys
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import Square
from fatiando.seismic import epicenter, traveltime
from fatiando import vis, logger, utils, inversion, gridder, ui

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Choose the location of the receivers")
pyplot.figure()
ax = pyplot.subplot(1, 1, 1)
pyplot.axis('scaled')
pyplot.suptitle("Choose the location of the receivers")
rec_points = ui.picker.points(area, ax, marker='^', color='r')

log.info("Choose the location of the receivers")
pyplot.figure()
ax = pyplot.subplot(1, 1, 1)
pyplot.axis('scaled')
pyplot.suptitle("Choose the location of the source")
vis.map.points(rec_points, '^r')
src = ui.picker.points(area, ax, marker='*', color='y')
if len(src) > 1:
    log.error("Don't be greedy! Pick only one point as the source")
    sys.exit()

log.info("Generating synthetic travel-time data")
srcs, recs = utils.connect_points(src, rec_points)
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
ttresiduals, error = utils.contaminate(stime - ptime, 0.10, percent=True,
                                       return_stddev=True)

log.info("Will solve the inverse problem using the Levenberg-Marquardt method")
solver = inversion.gradient.levmarq(initial=(0, 0), maxit=1000, tol=10**(-3))
result = epicenter.flat_earth(ttresiduals, recs, vp, vs, solver)
estimate, residuals = result
predicted = ttresiduals - residuals

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
goals = epicenter.mapgoal(xs, ys, ttresiduals, recs, vp, vs)

log.info("Plotting")
pyplot.figure(figsize=(10,4))
pyplot.subplot(1, 2, 1)
pyplot.title('Epicenter + %d recording stations' % (len(recs)))
pyplot.axis('scaled')
vis.map.contourf(xs, ys, goals, shape, 50)
vis.map.points(src, '*y', label="True")
vis.map.points(recs, '^r', label="Stations")
vis.map.points([estimate], '*g', label="Estimate")
vis.map.set_area(area)
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
