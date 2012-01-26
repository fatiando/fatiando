"""
Example of epicenter estimation assuming a homogeneous and flat Earth.
Show all steps of the non-linear solver algorithm.
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
error_level = 0.05
ttr, error = utils.contaminate(stime - ptime, error_level, percent=True,
                               return_stddev=True)

log.info("Will solve the inverse problem using Newton's method")
initial = (1, 1)
solver = inversion.gradient.newton(initial, maxit=1000, tol=10**(-3))
log.info("Record all iterations of the algorithm")
steps = [initial]
for e, r in epicenter.iterate_flathomogeneous(ttr, recs, vp, vs, solver):
    steps.append(e)
predicted = ttr - r

log.info("Plotting")
pyplot.figure(figsize=(10,4))
pyplot.subplot(1, 2, 1)
pyplot.title('Epicenter + recording stations')
pyplot.axis('scaled')
vis.points([src], '*y', label="True")
vis.points(recs, '^r', label="Stations")
vis.points(steps, '-*b', label="Steps")
vis.points([e], '*g', label="Estimate")
vis.set_area(area)
pyplot.legend(loc='lower right', shadow=True, numpoints=1, prop={'size':12})
pyplot.xlabel("X")
pyplot.ylabel("Y")
ax = pyplot.subplot(1, 2, 2)
pyplot.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.3
pyplot.bar(s - width, ttr, width, color='g', label="Observed", yerr=error)
pyplot.bar(s, predicted, width, color='r', label="Predicted")
ax.set_xticks(s)
pyplot.legend(loc='upper right', shadow=True, prop={'size':12})
pyplot.xlabel("Station number")
pyplot.ylabel("Travel-time residual")
pyplot.show()
