"""
Example of epicenter estimation assuming a homogeneous and flat Earth.
Show all steps of the non-linear solver algorithm.
"""
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import Square
from fatiando.seismic import epicenter, traveltime
from fatiando import vis, logger, utils, inversion, gridder

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Generating synthetic travel-time data")
src = (8, 6)
circ_area = (1, 9, 1, 9)
srcs, recs = utils.connect_points([src], utils.circular_points(circ_area, 4))
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
error_level = 0.05
ttr, error = utils.contaminate(stime - ptime, error_level, percent=True,
                               return_stddev=True)
    
initial = (0.5, 0.5)
log.info("Will solve the inverse problem using Newton's method")
nsolver = inversion.gradient.newton(initial)
newton = [initial]
iterator = epicenter.iterate_flat(ttr, recs, vp, vs, nsolver)
for e, r in iterator:
    newton.append(e)
newton_predicted = ttr - r

log.info("and the Steepest Descent method")
sdsolver = inversion.gradient.steepest(initial, step=0.1)
steepest = [initial]
iterator = epicenter.iterate_flat(ttr, recs, vp, vs, sdsolver)
for e, r in iterator:
    steepest.append(e)
steepest_predicted = ttr - r

log.info("... and also the Levemberg-Marquardt algorithm for comparison")
lmsolver = inversion.gradient.levmarq(initial, damp=0.1)
levmarq = [initial]
iterator = epicenter.iterate_flat(ttr, recs, vp, vs, lmsolver)
for e, r in iterator:
    levmarq.append(e)
levmarq_predicted = ttr - r

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
goals = epicenter.mapgoal(xs, ys, ttr, recs, vp, vs)

log.info("Plotting")
pyplot.figure(figsize=(14,6))
pyplot.subplot(1, 2, 1)
pyplot.title('Epicenter + recording stations')
pyplot.axis('scaled')
vis.pcolor(xs, ys, goals, shape)
vis.points(recs, '^r', label="Stations")
vis.points(newton, '.-c', size=5, label="Newton")
vis.points([newton[-1]], '*c')
vis.points(levmarq, '.-g', size=5, label="Lev-Marq")
vis.points([levmarq[-1]], '*g')
vis.points(steepest, '.-m', size=5, label="Steepest")
vis.points([steepest[-1]], '*m')
vis.points([src], '*y', label="True")
vis.set_area(area)
pyplot.legend(loc='upper left', shadow=True, numpoints=1, prop={'size':12})
pyplot.xlabel("X")
pyplot.ylabel("Y")
ax = pyplot.subplot(1, 2, 2)
pyplot.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.1
pyplot.bar(s - width, ttr, width, color='y', label="Observed", yerr=error)
pyplot.bar(s, newton_predicted, width, color='c', label="Newton")
pyplot.bar(s + width, levmarq_predicted, width, color='g', label="Lev-Marq")
pyplot.bar(s + 2*width, steepest_predicted, width, color='m', label="Steepest")
ax.set_xticks(s)
pyplot.legend(loc='upper left', shadow=True, prop={'size':12})
pyplot.xlabel("Station number")
pyplot.ylabel("Travel-time residual")
pyplot.savefig("sample-epicenter.png", dpi=300)
pyplot.show()
