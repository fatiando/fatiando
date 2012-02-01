"""
Example when epicenter estimation requires regularization.
Use an equality constraint on one of the coordinates to stabilize the problem.
Try setting equality=0 and see what happens! Make sure to run a few times since
the noise is random and you might get lucky!
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

log.info("The data are noisy and receiver locations are bad.")
log.info("So use a bit of regularization.")

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Generating synthetic travel-time data")
src = (8, 7)
stations = 10
srcs, recs = utils.connect_points([src], [(4, 6), (5, 5.9), (6, 6)])
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
error_level = 0.05
ttr_true = stime - ptime
ttr, error = utils.contaminate(ttr_true, error_level, percent=True,
                               return_stddev=True)
    
log.info("Choose the initial estimate for the gradient solvers")
pyplot.figure()
ax = pyplot.subplot(1, 1, 1)
pyplot.axis('scaled')
pyplot.suptitle("Choose the initial estimate for the gradient solvers")
vis.map.points(recs, '^r')
vis.map.points(srcs, '*y')
initial = ui.picker.points(area, ax, marker='*', color='k')
if len(initial) > 1:
    log.error("Don't be greedy! Pick only one initial estimate")
    sys.exit()
initial = initial[0]

ref = {'y':7}
equality = 0.1
log.info("Will solve the inverse problem using Newton's method")
log.info("and with equality constaints for stability")
nsolver = inversion.gradient.newton(initial)
newton = [initial]
iterator = epicenter.flat_earth(ttr, recs, vp, vs, nsolver, equality=equality,
                                ref=ref, iterate=True)
for e, r in iterator:
    newton.append(e)
newton_predicted = ttr - r

log.info("and the Steepest Descent method")
sdsolver = inversion.gradient.steepest(initial, step=0.1)
steepest = [initial]
iterator = epicenter.flat_earth(ttr, recs, vp, vs, sdsolver, equality=equality,
                                ref=ref, iterate=True)
for e, r in iterator:
    steepest.append(e)
steepest_predicted = ttr - r

log.info("... and also the Levenberg-Marquardt algorithm for comparison")
lmsolver = inversion.gradient.levmarq(initial)
levmarq = [initial]
iterator = epicenter.flat_earth(ttr, recs, vp, vs, lmsolver, equality=equality,
                                ref=ref, iterate=True)
for e, r in iterator:
    levmarq.append(e)
levmarq_predicted = ttr - r

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
goals = epicenter.mapgoal(xs, ys, ttr, recs, vp, vs, equality=equality, ref=ref)

log.info("Plotting")
pyplot.figure(figsize=(14, 4))
pyplot.subplot(1, 3, 1)
pyplot.title('Epicenter + recording stations')
pyplot.axis('scaled')
vis.map.contourf(xs, ys, goals, shape, 50)
vis.map.points(recs, '^r', label="Stations")
vis.map.points(newton, '.-r', size=5, label="Newton")
vis.map.points([newton[-1]], '*r')
vis.map.points(levmarq, '.-k', size=5, label="Lev-Marq")
vis.map.points([levmarq[-1]], '*k')
vis.map.points(steepest, '.-m', size=5, label="Steepest")
vis.map.points([steepest[-1]], '*m')
vis.map.points([src], '*y', label="True")
vis.map.set_area(area)
pyplot.legend(loc='upper left', shadow=True, numpoints=1, prop={'size':10})
pyplot.xlabel("X")
pyplot.ylabel("Y")
ax = pyplot.subplot(1, 3, 2)
pyplot.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.2
pyplot.bar(s - 2*width, ttr, width, color='y', label="Observed", yerr=error)
pyplot.bar(s - width, newton_predicted, width, color='r', label="Newton")
pyplot.bar(s, levmarq_predicted, width, color='k', label="Lev-Marq")
pyplot.bar(s + 1*width, steepest_predicted, width, color='m', label="Steepest")
pyplot.plot(s - 1.5*width, ttr_true, '^-y', linewidth=2, label="Noise-free")
ax.set_xticks(s)
pyplot.legend(loc='lower right', shadow=True, prop={'size':10})
pyplot.xlabel("Station number")
pyplot.ylabel("Travel-time residual")
ax = pyplot.subplot(1, 3, 3)
pyplot.title('Number of iterations')
width = 0.5
pyplot.bar(1, len(newton), width, color='r', label="Newton")
pyplot.bar(2, len(levmarq), width, color='k', label="Lev-Marq")
pyplot.bar(3, len(steepest), width, color='m', label="Steepest")
ax.set_xticks([])
pyplot.grid()
pyplot.legend(loc='upper left', shadow=True, prop={'size':10})
pyplot.show()
