"""
Seismic: Show steps taken by different algorithms for 2D epicenter estimation
on a flat Earth
"""
import sys
import numpy
from fatiando import logger, mesher, seismic, utils, gridder, vis, inversion

log = logger.get()
log.info(logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [mesher.Square(area, props={'vp':vp, 'vs':vs})]

log.info("Choose the location of the receivers")
vis.mpl.figure()
ax = vis.mpl.subplot(1, 1, 1)
vis.mpl.axis('scaled')
vis.mpl.suptitle("Choose the location of the receivers")
rec_points = vis.mpl.pick_points(area, ax, marker='^', color='r')

log.info("Choose the location of the receivers")
vis.mpl.figure()
ax = vis.mpl.subplot(1, 1, 1)
vis.mpl.axis('scaled')
vis.mpl.suptitle("Choose the location of the source")
vis.mpl.points(rec_points, '^r')
src = vis.mpl.pick_points(area, ax, marker='*', color='y')
if len(src) > 1:
    log.error("Don't be greedy! Pick only one point as the source")
    sys.exit()

log.info("Generating synthetic travel-time data")
srcs, recs = utils.connect_points(src, rec_points)
ptime = seismic.ttime2d.straight(model, 'vp', srcs, recs)
stime = seismic.ttime2d.straight(model, 'vs', srcs, recs)
error_level = 0.1
ttr_true = stime - ptime
ttr, error = utils.contaminate(ttr_true, error_level, percent=True,
                                  return_stddev=True)

log.info("Choose the initial estimate for the gradient solvers")
vis.mpl.figure()
ax = vis.mpl.subplot(1, 1, 1)
vis.mpl.axis('scaled')
vis.mpl.suptitle("Choose the initial estimate for the gradient solvers")
vis.mpl.points(rec_points, '^r')
vis.mpl.points(src, '*y')
initial = vis.mpl.pick_points(area, ax, marker='*', color='k')
if len(initial) > 1:
    log.error("Don't be greedy! Pick only one initial estimate")
    sys.exit()
initial = initial[0]

log.info("Will solve the inverse problem using Newton's method")
nsolver = inversion.gradient.newton(initial)
newton = [initial]
iterator = seismic.epic2d.homogeneous(ttr, recs, vp, vs, nsolver, iterate=True)
for e, r in iterator:
    newton.append(e)
newton_predicted = ttr - r

log.info("and the Steepest Descent method")
sdsolver = inversion.gradient.steepest(initial)
steepest = [initial]
iterator = seismic.epic2d.homogeneous(ttr, recs, vp, vs, sdsolver, iterate=True)
for e, r in iterator:
    steepest.append(e)
steepest_predicted = ttr - r

log.info("... and also the Levenberg-Marquardt algorithm for comparison")
lmsolver = inversion.gradient.levmarq(initial)
levmarq = [initial]
iterator = seismic.epic2d.homogeneous(ttr, recs, vp, vs, lmsolver, iterate=True)
for e, r in iterator:
    levmarq.append(e)
levmarq_predicted = ttr - r

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
goals = seismic.epic2d.mapgoal(xs, ys, ttr, recs, vp, vs)

log.info("Plotting")
vis.mpl.figure(figsize=(14,4))
vis.mpl.subplot(1, 3, 1)
vis.mpl.title('Epicenter + %d recording stations' % (len(rec_points)))
vis.mpl.axis('scaled')
vis.mpl.contourf(xs, ys, goals, shape, 50)
vis.mpl.points(recs, '^r', label="Stations")
vis.mpl.points(newton, '.-r', size=5, label="Newton")
vis.mpl.points([newton[-1]], '*r')
vis.mpl.points(levmarq, '.-k', size=5, label="Lev-Marq")
vis.mpl.points([levmarq[-1]], '*k')
vis.mpl.points(steepest, '.-m', size=5, label="Steepest")
vis.mpl.points([steepest[-1]], '*m')
vis.mpl.points(src, '*y', label="True")
vis.mpl.set_area(area)
vis.mpl.legend(loc='upper left', shadow=True, numpoints=1, prop={'size':10})
vis.mpl.xlabel("X")
vis.mpl.ylabel("Y")
ax = vis.mpl.subplot(1, 3, 2)
vis.mpl.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.2
vis.mpl.bar(s - 2*width, ttr, width, color='y', label="Observed", yerr=error)
vis.mpl.bar(s - width, newton_predicted, width, color='r', label="Newton")
vis.mpl.bar(s, levmarq_predicted, width, color='k', label="Lev-Marq")
vis.mpl.bar(s + 1*width, steepest_predicted, width, color='m', label="Steepest")
vis.mpl.plot(s - 1.5*width, ttr_true, '^-y', linewidth=2, label="Noise-free")
ax.set_xticks(s)
vis.mpl.legend(loc='lower right', shadow=True, prop={'size':10})
vis.mpl.ylim(0, 4.5)
vis.mpl.xlabel("Station number")
vis.mpl.ylabel("Travel-time residual")
ax = vis.mpl.subplot(1, 3, 3)
vis.mpl.title('Number of iterations')
width = 0.5
vis.mpl.bar(1, len(newton), width, color='r', label="Newton")
vis.mpl.bar(2, len(levmarq), width, color='k', label="Lev-Marq")
vis.mpl.bar(3, len(steepest), width, color='m', label="Steepest")
ax.set_xticks([])
vis.mpl.grid()
vis.mpl.legend(loc='lower right', shadow=True, prop={'size':10})
vis.mpl.show()
