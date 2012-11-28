"""
seismic: Show steps taken by different algorithms for 2D epicenter estimation
on a flat Earth
"""
import sys
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [ft.mesher.Square(area, props={'vp':vp, 'vs':vs})]

log.info("Choose the location of the receivers")
ft.vis.figure()
ax = ft.vis.subplot(1, 1, 1)
ft.vis.axis('scaled')
ft.vis.suptitle("Choose the location of the receivers")
rec_points = ft.vis.map.pick_points(area, ax, marker='^', color='r')

log.info("Choose the location of the receivers")
ft.vis.figure()
ax = ft.vis.subplot(1, 1, 1)
ft.vis.axis('scaled')
ft.vis.suptitle("Choose the location of the source")
ft.vis.points(rec_points, '^r')
src = ft.vis.map.pick_points(area, ax, marker='*', color='y')
if len(src) > 1:
    log.error("Don't be greedy! Pick only one point as the source")
    sys.exit()

log.info("Generating synthetic travel-time data")
srcs, recs = ft.utils.connect_points(src, rec_points)
ptime = ft.seismic.ttime2d.straight(model, 'vp', srcs, recs)
stime = ft.seismic.ttime2d.straight(model, 'vs', srcs, recs)
error_level = 0.1
ttr_true = stime - ptime
ttr, error = ft.utils.contaminate(ttr_true, error_level, percent=True,
                                  return_stddev=True)

log.info("Choose the initial estimate for the gradient solvers")
ft.vis.figure()
ax = ft.vis.subplot(1, 1, 1)
ft.vis.axis('scaled')
ft.vis.suptitle("Choose the initial estimate for the gradient solvers")
ft.vis.points(rec_points, '^r')
ft.vis.points(src, '*y')
initial = ft.vis.map.pick_points(area, ax, marker='*', color='k')
if len(initial) > 1:
    log.error("Don't be greedy! Pick only one initial estimate")
    sys.exit()
initial = initial[0]

log.info("Will solve the inverse problem using Newton's method")
nsolver = ft.inversion.gradient.newton(initial)
newton = [initial]
iterator = ft.seismic.epic2d.homogeneous(ttr, recs, vp, vs, nsolver, iterate=True)
for e, r in iterator:
    newton.append(e)
newton_predicted = ttr - r

log.info("and the Steepest Descent method")
sdsolver = ft.inversion.gradient.steepest(initial)
steepest = [initial]
iterator = ft.seismic.epic2d.homogeneous(ttr, recs, vp, vs, sdsolver, iterate=True)
for e, r in iterator:
    steepest.append(e)
steepest_predicted = ttr - r

log.info("... and also the Levenberg-Marquardt algorithm for comparison")
lmsolver = ft.inversion.gradient.levmarq(initial)
levmarq = [initial]
iterator = ft.seismic.epic2d.homogeneous(ttr, recs, vp, vs, lmsolver, iterate=True)
for e, r in iterator:
    levmarq.append(e)
levmarq_predicted = ttr - r

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = ft.gridder.regular(area, shape)
goals = ft.seismic.epic2d.mapgoal(xs, ys, ttr, recs, vp, vs)

log.info("Plotting")
ft.vis.figure(figsize=(14,4))
ft.vis.subplot(1, 3, 1)
ft.vis.title('Epicenter + %d recording stations' % (len(rec_points)))
ft.vis.axis('scaled')
ft.vis.contourf(xs, ys, goals, shape, 50)
ft.vis.points(recs, '^r', label="Stations")
ft.vis.points(newton, '.-r', size=5, label="Newton")
ft.vis.points([newton[-1]], '*r')
ft.vis.points(levmarq, '.-k', size=5, label="Lev-Marq")
ft.vis.points([levmarq[-1]], '*k')
ft.vis.points(steepest, '.-m', size=5, label="Steepest")
ft.vis.points([steepest[-1]], '*m')
ft.vis.points(src, '*y', label="True")
ft.vis.set_area(area)
ft.vis.legend(loc='upper left', shadow=True, numpoints=1, prop={'size':10})
ft.vis.xlabel("X")
ft.vis.ylabel("Y")
ax = ft.vis.subplot(1, 3, 2)
ft.vis.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.2
ft.vis.bar(s - 2*width, ttr, width, color='y', label="Observed", yerr=error)
ft.vis.bar(s - width, newton_predicted, width, color='r', label="Newton")
ft.vis.bar(s, levmarq_predicted, width, color='k', label="Lev-Marq")
ft.vis.bar(s + 1*width, steepest_predicted, width, color='m', label="Steepest")
ft.vis.plot(s - 1.5*width, ttr_true, '^-y', linewidth=2, label="Noise-free")
ax.set_xticks(s)
ft.vis.legend(loc='lower right', shadow=True, prop={'size':10})
ft.vis.ylim(0, 4.5)
ft.vis.xlabel("Station number")
ft.vis.ylabel("Travel-time residual")
ax = ft.vis.subplot(1, 3, 3)
ft.vis.title('Number of iterations')
width = 0.5
ft.vis.bar(1, len(newton), width, color='r', label="Newton")
ft.vis.bar(2, len(levmarq), width, color='k', label="Lev-Marq")
ft.vis.bar(3, len(steepest), width, color='m', label="Steepest")
ax.set_xticks([])
ft.vis.grid()
ft.vis.legend(loc='lower right', shadow=True, prop={'size':10})
ft.vis.show()
