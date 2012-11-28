"""
Seis: 2D epicenter estimation assuming a homogeneous and flat Earth
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
ptime = ft.seis.ttime2d.straight(model, 'vp', srcs, recs)
stime = ft.seis.ttime2d.straight(model, 'vs', srcs, recs)
ttresiduals, error = ft.utils.contaminate(stime - ptime, 0.10, percent=True,
                                          return_stddev=True)

log.info("Will solve the inverse problem using the Levenberg-Marquardt method")
solver = ft.inversion.gradient.levmarq(initial=(0, 0), maxit=1000, tol=10**(-3))
result = ft.seis.epic2d.homogeneous(ttresiduals, recs, vp, vs, solver)
estimate, residuals = result
predicted = ttresiduals - residuals

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = ft.gridder.regular(area, shape)
goals = ft.seis.epic2d.mapgoal(xs, ys, ttresiduals, recs, vp, vs)

log.info("Plotting")
ft.vis.figure(figsize=(10,4))
ft.vis.subplot(1, 2, 1)
ft.vis.title('Epicenter + %d recording stations' % (len(recs)))
ft.vis.axis('scaled')
ft.vis.contourf(xs, ys, goals, shape, 50)
ft.vis.points(src, '*y', label="True")
ft.vis.points(recs, '^r', label="Stations")
ft.vis.points([estimate], '*g', label="Estimate")
ft.vis.set_area(area)
ft.vis.legend(loc='lower right', shadow=True, numpoints=1, prop={'size':12})
ft.vis.xlabel("X")
ft.vis.ylabel("Y")
ax = ft.vis.subplot(1, 2, 2)
ft.vis.title('Travel-time residuals + 10% error')
s = numpy.arange(len(ttresiduals)) + 1
width = 0.3
ft.vis.bar(s - width, ttresiduals, width, color='g', label="Observed",
           yerr=error)
ft.vis.bar(s, predicted, width, color='r', label="Predicted")
ax.set_xticks(s)
ft.vis.legend(loc='upper right', shadow=True, prop={'size':12})
ft.vis.xlabel("Station number")
ft.vis.ylabel("Travel-time residual")
ft.vis.show()
