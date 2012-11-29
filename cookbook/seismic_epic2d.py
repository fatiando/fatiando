"""
Seismic: 2D epicenter estimation assuming a homogeneous and flat Earth
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
ttresiduals, error = utils.contaminate(stime - ptime, 0.10, percent=True,
                                          return_stddev=True)

log.info("Will solve the inverse problem using the Levenberg-Marquardt method")
solver = inversion.gradient.levmarq(initial=(0, 0), maxit=1000, tol=10**(-3))
result = seismic.epic2d.homogeneous(ttresiduals, recs, vp, vs, solver)
estimate, residuals = result
predicted = ttresiduals - residuals

log.info("Build a map of the goal function")
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
goals = seismic.epic2d.mapgoal(xs, ys, ttresiduals, recs, vp, vs)

log.info("Plotting")
vis.mpl.figure(figsize=(10,4))
vis.mpl.subplot(1, 2, 1)
vis.mpl.title('Epicenter + %d recording stations' % (len(recs)))
vis.mpl.axis('scaled')
vis.mpl.contourf(xs, ys, goals, shape, 50)
vis.mpl.points(src, '*y', label="True")
vis.mpl.points(recs, '^r', label="Stations")
vis.mpl.points([estimate], '*g', label="Estimate")
vis.mpl.set_area(area)
vis.mpl.legend(loc='lower right', shadow=True, numpoints=1, prop={'size':12})
vis.mpl.xlabel("X")
vis.mpl.ylabel("Y")
ax = vis.mpl.subplot(1, 2, 2)
vis.mpl.title('Travel-time residuals + 10% error')
s = numpy.arange(len(ttresiduals)) + 1
width = 0.3
vis.mpl.bar(s - width, ttresiduals, width, color='g', label="Observed",
           yerr=error)
vis.mpl.bar(s, predicted, width, color='r', label="Predicted")
ax.set_xticks(s)
vis.mpl.legend(loc='upper right', shadow=True, prop={'size':12})
vis.mpl.xlabel("Station number")
vis.mpl.ylabel("Travel-time residual")
vis.mpl.show()
