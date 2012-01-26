"""
Example when epicenter estimation requires regularization.
"""
from matplotlib import pyplot
import numpy
from fatiando.mesher.dd import Square
from fatiando.seismic import epicenter, traveltime
from fatiando import vis, logger, utils, inversion

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("The data are noisy and receiver locations are bad.")
log.info("So use the Minimum Distance from Receivers regularization.")

area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp':vp, 'vs':vs})]

log.info("Generating synthetic travel-time data")
src = (7, 5)
srcs, recs = utils.connect_points([src], [(0.5, 7), (5, 7), (9.5, 7)])
ptime = traveltime.straight_ray_2d(model, 'vp', srcs, recs)
stime = traveltime.straight_ray_2d(model, 'vs', srcs, recs)
error_level = 0.1
ttr, error = utils.contaminate(stime - ptime, error_level, percent=True,
                               return_stddev=True)
    
initial = (0, 1)
mindist = 5.
log.info("Will solve the inverse problem using Newton's method")
nsolver = inversion.gradient.newton(initial)
newton = [initial]
iterator = epicenter.iterate_flat(ttr, recs, vp, vs, nsolver, mindist=mindist)
for e, r in iterator:
    newton.append(e)
newton_predicted = ttr - r

log.info("... and also the Levemberg-Marquardt algorithm for comparison")
lmsolver = inversion.gradient.levmarq(initial, damp=0.1)
levmarq = [initial]
iterator = epicenter.iterate_flat(ttr, recs, vp, vs, lmsolver, mindist=mindist)
for e, r in iterator:
    levmarq.append(e)
levmarq_predicted = ttr - r

log.info("Plotting")
pyplot.figure(figsize=(14,6))
pyplot.subplot(1, 2, 1)
pyplot.title('Epicenter + recording stations')
pyplot.axis('scaled')
vis.points(recs, '^r', label="Stations")
vis.points(newton, '.-c', size=7)
vis.points([newton[-1]], '*c', label="Newton")
vis.points(levmarq, '.-g', size=7)
vis.points([levmarq[-1]], '*g', label="Lev-Marq")
vis.points([src], '*y', label="True")
vis.set_area(area)
pyplot.legend(loc='lower right', shadow=True, numpoints=1, prop={'size':12})
pyplot.xlabel("X")
pyplot.ylabel("Y")
ax = pyplot.subplot(1, 2, 2)
pyplot.title('Travel-time residuals + %g%s error' % (100.*error_level, '%'))
s = numpy.arange(len(ttr)) + 1
width = 0.2
pyplot.bar(s - width, newton_predicted, width, color='c', label="Newton")
pyplot.bar(s, ttr, width, color='y', label="Observed", yerr=error)
pyplot.bar(s + width, levmarq_predicted, width, color='g', label="Lev-Marq")
ax.set_xticks(s)
pyplot.legend(loc='upper center', shadow=True, prop={'size':12})
pyplot.xlabel("Station number")
pyplot.ylabel("Travel-time residual")
pyplot.show()
