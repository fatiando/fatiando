"""
Seismic: 2D epicenter estimation assuming a homogeneous and flat Earth
"""
import sys
import numpy
from fatiando import gridder, utils
from fatiando.mesher import Square
from fatiando.vis import mpl
from fatiando.seismic import ttime2d
from fatiando.seismic.epic2d import Homogeneous

# Make a velocity model to calculate traveltimes
area = (0, 10, 0, 10)
vp, vs = 2, 1
model = [Square(area, props={'vp': vp, 'vs': vs})]
# Pick the locations of the receivers
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Choose the location of the receivers")
rec_points = mpl.pick_points(area, mpl.gca(), marker='^', color='r')
# and the source
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Choose the location of the source")
mpl.points(rec_points, '^r')
src = mpl.pick_points(area, mpl.gca(), marker='*', color='y')
if len(src) > 1:
    print "Don't be greedy! Pick only one point as the source"
    sys.exit()
# Calculate the P and S wave traveltimes
srcs, recs = utils.connect_points(src, rec_points)
ptime = ttime2d.straight(model, 'vp', srcs, recs)
stime = ttime2d.straight(model, 'vs', srcs, recs)
# Calculate the residual time (S - P) with added noise
traveltime, error = utils.contaminate(stime - ptime, 0.05, percent=True,
                                      return_stddev=True)
solver = Homogeneous(traveltime, recs, vp, vs)
# Pick the initial estimate and fit
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Choose the initial estimate")
mpl.points(rec_points, '^r')
mpl.points(src, '*y')
initial = mpl.pick_points(area, mpl.gca(), marker='*', color='b')
if len(initial) > 1:
    print "Don't be greedy! Pick only one point"
    sys.exit()
estimate = solver.config('levmarq', initial=initial[0]).fit().estimate_

mpl.figure(figsize=(10, 4))
mpl.subplot(1, 2, 1)
mpl.title('Epicenter + %d recording stations' % (len(recs)))
mpl.axis('scaled')
mpl.points(src, '*y', label="True")
mpl.points(recs, '^r', label="Stations")
mpl.points(initial, '*b', label="Initial")
mpl.points([estimate], '*g', label="Estimate")
mpl.set_area(area)
mpl.legend(loc='lower right', shadow=True, numpoints=1, prop={'size': 12})
mpl.xlabel("X")
mpl.ylabel("Y")
ax = mpl.subplot(1, 2, 2)
mpl.title('Travel-time residuals + error bars')
s = numpy.arange(len(traveltime)) + 1
width = 0.3
mpl.bar(s - width, traveltime, width, color='g', label="Observed",
        yerr=error)
mpl.bar(s, solver.predicted(), width, color='r', label="Predicted")
ax.set_xticks(s)
mpl.legend(loc='upper right', shadow=True, prop={'size': 12})
mpl.xlabel("Station number")
mpl.ylabel("Travel-time residual")
mpl.show()
