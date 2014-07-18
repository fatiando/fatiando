"""
Seismic: Show steps taken by different algorithms for 2D epicenter estimation
on a flat Earth
"""
import sys
import numpy
from fatiando import gridder, utils
from fatiando.mesher import Square
from fatiando.vis import mpl
from fatiando.seismic import ttime2d, epic2d

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
solver = epic2d.Homogeneous(traveltime, recs, vp, vs)
# Pick the initial estimate
mpl.figure()
mpl.axis('scaled')
mpl.suptitle("Choose the initial estimate")
mpl.points(rec_points, '^r')
mpl.points(src, '*y')
initial = mpl.pick_points(area, mpl.gca(), marker='*', color='b')
if len(initial) > 1:
    print "Don't be greedy! Pick only one point"
    sys.exit()
# Fit using many different solvers
levmarq = [e for e in solver.levmarq(initial=initial[0], iterate=True)]
steepest = [e for e in solver.steepest(initial=initial[0], iterate=True)]
newton = [e for e in solver.newton(initial=initial[0], iterate=True)]
ACO_R = [e for e in solver.acor(bounds=area, maxit=100, iterate=True)]
# Make a map of the objective function
shape = (100, 100)
xs, ys = gridder.regular(area, shape)
obj = [solver.value(numpy.array([x, y])) for x, y in zip(xs, ys)]
mpl.figure()
mpl.title('Epicenter + %d recording stations' % (len(recs)))
mpl.axis('scaled')
mpl.contourf(xs, ys, obj, shape, 60)
mpl.points(recs, '^r')
mpl.points(initial, '*w')
mpl.points(levmarq, '.-k', label="Levemeber-Marquardt")
mpl.points(newton, '.-r', label="Newton")
mpl.points(steepest, '.-m', label="Steepest descent")
mpl.points(ACO_R, '.-c', label="Ant Colony")
mpl.points(src, '*y')
mpl.set_area(area)
mpl.legend(loc='lower right', shadow=True, numpoints=1, prop={'size': 12})
mpl.xlabel("X")
mpl.ylabel("Y")
mpl.show()
