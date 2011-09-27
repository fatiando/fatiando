"""
Time how long it takes to compute the effect of a prism on a list of points.
"""
import numpy
import time
import fatiando

def runbench(n=10000):
    from fatiando.potential import prism
    from fatiando.mesher.prism import Prism3D
    from numpy import arange, array
    xp = yp = array(range(n), dtype='f')
    zp = -1*numpy.ones_like(xp)
    prisms = [Prism3D(-1,1,-1,1,0,2,{'density':1})]
    start = time.clock()
    gzz = prism.gzz(xp, yp, zp, prisms)
    elapsed = time.clock() - start
    return elapsed

with open('bench_prism_results.txt', 'a') as res:
    res.write("\n\nLoop over points in C, loop over prisms in Python\n")
    res.write("===============================================================\n\n")
    date = time.asctime()
    res.write("%s\n" % (date))
    res.write("Changeset used: %s\n" % (fatiando.__changeset__))
    npoints, ntimes = 1000000, 10
    times = [runbench(n=npoints) for i in xrange(ntimes)]
    res.write("Time to calculate on %d points, %d times:\n" % (npoints,ntimes))
    res.write("Total time = %lf\n" % (sum(times)))
    res.write("Mean = %lf\n" % (numpy.mean(times)))
    res.write("Std = %lf\n" % (numpy.std(times)))
