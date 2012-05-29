"""
Compare the results of the Python + Numpy implementation vs the Cython one
"""
import sys
import timeit
import fatiando as ft

log = ft.log.get(stream=sys.stdout)
print ft.log.header()

print "------------------------------------"
print "Testing with 1 prism and many points"
print "------------------------------------"
setup = """
import fatiando as ft
prisms = [ft.msh.ddd.Prism(-2000,2000,-2000,2000,0,1000, {'density':1000})]
shape = (500, 500)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
from fatiando.potential import _prism
from fatiando.potential import _cprism
"""
n = 20
print "Average time of %d runs" % (n)
ctime = timeit.timeit("_cprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Cython:", ctime
pytime = timeit.timeit("_prism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy:", pytime
print "Cython is %lf%s faster" % (100.*(pytime - ctime)/pytime, r'%')

print "---------------------------------------"
print "Testing with many prism and many points"
print "---------------------------------------"
setup = """
import fatiando as ft
prisms = ft.msh.ddd.PrismMesh((-2000,2000,-2000,2000,0,1000), (10, 10, 10))
prisms.addprop('density', [1000]*prisms.size)
shape = (100, 100)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
from fatiando.potential import _prism
from fatiando.potential import _cprism
"""
n = 10
print "Average time of %d runs" % (n)
ctime = timeit.timeit("_cprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Cython:", ctime
pytime = timeit.timeit("_prism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy:", pytime
print "Cython is %lf%s faster" % (100.*(pytime - ctime)/pytime, r'%')
