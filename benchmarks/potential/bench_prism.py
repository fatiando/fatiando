"""
Compare the results of the Python + Numpy implementation vs the Cython one
"""
import sys
import timeit
import fatiando as ft

log = ft.log.get(stream=sys.stdout)
print ft.log.header()

print "-----------------------------------"
print "Testing with 1 prism and few points"
print "-----------------------------------"
setup = """
import fatiando as ft
prisms = [ft.msh.ddd.Prism(-2000,2000,-2000,2000,0,1000, {'density':1000})]
shape = (50, 50)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
from fatiando.pot import _prism
from fatiando.pot import _cprism
from fatiando.pot import _neprism
"""
n = 20
print "Average time of %d runs" % (n)
ctime = timeit.timeit("_cprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Cython:", ft.utils.sec2hms(ctime)
pytime = timeit.timeit("_prism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy:", ft.utils.sec2hms(pytime)
netime = timeit.timeit("_neprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy + Numexpr:", ft.utils.sec2hms(netime)
print "RESULTS:"
print "  Cython is %lf%s faster than Python + Numpy" \
    % (100.*(pytime - ctime)/pytime, r'%')
print "  Cython is %lf%s faster than Python + Numpy + Numexpr" \
    % (100.*(netime - ctime)/netime, r'%')

print "------------------------------------"
print "Testing with 1 prism and many points"
print "------------------------------------"
setup = """
import fatiando as ft
prisms = [ft.msh.ddd.Prism(-2000,2000,-2000,2000,0,1000, {'density':1000})]
shape = (500, 500)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
from fatiando.pot import _prism
from fatiando.pot import _cprism
from fatiando.pot import _neprism
"""
n = 20
print "Average time of %d runs" % (n)
ctime = timeit.timeit("_cprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Cython:", ft.utils.sec2hms(ctime)
pytime = timeit.timeit("_prism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy:", ft.utils.sec2hms(pytime)
netime = timeit.timeit("_neprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy + Numexpr:", ft.utils.sec2hms(netime)
print "RESULTS:"
print "  Cython is %lf%s faster than Python + Numpy" \
    % (100.*(pytime - ctime)/pytime, r'%')
print "  Cython is %lf%s faster than Python + Numpy + Numexpr" \
    % (100.*(netime - ctime)/netime, r'%')

print "---------------------------------------"
print "Testing with prism mesh and many points"
print "---------------------------------------"
setup = """
import fatiando as ft
prisms = ft.msh.ddd.PrismMesh((-2000,2000,-2000,2000,0,1000), (10, 10, 10))
prisms.addprop('density', [1000]*prisms.size)
shape = (500, 500)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-100)
from fatiando.pot import _prism
from fatiando.pot import _cprism
from fatiando.pot import _neprism
"""
n = 10
print "Average time of %d runs" % (n)
ctime = timeit.timeit("_cprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Cython:", ft.utils.sec2hms(ctime)
pytime = timeit.timeit("_prism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy:", ft.utils.sec2hms(pytime)
netime = timeit.timeit("_neprism.gz(xp, yp, zp, prisms)", setup, number=n)/float(n)
print "Python + Numpy + Numexpr:", ft.utils.sec2hms(netime)
print "RESULTS:"
print "  Cython is %lf%s faster than Python + Numpy" \
    % (100.*(pytime - ctime)/pytime, r'%')
print "  Cython is %lf%s faster than Python + Numpy + Numexpr" \
    % (100.*(netime - ctime)/netime, r'%')

