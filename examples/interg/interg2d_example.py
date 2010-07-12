import math
import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.interg import InterG2D
from fatiando.utils.geometry import Prism

#y1 = -10**(10)
#y2 = 10**(10)
##prism1 = Prism(x1=0, x2=200, y1=y1, y2=y2, z1=0, z2=200, dens=-1000)
##prism2 = Prism(x1=200, x2=400, y1=y1, y2=y2, z1=0, z2=300, dens=-1000)
#prism3 = Prism(x1=400, x2=600, y1=y1, y2=y2, z1=0, z2=1000, dens=-1000)
##prism4 = Prism(x1=600, x2=800, y1=y1, y2=y2, z1=0, z2=500, dens=-1000)
##prism5 = Prism(x1=800, x2=1000, y1=y1, y2=y2, z1=0, z2=100, dens=-1000)
#
#true_x = [0, 200, 200, 400, 400, 600, 600, 800, 800, 1000]
#true_z = [200, 200, 300, 300, 1000, 1000, 500, 500, 100, 100]
##prisms = [prism1, prism2, prism3, prism4, prism5]
#prisms = [prism3]


true_x = []
true_z = []
prisms = []
dens = -1000.
y1 = -10.**(6)
y2 = 10.**(6)
dx = 5.

for x in numpy.arange(0, 300, dx, dtype='float'):
    
    z = 100.
    
    true_x.append(x)
    true_x.append(x + dx)
    
    true_z.append(z)
    true_z.append(z)
    
    prisms.append(Prism(x1=x, x2=x + dx, y1=y1, y2=y2, z1=0., z2=z, dens=dens))


#for x in numpy.arange(300, 700, dx, dtype='float'):
#    
#    z = 100*numpy.log(1000*(x - 290))
#    
#    true_x.append(x)
#    true_x.append(x + dx)
#    
#    true_z.append(z)
#    true_z.append(z)
#    
#    prisms.append(Prism(x1=x, x2=x+dx, y1=y1, y2=y2, z1=0., z2=z, dens=dens))

for x in numpy.arange(300, 700, dx, dtype='float'):
    
    z = 1000.
    
    true_x.append(x)
    true_x.append(x + dx)
    
    true_z.append(z)
    true_z.append(z)
    
    prisms.append(Prism(x1=x, x2=x+dx, y1=y1, y2=y2, z1=0., z2=z, dens=dens))

for x in numpy.arange(700, 1000, dx, dtype='float'):
    
    z = 600.
    
    true_x.append(x)
    true_x.append(x + dx)
    
    true_z.append(z)
    true_z.append(z)
    
    prisms.append(Prism(x1=x, x2=x+dx, y1=y1, y2=y2, z1=0., z2=z, dens=dens))    
    

#pylab.plot(true_x, true_z, '-r')
#pylab.ylim(max(true_z), 0)
#pylab.show()

dx = 20
x = numpy.arange(-500, 1500 + dx, dx, dtype='float')
y = numpy.array([0.])

X, Y = pylab.meshgrid(x, y)

stddev = 0.05

zzdata = TensorComponent(component='z')
zzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-10, stddev=stddev, \
                       percent=False)

#xxdata = TensorComponent(component='xx')
#xxdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
#                       percent=False)
#
#yydata = TensorComponent(component='yy')
#yydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
#                       percent=False)
#
#xydata = TensorComponent(component='xy')
#xydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
#                       percent=False)
#
#xzdata = TensorComponent(component='xz')
#xzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
#                       percent=False)
#
#yzdata = TensorComponent(component='yz')
#yzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
#                       percent=False)
    
    
#pylab.figure()
#pylab.plot(zzdata.get_xarray(), zzdata.array, '.-k')

solver = InterG2D(x1=0, x2=1000, nx=20, dens=-1000, \
        gz=zzdata)

#solver.add_equality(x=501, z=100*numpy.log(1000*(501 - 290)))
#solver.add_equality(x=401, z=100*numpy.log(1000*(401 - 290)))
#solver.add_equality(x=401, z=100*numpy.log(1000*(401 - 290)))
solver.add_equality(x=501, z=1000)
solver.add_equality(x=901, z=600)

initial = 500*numpy.ones(20)

solver.solve(damping=0*10**(-10), smoothness=1*10**(-8), curvature=0*10**(-5), \
             sharpness=0*10**(-8), beta=10**(-7), equality=1, \
             initial_estimate=initial, apriori_var=stddev**2, contam_times=2, \
             max_it=100, max_lm_it=20, lm_start=10**(2), lm_step=10)

solver.plot_mean(true_x, true_z, title="Result n=5")
solver.plot_adjustment()

#initial = solver.split(2)
#
#solver2 = InterG2D(x1=0, x2=1000, nx=10, dens=-1000, \
#        gz=zzdata)
#
#solver2.add_equality(x=501, z=1000)
#solver2.add_equality(x=901, z=600)
#
#solver2.solve(damping=0*10**(-8), smoothness=1*10**(-10), curvature=0*10**(-5), \
#             sharpness=5*10**(-5), beta=10**(-3), equality=1, \
#             initial_estimate=initial, apriori_var=stddev**2, contam_times=2, \
#             max_it=100, max_lm_it=20, lm_start=10**(2), lm_step=10)

#solver2.plot_adjustment()
#solver2.plot_mean(true_x, true_z, title="Result n=10")
#
#initial = solver2.split(2)
#
#solver3 = InterG2D(x1=0, x2=1000, nx=20, dens=-1000, \
#        gz=zzdata)
#
#solver3.add_equality(x=501, z=1000)
#solver3.add_equality(x=901, z=600)
#
#
#
#solver3.solve(damping=0*10**(-8), smoothness=1*10**(-10), curvature=0*10**(-5), \
#             sharpness=1*10**(-5), beta=10**(-5), equality=1, \
#             initial_estimate=initial, apriori_var=stddev**2, contam_times=2, \
#             max_it=100, max_lm_it=20, lm_start=10**(0), lm_step=10)

#solver3.plot_residuals()
#solver3.plot_adjustment()
#solver3.plot_mean(true_x, true_z, title="Result n=20")
#
#initial = solver3.split(2)
#
#solver4 = InterG2D(x1=0, x2=1000, nx=40, dens=-1000, \
#        gz=zzdata)
#
#solver4.add_equality(x=501, z=1000)
#solver4.add_equality(x=901, z=600)
#
#
#
#solver4.solve(damping=0*10**(-8), smoothness=1*10**(-11), curvature=0*10**(-5), \
#             sharpness=1*10**(-5), beta=10**(-4), equality=1, \
#             initial_estimate=initial, apriori_var=stddev**2, contam_times=2, \
#             max_it=100, max_lm_it=20, lm_start=10**(0), lm_step=10)

#solver4.plot_residuals()
#solver4.plot_adjustment()
#solver4.plot_mean(true_x, true_z, title="Result n=40")
#
#initial = solver4.split(2)
#
#solver5 = InterG2D(x1=0, x2=1000, nx=80, dens=-1000, \
#        gz=zzdata)
#
#solver5.add_equality(x=501, z=1000)
#solver5.add_equality(x=901, z=600)
#
#solver5.solve(damping=0*10**(-8), smoothness=1*10**(-7), curvature=0*10**(-5), \
#             sharpness=1*10**(-3), beta=10**(-7), equality=1, \
#             initial_estimate=initial, apriori_var=stddev**2, contam_times=10, \
#             max_it=100, max_lm_it=20, lm_start=10**(-2), lm_step=10)
#
#solver5.plot_residuals()
#solver5.plot_adjustment()
#solver5.plot_mean(true_x, true_z, title="Result n=80")
#
#
#solver5.solve(damping=0*10**(-8), smoothness=5*10**(-8), curvature=0*10**(-5), \
#             sharpness=1*10**(-4), beta=10**(-5), equality=1, \
#             initial_estimate=None, apriori_var=stddev**2, contam_times=10, \
#             max_it=100, max_lm_it=20, lm_start=10**(2), lm_step=10)
#
#solver5.plot_residuals(title="Residuals (Non-adaptative)")
#solver5.plot_adjustment(title="Adjustment (Non-adaptative)")
#solver5.plot_mean(true_x, true_z, title="Result n=80 (Non-adaptative)")

pylab.show()