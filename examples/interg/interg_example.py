import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.interg import InterG
from fatiando.utils.geometry import Prism

    
prism1 = Prism(x1=200, x2=400, y1=200, y2=400, z1=0, z2=1000, dens=1000)
prism2 = Prism(x1=200, x2=400, y1=400, y2=600, z1=0, z2=1000, dens=1000)
prism3 = Prism(x1=200, x2=400, y1=600, y2=800, z1=0, z2=1000, dens=1000)
prism4 = Prism(x1=600, x2=800, y1=200, y2=400, z1=0, z2=1000, dens=1000)
prism5 = Prism(x1=600, x2=800, y1=400, y2=600, z1=0, z2=1000, dens=1000)
prism6 = Prism(x1=600, x2=800, y1=600, y2=800, z1=0, z2=1000, dens=1000)

prism7 = Prism(x1=400, x2=600, y1=200, y2=800, z1=0, z2=2000, dens=1000)

prisms = [prism1, prism2, prism3, prism4, prism5, prism6, prism7]

x = numpy.arange(-500, 1550, 200, 'f')
y = numpy.arange(-500, 1550, 200, 'f')
X, Y = pylab.meshgrid(x, y)

stddev = 0.05

zzdata = TensorComponent(component='zz')
zzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xxdata = TensorComponent(component='xx')
xxdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

yydata = TensorComponent(component='yy')
yydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xydata = TensorComponent(component='xy')
xydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xzdata = TensorComponent(component='xz')
xzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

yzdata = TensorComponent(component='yz')
yzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)
    
#pylab.figure()
#pylab.title("gzz")
#pylab.contourf(X, Y, zzdata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.figure()
#pylab.title("gxx")
#pylab.contourf(Y, X, xxdata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.figure()
#pylab.title("gxy")
#pylab.contourf(Y, X, xydata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.figure()
#pylab.title("gxz")
#pylab.contourf(Y, X, xzdata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.figure()
#pylab.title("gyy")
#pylab.contourf(Y, X, yydata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.figure()
#pylab.title("gyz")
#pylab.contourf(Y, X, yzdata.togrid(*X.shape), 30)
#pylab.colorbar()
#pylab.show()

solver = InterG(x1=0, x2=1000, y1=0, y2=1000, nx=10, ny=10, dens=1000, \
        gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)

solver.solve(damping=0, smoothness=0, curvature=10**(-10), \
             initial_estimate=None, apriori_var=stddev**2, contam_times=1, \
             max_it=100, max_lm_it=20, lm_start=10**(2), lm_step=10)

solver.plot_residuals()
solver.plot_adjustment(X.shape)
solver.plot_goal()

solver.plot_mean()
solver.plot_std()

pylab.show()