import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav
from fatiando.utils.geometry import Prism


    
prisma = Prism(x1=400, x2=600, y1=400, y2=600, z1=200, z2=400, dens=1000)

x = numpy.arange(-500, 1550, 100, 'f')
y = numpy.arange(-500, 1550, 100, 'f')
X, Y = pylab.meshgrid(x, y)

stddev = 0.05

zzdata = TensorComponent(component='zz')
zzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xxdata = TensorComponent(component='xx')
xxdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

yydata = TensorComponent(component='yy')
yydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xydata = TensorComponent(component='xy')
xydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

xzdata = TensorComponent(component='xz')
xzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)

yzdata = TensorComponent(component='yz')
yzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                       percent=False)


solver = PGrav(x1=400, x2=600, y1=400, y2=600, z1=0, z2=400, \
   nx=1, ny=1, nz=2, \
   gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)

Wp = solver.depth_weights(z0=300, power=6)

solver.add_equality(x=500, y=500, z=50, value=1000)

damping = 0
smoothness = 0
curvature = 0
equality = 0
weights = None# Wp#numpy.array([[10, 0], [0, 1]])

solver.solve(damping=damping, smoothness=smoothness, curvature=curvature, \
             equality=equality, param_weights=weights, apriori_var=stddev**2, \
             contam_times=10)

solver.map_goal(true=(0, 1000), res=solver.mean, lower=(-5000,-5000), \
                upper=(5000,5000), dp1=200, dp2=200, \
                damping=damping, smoothness=smoothness, curvature=curvature, \
                equality=equality, param_weights=weights)

solver.plot_residuals()
solver.plot_adjustment(X.shape)

solver.plot_std3d()
solver.plot_mean3d()
mlab.show_pipeline()
mlab.show()

pylab.show()