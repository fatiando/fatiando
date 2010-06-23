import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav
from fatiando.utils.geometry import Prism

prisma = Prism(x1=-100, x2=100, y1=-300, y2=300, z1=200, z2=400, dens=1000)

x = numpy.arange(-500,500,50)
y = numpy.arange(-500,500,50)
X, Y = pylab.meshgrid(x, y)

zzdata = TensorComponent(component='zz')
zzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)
xxdata = TensorComponent(component='xx')
xxdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)
yydata = TensorComponent(component='yy')
yydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)
xydata = TensorComponent(component='xy')
xydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)
xzdata = TensorComponent(component='xz')
xzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)
yzdata = TensorComponent(component='yz')
yzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.01)

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

solver = PGrav(x1=-500, x2=500, y1=-500, y2=500, z1=0, z2=600, nx=10, ny=10, nz=6, \
               gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)


solver.solve(damping=10**(-6), smoothness=0, curvature=0, \
             apriori_var=zzdata.std[0]**2, contam_times=5)

#solver.plot_residuals()
#solver.plot_adjustment(X.shape)
#pylab.show()
#
solver.plot_std3d()
solver.plot_mean3d()
#
#mlab.show_pipeline()
#mlab.show()

initial = 1*numpy.ones(10*10*6)
solver.sharpen(sharpness=10**(6), damping=0, beta=10**(15), \
               initial_estimate=solver.mean, apriori_var=zzdata.std[0]**2, contam_times=1, \
               max_it=100, max_marq_it=20, marq_start=10**5, marq_step=2)

solver.plot_goal(scale='linear')
solver.plot_residuals()
solver.plot_adjustment(X.shape)
pylab.show()
solver.plot_std3d()
solver.plot_mean3d()
mlab.show_pipeline()
mlab.show()