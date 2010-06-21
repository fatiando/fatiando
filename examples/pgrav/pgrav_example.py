import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav
from fatiando.utils.geometry import Prism

prisma = Prism(x1=-100, x2=100, y1=-200, y2=200, z1=100, z2=200, dens=1000)

x = numpy.arange(-500,500,20)
y = numpy.arange(-500,500,20)
Y, X = pylab.meshgrid(x, y)

zzdata = TensorComponent(component='zz')
zzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)
#xxdata = TensorComponent(component='xx')
#xxdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)
#xydata = TensorComponent(component='xy')
#xydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)
#xzdata = TensorComponent(component='xz')
#xzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)
#yydata = TensorComponent(component='yy')
#yydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)
#yzdata = TensorComponent(component='yz')
#yzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-10, stddev=0.01)

pylab.figure()
pylab.title("Dados")
pylab.contourf(Y, X, zzdata.togrid(*X.shape), 30)
pylab.colorbar()

solver = PGrav(x1=-500, x2=500, y1=-500, y2=500, z1=0, z2=500, nx=10, ny=10, nz=5, \
               gzz=zzdata)#, gxx=xxdata, gxy=xydata, gxz=xzdata, gyy=yydata, gyz=yzdata)

#solver.solve(damping=0.1, smoothness=0, curvature=0, \
#             apriori_var=zzdata.std[0]**2, contam_times=0)

solver.sharpen(sharpness=1, damping=0, initial_estimate=None, apriori_var=zzdata.std[0]**2,\
               contam_times=0, max_it=100, max_marq_it=20, marq_start=0.1, marq_step=10)

solver.plot_goal()
solver.plot_residuals()
pylab.show()
solver.plot_mean3d()