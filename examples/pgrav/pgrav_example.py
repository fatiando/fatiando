import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav

# Read the tensor component data from the data files
zzdata = TensorComponent(component='zz')
zzdata.load("gzz_data.txt")

xxdata = TensorComponent(component='xx')
xxdata.load("gxx_data.txt")

yydata = TensorComponent(component='yy')
yydata.load("gyy_data.txt")

xydata = TensorComponent(component='xy')
xydata.load("gxy_data.txt")

xzdata = TensorComponent(component='xz')
xzdata.load("gxz_data.txt")

yzdata = TensorComponent(component='yz')
yzdata.load("gyz_data.txt")

shape = (21,21)

stddev = 0.05
    
solver = PGrav(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, \
               nx=5, ny=5, nz=5, \
               gzz=zzdata, gxy=xydata, gxz=xzdata, \
               gxx=xxdata, gyy=yydata, gyz=yzdata)

Wp = solver.depth_weights(w0=1, \
z0=150, \
power=6, \
                          normalize=True)

solver.solve(damping=10**(-4), smoothness=10**(-5), curvature=0, equality=0, \
         param_weights=Wp, apriori_var=stddev**2, contam_times=10)

solver.plot_residuals()
#solver.plot_adjustment(shape)
pylab.show()

solver.plot_std3d()
solver.plot_mean3d()
mlab.show_pipeline()
mlab.show()

#initial = pylab.loadtxt("initial.txt").T
#initial = 1*numpy.ones(10*10*10)
#
#solver.sharpen(residual=0, sharpness=10**(5), damping=0, beta=10**(-10), \
#               param_weights=None, initial_estimate=initial, \
#               apriori_var=stddev**2, contam_times=0, \
#               max_it=200, max_marq_it=20, marq_start=10**(10), marq_step=10)
#
#solver.plot_goal(scale='linear')
#solver.plot_residuals()
#solver.plot_adjustment(shape)
#pylab.show()
#solver.plot_std3d()
#solver.plot_mean3d()
#mlab.show_pipeline()
#mlab.show()