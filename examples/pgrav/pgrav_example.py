"""
Example script for doing the inversion of synthetic FTG data
"""

import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.inversion.pgrav import PGrav3D, DepthWeightsCalculator
from fatiando.visualization import plot_prism

import make_data


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


# Make a solver class and define the model space discretization    
solver = PGrav3D(x1=0, x2=600, y1=0, y2=600, z1=0, z2=1000, \
                 nx=6, ny=6, nz=10, \
                 gzz=zzdata,\
                 gxy=xydata, gxz=xzdata, \
                 gxx=xxdata, gyy=yydata, gyz=yzdata)
# Compute the depth weight coefficients
dwsolver = DepthWeightsCalculator(pgrav_solver=solver, height=150)

dwsolver.solve_lm(initial=[10, 3], contam_times=0, \
                  lm_start=1, lm_step=10, it_p_step=20, max_it=100)

dwsolver.plot_adjustment(title="Depth Weights Adjustment")

z0, power = dwsolver.mean

Wp = solver.depth_weights(z0, power, normalize=True)

# Solve the linear inverse problem using Tikhonov regularization
solver.solve_linear(damping=10**(-10), \
                    smoothness=5*10**(-6), \
                    curvature=0, \
                    prior_weights=Wp, \
                    data_variance=1**2, \
                    contam_times=2)
#solver.dump("res_tk_10x10x10.txt")

# Compact the Tikhonov solution
#solver.solve_lm(damping=10**(-12), \
#                smoothness=10**(-8), \
#                curvature=0, \
#                sharpness=0, beta=10**(-7), \
#                compactness=10**(-5), epsilon=10**(-5), \
#                initial=None, \
#                prior_weights=Wp, \
#                data_variance=1**2, \
#                contam_times=1, \
#                lm_start=0.1, lm_step=10, it_p_step=10, max_it=100)
#solver.dump("res_compact_10x10x10.txt")

# Solve the non-linear problem
#solver.solve_lm(damping=10**(-10), \
#                smoothness=0, \
#                curvature=0, \
#                sharpness=10**(-4), beta=10**(-8), \
#                initial=None, \
#                prior_weights=Wp, \
#                data_variance=1**2, \
#                contam_times=1, \
#                lm_start=10, lm_step=10, it_p_step=10, max_it=100)
#solver.dump("res_vt_10x10x10.txt")

solver.plot_residuals()
#solver.plot_adjustment((21,21))
pylab.show()

solver.plot_stddev()
solver.plot_mean()

for prism in make_data.prisms:
    
    plot_prism(prism)
    
mlab.show_pipeline()
mlab.show()

