"""
Example script for doing the inversion of synthetic FTG data
"""

import pickle

import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav3D, DepthWeightsCalculator

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
solver = PGrav3D(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, \
                 nx=10, ny=10, nz=10, \
                 gzz=zzdata, gxy=xydata, gxz=xzdata, \
                 gxx=xxdata, gyy=yydata, gyz=yzdata)

# Compute the depth weight coefficients
dwsolver = DepthWeightsCalculator(pgrav_solver=solver, height=150)

dwsolver.solve_lm(initial=[150, 3], contam_times=0, \
                  lm_start=1, lm_step=10, it_p_step=20, max_it=100)

dwsolver.plot_adjustment(title="Depth Weights Adjustment")

z0, power = dwsolver.mean

Wp = solver.depth_weights(z0, power, normalize=True)

# Solve the linear inverse problem using Tikhonov regularization
#solver.solve_linear(damping=10**(-10), \
#                    smoothness=10**(-8), \
#                    curvature=0, \
#                    prior_weights=Wp, \
#                    data_variance=0.05**2, \
#                    contam_times=2)
#solver.dump("res_tk_10x10x10.txt")

# Solve the non-linear problem
solver.solve_lm(damping=10**(-8), \
                smoothness=0, \
                curvature=0, \
                sharpness=10**(-3), beta=10**(-7), \
                initial=None, \
                prior_weights=Wp, \
                data_variance=0.05**2, \
                contam_times=0, \
                lm_start=10, lm_step=10, it_p_step=10, max_it=100)
solver.dump("res_vt_10x10x10.txt")

solver.plot_residuals()
#solver.plot_adjustment((21,21))
pylab.show()

solver.plot_stddev()
solver.plot_mean()
mlab.show_pipeline()
mlab.show()

