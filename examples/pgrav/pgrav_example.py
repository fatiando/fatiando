import logging
logging.basicConfig()

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.data.gravity import TensorComponent
from fatiando.geoinv.pgrav import PGrav
from fatiando.utils.geometry import Prism
from fatiando.utils import contaminate


def main():
    
    prisma = Prism(dens=1000, x1=400, x2=600, y1=200, y2=800, z1=100, z2=400)
    
    x = numpy.arange(-500, 1550, 100, 'f')
    y = numpy.arange(-500, 1550, 100, 'f')
    X, Y = pylab.meshgrid(x, y)
    
    stddev = 0.05
    
    zzdata = TensorComponent(component='zz')
    zzdata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    zzdata.dump("gzz_data.txt")
#    zzdata.load("gzz_data.txt")
    
    xxdata = TensorComponent(component='xx')
    xxdata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xxdata.dump("gxx_data.txt")
#    xxdata.load("gxx_data.txt")
    
    yydata = TensorComponent(component='yy')
    yydata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    yydata.dump("gyy_data.txt")
#    yydata.load("gyy_data.txt")
    
    xydata = TensorComponent(component='xy')
    xydata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xydata.dump("gxy_data.txt")
#    xydata.load("gxy_data.txt")
    
    xzdata = TensorComponent(component='xz')
    xzdata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xzdata.dump("gxz_data.txt")
#    xzdata.load("gxz_data.txt")
    
    yzdata = TensorComponent(component='yz')
    yzdata.synthetic_prism(prisms=[prisma], X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    yzdata.dump("gyz_data.txt")
#    yzdata.load("gyz_data.txt")
    
    
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
    
    solver = PGrav(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, \
       nx=10, ny=10, nz=10, \
       gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)
    
    Wp = solver.depth_weights(z0=150, power=4)
        
    solver.solve(damping=0, smoothness=10**(-2), curvature=0, equality=0, \
                 param_weights=Wp, apriori_var=stddev**2, contam_times=10)

    solver.plot_residuals()
    solver.plot_adjustment(X.shape)
    pylab.show()
    
    solver.plot_std3d()
    solver.plot_mean3d()
    mlab.show_pipeline()
    mlab.show()
#    
#    initial = 1*numpy.ones(10*10*10)
#    
#    solver.sharpen(sharpness=10**(-3), damping=0, beta=10**(-7), \
#                   param_weights=Wp, initial_estimate=solver.mean, \
#                   apriori_var=stddev**2, contam_times=0, \
#                   max_it=200, max_marq_it=20, marq_start=10**(5), marq_step=10)
#    
#    solver.plot_goal(scale='linear')
#    solver.plot_residuals()
#    solver.plot_adjustment(X.shape)
#    pylab.show()
#    solver.plot_std3d()
#    solver.plot_mean3d()
#    mlab.show_pipeline()
#    mlab.show()
    
    
if __name__ == '__main__':
    
    main()