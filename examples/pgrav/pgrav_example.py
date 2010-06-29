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
    
    prisma = Prism(x1=400, x2=600, y1=400, y2=600, z1=400, z2=500, dens=1000)
    
    x = numpy.arange(0, 1050, 50, 'f')
    y = numpy.arange(0, 1050, 50, 'f')
    X, Y = pylab.meshgrid(x, y)
    
    zzdata = TensorComponent(component='zz')
#    zzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    zzdata.dump("gzz_data.txt")
    zzdata.load("gzz_data.txt")
    
    xxdata = TensorComponent(component='xx')
#    xxdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    xxdata.dump("gxx_data.txt")
    xxdata.load("gxx_data.txt")
    
    yydata = TensorComponent(component='yy')
#    yydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    yydata.dump("gyy_data.txt")
    yydata.load("gyy_data.txt")
    
    xydata = TensorComponent(component='xy')
#    xydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    xydata.dump("gxy_data.txt")
    xydata.load("gxy_data.txt")
    
    xzdata = TensorComponent(component='xz')
#    xzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    xzdata.dump("gxz_data.txt")
    xzdata.load("gxz_data.txt")
    
    yzdata = TensorComponent(component='yz')
#    yzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=0.02, \
#                           percent=False)
#    yzdata.dump("gyz_data.txt")
    yzdata.load("gyz_data.txt")
    
    
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
    
    solver = PGrav(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, nx=10, ny=10, nz=10, \
                   gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)
    
    solver.add_equality(450, 450, 450, 1000)
    
    Wp = solver.depth_weights(z0=150, power=6)
    
    solver.solve(damping=10**(-5), smoothness=10**(-5), curvature=0, \
                 param_weights=Wp, apriori_var=0.02**2, contam_times=5)
    
    solver.plot_residuals()
    solver.plot_adjustment(X.shape)
    pylab.show()
    
    solver.plot_std3d()
    solver.plot_mean3d()
    mlab.show_pipeline()
    mlab.show()
    
    #initial = 1*numpy.ones(10*10*5)
    #initial = numpy.zeros((5,10,10))
    #initial[2][4][4] = 1000
    #initial[2][4][5] = 1000
    #initial[2][5][4] = 1000
    #initial[2][5][5] = 1000
    #initial[3][4][4] = 1000
    #initial[3][4][5] = 1000
    #initial[3][5][4] = 1000
    #initial[3][5][5] = 1000
    #solver._estimates = [contaminate.gaussian(initial.ravel(), stddev=0.25)]
    #solver._estimates = [initial.ravel()]
    #solver.plot_mean3d()
    #mlab.show()
    #
    #solver.sharpen(sharpness=1*10**(-3), damping=0, beta=10**(-5), param_weights=Wp, \
    #               initial_estimate=initial, apriori_var=0.1**2, contam_times=1, \
    #               max_it=100, max_marq_it=20, marq_start=10**5, marq_step=10)
    #
    #solver.plot_goal(scale='linear')
    #solver.plot_residuals()
    #solver.plot_adjustment(X.shape)
    #pylab.show()
    #solver.plot_std3d()
    #solver.plot_mean3d()
    #mlab.show_pipeline()
    #mlab.show()
    
    
if __name__ == '__main__':
    
    main()