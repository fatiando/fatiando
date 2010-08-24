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
    
<<<<<<< local
    prisma = Prism(x1=400, x2=600, y1=200, y2=800, z1=200, z2=400, dens=1000)
    
    x = numpy.arange(-500, 1550, 100, 'f')
    y = numpy.arange(-500, 1550, 100, 'f')
    X, Y = pylab.meshgrid(x, y)
    
    stddev = 0.05
    
    zzdata = TensorComponent(component='zz')
    zzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    zzdata.dump("gzz_data.txt")
#    zzdata.load("gzz_data.txt")
    
    xxdata = TensorComponent(component='xx')
    xxdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xxdata.dump("gxx_data.txt")
#    xxdata.load("gxx_data.txt")
    
    yydata = TensorComponent(component='yy')
    yydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    yydata.dump("gyy_data.txt")
#    yydata.load("gyy_data.txt")
    
    xydata = TensorComponent(component='xy')
    xydata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xydata.dump("gxy_data.txt")
#    xydata.load("gxy_data.txt")
    
    xzdata = TensorComponent(component='xz')
    xzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
#    xzdata.dump("gxz_data.txt")
#    xzdata.load("gxz_data.txt")
    
    yzdata = TensorComponent(component='yz')
    yzdata.synthetic_prism(prism=prisma, X=X, Y=Y, z=-150, stddev=stddev, \
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
    
    solver = PGrav(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=500, \
       nx=15, ny=15, nz=15, \
       gzz=zzdata, gxy=xydata, gxz=xzdata, gxx=xxdata, gyy=yydata, gyz=yzdata)
    
    Wp = solver.depth_weights(z0=150, power=5.5)
    
    # Add equality to all the prisms
#    for z in numpy.arange(50, 500, 100, 'f'):
#        
#        for y in numpy.arange(50, 1000, 100, 'f'):
#            
#            for x in numpy.arange(50, 1000, 100, 'f'):
#                
#                solver.add_equality(x, y, z, 500)    
    
    solver.solve(damping=0, smoothness=10**(-2), curvature=0, equality=0, \
                 param_weights=Wp, apriori_var=stddev**2, contam_times=10)
=======
solver = PGrav(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, \
               nx=5, ny=5, nz=5, \
               gzz=zzdata, gxy=xydata, gxz=xzdata, \
               gxx=xxdata, gyy=yydata, gyz=yzdata)
>>>>>>> other

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