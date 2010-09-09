"""
Make some synthetic FTG data
"""

import logging
logging.basicConfig()

import numpy
import pylab

from fatiando.data.gravity import TensorComponent
from fatiando.utils.geometry import Prism


# Create a synthetic body
prisms = []
prism = Prism(dens=1000, x1=200, x2=400, y1=200, y2=400, z1=0, z2=200)
prisms.append(prism)
#prism = Prism(dens=1000, x1=400, x2=800, y1=400, y2=600, z1=400, z2=600)
#prisms.append(prism)
    
    
def main():
    
    # Make a computation grid
    step = 60
    x = numpy.arange(0, 600 + step, step, 'f')
    y = numpy.arange(0, 600 + step, step, 'f')
    X, Y = pylab.meshgrid(x, y)
        
    # Create the data classes and generate synthetic data (at 150 m altitude)
    stddev = 1
    
    zzdata = TensorComponent(component='zz')
    zzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    zzdata.dump("gzz_data.txt")
    
    xxdata = TensorComponent(component='xx')
    xxdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    xxdata.dump("gxx_data.txt")
    
    yydata = TensorComponent(component='yy')
    yydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    yydata.dump("gyy_data.txt")
    
    xydata = TensorComponent(component='xy')
    xydata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    xydata.dump("gxy_data.txt")
    
    xzdata = TensorComponent(component='xz')
    xzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    xzdata.dump("gxz_data.txt")
    
    yzdata = TensorComponent(component='yz')
    yzdata.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-150, stddev=stddev, \
                           percent=False)
    yzdata.dump("gyz_data.txt")
    
    # Make nice plots
    pylab.figure()
    pylab.title("Synthetic $g_{zz}$")
    pylab.contourf(X, Y, zzdata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.figure()
    pylab.title("Synthetic $g_{xx}$")
    pylab.contourf(Y, X, xxdata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.figure()
    pylab.title("Synthetic $g_{xy}$")
    pylab.contourf(Y, X, xydata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.figure()
    pylab.title("Synthetic $g_{xz}$")
    pylab.contourf(Y, X, xzdata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.figure()
    pylab.title("Synthetic $g_{yy}$")
    pylab.contourf(Y, X, yydata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.figure()
    pylab.title("Synthetic $g_{yz}$")
    pylab.contourf(Y, X, yzdata.togrid(*X.shape), 30)
    pylab.colorbar()
    
    pylab.show()
    
    
if __name__ == '__main__':
    
    main()