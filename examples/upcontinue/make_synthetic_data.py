"""
Generate synthetic gravity data at two different heights to test upward
continuation. 
"""

import pylab

import fatiando.grav.synthetic as synthetic
import fatiando.grav.io as io
import fatiando.utils as utils

log = utils.get_logger()

# Define a prism as the model to generate the gravity data
prism = {'x1':-100, 'x2':100, 'y1':-100, 'y2':100, 'z1':500, 'z2':700,
         'value':1000}

level1 = synthetic.from_prisms([prism], x1=-500, x2=500, y1=-500, y2=500, 
                               nx=50, ny=50, height=0, field='gz')

io.dump('gz0m.txt', level1)

level2 = synthetic.from_prisms([prism], x1=-500, x2=500, y1=-500, y2=500, 
                               nx=50, ny=50, height=1000, field='gz')

io.dump('gz1000m.txt', level2)

X1, Y1, Z1 = utils.extract_matrices(level1)
X2, Y2, Z2 = utils.extract_matrices(level2)

pylab.figure(figsize=(10,4))
pylab.subplot(1,2,1)
pylab.title('z = 0 m')
pylab.axis('scaled')
pylab.pcolor(X1, Y1, Z1)
pylab.xlim(X1.min(), X1.max())
pylab.ylim(Y1.min(), Y1.max())
pylab.colorbar()
pylab.xlabel('X')
pylab.ylabel('Y')

pylab.subplot(1,2,2)
pylab.title('z = 1000 m')
pylab.axis('scaled')
pylab.pcolor(X2, Y2, Z2)
pylab.xlim(X2.min(), X2.max())
pylab.ylim(Y2.min(), Y2.max())
pylab.colorbar()
pylab.xlabel('X')
pylab.ylabel('Y')

pylab.show()