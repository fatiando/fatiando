"""
Make some synthetic FTG data
"""

import logging
logging.basicConfig()

import pylab

from fatiando.gravity import synthetic, io
from fatiando.visualization import contour_grid
import fatiando.utils

prisms = []
prisms.append({'x1':-100, 'x2':100, 'y1':-100, 'y2':100, 'z1':300, 'z2':500,
               'density':100})

data = synthetic.from_prisms(prisms, x1=-500, x2=500, y1=-1000, y2=1000, 
                             nx=100, ny=200, height=150, field='gzz')

contam = data.copy()
contam['value'], error = fatiando.utils.contaminate(data['value'], stddev=0.01, 
                                                    percent=True, 
                                                    return_stddev=True)

io.dump('syntheticdata.txt', data)

pylab.figure()
pylab.title("Synthetic data with %g noise" % (error))
contour_grid(data, ncontours=5, color='b', width=1, style='dashed', fontsize=9, 
             label='Pure', alpha=1)
contour_grid(contam, ncontours=5, color='r', width=1, style='solid', 
             fontsize=9, label='Noisy', alpha=1)
pylab.legend()

pylab.show()
