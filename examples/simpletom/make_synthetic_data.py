"""
Generate some synthetic travel time data using the square-model.jpg image.
"""

import logging
logging.basicConfig()

import pylab

from fatiando.data.seismo import Cart2DTravelTime

            
ttdata = Cart2DTravelTime()

model = ttdata.synthetic_image('square-model.jpg', src_n=20, rec_n=10, \
                               dx=1, dy=1, vmin=1, vmax=5, stddev=0.005)

ttdata.dump('travel-time-data.txt')

ttdata.plot_rays(dx=1, dy=1, title='Ray paths')

ttdata.plot_synthetic(model, dx=1, dy=1, title='Synthetic model', \
                      cmap=pylab.cm.Greys)

ttdata.plot_traveltimes()

pylab.show()