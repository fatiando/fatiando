"""
Generate some synthetic travel time data using the square-model.jpg image.
"""

import logging
logging.basicConfig()

import pylab

from fatiando.data.seismo import Cart2DTravelTime

            
ttdata = Cart2DTravelTime()

model = ttdata.synthetic_image('square-model.jpg', src_n=60, rec_n=10, \
                               dx=1, dy=1, vmin=0, vmax=10, stddev=0.005, \
                               type='xray')

ttdata.dump('travel-time-data.txt')

ttdata.plot_rays(title='Ray paths')

ttdata.plot_rays(model=model, dx=1, dy=1, title='Ray coverage', \
                      cmap=pylab.cm.jet)

ttdata.plot_synthetic(model, dx=1, dy=1, title='Synthetic model', \
                      cmap=pylab.cm.jet)

ttdata.plot_traveltimes()

pylab.show()