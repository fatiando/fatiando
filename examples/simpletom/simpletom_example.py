import logging
logging.basicConfig()

import pylab
import numpy

from fatiando.geoinv.simpletom import SimpleTom
from fatiando.data.seismo import CartTravelTime

            
ttdata = CartTravelTime()

#model = ttdata.synthetic_image('square-model.jpg', src_n=20, rec_n=10, \
#                               dx=1, dy=1, vmin=1, vmax=5, stddev=0.005)
#ttdata.dump('travel-time-data.txt')
ttdata.load('travel-time-data.txt')
#ttdata.plot_rays(dx=1, dy=1, title='Ray paths')
#ttdata.plot_synthetic(model, dx=1, dy=1, title='Synthetic model', \
#                      cmap=pylab.cm.Greys)
#ttdata.plot_traveltimes()

stom = SimpleTom(ttdata, x1=0, x2=30, y1=0, y2=30, nx=30, ny=30)

stom.solve(damping=10**(-3), smoothness=10**(0), curvature=0, \
           apriori_var=ttdata.cov[0][0], contam_times=20)

stom.plot_mean(vmin=1, vmax=5, title='Tikhonov Result')
stom.plot_std(title='Tikhonov Standard Deviation', cmap=pylab.cm.jet)
stom.plot_residuals(title='Tikhonov Residuals')

stom.sharpen(sharpness=1*10**(0), damping=10**(-3), beta=10**(-7), \
             initial_estimate=None, \
             apriori_var=ttdata.cov[0][0], \
             max_it=50, max_marq_it=20, marq_start=10**(2), marq_step=10, \
             contam_times=1)

stom.plot_mean(vmin=1, vmax=5, title='Total Variation Result')
stom.plot_std(title='Total Variation Standard Deviation')
stom.plot_residuals(title='Total Variation Residuals')
stom.plot_goal(title='Total Variation Goal Function')

pylab.show()        