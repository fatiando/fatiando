"""
Integration test for SimpleTom. 
Run it on a known data set and see if the output is still good.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 16-Jun-2010'

import os
import logging

import pylab
import numpy

from fatiando.geoinv.simpletom import SimpleTom
from fatiando.data.seismo import CartTravelTime


def run():
    """
    Run SimpleTom on a known image file to check output.
    """
        
    this_dir = os.path.dirname(os.path.abspath(os.path.split(__file__)[-1]))
    
    log_file = os.path.join(this_dir, 'simpletom_integration.log')
    
    logging.basicConfig(filename=log_file, filemode='w')
            
    ttdata = CartTravelTime()

    datadir = os.path.join(os.path.dirname(__file__),'simpletom-testdata')

    ttdata.load(os.path.join(datadir, 'simpletom-traveltime-data.txt'))

    stom = SimpleTom(ttdata, x1=0, x2=30, y1=0, y2=30, nx=30, ny=30)

    stom.solve(damping=0.1, smoothness=1, curvature=1, \
               apriori_var=ttdata.cov[0][0], contam_times=20)

    stom.plot_mean(vmin=1, vmax=5, title='Tikhonov Result')
    stom.plot_std(title='Tikhonov Standard Deviation')
    stom.plot_residuals(title='Tikhonov Residuals')

    stom.sharpen(sharpen=1, initial_estimate=numpy.zeros_like(stom.mean), \
                 apriori_var=ttdata.cov[0][0], \
                 max_it=30, max_marq_it=20, marq_start=100, marq_step=10, \
                 contam_times=2)

    stom.plot_mean(vmin=1, vmax=5, title='Total Variation Result')
    stom.plot_std(title='Total Variation Standard Deviation')
    stom.plot_residuals(title='Total Variation Residuals')
    stom.plot_goal(title='Total Variation Goal Function')

    pylab.show()        
    

if __name__ == '__main__':

    run()