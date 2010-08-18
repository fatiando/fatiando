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

from fatiando.inversion.simpletom import SimpleTom
from fatiando.data.seismo import Cart2DTravelTime


def run():
    """
    Run SimpleTom on a known image file to check output.
    """
    
    # Setup the logging
    this_dir = os.path.dirname(os.path.abspath(os.path.split(__file__)[-1]))
    
    log_file = os.path.join(this_dir, 'simpletom_integration.log')
    
    logging.basicConfig(filename=log_file, filemode='w')
    
    imagefile = os.path.join(this_dir, \
                             os.path.join('simpletom-testdata', \
                                          'simpletom-square.jpg'))
    
    # Make some synthetic data
    ttdata = Cart2DTravelTime()
    
    model = ttdata.synthetic_image(imagefile, src_n=20, rec_n=10, \
                                   dx=1, dy=1, vmin=1, vmax=5, stddev=0.005)
   
    ttdata.plot_rays(dx=1, dy=1, title='Ray paths')
    
    ttdata.plot_synthetic(model, dx=1, dy=1, title='Synthetic model', \
                          cmap=pylab.cm.Greys)
    
    ttdata.plot_traveltimes()
            
    # Make a solver and set the model space discretization
    solver = SimpleTom(ttdata, x1=0, x2=30, y1=0, y2=30, nx=30, ny=30)
    
    # Solve the linear problem with only damping
    solver.solve_linear(damping=10**(-1), \
                        smoothness=0, \
                        curvature=0, \
                        prior_mean=None, \
                        prior_weights=None, \
                        data_variance=ttdata.cov[0][0], \
                        contam_times=20)
    
    # Plot the results
    solver.plot_mean(title='Damping Result (supposed to be bad)')
    solver.plot_stddev(title='Damping Standard Deviation', cmap=pylab.cm.jet)
    solver.plot_residuals(title='Damping Residuals')
    
    # Solve the linear problem with smoothness
    solver.solve_linear(damping=10**(-3), \
                        smoothness=10**(-1), \
                        curvature=0, \
                        prior_mean=None, \
                        prior_weights=None, \
                        data_variance=ttdata.cov[0][0], \
                        contam_times=20)
    
    # Plot the results
    solver.plot_mean(title='Smoothness Result')
    solver.plot_stddev(title='Smoothness Standard Deviation', cmap=pylab.cm.jet)
    solver.plot_residuals(title='Smoothness Residuals')
    
    # Solve the linear problem with curvature
    solver.solve_linear(damping=10**(-3), \
                        smoothness=0, \
                        curvature=10**(-1), \
                        prior_mean=None, \
                        prior_weights=None, \
                        data_variance=ttdata.cov[0][0], \
                        contam_times=20)
    
    # Plot the results
    solver.plot_mean(title='Curvature Result')
    solver.plot_stddev(title='Curvature Standard Deviation', cmap=pylab.cm.jet)
    solver.plot_residuals(title='Curvature Residuals')

    pylab.show()        
    

if __name__ == '__main__':

    run()