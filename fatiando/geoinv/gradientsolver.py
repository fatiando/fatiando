"""
GradientSolver:
    Class for generic solving of inverse problems using gradient methods. 
    Subclass them and implement the problem specific methods to solve a new 
    inverse problem.
    Includes:
        * solving linear inverse problems with Tikhonov regularization
        * total variation inversion
        * solving non-linear inverse problems with Levemberg-Marquardt algorithm
        * allows receiving prior parameter weights (useful for depth-weighing in
          gravity inversion)
        * automatically contaminates the data with Gaussian noise and re-runs
          the inversion to generate many estimates
        * calculate mean and standard deviation of the various estimates
        * plot histogram of residuals
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 15-Jul-2010'


import time
import logging
import math

import numpy
import pylab

import fatiando
from fatiando.utils import contaminate


# Add the default handler (a null handler) to the class loggers to ensure that
# they won't print verbose if the program calling them doesn't want it
logger = logging.getLogger('GradientSolver')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class GradientSolver():
    """
    Generic inverse problem solver.
    Subclass this and implement the methods:
        * _build_jacobian
        * _build_first_deriv
        * _calc_adjustment
        * _get_data_array
        * _get_data_cov
    To use equality constraints simply implement a method that fills
    self._equality_matrix and self._equality_values.
    """
    
    
    def __init__(self):
        
        # inversion parameters
        self._nparams = 0
        self._equality_matrix = None
        self._equality_values = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
        # The logger for this class
        self._log = logging.getLogger('GradientSolver')
        
    
    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        'estimate' is the the point in the parameter space where the Jacobian
        will be evaluated.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_jacobian was called before being implemented"
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_calc_adjusted_data was called before being implemented"
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_first_deriv was called before being implemented"
            
            
    def _get_data_array(self):
        """
        Return the data in a Numpy array so that the algorithm can access it
        in a general way
        """        
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_get_data_array was called before being implemented"
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_get_data_cov was called before being implemented"
        
                                      
    def _calc_means(self):
        """
        Calculate the means of the parameter estimates. 
        """
        
        means = []
        
        for param in numpy.transpose(self._estimates):
            
            means.append(param.mean())
            
        return means
    
    
    # Property for accessing the means of the parameter estimates
    mean = property(_calc_means)
    
            
    def _calc_stddevs(self):
        """
        Calculate the standard deviations of the parameter estimates. 
        """
        
        stds = []
        
        for param in numpy.transpose(self._estimates):
            
            stds.append(param.std())
            
        return stds
    
    
    # Property for accessing the standard deviations of the parameter estimates
    stddev = property(_calc_stddevs)
        
            
    def clear(self):
        """
        Erase the inversion results and parameters.
        """
                
        # inversion parameters
        self._nparams = 0
        self._equality_matrix = None
        self._equality_values = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
        
    def linear_solve(self, damping=0, smoothness=0, curvature=0, equality=0, \
                     adjustment=1, prior_mean=None, prior_weights=None, \
                     data_variance=0, contam_times=0):
        """
        Solve the linear inverse problem with Tikhonov regularization.
        
        Parameters:
            
            damping:
        
            smoothness:
                   
            curvature:
        
            equality:
        
            adjustment:
            
            prior_mean:
            
            prior_weights:
            
            data_variance:
            
            contam_times:
        """