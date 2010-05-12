"""
LinearSolver:
    Class for solving linear inverse problems. Used as a mother class for 
    solving specific problems. Includes estimators for Tikhonov regularization,
    compactness and minimum moment of inertia, and total variation.
    Also contains functions for the statistical analysis of the inversion result
    and plotting.
    (Coming soon: adptative discretizations) 
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-May-2010'


import logging
import exceptions
import math
import sys

import numpy

from fatiando.math import lu
from fatiando.utils.datamani import contaminate

class LinearSolverError(exceptions.Exception):
    """
    Standard exception for the LinearSolver class.
    """
    pass


# Set the default handler to the class logger. 
# By default, logging is set to stderr.
################################################################################ 
stlogger = logging.getLogger('linearsolver')
       
stlogger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
console_handler = logging.StreamHandler(strm=sys.stderr)

# Create formatter
formatter = logging.Formatter("linearsolver> %(message)s")

# Add formatter to the console handler
console_handler.setFormatter(formatter)

# Set verbose level
console_handler.setLevel(logging.INFO)

# Add the console handler to logger
stlogger.addHandler(console_handler)
################################################################################


class LinearSolver():
    """
    Mother class for linear inverse problem solvers.
    Methods that should be implemented by the child are:
        _build_sensibility()
        _build_first_deriv()
        _calc_distances(points, lines)
        _build_compact_weights(distances)
        set_discretization(...)
        _plot_data(folder)
        _plot_results(folder)
        some function for loading data
        some function(s) for creating synthetic models        
    """
    
    def __init__(self):
        """
        Initialize the paramters
        """        
        
        # Data parameters
        self._data = []
        self._data_cov = [[]]
        
        # Inversion parameters
        self._sensibility = [[]]
        self._first_deriv = [[]]
                
        # Inversion results
        self._estimates = []
        self._mean = []
        self._stddev = []
        self._residuals = []
        
        # The logger for this class
        self._log = logging.getLogger('linearsolver')    


    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "_build_sensibility was called before being implemented"
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "_build_first_deriv was called before being implemented"
        
        
    def _calc_distances(self, points, lines):
        """
        Calculate the distance from each model element to the closest point or 
        line and put the values in the compact_weights. 
        Also assign the target value for each parameter based on the closest 
        point or line.
        """       
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "_calc_distances was called before being implemented"
        
                
    def _build_compact_weights(self, distances, estimate):
        """
        Calculate the weights for the compactness and MMI regularizations.
        'estimate' is the current estimate for the parameters.
        """    
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "_build_compact_weights was called before being implemented"
                        
                        
    def set_discretization(self):
        """
        Set the discretization of the model space.
        Must be done before the inversion!
        """
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "set_discretization was called before being implemented"
                        
            
    def plot_data(self, outdir=None):
        """
        Plot the data.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "plot_data was called before being implemented"
             
                          
    def plot_results(self, outdir=None):
        """
        Plot the results.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise LinearSolverError, \
            "plot_results was called before being implemented"
                        
    
    def clear(self):
        """
        Erase the inversion results.
        """
        
        # Inversion parameters
        self._sensibility = [[]]
        self._first_deriv = [[]]
                
        # Inversion results
        self._estimates = [[]]
        self._mean = []
        self._stddev = []
        self._residuals = []
        
    
    def solve(self, damping=0, smoothness=0, apriori_var=1, contam_times=10):        
        """
        Perform the inversion with Tikhonov regularization:
        
            * 0th order: Ridge Regression (Damped);
            
            * 1st order: Smoothness;
            
        Parameters:
            
            damping: 0th order regularization parameter (how much damping to 
                     apply)
            
            smoothness: 1st order regularization parameter (how much smoothness
                        to apply)
            
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.
                          
        The means and standard deviations of the 'contam_times' estimates are
        kept in the self._mean and self._stddev
        Residuals are also calculated based on the mean value and kept in 
        self._residuals                         
        """
        
        self._sensibility = self._build_sensibility()
        
        ndata, nparams = self._sensibility.shape
        
        self._first_deriv = self._build_first_deriv()
        
        # The parameter weights
        Wp = damping*numpy.identity(nparams) + \
             smoothness*numpy.dot(self._first_deriv.T, self._first_deriv)
        
        # Overdetermined
        if nparams < ndata:
            
            # Data weight matrix
            Wd = apriori_var*numpy.linalg.inv(self._data_cov)
            
            # The normal equations
            N = numpy.dot(numpy.dot(self._sensibility.T, Wd), \
                          self._sensibility) + Wp
                          
            # Use LU decomposition to solve the system N*p=A.T*d
            N_LU, N_permut = lu.decomp(N.tolist())
            
            # An auxiliary variable
            y = numpy.dot(numpy.dot(self._sensibility.T, Wd), self._data)
            
            estimate = lu.solve(N_LU, N_permut, y.tolist())
            
            self._estimates.append(estimate)
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                y = numpy.dot(numpy.dot(self._sensibility.T, Wd), contam_data)
                
                estimate = lu.solve(N_LU, N_permut, y.tolist())
                
                self._estimates.append(estimate)
        
        # Underdetermined        
        else:
            
            # Inverse of the data weight matrix
            Wd_inv = self._data_cov/apriori_var
            
            # The inverse of the parameter weight matrix
            Wp_inv = numpy.linalg.inv(Wp)
            
            # The normal equations
            N = numpy.dot(numpy.dot(self._sensibility, Wp_inv), \
                          self._sensibility.T) + Wd_inv
                          
            # The inverse of the normal equations
            N_inv = numpy.linalg.inv(N)
                          
            # The pseudo-inverse
            pseudo_inv = numpy.dot(numpy.dot(Wp_inv, self._sensibility.T), \
                                   N_inv)
        
            estimate = numpy.dot(pseudo_inv, self._data)
            
            self._estimates.append(estimate)
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                estimate = numpy.dot(pseudo_inv, contam_data)
                
                self._estimates.append(estimate)
                
        # Compute means, standard deviations and residuals
        for param in numpy.transpose(self._estimates):
            
            self._mean.append(param.mean())
            
            self._stddev.append(param.std())
            
        self._residuals = self._data - numpy.dot(self._sensibility, self._mean)


    def compact(self, points, lines, compact=0, apriori_var=1, \
                contam_times=10, max_iterations=50):        
        """
        Perform a compact inversion. Parameters will be condensed around 
        geometric elements (points and lines).
        This is done with a non-linear process. The starting point for this can
        be either:
            1) previous estimate obtained by calling the solve function
            
            2) if 1 is absent, estimate initial solution with standard damped 
               least squares
               
        NOTE: The value of variables estimates, mean, stddev, and residuals will
            be altered!
            
        Parameters:
            
            points: list of points around which to enforce the compactness of 
                    the solution. Each point should be a tuple such as: 
                        ((x,y,z), value) 
                    The z coordinate should be omitted in 2 dimensional problems
                    'value' is the target value for the physical property 
                    (density, velocity, etc).
            
            lines: list of lines around which to enforce the minimum moment of
                   inertia. As in points, each line should be a tuple such as: 
                       ((x1,y1,z1), (x2,y2,z2), value)
                   The 2 (x,y,z) are the points that describe the line. Remarks
                   about the z coordinate and 'value' are the same as for points
                   
            compact: regularization parameter for the compactness (how much to
                     enforce the compactness)
                    
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.
                          
            max_iterations: maximum number of iterations that the will be run
             
        The means and standard deviations of the 'contam_times' estimates are
        kept in the self._mean and self._stddev
        Residuals are also calculated based on the mean value and kept in 
        self._residuals                         
        """
        
        # Clear the estimates
        self._estimates = []
        
        # Calculate the distances from the geometric elements and the target
        # values for the parameters
        distances, targets = self._calc_distances(points, lines)
        
        # Set the values for the first estimate in case there was none
        if len(self._mean) == 0:
            
            self._sensibility = self._build_sensibility()
            
            ndata, nparams = self._sensibility.shape
            
            # The parameter weights
            Wp = numpy.identity(nparams)
            
            # The first estimate
            estimate = numpy.zeros(nparams)
            
        else:
            
            # The first estimate
            estimate = numpy.copy(self._mean)
            
            # The parameter weight matrix
            Wp = compact*self._build_compact_weights(distances, estimate)
            
            ndata, nparams = self._sensibility.shape
            
        # The data weight matrix
        if nparams < ndata:
            
            Wd = apriori_var*numpy.linalg.inv(self._data_cov)
            
        else:
            
            Wd_inv = self._data_cov/apriori_var
            
        contam_data = numpy.copy(self._data)
            
        # Now compact the solution
        for contam_iteration in range(contam_times):
                
            for iteration in range(max_iterations):
                
                # Overdetermined
                if nparams < ndata:
                    
                    # The normal equations                
                    N = numpy.dot(numpy.dot(self._sensibility.T, Wd), \
                                  self._sensibility) + Wp
                                  
                    # Use LU decomposition to solve the system N*p=A.T*d
                    N_LU, N_permut = lu.decomp(N.tolist())
                    
                    # The data misfit of the current estimate
                    misfit = contam_data - numpy.dot(self._sensibility, \
                                                     estimate)
                    
                    # An auxiliary variable
                    y = numpy.dot(numpy.dot(self._sensibility.T, Wd), misfit)
                    
                    correction = lu.solve(N_LU, N_permut, y.tolist())
                    
                # Undertermined
                else:
                    
                    # The inverse of the parameter weights
                    Wp_inv = numpy.linalg.inv(Wp)
                    
                    # The normal equations
                    N = numpy.dot(numpy.dot(self._sensibility, Wp_inv), \
                                  self._sensibility.T) + Wd_inv
                                  
                    # The inverse of the normal equations
                    N_inv = numpy.linalg.inv(N)
                                  
                    # The pseudo-inverse
                    pseudo_inv = numpy.dot(numpy.dot(Wp_inv, \
                                           self._sensibility.T), N_inv)
                    
                    # The data misfit of the current estimate
                    misfit = contam_data - numpy.dot(self._sensibility, \
                                                     estimate)
                
                    correction = numpy.dot(pseudo_inv, misfit)
                                    
                estimate += correction
                
                # Stop if all estimates are bellow their targets\
                num = 0
                for i in range(nparams):
                    
                    if abs(estimate[i]) < 1.15*abs(targets[i]):
                        
                        num += 1
                
                if num == nparams:
                    
                    break
                
                # If not, update Wp and move on                        
                Wp = compact*self._build_compact_weights(distances, estimate)
                
                # Freeze the parameters that have passed the targets
                for i in range(nparams):
                    
                    if targets[i] > 0:
                        
                        if estimate[i] > 1.05*targets[i]:
                             
                            estimate[i] = targets[i]
                            
                            Wp[i][i] = 10**10
                             
                        elif estimate[i] < 0: 
                        
                            if estimate[i] > -0.1*targets[i]:
                            
                                estimate[i] = 0
                                
                                Wp[i][i] = 10**10
                            
                            else:
                                
                                estimate[i] = targets[i]
                                
                        elif  estimate[i] < 0.5*targets[i]:
                             
                            estimate[i] = 0
                            
                            Wp[i][i] = 10**10
                            
                            
                            
                    if targets[i] < 0:
                        
                        if estimate[i] < 1.05*targets[i]:
                                                     
                            estimate[i] = targets[i]
                            
                            Wp[i][i] = 10**10
                             
                        elif estimate[i] > 0:
                            
                            estimate[i] = 0
                            
                            Wp[i][i] = 10**10
                                    
            # Save the estimate contaminate the data and run again
            self._estimates.append(estimate)
            
            contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                               
        # Compute means, standard deviations and residuals
        self._mean = []
        self._stddev = []
        for param in numpy.transpose(self._estimates):
            
            self._mean.append(param.mean())
            
            self._stddev.append(param.std())
            
        self._residuals = self._data - numpy.dot(self._sensibility, self._mean)
                
        
            