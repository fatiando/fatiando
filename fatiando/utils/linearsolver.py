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
import time

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
    Methods that MUST be implemented by the child are:
        _build_sensibility()
        _build_first_deriv()
        _calc_distances(points, lines)
        _build_compact_weights(distances)
    Optional methods:
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
        self._goals = []
        
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
            
            
    def clear(self):
        """
        Erase the inversion results.
        """
        
        # Inversion parameters
        self._sensibility = [[]]
        self._first_deriv = [[]]
                
        # Inversion results
        self._estimates = []
        self._mean = []
        self._stddev = []
        self._residuals = []
        self._goals = []
        
    
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
        
        self._log.info("Regularization parameters:")
        self._log.info("  damping = %g" % (damping))
        self._log.info("  smoothness = %g" % (smoothness))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()
        
        # Clear previous results
        self.clear()
        
        self._sensibility = self._build_sensibility()
        
        ndata, nparams = self._sensibility.shape
        
        self._first_deriv = self._build_first_deriv()
        
        # The parameter weights
        Wp = damping*numpy.identity(nparams) + \
             smoothness*numpy.dot(self._first_deriv.T, self._first_deriv)
             
        # Data weight matrix
        Wd = apriori_var*numpy.linalg.inv(self._data_cov)
        
        # Overdetermined
        if nparams <= ndata:
            
            self._log.info("Solving overdetermined problem: %d d x %d p" % \
                           (ndata, nparams))            
                        
            # The normal equations
            N = numpy.dot(numpy.dot(self._sensibility.T, Wd), \
                          self._sensibility) + Wp
                          
            start = time.clock()
                          
            N_LU, N_permut = lu.decomp(N.tolist())
            
            end = time.clock()
            self._log.info("  LU decomposition of normal equations (%g s)" % \
                           (end - start))
            
            y = numpy.dot(numpy.dot(self._sensibility.T, Wd), self._data)
            
            start = time.clock()
            
            estimate = lu.solve(N_LU, N_permut, y.tolist())
            
            end = time.clock()
            self._log.info("  Solve linear system (%g s)" % (end - start))
            
            self._estimates.append(estimate)
            
            start = time.clock()
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                y = numpy.dot(numpy.dot(self._sensibility.T, Wd), contam_data)
                
                estimate = lu.solve(N_LU, N_permut, y.tolist())
                
                self._estimates.append(estimate)
                
            end = time.clock()
            self._log.info("  Contaminate data %d times " % (contam_times) + \
                           "with Gaussian noise (%g s)" % (end - start))
                   
        # Underdetermined        
        else:
            
            self._log.info("Solving underdetermined problem: %d d x %d p" % \
                           (ndata, nparams))            
            
            # Inverse of the data weight matrix
            Wd_inv = self._data_cov/apriori_var
            
            start = time.clock()
            
            # The inverse of the parameter weight matrix
            Wp_inv = numpy.linalg.inv(Wp)
            
            end = time.clock()
            self._log.info("  Invert parameter weights (%g s)" % (end - start))            
            
            # The normal equations
            N = numpy.dot(numpy.dot(self._sensibility, Wp_inv), \
                          self._sensibility.T) + Wd_inv
                          
            start = time.clock()
            
            N_inv = numpy.linalg.inv(N)
            
            end = time.clock()
            self._log.info("  Invert normal equations (%g s)" % (end - start))          
                          
            pseudo_inv = numpy.dot(numpy.dot(Wp_inv, self._sensibility.T), \
                                   N_inv)
        
            estimate = numpy.dot(pseudo_inv, self._data)
            
            self._estimates.append(estimate)
            
            start = time.clock()
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                estimate = numpy.dot(pseudo_inv, contam_data)
                
                self._estimates.append(estimate)
                
            end = time.clock()
            self._log.info("  Contaminate data %d times " % (contam_times) + \
                           "with Gaussian noise (%g s)" % (end - start))
            
        # Compute means, standard deviations and residuals
        for param in numpy.transpose(self._estimates):
            
            self._mean.append(param.mean())
            
            self._stddev.append(param.std())
            
        self._residuals = self._data - numpy.dot(self._sensibility, self._mean)

        total_end = time.clock()
        self._log.info("Total time: %g s" % (total_end - total_start))
        
        # Compute the goal function
        self._mean = numpy.array(self._mean)
        goal = numpy.dot(numpy.dot(self._residuals.T, Wd), self._residuals) + \
               numpy.dot(numpy.dot(self._mean.T, Wp), self._mean)
               
        self._log.info("Goal function: %g" % (goal))
        

    def compact(self, points, lines, compact=1, damping=0, smoothness=0, \
                apriori_var=1, contam_times=10, max_it=50):        
        """
        Perform a compact inversion. Parameters will be condensed around 
        geometric elements (points and lines).
        This is done with a non-linear process. The starting point for this can
        be either:
        
            1) previous estimate obtained by calling the solve function
            
            2) if 1 is absent, estimate initial solution with standard damped 
               least squares (damping=1)
               
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
            
            damping: 0th order regularization parameter (how much damping to 
                     apply)
            
            smoothness: 1st order regularization parameter (how much smoothness
                        to apply)
                    
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.
                          
            max_it: maximum number of iterations that the will be run
             
        The means and standard deviations of the 'contam_times' estimates are
        kept in the self._mean and self._stddev
        Residuals are also calculated based on the mean value and kept in 
        self._residuals                         
        """
        
        start = time.clock()
        
        # Clear the estimates
        self._estimates = []
        
        # Calculate the distances from the geometric elements and the target
        # values for the parameters
        distances, targets = self._calc_distances(points, lines)
        
        # Set the values for the first estimate in case there was none
        if len(self._mean) == 0:
            
            self._log.info("Starting estimate with null vector")
            
            self._sensibility = self._build_sensibility()
            
            self._first_deriv = self._build_first_deriv()
            
            ndata, nparams = self._sensibility.shape
            
            estimate = numpy.zeros(nparams)
            
            Wp = numpy.identity(nparams)
            
        else:
            
            self._log.info("Starting estimate with previous results")
                        
            estimate = numpy.copy(self._mean)
            
            ndata, nparams = self._sensibility.shape
            
            Wp = compact*self._build_compact_weights(distances, estimate)
                        
        # The data weight matrix
        if nparams <= ndata:
            
            self._log.info("Solving overdetermined problem: %d d x %d p" % \
                           (ndata, nparams))            
                        
            Wd = apriori_var*numpy.linalg.inv(self._data_cov)
            
        else:
            
            self._log.info("Solving underdetermined problem: %d d x %d p" % \
                           (ndata, nparams))            
                        
            Wd_inv = self._data_cov/apriori_var
            
        # First, start with the uncontaminated data
        contam_data = numpy.copy(self._data)
            
        self._log.info("Contaminate %d times (noise 0: original data)" \
                       % (contam_times))
        
        for contam_iteration in range(contam_times + 1):
            
            self._log.info("***** noise %d *****" % (contam_iteration))
            
            # Overdetermined  
            if nparams <= ndata:
                
                estimate = self._estimate_odet_compact(compact, \
                                                       damping, smoothness, \
                                                       estimate, Wp, \
                                                       contam_data, Wd, 
                                                       distances, targets, 
                                                       max_it)
                
            # Undertermined
            else:

                estimate = self._estimate_udet_compact(compact, \
                                                       damping, smoothness, \
                                                       estimate, Wp, \
                                                       contam_data, Wd_inv, \
                                                       distances, targets, \
                                                       max_it)
                        
            self._estimates.append(estimate)
                                            
            if contam_iteration == contam_times:
                
                break
                                   
            # Contaminate the data and run again            
            contam_data = contaminate(self._data, \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
            
            # Reset the first estimate
            if len(self._mean) == 0:
                
                estimate = numpy.zeros(nparams)
                
            else:
                
                estimate = numpy.copy(self._mean)                
                               
        # Compute means, standard deviations and residuals
        self._mean = []
        self._stddev = []
        for param in numpy.transpose(self._estimates):
            
            self._mean.append(param.mean())
            
            self._stddev.append(param.std())
            
        self._residuals = self._data - numpy.dot(self._sensibility, self._mean)
                        
        end = time.clock()
        self._log.info("Total time: %g s" % (end - start))   
                
                
    def _freeze_params(self, estimate, targets, param_weights):
        """
        Free the parameters that have passed their targets.
        """
        
        for i in range(len(estimate)):
            
            if targets[i] > 0:
                
                if abs(estimate[i]) > 1.05*targets[i]:
                     
                    estimate[i] = targets[i]
                    
                    param_weights[i][i] *= 10**3
                     
                elif estimate[i] < 0:                    
                    
                    estimate[i] = 0      
                
                    param_weights[i][i] *= 10**3         
                    
            if targets[i] < 0:
                
                if estimate[i] < 1.05*targets[i]:
                                             
                    estimate[i] = targets[i]
                    
                    param_weights[i][i] *= 10**3
                     
                elif estimate[i] > 0:
                    
                    estimate[i] = 0
                    
                    param_weights[i][i] *= 10**3
                    
                
    def _estimate_udet_compact(self, compact, damping, smoothness, \
                               initial_estimate, initial_param_weights, \
                               data, data_weights_inv, distances, targets, \
                               max_iterations=50):
        """
        Iterated to find the compact estimate in an underdetermined problem.
        
        Parameters:
        
            compact: regularization parameter
            
            initial_estimate: starting point for the iterations
            
            data: the data vector
        
            data_weights_inv: the inverse of the data weight matrix
            
            distances: distances of each parameter to the closest geometric
                       element
                       
            targets: target values for each parameter
            
        Returns the final estimate vector.
        """       
        
        nparams = len(initial_estimate)
        
        estimate = numpy.copy(initial_estimate)
        
        # The Tikhonov parameter weights
        Wp_tikho = damping*numpy.identity(nparams) + \
                   smoothness*numpy.dot(self._first_deriv.T, self._first_deriv)
        
        # Parameter weight matrix
        Wp = initial_param_weights + Wp_tikho
        
        # Iterate to compact the mass around the geometric elements
        for iteration in range(max_iterations):
            
            self._log.info("it %d:" % (iteration + 1))
            it_start = time.clock()
        
            start = time.clock()
            
            Wp_inv = numpy.linalg.inv(Wp)
            
            end = time.clock()
            self._log.info("  Invert parameter weights (%g s)" % (end - start))
            
            # The normal equations
            N = numpy.dot(numpy.dot(self._sensibility, Wp_inv), \
                          self._sensibility.T) + data_weights_inv
                          
            start = time.clock()                          
                          
            N_inv = numpy.linalg.inv(N)
            
            end = time.clock()
            self._log.info("  Invert normal equations (%g s)" % (end - start))
                          
            pseudo_inv = numpy.dot(numpy.dot(Wp_inv, \
                                   self._sensibility.T), N_inv)
            
            misfit = data - numpy.dot(self._sensibility, \
                                             estimate)
        
            correction = numpy.dot(pseudo_inv, misfit)
            
            estimate += correction
                        
            it_end = time.clock()
            self._log.info("  Iteration time: %g s" % (it_end - it_start))
            
            num = 0
            for i in range(nparams):
                
#                if abs(estimate[i]) <= 1.10*abs(targets[i]):
                if abs(correction[i]) <= 0.01*abs(targets[i]):# and \
#                   abs(estimate[i]) <= 1.15*abs(targets[i]):
                    
                    num += 1
            
            if num == nparams or iteration == max_iterations - 1:
                
                break
            
            else:            
                
                Wp = compact*self._build_compact_weights(distances, estimate) +\
                     Wp_tikho
            
                self._freeze_params(estimate, targets, Wp)
        
        return estimate
        
                
    def _estimate_odet_compact(self, compact, damping, smoothness, \
                               initial_estimate, initial_param_weights, \
                               data, data_weights, distances, targets, \
                               max_iterations=50):
        """
        Iterated to find the compact estimate in an overdetermined problem.
        
        Parameters:
        
            compact: regularization parameterDon_L._Anderson-New_theory_of_the_Earth-Cambridge_University_Press(2007).pdf
            
            initial_estimate: starting point for the iterations
            
            data: the data vector
        
            data_weights: the data weight matrix
            
            distances: distances of each parameter to the closest geometric
                       element
                       
            targets: target values for each parameter
            
        Returns the final estimate vector.
        """       
        
        nparams = len(initial_estimate)
        
        estimate = numpy.copy(initial_estimate)
                
        # The Tikhonov parameter weights
        Wp_tikho = damping*numpy.identity(nparams) + \
                   smoothness*numpy.dot(self._first_deriv.T, self._first_deriv)
                
        # Parameter weight matrix
        Wp = initial_param_weights + Wp_tikho
        
        # Iterate to compact the mass around the geometric elements
        for iteration in range(max_iterations):
            
            self._log.info("it %d:" % (iteration + 1))
            it_start = time.clock()
        
            # The normal equations                
            N = numpy.dot(numpy.dot(self._sensibility.T, data_weights), \
                          self._sensibility) + Wp
                          
            start = time.clock()
            
            N_LU, N_permut = lu.decomp(N.tolist())
            
            end = time.clock()
            self._log.info("  LU decomposition of normal equations (%g s)" \
                           % (end - start))
            
            misfit = data - numpy.dot(self._sensibility, estimate)
            
            y = numpy.dot(numpy.dot(self._sensibility.T, data_weights), misfit)
            
            start = time.clock()
            
            correction = lu.solve(N_LU, N_permut, y.tolist())
            
            end = time.clock()
            self._log.info("  Solve linear system (%g s)" % (end - start))
                        
            estimate += correction
                        
            it_end = time.clock()
            self._log.info("  Iteration time: %g s" % (it_end - it_start))
            
            num = 0
            for i in range(nparams):
                
                if abs(correction[i])/abs(estimate[i]) < 10*-(8):
                    
                    num += 1
            
            if num == nparams or iteration == max_iterations - 1:
                
                break
            
            else:            
                
                Wp = compact*self._build_compact_weights(distances, estimate) +\
                     Wp_tikho
            
                self._freeze_params(estimate, targets, Wp)
        
        return estimate
    
    
    def sharpen(self, sharpen=1, initial_estimate=[], apriori_var=1, 
                contam_times=10, max_it=100, \
                max_marq_it=10, marq_start=1, marq_step=10):
        """
        Invert with Total Variation to create a sharpened image. Uses the 
        Levenberg-Marquardt (LM) algorithm to optimize the goal (misfit) 
        function.
        
        Note: If the initial estimate is not given:

            1) initial estimate is the previous estimate obtained by calling the
               solve function
            
            2) if 1 is absent, estimate initial solution with standard damped 
               least squares (damping=1)
        
        Parameters:
        
            sharpen: the regularization parameter (how much to sharpen)
            
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.
                          
            max_it: maximum number of iterations when seaching for the minimum
                    of the goal (or misfit) function.
                    
            max_marq_it: maximum number of iterations in the LM when looking for
                         the right step size (Marquardt parameter).
                         
            marq_start: starting step size (Marquardt parameter).
            
            marq_step: how much to increase or decrease the step size at each
                       LM iteration.
        """
        
        self._log.info("Regularization parameters:")
        self._log.info("  sharpness = %g" % (sharpen))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()
                
        if initial_estimate == []:
                
            if len(self._mean) == 0:            
                
                self._log.info("Calculating initial estimate with Tikhonov " + \
                               "order 0 (damped)")
            
                self.solve(damping=1, smoothness=0, apriori_var=apriori_var, \
                           contam_times=0)
            
            else:              
                
                self._log.info("Using previous solution as initial estimate")
                    
            next = self._mean
                
        else:               
            
            self._sensibility = self._build_sensibility()
            
            self._first_deriv = self._build_first_deriv()
            
            next = initial_estimate
        
        # Clear the estimates
        self._estimates = []
            
        ndata, nparams = self._sensibility.shape
                
        nderivs = len(self._first_deriv)
        
        identity_nderivs = numpy.identity(nderivs)
        
        # The data weight matrix
        Wd = apriori_var*numpy.linalg.inv(self._data_cov)
        
        eps = 10**(-7)
        
        # The Hessian due to the residuals
        hessian_res = 2*numpy.dot(numpy.dot(self._sensibility.T, Wd), \
                              self._sensibility)
                
        data = self._data
                    
        self._log.info("Contaminate %d times (noise 0: original data)" \
                       % (contam_times))
                
        # Contaminate the data with Gaussian noite and run the inversion
        for contam_it in range(contam_times + 1):
            
            self._log.info("***** noise %d *****" % (contam_it))
                
            start = time.clock()
            
            residuals = data - numpy.dot(self._sensibility, next)
            
            # Part of the goal function due to the residuals
            goal = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
            
            # Part due to the Total Variation
            derivatives = numpy.dot(self._first_deriv, next)
            
            for deriv in derivatives:
                
                goal += abs(deriv)
            
            goals = [goal]
            
            marq_param = marq_start
            
            self._log.info("Starting goal function: %g" % (goal))
            self._log.info("Starting LM parameter: %g" % (marq_param))
            self._log.info("LM step size: %g" % (marq_step))
                                    
            for it in range(1, max_it + 1):
                
                self._log.info("it %d:" % (it))
                    
                inner_start = time.clock()
                
                prev = next
                
                misfit = data - numpy.dot(self._sensibility, prev)
                                
                # Auxiliary for calculating the Hessian and gradient
                d = numpy.zeros(nderivs)
                D = numpy.zeros((nderivs, nderivs))
                for l in range(len(self._first_deriv)):
                    
                    deriv = numpy.dot(self._first_deriv[l], prev)
                    
                    sqrt = math.sqrt(deriv**2 + eps)
                    
                    d[l] = deriv/sqrt
                    
                    D[l][l] = eps/(sqrt**3)
                                    
                grad = -2*numpy.dot(numpy.dot(self._sensibility.T, Wd), \
                                    misfit) + \
                        sharpen*numpy.dot(self._first_deriv.T, d)
                    
                hessian = hessian_res + \
                          numpy.dot(numpy.dot(self._first_deriv.T, \
                                              sharpen*D + identity_nderivs), \
                                    self._first_deriv)
                                       
                hessian_diag = numpy.diag(numpy.diag(hessian))
                
                inner_end = time.clock()
                self._log.info("  Build the gradient and Hessian (%g s)" \
                               % (inner_end - inner_start)) 
                                                       
                # LM loop
                for marq_it in range(1, max_marq_it + 1):
                    
                    self._log.info("  LM it %d: " % (marq_it))
                    self._log.info("    LM parameter: %g" % (marq_param))
                    
                    inner_start = time.clock()
                    
                    N = hessian + marq_param*hessian_diag
                    
#                    N_LU, N_permut = lu.decomp(N.tolist())
#                    
#                    correction = lu.solve(N_LU, N_permut, grad.tolist())
                    correction = numpy.linalg.solve(N, grad.tolist())
                    
                    inner_end = time.clock()
                    self._log.info("    Solve for correction (%g s)" \
                                   % (inner_end - inner_start)) 
                    
                    next = prev - numpy.array(correction)
                                
                    residuals = data - numpy.dot(self._sensibility, next)
                    
                    # Part of the goal function due to the residuals
                    goal = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
                    
                    # Part due to the Total Variation
                    derivatives = numpy.dot(self._first_deriv, next)
                    for deriv in derivatives:
                        
                        goal += abs(deriv)
                        
                    self._log.info("    Goal function: %f" % (goal))
                
                    if goal < goals[it - 1]:
                        
                        goals.append(goal)
                        
                        marq_param /= float(marq_step)
                        
                        break
                        
                    else:
                        
                        marq_param *= float(marq_step)
            
                # Stopping conditions for the LM algorithm
                
                # Got out of LM loop because of max_marq_it reached
                if len(goals) == it: 
                    
                    next = prev

                    break
                
                # Stagnation
                if abs((goals[it] - goals[it - 1])/goals[it - 1]) <= 10**(-6):
                    
                    break
                        
            self._estimates.append(next)
            
            # Keep the goals of the original data
            if contam_it == 0:
                
                self._goals = numpy.copy(goals)
            
            end = time.clock()
            self._log.info("Total time: %g s" % (end - start))            
                                            
            if contam_it == contam_times:
                
                break
            
            else:
                                   
                # Contaminate the data and run again            
                data = contaminate(self._data, \
                                   stddev=math.sqrt(apriori_var), \
                                   percent=False, return_stddev=False)
                
                # Reset the first estimate
                if initial_estimate == []:
                            
                    next = self._mean
                        
                else:               
                    
                    next = initial_estimate
                               
        # Compute means, standard deviations and residuals
        self._mean = []
        self._stddev = []
        for param in numpy.transpose(self._estimates):
            
            self._mean.append(param.mean())
            
            self._stddev.append(param.std())
            
        self._residuals = self._data - numpy.dot(self._sensibility, self._mean)
                        
        total_end = time.clock()
        self._log.info("Total time for inversion: %g s" \
                       % (total_end - total_start))   
                
            