"""
LMSolver:
    Class for solving non-linear inverse problems using the Levemberg-Marquardt
    algorithm. Used as a mother class for solving specific problems. 
    Includes estimators for Tikhonov regularization and total variation.
    Also contains functions for the statistical analysis of the inversion result
    and plotting.
    (Coming soon: adptative discretizations) 
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 08-Jun-2010'


import time
import logging
import math

import numpy

import fatiando
from fatiando.utils import contaminate

logger = logging.getLogger('lmsolver')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)




class LMSolver():
    """
    Mother class for non-linear inverse problem solvers using the 
    Levemberg-Marquardt algorithm.
    
    mean and standard deviation of the estimates made can be accessed via the
    mean and std class properties.
    
    Remember to set self._nparams!
    
    Methods that MUST be implemented by the child are:
        _build_sensibility(estimate)
        _build_first_deriv()
    Optional methods:
        set_discretization(...)
        _plot_data(folder)
        _plot_results(folder)
        some function for loading data
        some function(s) for creating synthetic models        
    """
    
    def __init__(self, data):
        """
        Parameters:
        
            - data: The data vector
        """        
        
        # Data parameters
        self._data = data
        
        # Inversion parameters
        self._first_deriv = None
        self._nparams = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
        # The logger for this class
        self._log = logging.getLogger('lmsolver')    


    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        'estimate' is the the point in the parameter space where the Jacobian
        will be evaluated.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_jacobian was called before being implemented"
            
    
    def _calc_residuals(self, data, estimate):
        """
        Calculate the residual vector based on the data and the current estimate
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_calc_residuals was called before being implemented"
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_first_deriv was called before being implemented"
                          
            
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
    
            
    def _calc_stds(self):
        """
        Calculate the standard deviations of the parameter estimates. 
        """
        
        stds = []
        
        for param in numpy.transpose(self._estimates):
            
            stds.append(param.std())
            
        return stds
    
    
    # Property for accessing the standard deviations of the parameter estimates
    std = property(_calc_stds)
        
            
    def clear(self):
        """
        Erase the inversion results.
        """
        
        # Inversion parameters
        self._first_deriv = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
    
    def solve(self, damping=0, smoothness=0, curvature=0, \
              initial_estimate=None, apriori_var=1, contam_times=10, \
              max_it=100, max_lm_it=10, lm_start=1, lm_step=10):       
        """
        Perform the inversion with Tikhonov regularization:
        
            * 0th order: Ridge Regression (Damped);
            
            * 1st order: Smoothness;
            
            * 2nd order: Curvature;
            
        Parameters:
            
            damping: 0th order regularization parameter (how much damping to 
                     apply)
            
            smoothness: 1st order regularization parameter (how much smoothness
                        to apply)
                        
            curvature: 2st order regularization parameter (how much to minimize
                       the curvature)
                       
            initial_estimate: an array with the initial estimate. If not given,
                              a zero array will be used. 
            
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.
                                                    
            max_it: maximum number of iterations when seaching for the minimum
                    of the goal (or misfit) function.
                    
            max_lm_it: maximum number of iterations in the LM when looking for
                         the right step size (Marquardt parameter).
                         
            lm_start: starting step size (Marquardt parameter).
            
            lm_step: how much to increase or decrease the step size at each
                       LM iteration.                   
        """       
        
        self._log.info("Regularization parameters:")
        self._log.info("  damping = %g" % (damping))
        self._log.info("  smoothness = %g" % (smoothness))
        self._log.info("  curvature = %g" % (curvature))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()        
        
        # Clear the estimates
        self._estimates = []
                    
        # Need to set the next estimate (p_k+1) because the first thing in the
        # loop is updating prev (previous estimate)
        if initial_estimate == None:
            
            self._log.info("Starting estimate with null vector")
            
            next = (10**(-10))*numpy.ones(self._nparams)
            
        else:
            
            next = initial_estimate       
            
        if self._first_deriv == None:
            
            self._first_deriv = self._build_first_deriv()
        
        # The data weight matrix is the inverse of the data covariance scaled
        # by the apriori variance factor
        Wd = apriori_var*numpy.linalg.inv(self._data.cov)
                                
        tmp = numpy.dot(self._first_deriv.T, self._first_deriv)
        # Make the parameter weight matrix with the regularization parameters
        Wp = damping*numpy.identity(self._nparams) + \
             smoothness*tmp + \
             curvature*numpy.dot(tmp.T, tmp)
    
        del tmp
                                
        data = self._data.array
                    
        self._log.info("Contaminate %d times (noise 0: original data)" \
                       % (contam_times))
                
        # Contaminate the data with Gaussian noise and run the inversion
        for contam_it in range(contam_times + 1):
            
            self._log.info("***** noise %d *****" % (contam_it))
                
            start = time.clock()
            
            residuals = self._calc_residuals(data, next)
            
            # Have to calculate the goal function for the initial estimate so
            # that I can compare it with the new estimate
            goal = numpy.dot(numpy.dot(residuals.T, Wd), residuals) + \
                   numpy.dot(numpy.dot(next.T, Wp), next)
            
            goals = [goal]
            
            lm_param = lm_start
            
            self._log.info("Starting goal function: %g" % (goal))
            self._log.info("Starting LM parameter: %g" % (lm_param))
            self._log.info("LM step size: %g" % (lm_step))
                                    
            for it in range(1, max_it + 1):
                                    
                inner_start = time.clock()
                
                prev = next
                
                jacobian = self._build_jacobian(prev)
                
                # The residuals were calculated in the previous iteration
                grad = 2*(numpy.dot(Wp, prev) - \
                          numpy.dot(numpy.dot(jacobian.T, Wd), residuals))
                    
                hessian = 2*(numpy.dot(numpy.dot(jacobian.T, Wd), jacobian) + \
                             Wp)
                                       
                hessian_diag = numpy.diag(numpy.diag(hessian))
                                                       
                # LM loop
                for lm_it in range(1, max_lm_it + 1):
                    
                    N = hessian + lm_param*hessian_diag
                    
                    correction = numpy.linalg.solve(N, -1*grad)
                    
                    next = prev + numpy.array(correction)
                                
                    residuals = self._calc_residuals(data, next)
                    
                    # Part of the goal function due to the residuals
                    goal = numpy.dot(numpy.dot(residuals.T, Wd), residuals) + \
                           numpy.dot(numpy.dot(next.T, Wp), next)
                                    
                    if goal < goals[it - 1]:
                        
                        goals.append(goal)
                        
                        lm_param /= float(lm_step)
                        
                        break
                        
                    else:
                        
                        lm_param *= float(lm_step)
                                        
                inner_end = time.clock()
                self._log.info("it %d: LM its = %d  LM param = %g  goal = %g" \
                               % (it, lm_it, lm_param, goal) + \
                               "  (%g s)" % (inner_end - inner_start))
                                            
                # Got out of LM loop because of max_marq_it reached
                if len(goals) == it: 
                    
                    next = prev

                    break
                
                # Stop if there is stagnation
                if abs((goals[it] - goals[it - 1])/goals[it - 1]) <= 10**(-4):
                    
                    break
                        
            self._estimates.append(next)
            
            # Keep the goals of the original data
            if contam_it == 0:
                
                self._goals = goals
            
            end = time.clock()
            self._log.info("Total time: %g s" % (end - start))            
                                            
            if contam_it == contam_times:
                
                break
            
            else:
                                   
                # Contaminate the data and run again            
                data = contaminate.gaussian( \
                                   self._data.array, \
                                   stddev=math.sqrt(apriori_var), \
                                   percent=False, return_stddev=False)
                
                # Reset the first estimate
                if initial_estimate == None:
                    
                    next = (10**(-10))*numpy.ones(self._nparams)
                        
                else:               
                    
                    next = initial_estimate
                               
        total_end = time.clock()
        self._log.info("Total time for inversion: %g s" \
                       % (total_end - total_start))   
        
