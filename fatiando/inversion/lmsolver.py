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
import pylab

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
        _plot_data(folder)
        _plot_results(folder)
        some function for loading data
        some function(s) for creating synthetic models        
    """
    
    def __init__(self):
        
        # Inversion parameters
        self._first_deriv = None
        self._nparams = None
        self._equality_matrix = None
        self._equality_values = None
                
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
        self._equality_matrix = None
        self._equality_values = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
    
    def solve(self, damping=0, smoothness=0, curvature=0, sharpness=0, \
              equality=0, param_weights=None, initial_estimate=None, \
              apriori_var=1, beta=10**(-7), \
              contam_times=10, max_it=100, max_lm_it=10, lm_start=1, \
              lm_step=10):       
        """
        Perform the inversion with Tikhonov and Total Variation regularization:
        
            * 0th order: Ridge Regression (Damped);
            
            * 1st order: Smoothness;
            
            * 2nd order: Curvature;
            
            * TV: Sharpness
            
        Uses the Levenberg-Marquardt (LM) algorithm to optimize the goal 
        (misfit) function.
            
        Parameters:
            
            damping: 0th order regularization parameter (how much damping to 
                     apply)
            
            smoothness: 1st order regularization parameter (how much smoothness
                        to apply)
                        
            curvature: 2st order regularization parameter (how much to minimize
                       the curvature)
                       
            sharpness: the TV regularization parameter (how much to sharpen)
            
            equality: how much to enforce the equality constraints
                       
            initial_estimate: an array with the initial estimate. If not given,
                              a zero array will be used. 
                              
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                
            beta: abs(R*p) (TV goal function) is substituted by a differential
                 form: sqrt((R*p)**2 + beta). Should be a small positive float
                                     
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
        self._log.info("  sharpness = %g" % (sharpness))
        self._log.info("  equality = %g" % (equality))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()        
        
        # Clear the estimates
        self._estimates = []
                    
        # Need to set the next estimate (p_k+1) because the first thing in the
        # loop is updating prev (previous estimate)
        if initial_estimate == None:
            
            self._log.info("Starting estimate with (almost) null vector")
            
            next = (10**(-10))*numpy.ones(self._nparams)
            
        else:
            
            next = initial_estimate       
                                        
        # Make the parameter weight matrix with the regularization parameters
        Wp = numpy.zeros((self._nparams, self._nparams))

        if damping:
            
            Wp = Wp + damping*numpy.identity(self._nparams)
            
        if smoothness or curvature or sharpness:
            
            self._first_deriv = self._build_first_deriv()
            
            nderivs = len(self._first_deriv)
             
            tmp = numpy.dot(self._first_deriv.T, self._first_deriv)
            
            if smoothness:
                
                Wp = Wp + smoothness*tmp
                
            if curvature:
                
                Wp = Wp + curvature*numpy.dot(tmp.T, tmp)
                            
            del tmp        
        
        if self._equality_matrix != None and equality:
            
            ref_params = numpy.dot(self._equality_matrix.T, \
                                   self._equality_values)         
            
            equality_weights = numpy.dot(self._equality_matrix.T, \
                                        self._equality_matrix)
                                
        data = self._get_data_array()
        
        Wd = numpy.identity(len(data))
                    
        self._log.info("Contaminate %d times (noise 0: original data)" \
                       % (contam_times))
                
        # Contaminate the data with Gaussian noise and run the inversion
        for contam_it in range(contam_times + 1):
            
            self._log.info("***** noise %d *****" % (contam_it))
                
            start = time.clock()            
            
            residuals = data - self._calc_adjusted_data(next)
            
            rms = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
            
            goal = rms
            
            # Part of the goal due to the different regularizations
            if damping or smoothness or curvature:
                
                goal_tk = numpy.dot(numpy.dot(next.T, Wp), next)
                
                goal += goal_tk
                
            if sharpness:
            
                derivatives = numpy.dot(self._first_deriv, next)
                            
                goal_tv = 0
                
                for deriv in derivatives:
                    
                    goal_tv += abs(deriv)
                            
                goal_tv *= sharpness
                
                goal += goal_tv      
                    
            # Part due to the equality constraints
            if self._equality_matrix != None and equality:
        
                aux = numpy.dot(self._equality_matrix, next) - \
                      self._equality_values
                      
                goal_eq = equality*numpy.dot(aux.T, aux)
                
                goal += goal_eq         
                            
            goals = [goal]
            
            lm_param = lm_start
            
            self._log.info("Starting goal function: %g" % (goal))
            self._log.info("  due to RMS: %g" % (rms))
            if damping or smoothness or curvature:
                
                self._log.info("  due to Tk reg: %g" % (goal_tk))
                
            if sharpness:
                
                self._log.info("  due to TV reg: %g" % (goal_tv))
                
            if self._equality_matrix != None and equality:
                
                self._log.info("  due to equality constraints: %g" % (goal_eq))
                
            self._log.info("Starting LM parameter: %g" % (lm_param))
            self._log.info("LM step size: %g" % (lm_step))
                                    
            for it in range(1, max_it + 1):
                                    
                inner_start = time.clock()
                
                prev = next
            
                # Hack for when mapping the goal function
                if contam_it == 0:
                    
                    if it == 1:
                        
                        steps_taken = []
                        
                    steps_taken.append(prev)
                
                jacobian = self._build_jacobian(prev)
                
                # The residuals were calculated in the previous iteration
                grad = -2*numpy.dot(numpy.dot(jacobian.T, Wd), residuals)
                    
                hessian = 2*numpy.dot(numpy.dot(jacobian.T, Wd), jacobian)
                
                if damping or smoothness or curvature:
                    
                    grad = grad + 2*numpy.dot(Wp, prev)
                    
                    hessian = hessian + 2*Wp
                    
                if sharpness:                    
                    
                    # Auxiliary for calculating the Hessian and gradient of the
                    # TV regularization
                    d = numpy.zeros(nderivs)
                    
                    D = numpy.zeros((nderivs, nderivs))
                                    
                    for l in xrange(nderivs):
                        
                        deriv = numpy.dot(self._first_deriv[l], prev)
                        
                        sqrt = math.sqrt(deriv**2 + beta)
                        
                        d[l] = deriv/sqrt
                        
                        D[l][l] = beta/(sqrt**3)
                        
                    grad = grad + sharpness*numpy.dot(self._first_deriv.T, d)
                    
                    hessian = hessian + sharpness*numpy.dot(\
                        numpy.dot(self._first_deriv.T, D), self._first_deriv)
                    
                if self._equality_matrix != None and equality:
                    
                    grad = grad + equality*2*( \
                           numpy.dot(equality_weights, prev) - ref_params)
                           
                    hessian = hessian + equality*2*equality_weights
                                       
                hessian_diag = numpy.diag(numpy.diag(hessian))
                                                       
                # LM loop
                for lm_it in range(1, max_lm_it + 1):
                    
                    N = hessian + lm_param*hessian_diag
                    
                    correction = numpy.linalg.solve(N, -1*grad)
                    
                    next = prev + correction
                                
                    residuals = data - self._calc_adjusted_data(next)
            
                    rms = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
                    
                    goal = rms
                    
                    # Part of the goal due to the different regularizations
                    if damping or smoothness or curvature:
                        
                        goal_tk = numpy.dot(numpy.dot(next.T, Wp), next)
                        
                        goal += goal_tk
                        
                    if sharpness:
                    
                        derivatives = numpy.dot(self._first_deriv, next)
                                    
                        goal_tv = 0
                        
                        for deriv in derivatives:
                            
                            goal_tv += abs(deriv)
                                    
                        goal_tv *= sharpness
                        
                        goal += goal_tv      
                            
                    # Part due to the equality constraints
                    if self._equality_matrix != None and equality:
                
                        aux = numpy.dot(self._equality_matrix, next) - \
                              self._equality_values
                              
                        goal_eq = equality*numpy.dot(aux.T, aux)
                        
                        goal += goal_eq
                                    
                    if goal < goals[it - 1]:
                        
                        goals.append(goal)
                        
                        if lm_param > 10**(-10):
                        
                            lm_param /= float(lm_step)
                        
                        break
                        
                    else:
                        
                        lm_param *= float(lm_step)
                                        
                inner_end = time.clock()
                msg = "it %d: LM its = %d  LM param = %g  goal = %g  rms = %g" \
                      % (it, lm_it, lm_param, goal, rms)
                      
                if damping or smoothness or curvature:
                    
                    msg = msg + "  Tk = %g" % (goal_tk)
                    
                if sharpness:
                    
                    msg = msg + "  TV = %g" % (goal_tv)
                    
                if self._equality_matrix != None and equality:
                    
                    msg = msg + "  Eq = %g" % (goal_eq)
                
                self._log.info(msg + "  (%g s)" % (inner_end - inner_start))
                                            
                # Got out of LM loop because of max_marq_it reached
                if len(goals) == it: 
                    
                    next = prev
                    
                    break
                
                # Stop if there is stagnation
                if abs((goals[it] - goals[it - 1])/goals[it - 1]) <= 10**(-7):
                    
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
                                   self._get_data_array(), \
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
        
        
        return steps_taken
                
    
    def plot_residuals(self, title="Residuals", bins=0):
        """
        Plot a histogram of the residuals.
        
        Parameters:
            
            - title: title of the figure
            
            - bins: number of bins (default to 10% len(residuals))
            
        Note: to view the image use pylab.show()
        """
                        
        residuals = self._get_data_array() - self._calc_adjusted_data(self.mean)
        
        if bins == 0:
        
            bins = int(0.1*len(residuals))
    
        pylab.figure()
        pylab.title(title)
        
        pylab.hist(residuals, bins=bins, facecolor='gray')
        
        pylab.xlabel("Residuals")
        pylab.ylabel("Count")    
        
    
    def plot_goal(self, title="Goal function", scale='log'):
        """
        Plot the goal function versus the iterations of the Levemberg-Marquardt
        algorithm. 
        
        scale is the scale type for the y axis. Can be either 'log' or 'linear'
        """
        
        pylab.figure()
        pylab.title(title)
        
        pylab.xlabel("LM iteration")
        pylab.ylabel("Goal")
        
        pylab.plot(self._goals, '.-k')
        
        if scale == 'log':
            
            ax = pylab.gca()
            
            ax.set_yscale('log')
            
            pylab.draw()
