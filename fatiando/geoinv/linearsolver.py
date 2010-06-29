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
import math
import time

import numpy
import pylab

from fatiando.utils import contaminate
import fatiando


logger = logging.getLogger('linearsolver')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class LinearSolver():
    """
    Mother class for linear inverse problem solvers.
    Receives a GeoData subclass instance with the data to invert.
    
    mean and standard deviation of the estimates made can be accessed via the
    mean and std class properties.
    
    Methods that MUST be implemented by the child are:
        _build_sensibility()
        _build_first_deriv()
        _get_data_array()
        _get_data_cov()
        _calc_distances(points, lines)
        _build_compact_weights(distances)
    Optional:
        some method to set the equality constraints.
            the constraints are set by setting self._equality_values to the 
            desired values and self._equality_matrix to the matrix mapping
            each parameter to its desired value.
    Suggested methods:
        set_discretization(...)
        _plot_data(folder)
        _plot_results(folder)
        some function for loading data
        some function(s) for creating synthetic models        
    """
    
    def __init__(self):
        
        # Inversion parameters
        self._sensibility = None
        self._first_deriv = None
        self._equality_values = None
        self._equality_matrix = None
                
        # Inversion results
        self._estimates = None
        self._goals = None
        
        # The logger for this class
        self._log = logging.getLogger('linearsolver')    
        
            
    def clear(self):
        """
        Erase the inversion results.
        """
        
        # Inversion parameters
        self._sensibility = None
        self._first_deriv = None
        self._equality_values = None
        self._equality_matrix = None
                
        # Inversion results
        self._estimates = None
        self._goals = None


    def _build_sensibility(self):
        """
        Return the sensibility matrix.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_sensibility was called before being implemented"
        
        
    def _build_first_deriv(self):
        """
        Return the first derivative matrix of the model parameters.
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
            
        return numpy.array(means)
    
    
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
        
            
    def plot_residuals(self, title="Residuals", bins=0):
        """
        Plot a histogram of the residuals.
        
        Parameters:
            
            - title: title of the figure
            
            - bins: number of bins (default to len(residuals)/8)
            
        Note: to view the image use pylab.show()
        """
                        
        residuals = self._get_data_array() - numpy.dot(self._sensibility, \
                                                       self.mean)
        
        if bins == 0:
        
            bins = len(residuals)/8
    
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
        
    
    def solve(self, damping=0, smoothness=0, curvature=0, param_weights=None, \
              apriori_var=1, contam_times=10):        
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
            
            param_weights: if you want to specify a parameter weight matrix. 
                This matrix will multiply the parameters in the regularizations
            
            apriori_var: the a-priori variance factor. Assumed variance of the
                         data. This will be the variance used to contaminate
                         the data.
                         
            contam_times: how many times to contaminate the data and run the 
                          inversion.                          
        """       
        
        self._log.info("*** Tikhonov Inversion ***")
        self._log.info("Regularization parameters:")
        self._log.info("  damping = %g" % (damping))
        self._log.info("  smoothness = %g" % (smoothness))
        self._log.info("  curvature = %g" % (curvature))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()
                
        self._estimates = []
        
        self._sensibility = self._build_sensibility()
        
        ndata, nparams = self._sensibility.shape
        
        start = time.clock()
        
        # The parameter weights
        Wp = numpy.zeros((nparams, nparams))
        
        if damping:
            
            Wp = Wp + damping*numpy.identity(nparams)
                
        if smoothness or curvature:
            
            self._first_deriv = self._build_first_deriv()
                    
            tmp = numpy.dot(self._first_deriv.T, self._first_deriv)
 
            if smoothness:
                       
                Wp = Wp + smoothness*tmp
                
            if curvature: 
                                                
                Wp = Wp + curvature*numpy.dot(tmp.T, tmp)
             
            del tmp
            
        if param_weights != None:
        
            Wp = numpy.dot(param_weights, Wp)
            
        if self._equality_matrix != None:
            
            Wp = Wp + numpy.dot(self._equality_matrix.T, self._equality_matrix)
        
        end = time.clock()
        self._log.info("Build parameter weight matrix (%g s)" % (end - start))
        
        # Overdetermined
        if nparams <= ndata:
            
            self._log.info("Solving overdetermined problem: %d d x %d p" % \
                           (ndata, nparams))      
              
            # Data weight matrix
            start = time.clock()
            
#            Wd = apriori_var*numpy.linalg.inv(self._get_data_cov())
            Wd = numpy.identity(ndata)
            
            end = time.clock()
            self._log.info("  Build data weight matrix (%g s)" % (end - start))          
              
            # The normal equations
            start = time.clock()
            
            aux = numpy.dot(self._sensibility.T, Wd)
            
            N = numpy.dot(aux, self._sensibility) + Wp
                          
            end = time.clock()
            self._log.info("  Build normal equations matrix (%g s)" \
                           % (end - start))  
            
            # Solve the system for the parameters
            start = time.clock()
            
            y = numpy.dot(aux, self._get_data_array())
            
            if self._equality_matrix != None:
                
                tmp_y_eq = numpy.dot(self._equality_matrix.T, \
                                      self._equality_values)
                
                y = y + tmp_y_eq
            
            estimate = numpy.linalg.solve(N, y)
            
            end = time.clock()
            self._log.info("  Solve linear system (%g s)" % (end - start))
            
            self._estimates.append(estimate)
            
            start = time.clock()
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate.gaussian(\
                                          self._get_data_array(), \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                y = numpy.dot(aux, contam_data)
                
                if self._equality_matrix != None:
                    
                    y = y + tmp_y_eq
                
                estimate = numpy.linalg.solve(N, y)
                
                self._estimates.append(estimate)
                
            end = time.clock()
            self._log.info("  Contaminate data %d times " % (contam_times) + \
                           "with Gaussian noise (%g s)" % (end - start))
                   
        # Underdetermined
        else:
            
            self._log.info("Solving underdetermined problem: %d d x %d p" % \
                           (ndata, nparams))            
            
            # Inverse of the data weight matrix
            start = time.clock()
            
            Wd_inv = self._get_data_cov()/apriori_var
            
            end = time.clock()
            self._log.info("  Inverse of data weight matrix (%g s)" \
                            % (end - start))
                        
            # The inverse of the parameter weight matrix
            start = time.clock()
            
            Wp_inv = numpy.linalg.inv(Wp)
            
            end = time.clock()
            self._log.info("  Inverse parameter weight matrix (%g s)" \
                            % (end - start))            
            
            # The normal equations
            start = time.clock()
            
            aux = numpy.dot(Wp_inv, self._sensibility.T)
            
            N = numpy.dot(self._sensibility, aux) + Wd_inv
            
            end = time.clock()
            self._log.info("  Build normal equations matrix (%g s)" \
                            % (end - start))

            start = time.clock()
            
            y = self._get_data_array()
            
            if self._equality_matrix != None:
                
                tmp_p_eq = numpy.dot(\
                            numpy.dot(Wp_inv, self._equality_matrix.T), \
                            self._equality_values)
                
                tmp_y_eq = numpy.dot(self._sensibility, tmp_p_eq)
                
                y = y + tmp_y_eq
                
            lamb = numpy.linalg.solve(N, y)
            
            end = time.clock()
            self._log.info("  Solve for Lagrange multipliers (%g s)" \
                           % (end - start))
            
            start = time.clock()
            
            estimate = numpy.dot(aux, lamb)
            
            if self._equality_matrix != None:
                
                estimate = estimate + tmp_p_eq
            
            self._estimates.append(estimate)
            
            end = time.clock()
            self._log.info("  Calculate the estimate (%g s)" \
                           % (end - start))
            
            start = time.clock()
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate.gaussian( \
                                          self._get_data_array(), \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                y = contam_data
            
                if self._equality_matrix != None:
                    
                    y = y + tmp_y_eq
                
                lamb = numpy.linalg.solve(N, y)
                    
                estimate = numpy.dot(aux, lamb)
            
                if self._equality_matrix != None:
                
                    estimate = estimate + tmp_p_eq
                
                self._estimates.append(estimate)
                
            end = time.clock()
            self._log.info("  Contaminate data %d times " % (contam_times) + \
                           "with Gaussian noise (%g s)" % (end - start))
                               
        residuals = self._get_data_array() - numpy.dot(self._sensibility, \
                                                       self.mean)
        
        rms = numpy.dot(residuals.T, residuals)
                
        self._log.info("RMS = %g" % (rms))
            
        if damping:
                
            goal_damp = numpy.dot(numpy.transpose(self.mean), self.mean)
                
            self._log.info("Tikhonov 0 goal = %g  (with damping coef = %g)" \
                           % (goal_damp, damping*goal_damp))
                
        if smoothness:
            
            tmp = numpy.dot(self._first_deriv, self.mean)
            
            goal_smooth = numpy.dot(tmp.T, tmp)
        
            self._log.info("Tikhonov 1 goal = %g  (with smoothness coef = %g)" \
                           % (goal_smooth, smoothness*goal_smooth))
            
        if curvature:
                
            tmp = numpy.dot(self._first_deriv, self.mean)
                
            tmp1 = numpy.dot(self._first_deriv.T, tmp)
            
            goal_curv = numpy.dot(tmp1.T, tmp1)
        
            self._log.info("Tikhonov 2 goal = %g  (with curvature coef = %g)" \
                           % (goal_curv, curvature*goal_curv))
            
        total_end = time.clock()
        self._log.info("Total time: %g s" % (total_end - total_start))
        
    
    
    def sharpen(self, sharpness=1, damping=0, initial_estimate=None, \
        param_weights=None, beta=10**(-7), apriori_var=1, contam_times=10, \
        max_it=100, max_marq_it=10, marq_start=1, marq_step=10):
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
        
            sharpness: the regularization parameter (how much to sharpen)
            
            damping: 0th order regularization parameter (how much damping to 
                     apply)
                     
            initial_estimate: see Note above
            
            param_weights: if you want to specify a parameter weight matrix. 
                This matrix will multiply the parameters in the regularizations
            
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
        
        self._log.info("*** Total Variation Inversion ***")
        self._log.info("Regularization parameters:")
        self._log.info("  sharpness = %g" % (sharpness))
        self._log.info("  damping = %g" % (damping))
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()    
        
        # Clear the estimates
        self._estimates = []           
        
        # Make the sensibility and first derivative if they are missing
        if self._sensibility == None:
            
            self._sensibility = self._build_sensibility()
            
        if self._first_deriv == None:
            
            self._first_deriv = self._build_first_deriv()
            
        ndata, nparams = self._sensibility.shape
        
        # Set the values for the first estimate in case there was none
        if initial_estimate == None:
            
            self._log.info("Starting estimate with (almost) null vector")
            
            next = 10**(-7)*numpy.ones(nparams)
                
        else:               
            
            next = initial_estimate
                
        nderivs = len(self._first_deriv)
                
        # The data weight matrix
#        Wd = apriori_var*numpy.linalg.inv(self._get_data_cov())
        Wd = numpy.identity(ndata)
                
        # The Hessian due to the residuals and damping
        start = time.clock()
        
        aux_AT_Wd = numpy.dot(self._sensibility.T, Wd)
        
        hessian_res = 2*numpy.dot(aux_AT_Wd, self._sensibility)
        
        if damping:
            
            if param_weights == None:
            
                hessian_res = hessian_res + 2*damping*numpy.identity(nparams)
                
            else:
                
                hessian_res = hessian_res + 2*damping*param_weights
                
        if self._equality_matrix != None:
            
            ref_params = numpy.dot(self._equality_matrix.T, \
                                   self._equality_values)         
            
            param_equality_weights = numpy.dot(self._equality_matrix.T, \
                                               self._equality_matrix)
            
            hessian_res = hessian_res + 2*param_equality_weights
        
        end = time.clock()
        self._log.info("Calculate misfit and damping portion of the Hessian" + \
                       " (%g s)" % (end - start))
                
        data = self._get_data_array()
                    
        self._log.info("Contaminate %d times (noise 0: original data)" \
                       % (contam_times))
                
        # Contaminate the data with Gaussian noite and run the inversion
        for contam_it in range(contam_times + 1):
            
            self._log.info("***** noise %d *****" % (contam_it))
                
            start = time.clock()
            
            residuals = data - numpy.dot(self._sensibility, next)
            
            # Part of the goal function due to the residuals
            rms = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
            
            # Part due to the Total Variation               
            if param_weights == None:
                
                derivatives = numpy.dot(self._first_deriv, next)
                
            else:
                
                aux_next = numpy.dot(param_weights, next)
                
                derivatives = numpy.dot(self._first_deriv, aux_next)
            
            goal_tv = 0
            
            for deriv in derivatives:
                
                goal_tv += abs(deriv)
                
            goal = rms + sharpness*goal_tv
            
            # Part due to Tikhonov 0 order (damping)
            if damping:        
                
                if param_weights == None:
                    
                    goal += damping*numpy.dot(next.T, next)
                    
                else:
                            
                    goal += damping*numpy.dot(next.T, aux_next)
                    
            # Part due to the equality constraints
            if self._equality_matrix != None:
                
                aux = numpy.dot(self._equality_matrix, next) - \
                      self._equality_values
                      
                goal += numpy.dot(aux.T, aux)
                        
            goals = [goal]
            
            marq_param = marq_start
            
            self._log.info("Starting RMS: %g" % (rms))
            self._log.info("Starting goal function: %g" % (goal))
            self._log.info("Starting TV goal function: %g" % (goal_tv))
            self._log.info("Starting LM parameter: %g" % (marq_param))
            self._log.info("LM step size: %g" % (marq_step))
                                    
            for it in range(1, max_it + 1):
                                    
                inner_start = time.clock()
                
                prev = next
                
                misfit = data - numpy.dot(self._sensibility, prev)
                                
                # Auxiliary for calculating the Hessian and gradient
                d = numpy.zeros(nderivs)
                D = numpy.zeros((nderivs, nderivs))
                                
                for l in range(len(self._first_deriv)):
                    
                    deriv = numpy.dot(self._first_deriv[l], prev)
                    
                    sqrt = math.sqrt(deriv**2 + beta)
                    
                    d[l] = deriv/sqrt
                    
                    D[l][l] = beta/(sqrt**3)
                    
                grad_tv = sharpness*numpy.dot(self._first_deriv.T, d)
                
                hessian_tv = numpy.dot(\
                        numpy.dot(self._first_deriv.T, \
                                  sharpness*D + numpy.identity(nderivs)), \
                        self._first_deriv)
                    
                if param_weights != None:
                    
                    grad_tv = numpy.dot(param_weights, grad_tv)
                    
                    hessian_tv = numpy.dot(param_weights, hessian_tv)
                                    
                grad = -2*numpy.dot(aux_AT_Wd, misfit) + grad_tv
                
                hessian = hessian_res + hessian_tv                        
                
                # Add the damping part of the gradient
                if damping:
                    
                    if param_weights == None:
                    
                        grad = grad + 2*damping*prev
                        
                    else:
                        
                        grad = grad + 2*damping*numpy.dot(param_weights, prev)
                        
                # Add the equality constraint part of the gradient
                if self._equality_matrix != None:
                    
                    grad = grad + 2*numpy.dot(param_equality_weights, prev) - \
                           2*ref_params
                                       
                hessian_diag = numpy.diag(numpy.diag(hessian))
                                                       
                # LM loop
                for marq_it in range(1, max_marq_it + 1):
                    
                    N = hessian + marq_param*hessian_diag
                    
                    correction = numpy.linalg.solve(N, -1*grad)
                    
                    next = prev + correction
                                
                    residuals = data - numpy.dot(self._sensibility, next)
                    
                    # Part of the goal function due to the residuals
                    rms = numpy.dot(numpy.dot(residuals.T, Wd), residuals)
                    
                    # Part due to the Total Variation
                    if param_weights == None:
                        
                        derivatives = numpy.dot(self._first_deriv, next)
                        
                    else:
                        
                        aux_next = numpy.dot(param_weights, next)
                        
                        derivatives = numpy.dot(self._first_deriv, aux_next)
                    
                    goal_tv = 0
                    
                    for deriv in derivatives:
                        
                        goal_tv += abs(deriv)
                        
                    goal = rms + sharpness*goal_tv
                    
                    # Part due to Tikhonov 0 order (damping)
                    if damping:
                        
                        if param_weights == None:
                            
                            goal += damping*numpy.dot(next.T, next)
                            
                        else:
                            
                            goal += damping*numpy.dot(next.T, aux_next)
                            
                    
                    # Part due to the equality constraints
                    if self._equality_matrix != None:
                
                        aux = numpy.dot(self._equality_matrix, next) - \
                              self._equality_values
                              
                        goal += numpy.dot(aux.T, aux)
                                            
                    # Accept the step if it's going down or reject and try again
                    if goal < goals[it - 1] and marq_param >= 10**(-6):
                        
                        goals.append(goal)
                        
                        marq_param /= float(marq_step)
                        
                        break
                        
                    else:
                        
                        marq_param *= float(marq_step)
                        
                
                inner_end = time.clock()
                self._log.info("it %d: LM its = %d  LM param = %g  RMS = %g" \
                               % (it, marq_it, marq_param, rms) + \
                               "  goal_tv = %g  goal = %g  (%g s)" \
                                % (goal_tv, goal, inner_end - inner_start))
                                            
                # Got out of LM loop because of max_marq_it reached
                if len(goals) == it: 
                    
                    next = prev

                    break
                
                # Stop if there is stagnation
                if abs((goals[it] - goals[it - 1])/goals[it - 1]) <= 10**(-10):
                    
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
                    
                    next = 10**(-7)*numpy.ones(nparams)
                        
                else:               
                    
                    next = initial_estimate
                               
        total_end = time.clock()
        self._log.info("Total time for inversion: %g s" \
                       % (total_end - total_start))   
                
            