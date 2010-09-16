# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Linear and non-linear generic solvers for inverse problems.

Implemented regularizations:

  * Tikhonov orders 0, 1 and 2: imposes minimum norm (damping), smoothness and 
      minimum curvature, respectively, of the solution
  * Total Variation: imposes minimum l1 norm of the model derivatives 
      (discontinuities)
  * Compact: imposes minimum area (volume) of the solution (as in Last and Kubic
      (1983))

Functions:
  * lm: Levemberg-Marquardt solver
  * set_bounds: Set upper and lower bounds on the parameter values
  * clear: Reset all globals to their default
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 10-Sep-2010'


import logging
import time

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.solvers')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


# The regularization parameters are global so that they can be set by the caller
# module and then the solver doesn't have to know them
damping = 0
smoothness = 0
curvature = 0
sharpness = 0
beta = 10**(-5)
compactness = 0
epsilon = 10**(-5)
equality = 0

# Need access the lower and upper bounds on the parameters
_lower = None
_upper = None

# These globals are things that only need to be calculated once per inversion
_tk_weights = None
_first_deriv = None


def clear():
    """
    Reset all globals to their default.
    """
    
    global damping, smoothness, curvature, \
           sharpness, beta, compactness, epsilon, \
           equality
    global _tk_weights, _first_deriv
           
    damping = 0
    smoothness = 0
    curvature = 0
    sharpness = 0
    beta = 10**(-5)
    compactness = 0
    epsilon = 10**(-5)
    equality = 0
    _tk_weights = None
    _first_deriv = None
    

def _build_jacobian(estimate):
    """
    Build the Jacobian matrix of the mathematical model (geophysical function)
    that we're trying to fit to the data.
    
    Parameters:
        
      estimate: array-like current estimate where the Jacobian will be
                evaluated                      
    """
    
    raise NotImplementedError(
          "_build_jacobian was called before being implemented")


def _build_first_deriv_matrix():
    """
    Build the finite differences approximation of the first derivative matrix 
    of the model parameters. 
    """
    
    raise NotImplementedError(
          "_build_first_deriv_matrix was called before being implemented")
    

def _calc_adjustment(estimate):
    """
    Calculate the adjusted data produced by a given estimate.
    
    Parameters:
        
      estimate: array-like current estimate
    """
    
    raise NotImplementedError(
          "_calc_adjustment was called before being implemented")
    
    
def _build_tk_weights(nparams):
    """
    Build the parameter weight matrix of Tikhonov regularization
    
    Parameters:
    
      nparams: number of parameters
    """
    
    global _first_deriv
        
    weights = numpy.zeros((nparams, nparams))
    
    if damping > 0:
        
        for i in xrange(nparams):
        
            weights[i][i] += damping
            
    if smoothness > 0:
        
        if _first_deriv is None:
            
            _first_deriv = _build_first_deriv_matrix()
            
        tmp = numpy.dot(_first_deriv.T, _first_deriv)            
            
        weights += smoothness*tmp
    
    if curvature > 0:
                    
        if _first_deriv is None:
            
            _first_deriv = _build_first_deriv_matrix()
            
        if smoothness == 0:
            
            tmp = numpy.dot(_first_deriv.T, _first_deriv)
                            
        tmp = numpy.dot(tmp.T, tmp)
            
        weights += curvature*tmp
        
    return weights

    
def _calc_tk_goal(estimate):
    """Portion of the goal function due to Tikhonov regularization"""
    
    global _tk_weights
    
    if _tk_weights is None:
        
        _tk_weights = _build_tk_weights(len(estimate))
        
    # No need to multiply by the regularization parameters because they are
    # already in the _tk_weights
    goal = (numpy.dot(estimate.T, _tk_weights)*estimate).sum()
    
    msg = "TK=%g" % (goal)
    
    return goal, msg
    


def _calc_tv_goal(estimate):
    """Portion of the goal function due to Total Variation regularization"""
    
    global _first_deriv

    if _first_deriv is None:
        
        _first_deriv = _build_first_deriv_matrix()

    derivatives = numpy.dot(_first_deriv, estimate)
    
    goal = sharpness*abs(derivatives).sum()
    
    msg = "TV=%g" % (goal)
    
    return goal, msg


def _calc_compact_goal(estimate):    
    """Portion of the goal function due to Compact regularization"""
        
    estimate_sqr = estimate**2
    
    goal = compactness*(estimate_sqr/(estimate_sqr + epsilon)).sum()    
    
    msg = "CP=%g" % (goal)
    
    return goal, msg


def _calc_eq_goal(estimate):
    """Portion of the goal function due to equality constraints"""
    
    raise NotImplementedError(
          "_calc_eq_goal was called before being implemented")

    
def _calc_regularizer_goal(estimate):
    """
    Calculate the portion of the goal function due to the regularizers
    
    Parameters:
        
      estimate: array-like current estimate
    """
    
    goal = 0
    
    msg = ''
    
    if damping > 0 or smoothness > 0 or curvature > 0:
        
        reg_goal, reg_msg = _calc_tk_goal(estimate)
        
        goal += reg_goal
        
        msg = ' '.join([msg, reg_msg])
        
    if sharpness > 0:
        
        reg_goal, reg_msg = _calc_tv_goal(estimate)
        
        goal += reg_goal
        
        msg = ' '.join([msg, reg_msg])
        
    if compactness > 0:
        
        reg_goal, reg_msg = _calc_compact_goal(estimate)
        
        goal += reg_goal
        
        msg = ' '.join([msg, reg_msg])
        
    if equality > 0:
        
        reg_goal, reg_msg = _calc_eq_goal(estimate)
        
        goal += reg_goal
        
        msg = ' '.join([msg, reg_msg])
    
    return goal, msg
    
    
def _sum_tk_hessian(hessian):
    """
    Sum the Tikhonov regularization Hessian to hessian.
    
    Parameters:
    
      hessian: array-like Hessian matrix
    """    
    
    global _tk_weights
    
    if _tk_weights is None:
        
        _tk_weights = _build_tk_weights(len(hessian))
    
    hessian += _tk_weights    
    

def _sum_tv_hessian(hessian, estimate):
    """
    Sum the Total Variation regularization Hessian to hessian.
    
    Parameters:
    
      hessian: array-like Hessian matrix
        
      estimate: array-like current estimate
    """
        
    global _first_deriv
    
    if _first_deriv is None:
        
        _first_deriv = _build_first_deriv_matrix()

    derivatives = numpy.dot(_first_deriv, estimate)
    
    tmp = _first_deriv.copy()
    
    for i, deriv in enumerate(derivatives):
        
        sqrt = numpy.sqrt(deriv**2 + beta)
                
        tmp[i] *= beta/(sqrt**3)
                    
    hessian += sharpness*numpy.dot(_first_deriv.T, tmp)
        

def _sum_compact_hessian(hessian, estimate):
    """
    Sum the Compact regularization Hessian to hessian.
    
    Parameters:
    
      hessian: array-like Hessian matrix
        
      estimate: array-like current estimate
    """
    
    for i, param in enumerate(estimate):
        
        hessian[i][i] += compactness/(param**2 + epsilon)


def _sum_eq_hessian(hessian):
    """
    Sum the Equality Constraints Hessian to hessian.
    
    Parameters:
    
      hessian: array-like Hessian matrix
        
      estimate: array-like current estimate
    """
       
    raise NotImplementedError(
          "_sum_eq_hessian was called before being implemented")
    
    
def _sum_reg_hessians(hessian, estimate):
    """
    Sum the Hessians of the regularizers to the Hessian of the adjustment.
    
    Parameters:
    
      hessian: array-like Hessian matrix of the adjustment
        
      estimate: array-like current estimate
    """
    
    if damping > 0 or smoothness > 0 or curvature > 0:
        
        _sum_tk_hessian(hessian)
        
    if sharpness > 0:
        
        _sum_tv_hessian(hessian, estimate)
        
    if compactness > 0:
        
        _sum_compact_hessian(hessian, estimate)
        
    if equality > 0:
        
        _sum_eq_hessian(hessian)
    
    
def _sum_tk_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Tikhonov regularizers to gradient
    
    Parameters:
    
      gradient: array-like gradient vector
        
      estimate: array-like current estimate
    """
        
    global _tk_weights
    
    if _tk_weights is None:
        
        _tk_weights = _build_tk_weights(len(estimate))    
    
    gradient += numpy.dot(_tk_weights, estimate)


def _sum_tv_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Total Variation regularizer to gradient
    
    Parameters:
    
      gradient: array-like gradient vector
        
      estimate: array-like current estimate
    """
        
    global _first_deriv
    
    if _first_deriv is None:
        
        _first_deriv = _build_first_deriv_matrix()
        
    derivatives = numpy.dot(_first_deriv, estimate)
        
    d = derivatives/numpy.sqrt(derivatives**2 + beta)
    
    gradient += sharpness*numpy.dot(_first_deriv.T, d)
    

def _sum_compact_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Compact regularizer to gradient
    
    Parameters:
    
      gradient: array-like gradient vector
        
      estimate: array-like current estimate
    """
    
    gradient += compactness*estimate/(estimate**2 + epsilon)


def _sum_eq_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Equality Constraints to gradient
    
    Parameters:
    
      gradient: array-like gradient vector
        
      estimate: array-like current estimate
    """
       
    raise NotImplementedError(
          "_sum_eq_gradient was called before being implemented")
    
    
def _sum_reg_gradients(gradient, estimate):
    """
    Sum the gradients of the regularizers to the gradient of the adjustment.
    
    Parameters:
    
      gradient: array-like gradient due to the adjustment
        
      estimate: array-like current estimate
    """
    
    if damping > 0 or smoothness > 0 or curvature > 0:
        
        _sum_tk_gradient(gradient, estimate)
        
    if sharpness > 0:
        
        _sum_tv_gradient(gradient, estimate)
        
    if compactness > 0:
        
        _sum_compact_gradient(gradient, estimate)
        
    if equality > 0:
        
        _sum_eq_gradient(gradient, estimate)
        
        
def _apply_variable_change(system, y, prev):
    """Apply a variable change of the parameters to the equation system"""
    pass


def _revert_variable_change(correction, prev):
    """Change back the estimated correction"""
    
    return correction
    

def _apply_log_barrier(hessian, y, prev):
    """Apply a log barrier to the equation system"""
    
    jacobian_diag = (prev - _lower + 10**(-8))*(_upper - prev + 10**(-8))/ \
                    float(_upper - _lower)
                    
    y *= jacobian_diag
    
    for i, row in enumerate(hessian):
        
        row *= jacobian_diag[i]*jacobian_diag
        
    
def _revert_log_barrier(correction, prev):
    """Revert the log barrier from the correction"""
    
    changed = -1*numpy.log((_upper - prev)/(prev - _lower))
    
    delta = float(_upper - _lower)
    
    corr_reverted = delta/(1. + numpy.exp(-1*(changed + correction))) - \
                    delta/(1. + numpy.exp(-1*changed))
    
    return corr_reverted


def set_bounds(lower, upper):
    """Set upper and lower bounds on the parameter values."""
    
    global _lower, _upper, _apply_variable_change, _revert_variable_change
    
    _lower = lower
    _upper = upper
    
    _apply_variable_change = _apply_log_barrier
    _revert_variable_change = _revert_log_barrier
    
    
    
def lm(data, cov, initial, lm_start=100, lm_step=10, max_steps=20, max_it=100):
    """
    Solve using the Levemberg-Marquardt algorithm.
    
    Parameters:
    
        data: array-like data vector
        
        cov: array-like covariance matrix of the data
        
        initial: array-like initial estimate
        
        lm_start: initial Marquardt parameter (controls the step size)
        
        lm_step: factor by which the Marquardt parameter will be reduced with
                 each successful step
                 
        max_steps: how many times to try giving a step before exiting
        
        max_it: maximum number of iterations 
    """
    
    total_start = time.time()
    
    log.info("Levemberg-Marquardt Inversion:")

    next = initial
    
    residuals = data - _calc_adjustment(next)

    rms = (residuals*residuals).sum()
        
    reg_goal, msg = _calc_regularizer_goal(next)
    
    goals = [rms + reg_goal]
    
    log.info("  Initial RMS: %g" % (rms))
    log.info("  Initial regularizers:%s" % (msg))
    log.info("  Total initial goal function: %g" % (goals[0]))
    log.info("  Initial Marquardt parameter: %g" % (lm_start))
    log.info("  Marquardt parameter step: %g" % (lm_step))
    
    lm_param = lm_start
    
    for iteration in xrange(max_it):
                
        it_start = time.time()
        
        prev = next
        
        jacobian = _build_jacobian(prev)
        
        gradient = -1*numpy.dot(jacobian.T, residuals)
        
        _sum_reg_gradients(gradient, prev)
        
        hessian = numpy.dot(jacobian.T, jacobian)
    
        _sum_reg_hessians(hessian, prev)
            
        gradient *= -1
            
        _apply_variable_change(hessian, gradient, prev)
        
        hessian_diag = numpy.diag(numpy.diag(hessian))
        
        stagnation = True
        
        for lm_iteration in xrange(max_steps):
            
            system = hessian + lm_param*hessian_diag
            
            correction = numpy.linalg.solve(system, gradient)
            
            correction = _revert_variable_change(correction, prev)
            
            next = prev + correction
            
            residuals = data - _calc_adjustment(next)
            
            rms = (residuals*residuals).sum()
            
            msg = ''
            
            reg_goal, msg = _calc_regularizer_goal(next)
            
            goal = rms + reg_goal
            
            if goal < goals[-1]:
                
                if lm_param > 10**(-10):
                
                    lm_param /= lm_step
                
                stagnation = False
                
                break
            
            else:
                
                if lm_param < 10**(10):
                
                    lm_param *= lm_step
            
        if stagnation:
            
            next = prev
            
            log.warning("  Exited because couldn't take a step")
            
            break
        
        goals.append(goal)
        
        it_finish = time.time()
        
        log.info("  it %d: RMS=%g%s TOTAL=%g (%.3lf s)" % 
                 (iteration + 1, rms, msg, goal, it_finish - it_start))
        
        if abs((goals[-1] - goals[-2])/goals[-2]) <= 10**(-4):
            
            break
        
    total_finish = time.time()
    
    log.info("  Total time: %g s" % (total_finish - total_start))
        
    return next, goals