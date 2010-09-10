# Copyright 2010 Leonardo Uieda
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
Linear and non-linear solvers for inverse problems.
Includes:
    - linear solver
    - Levemberg-Marquardt solver
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 10-Sep-2010'


import logging

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

# These globals are things that only need to be calculated once per inversion
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

    
def _calc_tk_goal(estimate, msg):
    """Portion of the goal function due to Tikhonov regularization"""
    
    if _tk_weights is None:
        
        _tk_weights = _build_tk_weights(len(estimate))
        
    # No need to multiply by the regularization parameters because they are
    # already in the _tk_weights
    goal = (numpy.dot(estimate.T, _tk_weights)*estimate).sum()
    
    msg.join(" TK=%g" % (goal))
    
    return goal
    


def _calc_tv_goal(estimate, msg):
    """Portion of the goal function due to Total Variation regularization"""

    if _first_deriv is None:
        
        _first_deriv = _build_first_deriv_matrix()

    derivatives = numpy.dot(_first_deriv, estimate)
    
    goal = sharpness*abs(derivatives).sum()
    
    msg.join(" TV=%g" % (goal))
    
    return goal


def _calc_compact_goal(estimate, msg):    
    """Portion of the goal function due to Compact regularization"""
        
    estimate_sqr = estimate**2
    
    goal = compactness*(estimate_sqr/(estimate_sqr + epsilon)).sum()    
    
    msg.join(" CP=%g" % (goal))
    
    return goal


def _calc_eq_goal(estimate, msg):
    """Portion of the goal function due to equality constraints"""
    pass

    
def _calc_regularizer_goal(estimate, msg):
    """
    Calculate the portion of the goal function due to the regularizers
    
    Parameters:
        
      estimate: array-like current estimate
      
      msg: string that will be printed in the current iteration verbose.
           use it to inform the value of each regularizer 
    """
    
    goal = 0
    goal += _calc_tk_goal(estimate, msg)
    goal += _calc_tv_goal(estimate, msg)
    goal += _calc_compact_goal(estimate, msg)
#    goal += _calc_eq_goal(estimate, msg)
    
    return goal
    
    
def _sum_tk_hessian(hessian):
    """
    Sum the Tikhonov regularization Hessian to hessian.
    
    Parameters:
    
      hessian: array-like Hessian due to the adjustment
    """
    
    if _tk_weights is None:
        
        _tk_weights = _build_tk_weights(len(hessian))
    
    hessian += _tk_weights    
    

def _sum_tv_hessian(hessian):
    pass

def _sum_compact_hessian(hessian):
    pass

def _sum_eq_hessian(hessian):
    pass
    
    
def _sum_reg_hessians(hessian):
    """
    Sum the Hessians of the regularizers to the given Hessian.
    
    Parameters:
    
      hessian: array-like Hessian due to the adjustment
    """
    
    _sum_tk_hessian(hessian)
    _sum_tv_hessian(hessian)
    _sum_compact_hessian(hessian)
    _sum_eq_hessian(hessian)
    
    
def _sum_tk_gradient(gradient):
    pass

def _sum_tv_gradient(gradient):
    pass

def _sum_compact_gradient(gradient):
    pass

def _sum_eq_gradient(gradient):
    pass
    
    
def _sum_reg_gradients(gradient):
    """
    Sum the gradients of the regularizers to the given gradient.
    
    Parameters:
    
      gradient: array-like gradient due to the adjustment
    """
    
    _sum_tk_gradient(gradient)
    _sum_tv_gradient(gradient)
    _sum_compact_gradient(gradient)
    _sum_eq_gradient(gradient)
    
    
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
    
    log.info("Levemberg-Marquardt Inversion:")

    next = initial
    
    residuals = data - _calc_adjustment(next)

    rms = (residuals*residuals).sum()
    
    msg = ''
    
    reg_goal = _calc_regularizer_goal(next, msg)
    
    goals = [rms + reg_goal]
    
    log.info("  Initial RMS: %g" % (rms))
    log.info("  Initial regularizers: %s" % (msg))
    log.into("  Total initial goal function: %g" % (goals[0]))
    log.info("  Initial LM param: %g" % (lm_start))
    log.info("  LM param step: %g" % (lm_step))
    
    lm_param = lm_start
    
    for iteration in xrange(max_it):
        
        prev = next
        
        jacobian = _build_jacobian(prev)
        
        gradient = -1*numpy.dot(jacobian.T, residuals)
        
        _sum_reg_gradients(gradient)
        
        hessian = numpy.dot(jacobian.T, jacobian)
    
        _sum_reg_hessians(hessian)
        
        hessian_diag = numpy.diag(numpy.diag(hessian))
        
        stagnation = True
        
        for lm_iteration in xrange(max_steps):
            
            system = hessian + lm_param*hessian_diag
            
            correction = numpy.linalg.solve(system, -1*gradient)
            
            next = prev + correction
            
            residuals = data - _calc_adjustment(next)
            
            rms = (residuals*residuals).sum()
            
            msg = ''
            
            reg_goal = _calc_regularizer_goal(next, msg)
            
            goal = rms + reg_goal
            
            if goal < goals[-1]:
                
                stagnation = False
                
                break
            
        if stagnation:
            
            next = prev
            
            log.warning("  Exited because couldn't take a step")
            
            break
        
        goals.append(goal)
        
        log.info("  it %d: RMS=%g %s" % (iteration + 1, rms, msg))
        
        if abs((goals[-1] - goals[-2])/goals[-2]) <= 10**(-4):
            
            break
        
    return next, goals


if __name__ == '__main__':
    pass