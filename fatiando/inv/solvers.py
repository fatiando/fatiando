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

* Tikhonov orders 0, 1 and 2
    Imposes minimum norm (damping), smoothness and minimum curvature,
    respectively, on the solution.

* Total Variation
    Imposes minimum l1 norm of the model derivatives (ie, discontinuities)

* Compact
    Imposes minimum area (or volume) of the solution (as in Last and Kubic
    (1983) without parameter freezing. To achieve the same effect, use
    the specific ``set_bounds`` functions instead)

Functions:

* :func:`fatiando.inv.solvers.clear`
    Reset all globals to their default.

* :func:`fatiando.inv.solvers.lm`
    Levemberg-Marquardt solver

* :func:`fatiando.inv.solvers.linear_overdet`
    Solve a linear over-determined problem.

* :func:`fatiando.inv.solvers.linear_underdet`
    Solve a linear under-determined problem

* :func:`fatiando.inv.solvers.set_bounds`
    Set upper and lower bounds on the parameter values (log barrier)

* :func:`fatiando.inv.solvers.clear`
    Reset all globals to their default

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 10-Sep-2010'


import logging
import time
import math

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.inv.solvers')
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

# These globals are things that only need to be calculated once per inversion:
# Tikhonov regularization weight matrix combining all orders
_tk_weights = None
# First derivative (finite differences) matrix of the parameters
_first_deriv = None
# Equality constraints matrix and reference parameter values
_eq_matrix = None
_eq_ref_values = None
_eq_hessian = None


def clear():
    """
    Reset all globals to their default.
    """

    global damping, smoothness, curvature, \
        sharpness, beta, compactness, epsilon, \
        equality
    global _tk_weights, _first_deriv, \
        _eq_matrix, _eq_ref_values, _eq_hessian

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
    _eq_matrix = None
    _eq_ref_values = None
    _eq_hessian = None


def _build_jacobian(estimate):
    """
    Build the Jacobian matrix of the mathematical model (geophysical function)
    that we're trying to fit to the data.

    Overwrite this function in the specific solvers.

    If called as is, will raise a ``NotImplementedError``.

    Parameters:

    * estimate
        1D array-like current estimate where the Jacobian will be evaluated

    Returns:

    * jacobian
        2D array-like Jacobian matrix

    """

    raise NotImplementedError(
        "_build_jacobian was called before being implemented")


def _build_first_deriv_matrix():
    """
    Build the finite differences approximation of the first derivative matrix
    of the model parameters.

    Overwrite this function in the specific solvers.

    If called as is, will raise a ``NotImplementedError``.

    Returns:

    * first_deriv
        2D array-like finite differences first derivative of model parameters

    """

    raise NotImplementedError(
        "_build_first_deriv_matrix was called before being implemented")


def _calc_adjustment(estimate):
    """
    Calculate the adjusted data produced by a given estimate.

    Overwrite this function in the specific solvers.

    If called as is, will raise a ``NotImplementedError``.

    Parameters:

    * estimate
        1D array-like current estimate

    Returns:

    * adjusted_data
        1D array-like adjusted data vector

    """

    raise NotImplementedError(
        "_calc_adjustment was called before being implemented")


def _build_tk_weights(nparams):
    """
    Build the parameter weight matrix of Tikhonov regularization

    Parameters:

    * nparams
        Number of parameters

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

    * estimate
        1D array-like current estimate

    Returns:

    * [goal, msg]
        *msg* is a string with the individual regularizers information

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

    * hessian
        2D array-like Hessian matrix

    """

    global _tk_weights

    if _tk_weights is None:

        _tk_weights = _build_tk_weights(len(hessian))

    hessian += _tk_weights


def _sum_tv_hessian(hessian, estimate):
    """
    Sum the Total Variation regularization Hessian to hessian.

    Parameters:

    * hessian
        2D array-like Hessian matrix

    * estimate
        1D array-like current estimate

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

    * hessian
        2D array-like Hessian matrix

    * estimate
        1D array-like current estimate

    """

    for i, param in enumerate(estimate):

        hessian[i][i] += compactness/(param**2 + epsilon)


def _sum_eq_hessian(hessian):
    """
    Sum the Equality Constraints Hessian to hessian.

    Parameters:

    * hessian
        2D array-like Hessian matrix

    * estimate
        1D array-like current estimate

    """

    assert equality > 0, "'equality' regularization parameter needs to be > 0"

    assert _eq_matrix is not None and _eq_ref_values is not None, \
        "Tried to use equality constraints before setting them."

    global _eq_hessian

    if _eq_hessian is None:

        _eq_hessian = equality*numpy.dot(_eq_matrix.T, _eq_matrix)

    hessian += _eq_hessian


def _sum_reg_hessians(hessian, estimate):
    """
    Sum the Hessians of the regularizers to the Hessian of the adjustment.

    Parameters:

    * hessian
        2D array-like Hessian matrix

    * estimate
        1D array-like current estimate

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

    * gradient
        1D array-like gradient vector

    * estimate
        1D array-like current estimate

    """

    global _tk_weights

    if _tk_weights is None:

        _tk_weights = _build_tk_weights(len(estimate))

    if estimate is not None:

        gradient += numpy.dot(_tk_weights, estimate)


def _sum_tv_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Total Variation regularizer to gradient

    Parameters:

    * gradient
        1D array-like gradient vector

    * estimate
        1D array-like current estimate

    """

    global _first_deriv, beta, sharpness

    if _first_deriv is None:

        _first_deriv = _build_first_deriv_matrix()

    derivatives = numpy.dot(_first_deriv, estimate)

    d = derivatives/numpy.sqrt(derivatives**2 + beta)

    gradient += sharpness*numpy.dot(_first_deriv.T, d)


def _sum_compact_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Compact regularizer to gradient

    Parameters:

    * gradient
        1D array-like gradient vector

    * estimate
        1D array-like current estimate

    """

    gradient += compactness*estimate/(estimate**2 + epsilon)


def _sum_eq_gradient(gradient, estimate):
    """
    Sum the gradient vector of the Equality Constraints to gradient

    Parameters:

    * gradient
        1D array-like gradient vector

    * estimate
        1D array-like current estimate

    """

    assert equality > 0, "'equality' regularization parameter needs to be > 0"

    assert _eq_matrix is not None and _eq_ref_values is not None, \
        "Tried to use equality constraints before setting them."

    if estimate is None:

        gradient -= equality*numpy.dot(_eq_matrix.T, _eq_ref_values)

    else:

        eq_residuals = _eq_ref_values - numpy.dot(_eq_matrix, estimate)

        gradient -= equality*numpy.dot(_eq_matrix.T, eq_residuals)


def _sum_reg_gradients(gradient, estimate):
    """
    Sum the gradients of the regularizers to the gradient of the adjustment.

    Parameters:

    * gradient
        1D array-like gradient vector

    * estimate
        1D array-like current estimate

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

    # Check for Infs and NaNs caused by prev being too close to _upper or _lower
    for i in xrange(len(corr_reverted)):

        if math.isinf(corr_reverted[i]) or math.isnan(corr_reverted[i]):

            corr_reverted[i] = 0

    return corr_reverted


def set_bounds(vmin, vmax):
    """
    Set bounds on the parameter values.

    Parameters:

    * vmin
        Lowest value the parameter can assume

    * vmax
        Highest value the parameter can assume

    """

    global _lower, _upper, _apply_variable_change, _revert_variable_change

    _lower = vmin
    _upper = vmax

    _apply_variable_change = _apply_log_barrier
    _revert_variable_change = _revert_log_barrier


def lm(data, cov, initial, lm_start=100, lm_step=10, max_steps=20, max_it=100):
    """
    Solve using the Levemberg-Marquardt algorithm.

    Parameters:

    * data
        1D array-like data vector

    * cov
        2D array-like covariance matrix of the data

    * initial
        1D array-like initial estimate

    * lm_start
        Initial Marquardt parameter (ie, step size)

    * lm_step
        Factor by which the Marquardt parameter will be reduced with each
        successful step

    * max_steps
        How many times to try giving a step before exiting

    * max_it
        Maximum number of iterations

    Return:

    * [estimate, goals]
        estimate = array-like estimated parameter vector
        goals = list of the goal function value per iteration

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

    return next, residuals, goals


def linear_overdet(data, cov=None):
    """
    Solve a linear over-determined problem.

    Only supports Tikhonov regularization and Equality constraints

    Parameters:

    * data
        1D array-like data vector

    * cov
        2D array-like covariance matrix of the data

    Returns:

    * estimate
        1D array-like estimated parameter vector

    """

    log.info("Linear Over-determined Inversion:")

    total_start = time.time()

    sensibility = _build_jacobian(None)

    ndata, nparams = sensibility.shape

    assert len(data) == ndata, \
        "Size of data vector doesn't match number of lines in the " + \
        "sensibility (Jacobian) matrix."

    start = time.time()

    # Put together the normal equation system
    hessian = numpy.dot(sensibility.T, sensibility)

    _sum_reg_hessians(hessian, None)

    gradient = -1*numpy.dot(sensibility.T, data)

    _sum_reg_gradients(gradient, None)

    end = time.time()

    log.info("  Assemble normal equation system (%g s)" % (end - start))

    start = time.time()

    estimate = numpy.linalg.solve(hessian, -1*gradient)

    end = time.time()
    log.info("  Solve for the parameters (%g s)" % (end - start))

    residuals = data - numpy.dot(sensibility, estimate)

    rms = (residuals*residuals).sum()

    reg_goal, msg = _calc_regularizer_goal(estimate)

    goal = rms + reg_goal

    log.info("  RMS: %g" % (rms))
    log.info("  Regularizers:%s" % (msg))
    log.info("  Total goal function: %g" % (goal))

    total_end = time.time()

    log.info("  Total time for inversion: %g s" % (total_end - total_start))

    return estimate, residuals, [goal]


def linear_underdet(data, cov=None):
    """
    Solve a linear under-determined problem by using prior information in the
    form of regularization.

    Only supports Tikhonov regularization and Equality constraints.

    Parameters:

    * data
        1D array-like data vector

    * cov
        2D array-like covariance matrix of the data

    Returns:

    * estimate
        1D array-like estimated parameter vector

    """

    log.info("Linear Under-determined Inversion:")

    assert damping > 0 or smoothness > 0 or curvature > 0, \
        "Can't solve under-determined problem without regularization." + \
        "Use damping or smoothness or curvature."

    total_start = time.time()

    sensibility = _build_jacobian(None)

    ndata, nparams = sensibility.shape

    assert len(data) == ndata, \
        "Size of data vector doesn't match number of lines in the " + \
        "sensibility (Jacobian) matrix."

    start = time.time()

    hessian = numpy.zeros((nparams, nparams))

    _sum_reg_hessians(hessian, None)

    hessian = numpy.linalg.inv(hessian)

    end = time.time()

    log.info("  Calculating parameter weights matrix (%g s)" % (end - start))

    start = time.time()

    # Build the equation system to solve for the Lagrange multipliers vector
    aux = numpy.dot(sensibility, hessian)

    A = numpy.dot(aux, sensibility.T) + \
        numpy.identity(ndata)

    b = data

    # equality: equality constraints regularization param (global variable)
    if equality != 0:

        eq_gradient = numpy.zeros(ndata)

        _sum_eq_gradient(eq_gradient, None)

        b += numpy.dot(aux, -1*eq_gradient)

    lagrande_mult = numpy.linalg.solve(A, b)

    end = time.time()

    log.info("  Solving for Lagrange multiplier vector (%g s)" % (end - start))

    start = time.time()

    # Apply the multipliers to get the estimate
    estimate = numpy.dot(numpy.dot(hessian, sensibility.T), lagrande_mult)

    if equality != 0:

        estimate += numpy.dot(hessian, -1*eq_gradient)

    end = time.time()

    log.info("  Multiply to get the parameters (%g s)" % (end - start))

    residuals = data - numpy.dot(sensibility, estimate)

    rms = (residuals*residuals).sum()

    reg_goal, msg = _calc_regularizer_goal(estimate)

    goal = rms + reg_goal

    log.info("  RMS: %g" % (rms))
    log.info("  Regularizers:%s" % (msg))
    log.info("  Total goal function: %g" % (goal))

    total_end = time.time()

    log.info("  Total time for inversion: %g s" % (total_end - total_start))

    return estimate, residuals, [goal]