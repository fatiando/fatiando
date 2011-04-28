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
Gradient solvers for inverse problems with optional regularization.

Implemented algorithms:

* Levemberg-Marquardt

Uses higher order functions (functions that take functions as arguments) to
implement generic solvers and to apply generic regularization.

Functions:

* :func:`fatiando.inv.gsolvers.marq`
    Solve the (non)-linear system f(p) = d with optional constraints
    (regularization) in a least-squares sense using the Levemberg-Marquardt
    algorithm.

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 25-Apr-2011'


import logging


import numpy
from numpy import dot as dot_product
from numpy.linalg import solve as linsys_solver


import fatiando

log = logging.getLogger('fatiando.inv.gsolvers')
log.addHandler(fatiando.default_log_handler)


def linlsq(data, func, jac, reggrad=None, reghess=None):
    """
    Solve the regularized linear inverse problem using least-squares.

    Estimates a parameter vector p that satisfies

    .. math::

        \mathbf{d} = \mathbf{A}\mathbf{p}

    by finding the minimum of the goal function

    .. math::

        \Gamma = \mathbf{r}^T\mathbf{r} + \mu_1\phi_1(\mathbf{p}) + ... +
        \mu_N\phi_N(\mathbf{p})

    Functions :math:`\phi_1, ..., \phi_N` are linear regularizing functions of
    the form

    .. math::

        \phi_k=(\mathbf{p}-\mathbf{p}_k)^T\mathbf{W_k}(\mathbf{p}-\mathbf{p}_k)

    :math:`\mathbf{W}_k` is a weight matrix and :math:`\mathbf{p}_k` is a
    reference parameter vector.

    The regularizing functions are included in the inversion using functions
    *reggrad* and *reghess* that implement the combined gradients and Hessian
    matries of all regularizing functions.

    """
    # Make lambdas that do nothing if no regularization is given
    if reghess is None:
        reghess = lambda x: x
    if reggrad is None:
        reggrad = lambda x: x
    estimate = linsys_solver(reghess(dot_product(jac.T, jac)),
                             -1*reggrad(-1*dot_product(jac.T, data)))
    residuals = data - func(estimate)
    return {'estimate':estimate, 'residuals':residuals}


def marq(data, init, func, jac, lmstart=100, lmstep=10, maxsteps=20, maxit=100,
         tol=10**(-5), regnorm=None, reggrad=None, reghess=None):
    """
    Solve the (non)-linear system f(p) = d with optional constraints
    (regularization) in a least-squares sense using the Levemberg-Marquardt
    algorithm.

    Parameters:

    * data
        1D array with the data that will be fitted

    * init
        1D array with the initial estimate of the parameters

    * func
        Function f(p) that calculates the predicted data from the parameters.
        Should receive one argument: a 1D array with the current estimate; and
        return the predicted data vector.

    * jac
        Function that returns the Jacobian matrix of f(p). Should receive a 1D
        array with the current estimate and return a 2D array with the Jacobian
        matrix.

    * lmstart
        Initial Marquardt parameter (step size)

    * lmstep
        How much to increase or decrease the Marquardt parameter with each
        iteration.

    * maxsteps
        Maximum number of times that will try to take a step

    * maxit
        Maximum iterations

    * tol
        Relative tolerance for decreasing the goal function to achieve before
        terminating

    * regnorm
        Function that calculates the total value of the regularizing functions
        for the current estimate. Should receive a 1D array with the estimate
        and return a float.

    * reggrad
        Function that sums the total gradient of the regularizing functions
        to the given gradient vector. Call signature:
            grad = reggrad(p, grad)

    * reghess
        Function that sums the total Hessian of the regularizing functions
        to the given Hessian matrix.

    """
    # Make lambdas that do nothing if no regularization is given
    if regnorm is None or reggrad is None or reghess is None:
        regnorm = lambda x: 0
        reggrad = lambda g, x: g
        reghess = lambda h, x: h
    residuals = data - func(init)
    rms = numpy.linalg.norm(residuals)**2
    reg = regnorm(init)
    goals = [rms + reg]
    lm_param = lmstart
    next = init
    for iteration in xrange(maxit):
        prev = next
        jacobian = jac(prev)
        gradient = reggrad(-1*dot_product(jacobian.T, residuals), prev)
        hessian = reghess(dot_product(jacobian.T, jacobian), prev)
        # Don't calculate things twice
        hessian_diag = hessian.diagonal()
        gradient *= -1
        # Enter the Marquardt loop to find the best step size
        stagnation = True
        for lm_iteration in xrange(maxsteps):
            delta = linsys_solver(hessian + lm_param*hessian_diag, gradient)
            next = prev + delta
            residuals = data - func(next)
            rms = numpy.linalg.norm(residuals)**2
            reg = regnorm(next)
            goal = rms + reg
            if goal < goals[-1]:
                # Don't let lm_param be smaller than this
                if lm_param > 10**(-10):
                    lm_param /= lmstep
                stagnation = False
                break
            else:
                # Don't let lm_param be larger than this
                if lm_param < 10**(10):
                    lm_param *= lmstep
        if stagnation:
            next = prev
            log.warning("WARNING: convergence tolerance not achieved")
            break
        else:
            goals.append(goal)
            # Check if goal function decreases more than a threshold
            if abs((goals[-1] - goals[-2])/goals[-2]) <= tol:
                break
    result = {'estimate':next, 'residuals':residuals, 'goal_p_it':goals}
    return result
