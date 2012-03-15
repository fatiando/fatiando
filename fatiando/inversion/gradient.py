# Copyright 2012 The Fatiando a Terra Development Team
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
Factory functions for generic inverse problem solvers using gradient
optimization methods.

* :func:`fatiando.inversion.gradient.newton`
* :func:`fatiando.inversion.gradient.levmarq`
* :func:`fatiando.inversion.gradient.steepest`

The factory functions produce the actual solver functions. Solver functions are
Python generator functions that have the general format::

    def solver(dms, regs, **kwargs):
        # Start-up
        ...
        # yield the initial estimate
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        for i in xrange(maxit):
            # Perform an iteration
            ...
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}

Parameters:

* dms
    List of data modules. Data modules should be child-classes of the
    :class:`fatiando.inversion.datamodule.DataModule` class.
* regs
    List of regularizers. Regularizers should be child-classes of the
    :class:`fatiando.inversion.regularizer.Regularizer` class.
* kwargs
    Are how the factory functions pass the needed parameters to the solvers.
    Not to be altered outside the factory functions.

Yields:

* changeset
    A dictionary with the solution at the current iteration:
    
    ``changeset = {'estimate':p, 'misfits':misfits, 'goals':goals,
    'residuals':residuals}``
    
    * ``p`` is the parameter vector.
    * ``misfits`` list with data-misfit function values per iteration
    * ``goals`` list with goal function values per iteration
    * ``residuals`` list with the residual vectors at this iteration

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'


import itertools

import numpy
from numpy.linalg import solve as linsys_solver
import scipy.sparse
import scipy.sparse.linalg

from fatiando import logger


def _sparse_linsys_solver(A, x):
    res = scipy.sparse.linalg.cgs(A, x)
    if res[1] > 0:
        log = logger.dummy('fatiando.inversion.gradient._sparse_linsys_solver')
        log.warning("Conjugate Gradient convergence not achieved")
    if res[1] < 0:
        log = logger.dummy('fatiando.inversion.gradient._sparse_linsys_solver')
        log.error("Conjugate Gradient illegal input or breakdown")
    return res[0]

def _zerovector(n):
    return numpy.zeros(n)

def _zeromatrix(shape):
    return numpy.zeros(shape)
    
def use_sparse():
    """
    Configure the gradient solvers to use the sparse conjugate gradient linear
    system solver from Scipy.

    Note that this does not make the DataModules use sparse matrices! That must
    be implemented for each inverse problem separately.
    
    """
    log = logger.dummy('fatiando.inversion.gradient.use_sparse')
    log.info("Using sparse conjugate gradient solver")
    global linsys_solver, _zeromatrix
    linsys_solver = _sparse_linsys_solver
    _zeromatrix = scipy.sparse.csr_matrix

def steepest(initial, step=0.1, maxit=500, tol=10**(-5), armijo=True,
    maxsteps=20):
    """
    Factory function for the non-linear inverse problem solver using the
    Steepest Descent algorithm.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\Delta\\bar{p} = -\\lambda\\bar{g}

    where :math:`\\lambda` is the step size and :math:`\\bar{g}` is the
    gradient vector.

    Optionally, the step size can be determined thought a line search algorithm
    using the Armijo rule. In this case

    .. math::
    
        \\lambda = \\beta^m

    where :math:`1 > \\beta > 0` and :math:`m \\ge 0` is an interger that
    controls the step size.
    The line search finds the smallest :math:`m` that satisfies the condition
    (Armijo rule)

    .. math::

        \\Gamma(\\bar{p} + \\Delta\\bar{p}) - \\Gamma(\\bar{p}) <
        \\alpha\\beta^m ||\\bar{g}(\\bar{p})||^2

    where :math:`\\Gamma(\\bar{p})` is the goal function evaluated for parameter
    vector :math:`\\bar{p}`, :math:`\\alpha = 10^{-4}`, and 
    :math:`\\bar{g}(\\bar{p})` is the gradient vector of
    :math:`\\Gamma(\\bar{p})`.

    Parameters:
    
    * initial
        The initial estimate of the parameters
    * step
        The step size (if using Armijo, :math:`\\beta`, else :math:`\\lambda`)
    * maxit
        Maximum number of iterations
    * tol
        Relative tolerance for decreasing the goal function to before
        terminating
    * armijo
        If True, will use the Armijo rule to determine the best step size at
        each iteration. It's highly recommended to use this.
    * maxsteps
        If using Armijo, the maximum number os times to try to take a step
        before giving up (i.e., the maximum value of :math:`m`). If not using
        Armijo, maxsteps is ignored

    Returns:

    * solver
        A Python generator function that solves an inverse problem using the
        parameters given above.

    References:

    * Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.
    
    """
    log = logger.dummy('fatiando.inversion.gradient.steepest')
    if tol <= 0.0:
        raise ValueError, "tol parameter should be > 0"
    if maxit <= 0:
        raise ValueError, "maxit parameter should be > 0"
    if maxsteps <= 0:
        raise ValueError, "maxsteps parameter should be > 0"
    if step <= 0 or step >= 1.:
        raise ValueError, "step parameter should be 1 > step > 0"
    initial_array = numpy.array(initial, dtype='f')
    log.info("Generating Steepest Descent solver:")
    log.info("  initial step size: %g" % (step))
    log.info("  max iterations: %d" % (maxit))
    log.info("  convergence tolerance: %g" % (tol))
    log.info("  using Armijo rule: %s" % (str(armijo)))
    if armijo:
        log.info("  max step iterations: %d" % (maxsteps))
        return _steepest_armijo(initial_array, float(step), maxsteps, maxit,
            float(tol))
    else:
        return _steepest(initial_array, float(step), maxit, float(tol))

def _steepest(initial, step, maxit, tol):
    """
    Factory for the simple Steepest Descent algorithm without using backtracking
    to find the best step sizes.
    """
    def solver(dms, regs, initial=initial, step=step, maxit=maxit, tol=tol):
        if len(dms) == 0:
            raise ValueError, "Need at least 1 data module. None given"
        p = initial
        nparams = len(p)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res)
                     for d, res in itertools.izip(dms, residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        misfits = [misfit]
        goals = [goal]
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        for it in xrange(maxit):
            gradient = _zerovector(nparams)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            p -= step*gradient
            residuals = [d.data - d.get_predicted(p) for d in dms]
            misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                         residuals))
            goal = misfit + sum(r.value(p) for r in regs)
            misfits.append(misfit)
            goals.append(goal)            
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}
            # Check if goal function decreases more than a threshold
            if abs((goals[-1] - goals[-2])/goals[-2]) <= tol:
                break
    return solver

def _steepest_armijo(initial, step, maxsteps, maxit, tol):
    """
    Factory for the Steepest Descent algorithm with line search using the Armijo
    rule.
    """
    def solver(dms, regs, initial=initial, step=step, maxsteps=maxsteps,
        maxit=maxit, tol=tol, alpha=10**(-4)):
        if len(dms) == 0:
            raise ValueError, "Need at least 1 data module. None given"
        p = initial
        nparams = len(p)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res)
                     for d, res in itertools.izip(dms, residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        misfits = [misfit]
        goals = [goal]
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        for it in xrange(maxit):
            gradient = _zerovector(nparams)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            # Calculate now to avoid computing inside the loop
            gradnorm = numpy.linalg.norm(gradient)**2
            stagnation = True
            # The loop to determine the best step size
            for m in xrange(maxsteps):
                factor = step**m
                ptmp = p - factor*gradient
                restmp = [d.data - d.get_predicted(ptmp) for d in dms]
                misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                             restmp))
                goal = misfit + sum(r.value(ptmp) for r in regs)
                if goal - goals[-1] < alpha*factor*gradnorm:
                    stagnation = False
                    break
            if stagnation:
                if it == 0:
                    msg = "  Steepest Descent didn't take any steps"
                else:
                    msg = "  Steepest Descent finished: couldn't take a step"
                log = logger.dummy('fatiando.inversion.gradient.steepest')
                log.warning(msg)
                break
            p = ptmp
            residuals = restmp     
            misfits.append(misfit)
            goals.append(goal)            
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}
            # Check if goal function decreases more than a threshold
            if abs((goals[-1] - goals[-2])/goals[-2]) <= tol:
                break
    return solver

def levmarq(initial, damp=1., factor=10., maxsteps=20, maxit=100, tol=10**(-5),
    diag=False):
    """
    Factory function for the non-linear inverse problem solver using the
    Levenberg-Marquardt algorithm.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\Delta\\bar{p} = -[\\bar{\\bar{H}} +
        \\lambda \\bar{\\bar{I}}]^{-1}\\bar{g}

    where :math:`\\lambda` is a damping factor (step size),
    :math:`\\bar{\\bar{H}}` is the Hessian matrix,
    :math:`\\bar{\\bar{I}}` is the identity matrix,
    and :math:`\\bar{g}` is the gradient vector.

    Parameters:
    
    * initial
        The initial estimate of the parameters
    * damp
        The initial damping factor (:math:`\\lambda`)
    * factor
        The increment/decrement to the damping factor at each iteration.
        Should be ``factor > 1``
    * maxsteps
        The maximum number os times to try to take a step before giving up
    * maxit
        Maximum number of iterations
    * tol
        Relative tolerance for decreasing the goal function to before
        terminating
    * diag
        If True, will use the diagonal of the Hessian matrix instead of the
        identity matrix. Only use this is the parameters are different physical
        quantities (like time and distance, for example)

    Returns:

    * solver
        A Python generator function that solves an inverse problem using the
        parameters given above.

    References:

    * Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.
    
    """
    log = logger.dummy('fatiando.inversion.gradient.levmarq')
    if tol <= 0.0:
        raise ValueError, "tol parameter should be > 0"
    if maxit <= 0:
        raise ValueError, "maxit parameter should be > 0"
    if maxsteps <= 0:
        raise ValueError, "maxsteps parameter should be > 0"
    if damp <= 0:
        raise ValueError, "damp parameter should be > 0"
    if factor <= 0:
        raise ValueError, "factor parameter should be > 0"
    initial_array = numpy.array(initial, dtype='f')
    log.info("Generating Levenberg-Marquardt solver:")
    log.info("  initial damping factor: %g" % (damp))
    log.info("  damping factor increment/decrement: %g" % (factor))
    log.info("  max step iterations: %d" % (maxsteps))
    log.info("  max iterations: %d" % (maxit))
    log.info("  convergence tolerance: %g" % (tol))
    log.info("  use diagonal of Hessian: %s" % (diag))
    if diag:        
        return _levmarq_diag(initial_array, float(damp), float(factor),
            maxsteps, maxit, float(tol))
    else:
        return _levmarq(initial_array, float(damp), float(factor), maxsteps,
            maxit, float(tol))

def _levmarq(initial, damp, factor, maxsteps, maxit, tol):
    """
    Factory function for the Levenberg-Marquardt solver using the identity
    matrix for damping.
    """
    def solver(dms, regs, initial=initial, damp=damp, factor=factor,
        maxsteps=maxsteps, maxit=maxit, tol=tol):
        if len(dms) == 0:
            raise ValueError, "Need at least 1 data module. None given"
        p = initial
        nparams = len(p)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res)
                     for d, res in itertools.izip(dms, residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        misfits = [misfit]
        goals = [goal]
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        identity = numpy.identity(nparams)
        step = damp
        for it in xrange(maxit):
            gradient = _zerovector(nparams)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            # Multiply the gradient now so that doesn't do this inside the loop
            gradient *= -1
            hessian = _zeromatrix((nparams, nparams))
            for m in itertools.chain(dms, regs):
                hessian = m.sum_hessian(hessian, p)
            stagnation = True
            # The loop to determine the best step size
            for itstep in xrange(maxsteps):
                ptmp = p + linsys_solver(hessian + step*identity, gradient)
                restmp = [d.data - d.get_predicted(ptmp) for d in dms]
                misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                             restmp))
                goal = misfit + sum(r.value(ptmp) for r in regs)
                if goal < goals[-1]:
                    # Don't let the damping factor be smaller than this
                    if step > 10.**(-10):
                        step /= factor
                    stagnation = False
                    break
                else:
                    # Don't let the damping factor be larger than this
                    if step < 10**(10):
                        step *= factor
                    else:
                        break
            if stagnation:
                if it == 0:
                    msg = "  Levenberg-Marquardt didn't take any steps"
                else:
                    msg = "  Levenberg-Marquardt finished: couldn't take a step"
                log = logger.dummy('fatiando.inversion.gradient.levmarq')
                log.warning(msg)
                break
            p = ptmp
            residuals = restmp     
            misfits.append(misfit)
            goals.append(goal)            
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}
            # Check if goal function decreases more than a threshold
            if abs((goals[-1] - goals[-2])/goals[-2]) <= tol:
                break
    return solver
    
def _levmarq_diag(initial, damp, factor, maxsteps, maxit, tol):
    """
    Factory function for the Levenberg-Marquardt solver using the diagonal of
    the Hessian for damping.
    """
    def solver(dms, regs, initial=initial, damp=damp, factor=factor,
        maxsteps=maxsteps, maxit=maxit, tol=tol):
        if len(dms) == 0:
            raise ValueError, "Need at least 1 data module. None given"
        p = initial
        nparams = len(p)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res)
                     for d, res in itertools.izip(dms, residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        misfits = [misfit]
        goals = [goal]
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        step = damp
        for it in xrange(maxit):
            gradient = _zerovector(nparams)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            # Multiply the gradient now so that doesn't do this inside the loop
            gradient *= -1
            hessian = _zeromatrix((nparams, nparams))
            for m in itertools.chain(dms, regs):
                hessian = m.sum_hessian(hessian, p)
            hessian_diag = hessian.diagonal()
            stagnation = True
            # The loop to determine the best step size
            for itstep in xrange(maxsteps):
                ptmp = p + linsys_solver(hessian + step*hessian_diag, gradient)
                restmp = [d.data - d.get_predicted(ptmp) for d in dms]
                misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                             restmp))
                goal = misfit + sum(r.value(ptmp) for r in regs)
                if goal < goals[-1]:
                    # Don't let the damping factor be smaller than this
                    if step > 10.**(-10):
                        step /= factor
                    stagnation = False
                    break
                else:
                    # Don't let the damping factor be larger than this
                    if step < 10**(10):
                        step *= factor
                    else:
                        break
            if stagnation:
                if it == 0:
                    msg = "  Levenberg-Marquardt didn't take any steps"
                else:
                    msg = "  Levenberg-Marquardt finished: couldn't take a step"
                log = logger.dummy('fatiando.inversion.gradient.levmarq')
                log.warning(msg)
                break
            p = ptmp
            residuals = restmp     
            misfits.append(misfit)
            goals.append(goal)            
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}
            # Check if goal function decreases more than a threshold
            if abs((goals[-1] - goals[-2])/goals[-2]) <= tol:
                break
    return solver

def newton(initial, maxit=100, tol=10**(-5)):
    """
    Factory function for the non-linear inverse problem solver using Newton's
    method.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\Delta\\bar{p} = -\\bar{\\bar{H}}^{-1}\\bar{g}

    where :math:`\\bar{\\bar{H}}` is the Hessian matrix and :math:`\\bar{g}` is
    the gradient vector.
    
    Parameters:
    
    * initial
        The initial estimate of the parameters
    * maxit
        Maximum number of iterations
    * tol
        Relative tolerance for decreasing the goal function to before
        terminating

    Returns:

    * solver
        A Python generator function that solves an inverse problem using the
        parameters given above.

    References:

    * Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.
    
    """
    log = logger.dummy('fatiando.inversion.gradient.newton')
    if tol <= 0.0:
        raise ValueError, "tol parameter should be > 0"
    if maxit <= 0:
        raise ValueError, "maxit parameter should be > 0"
    log.info("Generating Newton's method solver:")
    log.info("  max iterations: %d" % (maxit))
    log.info("  convergence tolerance: %g" % (tol))
    initial_array = numpy.array(initial, dtype='f')
    def solver(dms, regs, initial=initial_array, maxit=maxit, tol=tol):
        """
        Inverse problem solver using Newton's method.
        """
        if len(dms) == 0:
            raise ValueError, "Need at least 1 data module. None given"
        p = initial
        nparams = len(p)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res)
                     for d, res in itertools.izip(dms, residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        misfits = [misfit]
        goals = [goal]
        yield {'estimate':p, 'misfits':misfits, 'goals':goals,
               'residuals':residuals}
        for it in xrange(maxit):
            gradient = _zerovector(nparams)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            hessian = _zeromatrix((nparams, nparams))
            for m in itertools.chain(dms, regs):
                hessian = m.sum_hessian(hessian, p)
            p = p + linsys_solver(hessian, -1*gradient)
            residuals = [d.data - d.get_predicted(p) for d in dms]
            misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                         residuals))
            goal = misfit + sum(r.value(p) for r in regs)
            misfits.append(misfit)
            goals.append(goal)            
            yield {'estimate':p, 'misfits':misfits, 'goals':goals,
                   'residuals':residuals}
            # Check if goal function decreases more than a threshold
            if (goals[-1] < goals[-2] and
                abs((goals[-1] - goals[-2])/goals[-2]) <= tol):
                break
    return solver
