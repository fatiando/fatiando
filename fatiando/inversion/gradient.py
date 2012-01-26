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

The factory functions produce the actual solver functions. Solver functions are
Python generator functions that have the general format::

    def solver(dms, regs, **kwargs):
        ...
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


import numpy
from numpy import dot as dot_product
from numpy.linalg import solve as linsys_solver
import itertools

from fatiando import logger

log = logger.dummy()


def steepest(initial, step=1., maxsteps=20, maxit=100, tol=10**(-3)):
    """
    Factory function for the non-linear inverse problem solver using the
    Steepest Descent algorithm.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\Delta\\bar{p} = -\\lambda\\bar{g}

    where :math:`\\lambda` is the step size and :math:`\\bar{g}` is the
    gradient vector.

    Parameters:
    
    * initial
        The initial estimate of the parameters
    * step
        The initial step size (:math:`\\lambda`)
    * maxsteps
        The maximum number os times to try to take a step before giving up
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
    if tol <= 0.0:
        raise ValueError, "tol parameter should be > 0"
    if maxit <= 0:
        raise ValueError, "maxit parameter should be > 0"
    if maxsteps <= 0:
        raise ValueError, "maxsteps parameter should be > 0"
    if step <= 0 or step > 1.:
        raise ValueError, "step parameter should be 1 >= step > 0"
    log.info("Generating Steepest Descent solver:")
    log.info("  initial step size: %g" % (step))
    log.info("  max step iterations: %d" % (maxsteps))
    log.info("  max iterations: %d" % (maxit))
    log.info("  convergence tolerance: %g" % (tol))
    initial_array = numpy.array(initial, dtype='f')
    def solver(dms, regs, initial=initial_array, step=float(step),
        maxsteps=maxsteps, maxit=maxit, tol=tol):
        """
        Inverse problem solver using the Steepest Descent algorithm.
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
        for it in xrange(maxit):
            gradient = numpy.zeros_like(p)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            stagnation = True
            # The loop to determine the best step size
            m = 1
            for itstep in xrange(maxsteps):
                ptmp = p - (step**m)*gradient
                restmp = [d.data - d.get_predicted(ptmp) for d in dms]
                misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                             restmp))
                goal = misfit + sum(r.value(ptmp) for r in regs)
                if goal - goals[-1] < (step**m)*numpy.linalg.norm(gradient)**2:
                    # Don't let the damping factor be smaller than this
                    if damp > 10.**(-10):
                        damp /= factor
                    stagnation = False
                    break
                else:
                    # Don't let the damping factor be larger than this
                    if damp < 10**(10):
                        damp *= factor
            if stagnation:
                msg = "Steepest Descent exited because couldn't take a step"
                if it == 0:
                    raise ValueError, msg
                else:
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

def levmarq(initial, damp=1., factor=10., maxsteps=20, maxit=100, tol=10**(-3)):
    """
    Factory function for the non-linear inverse problem solver using the
    Levemberg-Marquardt algorithm.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\Delta\\bar{p} = -[\\bar{\\bar{H}} +
        \\lambda\\cdot diag(\\bar{\\bar{H}})]^{-1}\\bar{g}

    where :math:`\\lambda` is a damping factor (step size),
    :math:`\\bar{\\bar{H}}` is the Hessian matrix,
    :math:`diag(\\bar{\\bar{H}})` is a matrix with the diagonal of the Hessian,
    and :math:`\\bar{g}` is the gradient vector.

    Parameters:
    
    * initial
        The initial estimate of the parameters
    * damp
        The initial damping factor (:math:`\\lambda`)
    * factor
        The increment/decrement to the damping factor at each iteration
    * maxsteps
        The maximum number os times to try to take a step before giving up
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
    log.info("Generating Levemberg-Marquardt solver:")
    log.info("  initial damping factor: %g" % (damp))
    log.info("  damping factor increment/decrement: %g" % (factor))
    log.info("  max step iterations: %d" % (maxsteps))
    log.info("  max iterations: %d" % (maxit))
    log.info("  convergence tolerance: %g" % (tol))
    initial_array = numpy.array(initial, dtype='f')
    def solver(dms, regs, initial=initial_array, damp=float(damp),
        factor=float(factor), maxsteps=maxsteps, maxit=maxit, tol=tol):
        """
        Inverse problem solver using the Levemberg-Marquardt algorithm.
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
        for it in xrange(maxit):
            gradient = numpy.zeros_like(p)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            # Multiply the gradient now so that doesn't do this inside the loop
            gradient *= -1
            hessian = numpy.zeros((nparams, nparams))
            for m in itertools.chain(dms, regs):
                hessian = m.sum_hessian(hessian, p)
            hessian_diag = hessian.diagonal()
            stagnation = True
            # The loop to determine the best step size
            for itstep in xrange(maxsteps):
                ptmp = p + linsys_solver(hessian + damp*hessian_diag, gradient)
                restmp = [d.data - d.get_predicted(ptmp) for d in dms]
                misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                             restmp))
                goal = misfit + sum(r.value(ptmp) for r in regs)
                if goal < goals[-1]:
                    # Don't let the damping factor be smaller than this
                    if damp > 10.**(-10):
                        damp /= factor
                    stagnation = False
                    break
                else:
                    # Don't let the damping factor be larger than this
                    if damp < 10**(10):
                        damp *= factor
            if stagnation:
                msg = "Levemberg-Marquardt exited because couldn't take a step"
                if it == 0:
                    raise ValueError, msg
                else:
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

def newton(initial, maxit=100, tol=10**(-3)):
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
        for it in xrange(maxit):
            gradient = numpy.zeros_like(p)
            for d, res in itertools.izip(dms, residuals):
                gradient = d.sum_gradient(gradient, p, res)
            for r in regs:
                gradient = r.sum_gradient(gradient, p)
            hessian = numpy.zeros((nparams, nparams))
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
    
            
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
