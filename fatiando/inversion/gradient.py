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
Gradient solvers for generic inverse problems.

* :func:`fatiando.inversion.gradient.newton`
* :func:`fatiando.inversion.gradient.levmarq`

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


def levmarq(dms, initial, regs=[], step=1, maxit=100, tol=10**(-5)):
    """
    Solve the non-linear inverse problem using the Levemberg-Marquardt algorithm

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\bar{\\Delta\\bar{p}} = -[\\bar{\\bar{H}} +
        \\lambda\\cdot diag(\\bar{\\bar{H}})]^{-1}\\bar{g}

    where :math:`\\lambda` is the Marquardt parameter, :math:`\\bar{\\bar{H}}`
    is the Hessian matrix and :math:`\\bar{g}` is the gradient vector.

    This function is a generator and should be used inside a loop.
    It yields one step of the algorithm per iteration.

    Example::

        Need example
    

    Parameters:

    * dms
        List of data modules. Data modules should be child-classes of the
        :class:`fatiando.inversion.datamodule.DataModule` class.
    * initial
        The initial estimate of the parameters
    * regs
        List of regularizers. Regularizers should be child-classes of the
        :class:`fatiando.inversion.regularizer.Regularizer` class.
    * step
        Step size.
    * maxit
        Maximum number of iterations
    * tol
        Relative tolerance for decreasing the goal function to before
        terminating

    Yields:

    * changeset
        A dictionary with the current solution.        
        ``{'estimate':p, 'misfits':misfits, 'goals':goals, 'dms':dms}``
        
        * ``p`` is the current parameter vector.
        * ``misfits`` list with data-misfit function values per iteration
        * ``goals`` list with goal function values per iteration
        * ``residuals`` list with the residual vectors at this iteration
    
    References:

    * Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.
    
    """
    if len(dms) == 0:
        raise ValueError, "Need at least 1 data module. None given"
    p = initial
    nparams = len(p)
    residuals = [d.data - d.get_predicted(p) for d in dms]
    misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms, residuals))
    goal = misfit + sum(r.value(p) for r in regs)
    misfits = [misfit]
    goals = [goal]
    for i in xrange(maxit):
        gradient = numpy.zeros_like(p)
        for d, res in itertools.izip(dms, residuals):
            gradient = d.sum_gradient(gradient, p, res)
        for r in regs:
            gradient = r.sum_gradient(gradient, p)
        hessian = numpy.zeros((nparams, nparams))
        for m in itertools.chain(dms, regs):
            hessian = m.sum_hessian(hessian, p)
        p += step*linsys_solver(hessian, -1*gradient)
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

def newton(initial, maxit=100, tol=10**(-5)):
    """
    Factory function for the non-linear inverse problem solver using Newton's
    method.

    The increment to the parameter vector :math:`\\bar{p}` is calculated by
    
    .. math::

        \\bar{\\Delta\\bar{p}} = -\\bar{\\bar{H}}^{-1}\\bar{g}

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

    The solver generator function has the following specifications:

    ``solver(dms, regs, initial=initial, maxit=maxit, tol=tol)``
    
    Parameters:

    * dms
        List of data modules. Data modules should be child-classes of the
        :class:`fatiando.inversion.datamodule.DataModule` class.
    * regs
        List of regularizers. Regularizers should be child-classes of the
        :class:`fatiando.inversion.regularizer.Regularizer` class.
    * initial
        The initial estimate of the parameters.
        Default is the value given to the factory function.
    * maxit
        Maximum number of iterations.
        Default is the value given to the factory function.
    * tol
        Relative tolerance for decreasing the goal function to before
        terminating.        
        Default is the value given to the factory function.

    Yields:

    * changeset
        A dictionary with the current solution.        
        ``{'estimate':p, 'misfits':misfits, 'goals':goals, 'dms':dms}``
        
        * ``p`` is the current parameter vector.
        * ``misfits`` list with data-misfit function values per iteration
        * ``goals`` list with goal function values per iteration
        * ``residuals`` list with the residual vectors at this iteration

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
            p += linsys_solver(hessian, -1*gradient)
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
    
            
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
