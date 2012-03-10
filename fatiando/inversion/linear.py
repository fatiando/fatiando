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
Factory functions for generic linear inverse problem solvers.

* :func:`fatiando.inversion.linear.overdet`
* :func:`fatiando.inversion.linear.underdet`

The factory functions produce the actual solver functions.
These solver functions are Python generator functions that yield only once.
This is so they are compatible with gradient
(:mod:`fatiando.inversion.gradient`) and heurist
(:mod:`fatiando.inversion.heuristic`) solvers.
In the case of linear problems, solver functions have the general format::

    def solver(dms, regs):
        # Start-up
        ...
        # Calculate the estimate
        ...
        # yield the results
        yield {'estimate':p, 'misfits':[misfit], 'goals':[goal],
               'residuals':residuals}

Parameters:

* dms
    List of data modules. Data modules should be child-classes of the
    :class:`fatiando.inversion.datamodule.DataModule` class.
* regs
    List of regularizers. Regularizers should be child-classes of the
    :class:`fatiando.inversion.regularizer.Regularizer` class.

Yields:

* solution
    A dictionary with the final solution:
    
    ``changeset = {'estimate':p, 'misfits':[misfit], 'goals':[goal],
    'residuals':residuals}``
    
    * ``p`` is the parameter vector.
    * ``misfit`` the data-misfit function value
    * ``goal`` the goal function value
    * ``residuals`` list with the residual vectors

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Jan-2012'


import itertools

import numpy
from numpy.linalg import solve as linsys_solver
import scipy.sparse
import scipy.sparse.linalg

from fatiando import logger


def _sparse_linsys_solver(A, x):
    res = scipy.sparse.linalg.cgs(A, x)
    if res[1] > 0:
        log = logger.dummy('fatiando.inversion.linear._sparse_linsys_solver')
        log.warning("Conjugate Gradient convergence not achieved")
    if res[1] < 0:
        log = logger.dummy('fatiando.inversion.linear._sparse_linsys_solver')
        log.error("Conjugate Gradient illegal input or breakdown")
    return res[0]

def _zerovector(n):
    return numpy.zeros(n)

def _zeromatrix(shape):
    return numpy.zeros(shape)
    
def use_sparse():
    """
    Configure the linear solvers to use the sparse conjugate gradient linear
    linear system solver from Scipy.

    Note that this does not make the DataModules use sparse matrices! That must
    be implemented for each inverse problem.
    
    """
    log = logger.dummy('fatiando.inversion.linear.use_sparse')
    log.info("Using sparse conjugate gradient solver")
    global linsys_solver, _zeromatrix
    linsys_solver = _sparse_linsys_solver
    _zeromatrix = scipy.sparse.csr_matrix
    
def overdet(nparams):
    """
    Factory function for a linear least-squares solver to an overdetermined
    problem (more data than parameters).

    The least-squares estimate is found by solving the linear system

    .. math::

        \\bar{\\bar{H}}\\hat{\\bar{p}} = -\\bar{g}

    where :math:`\\bar{\\bar{H}}` is the Hessian matrix of the goal function,
    :math:`\\bar{g}` is the gradient vector of the goal function, and
    :math:`\\bar{p}` is the estimated parameter vector.
    
    Parameters:

    * nparams
        The number of parameters in vector :math:`\\bar{p}` (usually something
        like ``mesh.size``)

    Returns
    
    * solver
        A Python generator function that solves an linear overdetermined inverse
        problem using the parameters given above.
    
    """
    log = logger.dummy('fatiando.inversion.linear.overdet')
    if nparams <= 0:
        raise ValueError, "nparams must be > 0"
    log.info("Generating linear solver for overdetermined problems")
    log.info("  number of parameters: %d" % (nparams))
    def solver(dms, regs, nparams=nparams):
        gradientchain = [_zerovector(nparams)]
        gradientchain.extend(itertools.chain(dms, regs))
        gradient = reduce(lambda g, m: m.sum_gradient(g), gradientchain)
        hessianchain = [_zeromatrix((nparams, nparams))]
        hessianchain.extend(itertools.chain(dms, regs))
        hessian = reduce(lambda h, m: m.sum_hessian(h), hessianchain)
        p = linsys_solver(hessian, -1*gradient)
        residuals = [d.data - d.get_predicted(p) for d in dms]
        misfit = sum(d.get_misfit(res) for d, res in itertools.izip(dms,
                     residuals))
        goal = misfit + sum(r.value(p) for r in regs)
        yield {'estimate':p, 'misfits':[misfit], 'goals':[goal],
               'residuals':residuals}
    return solver
            
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
