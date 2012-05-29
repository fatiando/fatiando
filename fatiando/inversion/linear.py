"""
Factory functions for generic linear inverse problem solvers.

* :func:`~fatiando.inversion.linear.overdet`
* :func:`~fatiando.inversion.linear.underdet`

The factory functions produce the actual solver functions.
These solver functions are Python generator functions that yield only once.
This might seem unnecessary but it is done so that the linear solvers are
compatible with the non-linear solvers (e.g.,
:mod:`~fatiando.inversion.gradient` and :mod:`~fatiando.inversion.heuristic`).

This module uses dense matrices (:mod:`numpy` arrays) by default. If you want
to enable the use of sparse matrices from :mod:`scipy.sparse`, call function
:func:`fatiando.inversion.linear.use_sparse` **before** creating any solver
functions!

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

* dms : list
    List of data modules. Data modules should be child-classes of the
    :class:`~fatiando.inversion.datamodule.DataModule` class.
* regs : list
    List of regularizers. Regularizers should be child-classes of the
    :class:`~fatiando.inversion.regularizer.Regularizer` class.

    .. note:: The regularizing functions must also be linear!
    
Yields:

* changeset : dict
    A dictionary with the final solution:    
    ``changeset = {'estimate':p, 'misfits':[misfit], 'goals':[goal],
    'residuals':residuals}``
    
    * p : array
        The parameter vector.
    * misfit : list
        The data-misfit function value
    * goal : list
        The goal function value
    * residuals : list
        List with the residual vectors from each data module at this iteration

----

"""

import itertools

import numpy
from numpy.linalg import solve as linsys_solver
import scipy.sparse
import scipy.sparse.linalg

from fatiando import logger


log = logger.dummy('fatiando.inversion.linear')

def _sparse_linsys_solver(A, x):
    res = scipy.sparse.linalg.cgs(A, x)
    if res[1] > 0:
        log.warning("Conjugate Gradient convergence not achieved")
    if res[1] < 0:
        log.error("Conjugate Gradient illegal input or breakdown")
    return res[0]

def _zerovector(n):
    return numpy.zeros(n)

def _zeromatrix(shape):
    return numpy.zeros(shape)
    
def use_sparse():
    """
    Configure the gradient solvers to use the sparse conjugate gradient linear
    system solver from `scipy.sparse`.

    .. note:: This does not make the data modules use sparse matrices! That must
        be implemented for each inverse problem separately.
        
    """
    log.info("Using sparse conjugate gradient solver")
    global linsys_solver, _zeromatrix
    linsys_solver = _sparse_linsys_solver
    _zeromatrix = scipy.sparse.csr_matrix
    
def overdet(nparams):
    r"""
    Factory function for a linear least-squares solver to an overdetermined
    problem (more data than parameters).

    The problem at hand is finding the vector :math:`\bar{p}` that produces a
    predicted data vector :math:`\bar{d}` as close as possible to an observed
    data vector :math:`\bar{d}^o`.

    In linear problems, the relation between the parameters and the predicted
    data are expressed through the linear system

    .. math::

        \bar{\bar{G}}\bar{p} = \bar{d}

    where :math:`\bar{\bar{G}}` is the Jacobian (or sensitivity) matrix. In the
    **over determined** case, matrix :math:`\bar{\bar{G}}` has more lines
    than columns, i.e., more equations than unknowns.

    The least-squares estimate of :math:`\bar{p}` can be found by minimizing
    the goal function

    .. math::

        \Gamma(\bar{p}) = \bar{r}^T\bar{r} + \sum\limits_{k=1}^L
        \mu_k\theta_k(\bar{p})

    where :math:`\bar{r} = \bar{d}^o - \bar{d}` is the residual vector,
    :math:`\theta_k(\bar{p})` are regularizing functions, and :math:`\mu_k` are
    regularizing parameters (positive scalars).

    The mininum of the goal function can be calculated by solving the linear
    system

    .. math::

        \bar{\bar{H}}\hat{\bar{p}} = -\bar{g}

    where :math:`\bar{\bar{H}}` is the Hessian matrix of the goal function,
    :math:`\bar{g}` is the gradient vector of the goal function, and
    :math:`\hat{\bar{p}}` is the estimated parameter vector.

    The Hessian of the goal function is given by
    
    .. math::

        \bar{\bar{H}} = 2\bar{\bar{G}}^T\bar{\bar{G}} + \sum\limits_{k=1}^L
        \mu_k \bar{\bar{H}}_k

    where :math:`\bar{\bar{H}}_k` are the Hessian matrices of the regularizing
    functions.

    The gradient vector of the goal function is given by

    .. math::

        \bar{g} = -2\bar{\bar{G}}^T\bar{d}^o + \sum\limits_{k=1}^L
        \mu_k \bar{g}_k
        
    where :math:`\bar{g}_k` are the gradient vectors of the regularizing
    functions.
    
    Parameters:

    * nparams : int
        The number of parameters in vector :math:`\bar{p}` (usually something
        like ``mesh.size``)

    Returns
    
    * solver : function
        A Python generator function that solves an linear overdetermined inverse
        problem using the parameters given above.
    
    """
    if nparams <= 0:
        raise ValueError("nparams must be > 0")
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
