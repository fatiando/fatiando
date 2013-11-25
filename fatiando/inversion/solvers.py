"""
Gradient descent and heuristic optimizing functions.
"""
import numpy
import scipy.sparse

from ..utils import safe_solve, safe_diagonal, safe_dot


def linear(hessian, gradient, precondition=True):
    r"""
    Find the parameter vector that minimizes a linear objective function.

    The parameter vector :math:`\bar{p}` that minimizes this objective
    function :math:`\phi` is the one that solves the linear system

    .. math::

        \bar{\bar{H}} \bar{p} = -\bar{g}

    where :math:`\bar{\bar{H}}` is the Hessian matrix of :math:`\phi` and
    :math:`\bar{g}` is the gradient vector of :math:`\phi`.

    Parameters:

    * hessian : 2d-array
        The Hessian matrix of the objective function.
    * gradient : 1d-array
        The gradient vector of the objective function.
    * precondition : True or False
        If True, will use Jacobi preconditioning.

    Returns:

    * estimate : 1d-array
        The estimated parameter vector

    """
    if precondition:
        diag = numpy.abs(safe_diagonal(hessian))
        diag[diag < 10**-10] = 10**-10
        precond = scipy.sparse.diags(1./diag, 0).tocsr()
        hessian = safe_dot(precond, hessian)
        gradient = safe_dot(precond, gradient)
    p = safe_solve(hessian, -gradient)
    return p

def newton(initial, hessian, gradient, value, maxit=30, tol=10**-5,
           precontition=True, stats=None):
    """
    Minimize an objective function using Newton's method.

    Newton's method searches for the minimum of an objective function
    :math:`\phi(\bar{p})` by successively incrementing the initial estimate.
    The increment is the solution of the linear system

    .. math::

        \bar{\bar{H}}(\bar{p}^k) \bar{\Delta p}^k = -\bar{g}(\bar{p}^k)

    where :math:`\bar{\bar{H}}` is the Hessian matrix of :math:`\phi` and
    :math:`\bar{g}` is the gradient vector of :math:`\phi`. Both are evaluated
    at the previous estimate :math:`\bar{p}^k`.


    Parameters:

    * initial : 1d-array
        The initial estimate for the gradient descent.
    * hessian : function
        A function that returns the Hessian matrix of the objective function
        when given a parameter vector.
    * gradient : function
        A function that returns the gradient vector of the objective function
        when given a parameter vector.
    * value : function
        A function that returns the value of the objective function evaluated
        at a given parameter vector.
    * maxit : int
        The maximum number of iterations allowed.
    * tol : float
        The convergence criterion. The lower it is, the more steps are
        permitted.
    * precondition : True or False
        If True, will use Jacobi preconditioning.
    * stats : None or a dictionary
        If a dict, will fill the dictionary with information about the
        optimization procedure.

    Yields:

    * estimate : 1d-array
        The estimated parameter vector at the current iteration.

    """
    p = initial.astype(numpy.float)
    misfit = value(p)
    if stats is not None:
        stats['method'] = 'Newton'
        stats['iterations'] = 0
        stats['misfit/iteration'] = [misfit]
        stats['preconditioned'] = precondition
    yield p
    for iteration in xrange(maxit):
        hess = hessian(p)
        grad = gradient(p)
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10**-10] = 10**-10
            precond = scipy.sparse.diags(1./diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            grad = safe_dot(precond, grad)
        p = p + safe_solve(hess, -grad)
        newmisfit = self.value(p)
        if newmisfit > misfit or abs((newmisfit - misfit)/misfit) < tol:
            break
        misfit = newmisfit
        if stats is not None:
            stats['iterations'] += 1
            stats['misfit/iteration'].append(misfit)
        yield p
