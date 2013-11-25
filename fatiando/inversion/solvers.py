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
