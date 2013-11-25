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
        newmisfit = value(p)
        if newmisfit > misfit or abs((newmisfit - misfit)/misfit) < tol:
            break
        misfit = newmisfit
        if stats is not None:
            stats['iterations'] += 1
            stats['misfit/iteration'].append(misfit)
        yield p

def levmarq(initial, hessian, gradient, value, maxit=30, maxsteps=10, lamb=10,
            dlamb=2, tol=10**-5, precondition=True, stats=None):
    """
    Minimize an objective function using the Levemberg-Marquardt algorithm.

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
    * maxsteps : int
        The maximum number of times to try to take a step before giving up.
    * lamb : float
        Initial amount of step regularization. The larger this is, the more the
        algorithm will resemble Steepest Descent in the initial iterations.
    * dlamb : float
        Factor by which *lamb* is divided or multiplied when taking steps.
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
    p = initial
    misfit = value(p)
    if stats is not None:
        stats['method'] = 'Levemberg-Marquardt'
        stats['iterations'] = 0
        stats['misfit/iteration'] = [misfit]
        stats['step search stagnation'] = False
        stats['trial steps/iteration'] = []
        stats['preconditioned'] = precondition
    yield p
    for iteration in xrange(maxit):
        hess = hessian(p)
        minus_gradient = -gradient(p)
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10**-10] = 10**-10
            precond = scipy.sparse.diags(1./diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            minus_gradient = safe_dot(precond, minus_gradient)
        stagnation = True
        diag = scipy.sparse.diags(safe_diagonal(hess), 0).tocsr()
        for step in xrange(maxsteps):
            newp = p + safe_solve(hess + lamb*diag, minus_gradient)
            newmisfit = value(newp)
            if newmisfit >= misfit:
                if lamb < 10**15:
                    lamb *= dlamb
            else:
                if lamb > 10**-15:
                    lamb /= dlamb
                stagnation = False
                break
        if stagnation:
            stop = True
            if stats is not None:
                stats['step search stagnation'] = True
        else:
            stop = newmisfit > misfit or abs((newmisfit - misfit)/misfit) < tol
            p = newp
            misfit = newmisfit
            if stats is not None:
                stats['iterations'] += 1
                stats['misfit/iteration'].append(newmisfit)
                stats['trial steps/iteration'].append(step + 1)
            yield p
        if stop:
            break

def steepest(initial, gradient, value, stepsize=0.1, maxsteps=30, maxit=1000,
             tol=10**-5, stats=None):
    r"""
    Minimize an objective function using the Steepest Descent method.

    The increment to the initial estimate of the parameter vector
    :math:`\bar{p}` is calculated by (Kelley, 1999)

    .. math::

        \Delta\bar{p} = -\lambda\bar{g}

    where :math:`\lambda` is the step size and :math:`\bar{g}` is the gradient
    vector.

    The step size can be determined thought a line search algorithm using the
    Armijo rule (Kelley, 1999). In this case,

    .. math::

        \lambda = \beta^m

    where :math:`1 > \beta > 0` and :math:`m \ge 0` is an integer that controls
    the step size. The line search finds the smallest :math:`m` that satisfies
    the Armijo rule

    .. math::

        \phi(\bar{p} + \Delta\bar{p}) - \Gamma(\bar{p}) <
        \alpha\beta^m ||\bar{g}(\bar{p})||^2

    where :math:`\phi(\bar{p})` is the objective function evaluated at
    :math:`\bar{p}` and :math:`\alpha = 10^{-4}`.

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
    * maxsteps : int
        The maximum number of times to try to take a step before giving up.
    * lamb : float
        Initial amount of step regularization. The larger this is, the more the
        algorithm will resemble Steepest Descent in the initial iterations.
    * dlamb : float
        Factor by which *lamb* is divided or multiplied when taking steps.
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
    p = initial
    misfit = value(p)
    if stats is not None:
        stats['method'] = 'Steepest Descent'
        stats['iterations'] = 0
        stats['misfit/iteration'] = [misfit]
        stats['step search stagnation'] = False
        stats['trial steps/iteration'] = []
    yield p
    # This is a mystic parameter of the Armijo rule
    alpha = 10**(-4)
    for iteration in xrange(maxit):
        grad = gradient(p)
        # Calculate now to avoid computing inside the loop
        gradnorm = numpy.linalg.norm(grad)**2
        stagnation = True
        # Determine the best step size
        for i in xrange(maxsteps):
            factor = stepsize**i
            newp = p - factor*grad
            newmisfit = value(newp)
            if newmisfit - misfit < alpha*factor*gradnorm:
                stagnation = False
                break
        if stagnation:
            stop = True
            if stats is not None:
                stats['step search stagnation'] = True
        else:
            stop = abs((newmisfit - misfit)/misfit) < tol
            p = newp
            misfit = newmisfit
            if stats is not None:
                stats['iterations'] += 1
                stats['misfit/iteration'].append(misfit)
                stats['trial steps/iteration'].append(i + 1)
            yield p
        if stop:
            break
