"""
Methods to optimize a given objective function.

**Gradient descent**

* :func:`~fatiando.inversion.solvers.linear`: Solver for a linear problem
* :func:`~fatiando.inversion.solvers.newton`: Newton's method
* :func:`~fatiando.inversion.solvers.levmarq`: Levemberg-Marquardt algorithm
* :func:`~fatiando.inversion.solvers.steepest`: Steepest Descent method

**Heuristic methods**

* :func:`~fatiando.inversion.solvers.acor`: ACO-R: Ant Colony Optimization for
  Continuous Domains (Socha and Dorigo, 2008)


**References**

Socha, K., and M. Dorigo (2008), Ant colony optimization for continuous
domains, European Journal of Operational Research, 185(3), 1155-1173,
doi:10.1016/j.ejor.2006.06.046.


----

"""
from __future__ import division
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
        diag[diag < 10 ** -10] = 10 ** -10
        precond = scipy.sparse.diags(1. / diag, 0).tocsr()
        hessian = safe_dot(precond, hessian)
        gradient = safe_dot(precond, gradient)
    p = safe_solve(hessian, -gradient)
    return p


def newton(hessian, gradient, value, initial, maxit=30, tol=10 ** -5,
           precondition=True):
    r"""
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

    * hessian : function
        A function that returns the Hessian matrix of the objective function
        when given a parameter vector.
    * gradient : function
        A function that returns the gradient vector of the objective function
        when given a parameter vector.
    * value : function
        A function that returns the value of the objective function evaluated
        at a given parameter vector.
    * initial : 1d-array
        The initial estimate for the gradient descent.
    * maxit : int
        The maximum number of iterations allowed.
    * tol : float
        The convergence criterion. The lower it is, the more steps are
        permitted.
    * precondition : True or False
        If True, will use Jacobi preconditioning.

    Yields:

    * estimate : 1d-array
        The estimated parameter vector at the current iteration.

    """
    p = numpy.array(initial, dtype=numpy.float)
    misfit = value(p)
    yield p
    for iteration in xrange(maxit):
        hess = hessian(p)
        grad = gradient(p)
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = scipy.sparse.diags(1. / diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            grad = safe_dot(precond, grad)
        p = p + safe_solve(hess, -grad)
        newmisfit = value(p)
        if newmisfit > misfit or abs((newmisfit - misfit) / misfit) < tol:
            break
        misfit = newmisfit
        yield p


def levmarq(hessian, gradient, value, initial, maxit=30, maxsteps=10, lamb=10,
            dlamb=2, tol=10 ** -5, precondition=True):
    r"""
    Minimize an objective function using the Levemberg-Marquardt algorithm.

    Parameters:

    * hessian : function
        A function that returns the Hessian matrix of the objective function
        when given a parameter vector.
    * gradient : function
        A function that returns the gradient vector of the objective function
        when given a parameter vector.
    * value : function
        A function that returns the value of the objective function evaluated
        at a given parameter vector.
    * initial : 1d-array
        The initial estimate for the gradient descent.
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

    Yields:

    * estimate : 1d-array
        The estimated parameter vector at the current iteration.

    """
    p = numpy.array(initial, dtype=numpy.float)
    misfit = value(p)
    yield p
    for iteration in xrange(maxit):
        hess = hessian(p)
        minus_gradient = -gradient(p)
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = scipy.sparse.diags(1. / diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            minus_gradient = safe_dot(precond, minus_gradient)
        stagnation = True
        diag = scipy.sparse.diags(safe_diagonal(hess), 0).tocsr()
        for step in xrange(maxsteps):
            newp = p + safe_solve(hess + lamb * diag, minus_gradient)
            newmisfit = value(newp)
            if newmisfit >= misfit:
                if lamb < 10 ** 15:
                    lamb *= dlamb
            else:
                if lamb > 10 ** -15:
                    lamb /= dlamb
                stagnation = False
                break
        if stagnation:
            stop = True
        else:
            stop = newmisfit > misfit or abs(
                (newmisfit - misfit) / misfit) < tol
            p = newp
            misfit = newmisfit
            yield p
        if stop:
            break


def steepest(gradient, value, initial, maxit=1000, maxsteps=30, stepsize=0.1,
             tol=10 ** -5):
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

    * gradient : function
        A function that returns the gradient vector of the objective function
        when given a parameter vector.
    * value : function
        A function that returns the value of the objective function evaluated
        at a given parameter vector.
    * initial : 1d-array
        The initial estimate for the gradient descent.
    * maxit : int
        The maximum number of iterations allowed.
    * maxsteps : int
        The maximum number of times to try to take a step before giving up.
    * stepsize : float
        Initial amount of step step size.
    * tol : float
        The convergence criterion. The lower it is, the more steps are
        permitted.

    Yields:

    * estimate : 1d-array
        The estimated parameter vector at the current iteration.

    References:

    Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.

    """
    p = numpy.array(initial, dtype=numpy.float)
    misfit = value(p)
    yield p
    # This is a mystic parameter of the Armijo rule
    alpha = 10 ** (-4)
    for iteration in xrange(maxit):
        grad = gradient(p)
        # Calculate now to avoid computing inside the loop
        gradnorm = numpy.linalg.norm(grad) ** 2
        stagnation = True
        # Determine the best step size
        for i in xrange(maxsteps):
            factor = stepsize ** i
            newp = p - factor * grad
            newmisfit = value(newp)
            if newmisfit - misfit < alpha * factor * gradnorm:
                stagnation = False
                break
        if stagnation:
            stop = True
        else:
            stop = abs((newmisfit - misfit) / misfit) < tol
            p = newp
            misfit = newmisfit
            yield p
        if stop:
            break


def acor(value, bounds, nparams, nants=None, archive_size=None, maxit=1000,
         diverse=0.5, evap=0.85, seed=None):
    """
    Minimize the objective function using ACO-R.

    ACO-R stands for Ant Colony Optimization for Continuous Domains (Socha and
    Dorigo, 2008).

    Parameters:

    * value : function
        Returns the value of the objective function at a given parameter vector
    * bounds : list
        The bounds of the search space. If only two values are given, will
        interpret as the minimum and maximum, respectively, for all parameters.
        Alternatively, you can given a minimum and maximum for each parameter,
        e.g., for a problem with 3 parameters you could give
        `bounds = [min1, max1, min2, max2, min3, max3]`.
    * nparams : int
        The number of parameters that the objective function takes.
    * nants : int
        The number of ants to use in the search. Defaults to the number of
        parameters.
    * archive_size : int
        The number of solutions to keep in the solution archive. Defaults to
        10 x nants
    * maxit : int
        The number of iterations to run.
    * diverse : float
        Scalar from 0 to 1, non-inclusive, that controls how much better
        solutions are favored when constructing new ones.
    * evap : float
        The pheromone evaporation rate (evap > 0). Controls how spread out the
        search is.
    * seed : None or int
        Seed for the random number generator.

    Yields:

    * estimate : 1d-array
        The best estimate at each iteration

    """
    numpy.random.seed(seed)
    # Set the defaults for number of ants and archive size
    if nants is None:
        nants = nparams
    if archive_size is None:
        archive_size = 10 * nants
    # Check is giving bounds for each parameter or one for all
    bounds = numpy.array(bounds)
    if bounds.size == 2:
        low, high = bounds
        archive = numpy.random.uniform(low, high, (archive_size, nparams))
    else:
        archive = numpy.empty((archive_size, nparams))
        bounds = bounds.reshape((nparams, 2))
        for i, bound in enumerate(bounds):
            low, high = bound
            archive[:, i] = numpy.random.uniform(low, high, archive_size)
    # Compute the inital pheromone trail based on the objetive function value
    trail = numpy.fromiter((value(p) for p in archive), dtype=numpy.float)
    # Sort the archive
    order = numpy.argsort(trail)
    archive = [archive[i] for i in order]
    trail = trail[order].tolist()
    # The first of the archive is the best solution found
    yield archive[0]
    # Compute the weights (probabilities) of the solutions in the archive
    amp = 1. / (diverse * archive_size * numpy.sqrt(2 * numpy.pi))
    variance = 2 * diverse ** 2 * archive_size ** 2
    weights = amp * numpy.exp(-numpy.arange(archive_size) ** 2 / variance)
    weights /= numpy.sum(weights)
    for iteration in xrange(maxit):
        for k in xrange(nants):
            # Sample the propabilities to produce new estimates
            ant = numpy.empty(nparams, dtype=numpy.float)
            # 1. Choose a pdf from the archive
            pdf = numpy.searchsorted(
                numpy.cumsum(weights),
                numpy.random.uniform())
            for i in xrange(nparams):
                # 2. Get the mean and stddev of the chosen pdf
                mean = archive[pdf][i]
                std = (evap / (archive_size - 1)) * numpy.sum(
                    abs(p[i] - archive[pdf][i]) for p in archive)
                # 3. Sample the pdf until the samples are in bounds
                for atempt in xrange(100):
                    ant[i] = numpy.random.normal(mean, std)
                    if bounds.size == 2:
                        low, high = bounds
                    else:
                        low, high = bounds[i]
                    if ant[i] >= low and ant[i] <= high:
                        break
            pheromone = value(ant)
            # Place the new estimate in the archive
            place = numpy.searchsorted(trail, pheromone)
            if place == archive_size:
                continue
            trail.insert(place, pheromone)
            trail.pop()
            archive.insert(place, ant)
            archive.pop()
        yield archive[0]
