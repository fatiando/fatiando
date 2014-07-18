"""
The base classes for inverse problem solving.

All classes derive from :class:`~fatiando.inversion.base.Objective`. This class
represents an objective function, a scalar function of a parameter vector.

The main powerhouse of this module is the
:class:`~fatiando.inversion.base.Misfit` class. It represents a data misfit
function and knows how to fit a specified model to data using various solvers
(see :mod:`~fatiando.inversion.solvers`). A model is specified by subclassing
Misfit and implementing the
:meth:`~fatiando.inversion.base.Misfit._get_predicted` and
:meth:`~fatiando.inversion.base.Misfit._get_jacobian` methods, which are
problem specific.

See :mod:`fatiando.inversion` for examples, regularization, and more.

----

"""
from __future__ import division
import hashlib
import types
import copy
import numpy
import scipy.sparse

from .solvers import linear, levmarq, steepest, newton, acor
from ..utils import safe_dot


class Objective(object):

    """
    An objective function for an inverse problem.

    Objective functions know how to calculate their value, gradient and/or
    Hessian matrix for a given parameter vector *p*. The methods that implement
    these should have the following format::

        def _get_value(self, p):
            '''
            Calculate the value of the objetive function.

            Parameters:

            * p : 1d-array or None
                The parameter vector.

            Returns:

            * value : float

            '''
            ...

        def _get_hessian(self, p):
            '''
            Calculates the Hessian matrix.

            Parameters:

            * p : 1d-array
                The parameter vector where the Hessian is evaluated

            Returns:

            * hessian : 2d-array

            '''
            ...

        def _get_gradient(self, p):
            '''
            The gradient vector.

            Parameters:

            * p : 1d-array
                The parameter vector where the gradient is evaluated

            Returns:

            * gradient : 1d-array

            '''
            ...

    These methods are problem specific and need to be implemented when
    subclassing *Objective*.

    *Objective* has methods that find the parameter vector that minimizes it:

    * :meth:`~fatiando.inversion.base.Objective.newton`
    * :meth:`~fatiando.inversion.base.Objective.levmarq`
    * :meth:`~fatiando.inversion.base.Objective.steepest`
    * :meth:`~fatiando.inversion.base.Objective.acor`

    Parameters:

    * nparams : int
        The number of parameters the objective function takes.
    * islinear : True or False
        Wether the functions is linear with respect to the parameters.

    Operations:

    For joint inversion and regularization, you can add *Objective*
    instances together and multiply them by scalars (i.e., regularization
    parameters):

    >>> class MyObjective(Objective):
    ...     def __init__(self, scale):
    ...         super(MyObjective, self).__init__(10, True)
    ...         self._scalar = scale
    ...     def _get_value(self, p):
    ...         return self._scalar*p
    >>> a = MyObjective(2)
    >>> a.value(3)
    6
    >>> b = MyObjective(-3)
    >>> b.value(3)
    -9
    >>> c = a + b
    >>> c.value(3)
    -3
    >>> d = 0.5*c
    >>> d.value(3)
    -1.5
    >>> e = a + 2*b
    >>> e.value(3)
    -12

    """

    def __init__(self, nparams, islinear):
        self.islinear = islinear
        self.nparams = nparams
        self.ndata = 0
        self._scale = None
        self._parents = None

    def __repr__(self):
        return 'Objective instance'

    def value(self, p):
        """
        The value (scalar) of this objective function at *p*

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of this Objective function. If this is the sum of 1 or
            more objective functions, value will be the sum of the values.

        """
        if self._parents is None:
            return self._get_value(p)
        else:
            if self._scale is None:
                obj1, obj2 = self._parents
                return obj1.value(p) + obj2.value(p)
            else:
                assert len(self._parents) == 1, \
                    'Result of multiplying Objective produces > one parent.'
                return self._scale * self._parents[0].value(p)

    def gradient(self, p):
        """
        The gradient vector of this objective function at *p*

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * gradient : 1d-array
            The gradient of this Objective function. If this is the sum of 1 or
            more objective functions, gradient will be the sum of the gradients

        """
        if self._parents is None:
            return self._get_gradient(p)
        else:
            if self._scale is None:
                obj1, obj2 = self._parents
                return obj1.gradient(p) + obj2.gradient(p)
            else:
                assert len(self._parents) == 1, \
                    'Result of multiplying Objective produces > one parent.'
                return self._scale * self._parents[0].gradient(p)

    def hessian(self, p):
        """
        The Hessian matrix of this objective function at *p*

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * hessian : 2d-array
            The Hessian of this Objective function. If this is the sum of 1 or
            more objective functions, hessian will be the sum of the Hessians

        """
        if self._parents is None:
            return self._get_hessian(p)
        else:
            if self._scale is None:
                obj1, obj2 = self._parents
                return obj1.hessian(p) + obj2.hessian(p)
            else:
                assert len(self._parents) == 1, \
                    'Result of multiplying Objective produces > one parent.'
                return self._scale * self._parents[0].hessian(p)

    def __add__(self, other):
        """
        Examples:

        >>> class MyObjective(Objective):
        ...     def __init__(self, n, scale):
        ...         super(MyObjective, self).__init__(n, True)
        ...         self._scalar = scale
        ...     def _get_value(self, p):
        ...         return self._scalar*p
        >>> a = MyObjective(10, 2)
        >>> b = MyObjective(10, -3)
        >>> c = a + b
        >>> c.value(3)
        -3
        >>> a.value(3) + b.value(3)
        -3

        Every Objective should be a copy:

        >>> c is a
        False
        >>> c is b
        False
        >>> c._parents[0] is a
        False
        >>> c._parents[1] is b
        False

        Modifying the 2 Objectives should not alter the sum:

        >>> a._scalar = 10
        >>> b._scalar = 20
        >>> a.value(3) + b.value(3)
        90
        >>> c.value(3)
        -3

        """
        if self.nparams != other.nparams:
            raise ValueError(
                "Can only add functions with same number of parameters")
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        tmp = copy.copy(self)
        tmp._scale = None
        tmp._parents = [copy.copy(self), copy.copy(other)]
        return tmp

    def __mul__(self, other):
        """
        Examples:

        >>> class MyObjective(Objective):
        ...     def __init__(self, n, scale):
        ...         super(MyObjective, self).__init__(n, True)
        ...         self._scalar = scale
        ...     def _get_value(self, p):
        ...         return self._scalar*p
        >>> a = MyObjective(10, 2)
        >>> b = MyObjective(10, -3)
        >>> d = 0.5*a
        >>> d.value(3)
        3.0
        >>> e = a + 2*b
        >>> e.value(3)
        -12
        >>> f = 3*a + b
        >>> f.value(3)
        9

        Every Objective should be a copy:

        >>> d is a
        False
        >>> e is a
        False
        >>> e is b
        False
        >>> f is a
        False
        >>> f is b
        False

        Modifying the 2 Objectives should not alter the multiplication:

        >>> a._scalar = 10
        >>> b._scalar = 20
        >>> 0.5*a.value(3)
        15.0
        >>> d.value(3)
        3.0
        >>> a.value(3) + 2*b.value(3)
        150
        >>> e.value(3)
        -12
        >>> f.value(3)
        9

        """
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Can only multiply a Objective by a float or int')
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        tmp = copy.copy(self)
        tmp._scale = other
        tmp._parents = [copy.copy(self)]
        return tmp

    def __rmul__(self, other):
        return self.__mul__(other)

    def levmarq(self, initial, maxit=30, maxsteps=10, lamb=1, dlamb=2,
                tol=10 ** -5, precondition=True, iterate=False):
        """
        Solve using the Levemberg-Marquardt algorithm.

        See :func:`fatiando.inversion.solvers.levmarq` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * maxsteps : int
            The maximum number of times to try to take a step before giving
            up
        * lamb : float
            Initial amount of step regularization. The larger this is, the
            more the algorithm will resemble Steepest Descent in the initial
            iterations.
        * dlamb : float
            Factor by which *lamb* is divided or multiplied when taking steps
        * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted
        * precondition : True or False
            If True, will use Jacobi preconditioning
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = levmarq(self.hessian, self.gradient, self.value, initial,
                         maxit=maxit, maxsteps=maxsteps, lamb=lamb,
                         dlamb=dlamb, tol=tol, precondition=precondition)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def newton(self, initial, maxit=30, tol=10 ** -5, precondition=True,
               iterate=False):
        """
        Minimize an objective function using Newton's method.

        See :func:`fatiando.inversion.solvers.newton` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted.
        * precondition : True or False
            If True, will use Jacobi preconditioning.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = newton(self.hessian, self.gradient, self.value, initial,
                        maxit=maxit, tol=tol, precondition=precondition)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def steepest(self, initial, stepsize=0.1, maxsteps=30, maxit=1000,
                 tol=10 ** -5, iterate=False):
        """
        Minimize an objective function using the Steepest Descent method.

        See :func:`fatiando.inversion.solvers.steepest` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * maxsteps : int
            The maximum number of times to try to take a step before giving
            up
        * stepsize : float
            Initial amount of step step size.
        * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = steepest(self.gradient, self.value, initial, maxit=maxit,
                          maxsteps=maxsteps, stepsize=stepsize, tol=tol)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def acor(self, bounds, nants=None, archive_size=None, maxit=1000,
             diverse=0.5, evap=0.85, seed=None, iterate=False):
        """
        Minimize the objective function using ACO-R.

        See :func:`fatiando.inversion.solvers.acor` for more details.

        Parameters:

        * bounds : list
            The bounds of the search space. If only two values are given,
            will interpret as the minimum and maximum, respectively, for all
            parameters.
            Alternatively, you can given a minimum and maximum for each
            parameter, e.g., for a problem with 3 parameters you could give
            `bounds = [min1, max1, min2, max2, min3, max3]`.
        * nants : int
            The number of ants to use in the search. Defaults to the number
            of parameters.
        * archive_size : int
            The number of solutions to keep in the solution archive.
            Defaults to 10 x nants
        * maxit : int
            The number of iterations to run.
        * diverse : float
            Scalar from 0 to 1, non-inclusive, that controls how much better
            solutions are favored when constructing new ones.
        * evap : float
            The pheromone evaporation rate (evap > 0). Controls how spread
            out the search is.
        * seed : None or int
            Seed for the random number generator.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The best estimate

        """
        solver = acor(self.value, bounds, self.nparams, nants=nants,
                      archive_size=archive_size, maxit=maxit, diverse=diverse,
                      evap=evap, seed=seed)
        if iterate:
            return solver
        for p in solver:
            continue
        return p


class Misfit(Objective):

    r"""
    An l2-norm data-misfit function.

    This is a kind of objective function that measures the misfit between
    observed data :math:`\bar{d}^o` and data predicted by a set of model
    parameters :math:`\bar{d} = \bar{f}(\bar{p})`.

    The l2-norm data-misfit is defined as:

    .. math::

        \phi(\bar{p}) = \dfrac{\bar{r}^T\bar{r}}{N}

    where :math:`\bar{r} = \bar{d}^o - \bar{d}` is the residual vector and
    :math:`N` is the number of data.

    This class inherits the solvers from
    :class:`~fatiando.inversion.base.Objective` that estimate a parameter
    vector :math:`\bar{p}` that minimizes it.
    See :class:`~fatiando.inversion.base.Objective` for more details.

    When subclassing this class, you must implement two methods:

    * ``_get_predicted(self, p)``: calculates the predicted data
      :math:`\bar{d}` for a given parameter vector ``p``
    * ``_get_jacobian(self, p)``: calculates the Jacobian matrix of
      :math:`\bar{f}(\bar{p})` evaluated at ``p``

    If :math:`\bar{f}` is linear, then the Jacobian will be cached in memory so
    that it is only calculated once when using the class multiple times. So
    solving the same problem with different methods or using an iterative
    method doesn't have the penalty of recalculating the Jacobian.

    .. tip::

        See :mod:`fatiando.inversion` for examples of usage.

    Parameters:

    * data : 1d-array
        The observed data vector :math:`\bar{d}^o`
    * positional : dict
        A dictionary with the positional arguments of the data, for example, x,
        y coordinates, depths, etc. Keys should the string name of the argument
        and values should be 1d-arrays with the same size as *data*.
    * model : dict
        A dictionary with the model parameters, like the mesh, physical
        properties, etc.
    * nparams : int
        The number of parameters in parameter vector :math:`\bar{p}`
    * weights : 1d-array
        Weights to be applied to the each element in *data* when computing the
        l2-norm. Effectively the diagonal of a matrix :math:`\bar{\bar{W}}`
        such that :math:`\phi = \bar{r}^T\bar{\bar{W}}\bar{r}`
    * islinear : True or False
        Whether :math:`\bar{f}` is linear or not.

    """

    def __init__(self, data, positional, model, nparams, weights=None,
                 islinear=False):
        super(Misfit, self).__init__(nparams, islinear=islinear)
        self.data = data
        self.ndata = len(data)
        self.positional = positional
        self.model = model
        # To cache the latest computations (or for linear problems)
        self._cache = {
            'predicted': {'hash': '', 'array': None},
            'jacobian': {'hash': '', 'array': None},
            'hessian': {'hash': '', 'array': None}}
        # Set default arguments for fit
        self.default_solver_args = {
            'linear': {'precondition': True},
            'newton': {'initial': None,
                       'maxit': 30,
                       'tol': 10 ** -5,
                       'precondition': True},
            'levmarq': {'initial': None,
                        'maxit': 30,
                        'maxsteps': 10,
                        'lamb': 1,
                        'dlamb': 2,
                        'tol': 10 ** -5,
                        'precondition': True},
            'steepest': {'initial': None,
                         'stepsize': 0.1,
                         'maxsteps': 30,
                         'maxit': 1000,
                         'tol': 10 ** -5},
            'acor': {'bounds': None,
                     'nants': None,
                     'archive_size': None,
                     'maxit': 1000,
                     'diverse': 0.5,
                     'evap': 0.85,
                     'seed': None}}
        # Data weights
        self.weights = None
        if weights is not None:
            self.set_weights(weights)
        # So that don't need to use config on linear problems
        if islinear:
            self.config(method='linear')

    def hasher(self, x):
        return hashlib.sha1(x).hexdigest()

    def _clear_cache(self):
        "Reset the cached matrices"
        self._cache = {
            'predicted': {'hash': '', 'array': None},
            'jacobian': {'hash': '', 'array': None},
            'hessian': {'hash': '', 'array': None}}

    def __repr__(self):
        return 'Misfit instance'

    def subset(self, indices):
        """
        Produce a shallow copy of this object with only a subset of the data.

        Additionally cuts the *positional* arguments and Jacobian matrix (if it
        is present in the cache).

        Parameters:

        * indices : list of ints or 1d-array of bools
            The indices that correspond to the subset.

        Returns:

        * subset : Misfit
            A copy of this object

        Examples:

        >>> import numpy as np
        >>> solver = Misfit(np.array([1, 2, 3, 10]),
        ...                 positional={'x':np.array([4, 5, 6, 100])},
        ...                 model={'d':12},
        ...                 nparams=2)
        >>> # Populate the cache to show what happens to it
        >>> solver._cache['jacobian']['array'] = np.array(
        ...     [[1, 2],
        ...      [3, 4],
        ...      [5, 6],
        ...      [7, 8]])
        >>> solver._cache['hessian']['array'] = np.ones((2, 2))
        >>> solver._cache['predicted']['array'] = np.ones(4)
        >>> # Get the subset
        >>> sub = solver.subset([1, 3])
        >>> sub.ndata
        2
        >>> sub.data
        array([ 2, 10])
        >>> sub.positional
        {'x': array([  5, 100])}
        >>> sub.model
        {'d': 12}
        >>> sub._cache['jacobian']['array']
        array([[3, 4],
               [7, 8]])
        >>> sub._cache['hessian']['array'] is None
        True
        >>> sub._cache['predicted']['array'] is None
        True
        >>> # The original solver stays the same
        >>> solver.data
        array([ 1,  2,  3, 10])
        >>> solver.positional
        {'x': array([  4,   5,   6, 100])}
        >>> solver.model
        {'d': 12}
        >>> solver._cache['jacobian']['array']
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        >>> solver._cache['hessian']['array']
        array([[ 1.,  1.],
               [ 1.,  1.]])
        >>> solver._cache['predicted']['array']
        array([ 1.,  1.,  1.,  1.])
        >>> # Can also use a numpy array of booleans
        >>> sub = solver.subset(np.array([False, True, False, True]))
        >>> sub.ndata
        2
        >>> sub.data
        array([ 2, 10])
        >>> sub.positional
        {'x': array([  5, 100])}
        >>> sub.model
        {'d': 12}
        >>> sub._cache['jacobian']['array']
        array([[3, 4],
               [7, 8]])
        >>> sub._cache['hessian']['array'] is None
        True
        >>> sub._cache['predicted']['array'] is None
        True

        """
        sub = copy.copy(self)
        sub.model = self.model.copy()
        sub._cache = self._cache.copy()
        sub.data = sub.data[indices]
        sub.positional = dict((k, sub.positional[k][indices])
                              for k in sub.positional)
        sub._clear_cache()
        if self._cache['jacobian']['array'] is not None:
            sub._cache['jacobian']['array'] = \
                self._cache['jacobian']['array'][indices]
        sub.ndata = sub.data.size
        return sub

    def set_weights(self, weights):
        """
        Set the data weights array.

        See :class:`~fatiando.inversion.base.Misfit` for more information.

        Parameters:

        * weights : 1d-array
            A vector with the data weights.

        """
        self.weights = scipy.sparse.diags(weights, 0)
        # Weights change the Hessian
        self._cache['hessian'] = {'hash': '', 'array': None}
        return self

    def residuals(self, p=None):
        """
        Calculate the residuals vector (observed - predicted data).

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the residuals. If None, will
            use the current estimate stored in ``estimate_``.

        Returns:

        * residuals : 1d-array or list of 1d-arrays
            The residual vector. If this is the sum of 1 or more Misfit
            instances, will return the residual vector from each of the summed
            misfits in the order of the sum.

        """
        if p is None:
            p = self.p_
        if self._parents is None:
            res = self.data - self.predicted(p)
        else:
            res = []
            for o in self._parents:
                if hasattr(o, 'residuals'):
                    aux = o.residuals(p)
                    if isinstance(aux, list):
                        res.extend(aux)
                    else:
                        res.append(aux)
            if len(res) == 1:
                res = res[0]
        return res

    def predicted(self, p=None):
        """
        Calculate the predicted data for a given parameter vector.

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the predicted data. If None,
            will use the current estimate stored in ``estimate_``.

        Returns:

        * predicted : 1d-array or list of 1d-arrays
            The predicted data. If this is the sum of 1 or more Misfit
            instances, will return the predicted data from each of the summed
            misfits in the order of the sum.

        """
        if p == 'null':
            pred = 0
        else:
            if p is None:
                p = self.p_
            if self._parents is None:
                hash = self.hasher(p)
                if hash != self._cache['predicted']['hash']:
                    self._cache['predicted']['array'] = self._get_predicted(p)
                    self._cache['predicted']['hash'] = hash
                pred = self._cache['predicted']['array']
            else:
                pred = []
                for o in self._parents:
                    if hasattr(o, 'predicted'):
                        aux = o.predicted(p)
                        if isinstance(aux, list):
                            pred.extend(aux)
                        else:
                            pred.append(aux)
                if len(pred) == 1:
                    pred = pred[0]
        return pred

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix evaluated at a given parameter vector.

        The Jacobian matrix is cached in memory, so passing the same
        parameter vector again will not trigger a re-calculation. However, only
        one matrix is cached at a time.

        Parameters:

        * p : 1d-array or None
            The parameter vector. If the problem is linear, pass ``None``

        Returns:

        * jacobian : 2d-array
            The Jacobian matrix

        """
        if self.islinear:
            hash = ''
        else:
            hash = self.hasher(p)
        if (hash != self._cache['jacobian']['hash'] or
                self._cache['jacobian']['array'] is None):
            self._cache['jacobian']['array'] = self._get_jacobian(p)
            self._cache['jacobian']['hash'] = hash
        return self._cache['jacobian']['array']

    def _get_value(self, p):
        r"""
        Calculate the value of the misfit for a given parameter vector.

        The value is given by:

        .. math::

            \phi(\bar{p}) = \dfrac{\bar{r}^T\bar{\bar{W}}\bar{r}}{N}


        where :math:`\bar{r}` is the residual vector and :math:`bar{\bar{W}}`
        are optional data weights.

        Parameters:

        * p : 1d-array or None
            The parameter vector.

        Returns:

        * value : float
            The value of the misfit function.

        """
        if self.weights is None:
            return numpy.linalg.norm(
                self.data - self.predicted(p)
            ) ** 2 / self.ndata
        else:
            return numpy.sum(self.weights * (
                (self.data - self.predicted(p)) ** 2)
            ) / self.ndata

    def _get_hessian(self, p):
        r"""
        The Hessian of the misfit function with respect to the parameters.

        Calculated using the Gauss approximation:

        .. math::

            \bar{\bar{H}} \approx 2\bar{\bar{J}}^T\bar{\bar{J}}

        where :math:`\bar{\bar{J}}` is the Jacobian matrix.

        For linear problems, the Hessian matrix is cached in memory, so calling
        this method again will not trigger a re-calculation.

        Parameters:

        * p : 1d-array
            The parameter vector where the Hessian is evaluated

        Returns:

        * hessian : 2d-array
            The Hessian matrix

        """
        if self.islinear and self._cache['hessian']['array'] is not None:
            hessian = self._cache['hessian']['array']
        else:
            jacobian = self.jacobian(p)
            if self.weights is None:
                hessian = (2 / self.ndata) * safe_dot(jacobian.T, jacobian)
            else:
                hessian = (2 / self.ndata) * safe_dot(
                    jacobian.T, self.weights * jacobian)
            if self.islinear:
                self._cache['hessian']['array'] = hessian
        return hessian

    def _get_gradient(self, p):
        r"""
        The gradient vector of the misfit function.

        .. math::

            \bar{g} = -2\bar{\bar{J}}^T\bar{r}

        where :math:`\bar{\bar{J}}` is the Jacobian matrix and :math:`\bar{r}`
        is the residual vector.

        Parameters:

        * p : 1d-array
            The parameter vector where the gradient is evaluated

        Returns:

        * gradient : 1d-array
            The gradient vector.

        """
        jacobian = self.jacobian(p)
        if self.weights is None:
            grad = (-2 / self.ndata) * safe_dot(
                jacobian.T, self.data - self.predicted(p))
        else:
            grad = (-2 / self.ndata) * safe_dot(
                jacobian.T, self.weights * (self.data - self.predicted(p)))
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break loose
            grad = numpy.array(grad).ravel()
        return grad

    # Addition needs some tweaks
    def __add__(self, other):
        """
        Examples:

        Added Misfits should not share a cache. Bellow are some tests for this
        behaviour:

        >>> import numpy as np
        >>> class MyMisfit(Misfit):
        ...     def __init__(self, data, factor):
        ...         super(MyMisfit, self).__init__(data, {}, {}, 2,
        ...                                        islinear=True)
        ...         self._factor = factor
        ...     def _get_predicted(self, p):
        ...         return 2*p
        ...     def _get_jacobian(self, p):
        ...         shape = (self.ndata, a.nparams)
        ...         return self._factor*np.ones(shape)
        >>> a = MyMisfit([1, 2, 3], np.array([1, 2]))
        >>> a._cache['jacobian']['array'] is None
        True
        >>> a._cache['hessian']['array'] is None
        True
        >>> b = MyMisfit([1, 2, 3], 1)
        >>> b._cache['jacobian']['array'] is None
        True
        >>> b._cache['hessian']['array'] is None
        True
        >>> c = a + b
        >>> c._cache['jacobian']['array'] is None
        True
        >>> c._cache['hessian']['array'] is None
        True
        >>> [p._cache['jacobian']['array'] is None for p in c._parents]
        [True, True]
        >>> [p._cache['hessian']['array'] is None for p in c._parents]
        [True, True]

        Calling a.jacobian and a.hessian should fill the cache of a but not
        of the parents of c:

        >>> p = np.array([1, 2])
        >>> a.jacobian(p)
        array([[ 1.,  2.],
               [ 1.,  2.],
               [ 1.,  2.]])
        >>> a._cache['jacobian']['array']
        array([[ 1.,  2.],
               [ 1.,  2.],
               [ 1.,  2.]])
        >>> [p._cache['jacobian']['array'] is None for p in c._parents]
        [True, True]
        >>> a.hessian(p)
        array([[ 2.,  4.],
               [ 4.,  8.]])
        >>> a._cache['hessian']['array']
        array([[ 2.,  4.],
               [ 4.,  8.]])
        >>> [p._cache['hessian']['array'] is None for p in c._parents]
        [True, True]

        The same goes for b:

        >>> b.jacobian(p)
        array([[ 1.,  1.],
               [ 1.,  1.],
               [ 1.,  1.]])
        >>> b._cache['jacobian']['array']
        array([[ 1.,  1.],
               [ 1.,  1.],
               [ 1.,  1.]])
        >>> [p._cache['jacobian']['array'] is None for p in c._parents]
        [True, True]
        >>> b.hessian(p)
        array([[ 2.,  2.],
               [ 2.,  2.]])
        >>> b._cache['hessian']['array']
        array([[ 2.,  2.],
               [ 2.,  2.]])
        >>> [p._cache['hessian']['array'] is None for p in c._parents]
        [True, True]

        Just as well, calling c.hessian should not fill the cache of a or b:

        >>> a._clear_cache()
        >>> b._clear_cache()
        >>> c.hessian(p)
        array([[  4.,   6.],
               [  6.,  10.]])
        >>> c._parents[0]._cache['hessian']['array']
        array([[ 2.,  4.],
               [ 4.,  8.]])
        >>> c._parents[1]._cache['hessian']['array']
        array([[ 2.,  2.],
               [ 2.,  2.]])
        >>> a._cache['hessian']['array'] is None
        True
        >>> b._cache['hessian']['array'] is None
        True

        But creating c after filling the cache of a and b should result in c
        whose parents have their cache filled:

        >>> a.hessian(p)
        array([[ 2.,  4.],
               [ 4.,  8.]])
        >>> c = a + b
        >>> c._parents[0]._cache['jacobian']['array']
        array([[ 1.,  2.],
               [ 1.,  2.],
               [ 1.,  2.]])
        >>> c._parents[0]._cache['hessian']['array']
        array([[ 2.,  4.],
               [ 4.,  8.]])
        >>> c._parents[1]._cache['jacobian']['array'] is None
        True
        >>> c._parents[1]._cache['hessian']['array'] is None
        True
        >>> b.hessian(p)
        array([[ 2.,  2.],
               [ 2.,  2.]])
        >>> c = a + b
        >>> c._parents[1]._cache['jacobian']['array']
        array([[ 1.,  1.],
               [ 1.,  1.],
               [ 1.,  1.]])
        >>> c._parents[1]._cache['hessian']['array']
        array([[ 2.,  2.],
               [ 2.,  2.]])

        This should also work when adding something that is not a Misfit:

        >>> from fatiando.inversion.regularization import Damping
        >>> c = Damping(2)
        >>> a._clear_cache()
        >>> b._clear_cache()
        >>> d = a + b + 3*c
        >>> d.hessian(p)
        matrix([[ 10.,   6.],
                [  6.,  16.]])
        >>> a._cache['jacobian']['array'] is None
        True
        >>> b._cache['jacobian']['array'] is None
        True
        >>> d._parents[0]._parents[0]._cache['jacobian']['array']
        array([[ 1.,  2.],
               [ 1.,  2.],
               [ 1.,  2.]])
        >>> d._parents[0]._parents[1]._cache['jacobian']['array']
        array([[ 1.,  1.],
               [ 1.,  1.],
               [ 1.,  1.]])

        """
        tmp = super(Misfit, self).__add__(other)
        # Cache is not shared. This would cause a problem if I made 2 or more
        # sums and then tried to alter the cache of both at the same time (in a
        # parallel run, for example).
        # A shallow copy is used so the cached matrices is not copied.
        for o in tmp._parents:
            if hasattr(o, '_cache'):
                o._cache = dict([k, o._cache[k].copy()] for k in o._cache)
            if isinstance(o, Misfit):
                o.model = o.model.copy()
                o.positional = o.positional.copy()
        tmp._clear_cache()
        return tmp

    def config(self, method, **kwargs):
        """
        Configure the optimization method and its parameters.

        This sets the method used by
        :meth:`~fatiando.inversion.base.Misfit.fit` and the keyword arguments
        that are passed to it.

        Parameters:

        * method : string
            The optimization method. One of: ``'linear'``, ``'newton'``,
            ``'levmarq'``, ``'steepest'``, ``'acor'``

        Other keyword arguments that can be passed are the ones allowed by each
        method.

        Some methods have required arguments:

        * *newton*, *levmarq* and *steepest* require the ``initial`` argument
          (an initial estimate for the gradient descent)
        * *acor* requires the ``bounds`` argument (min/max values for the
          search space)

        See the corresponding docstrings for more information:

        * :meth:`~fatiando.inversion.base.Misfit.linear`
        * :meth:`~fatiando.inversion.base.Objective.newton`
        * :meth:`~fatiando.inversion.base.Objective.levmarq`
        * :meth:`~fatiando.inversion.base.Objective.steepest`
        * :meth:`~fatiando.inversion.base.Objective.acor`

        .. note::

            The *iterate* keyword is not supported by *fit*.
            Use the individual methods to step through iterations.


        Examples:

        >>> s = Misfit([1, 2], {}, {}, 2).config(
        ...     method='newton', precondition=False,
        ...     initial=[0, 0], maxit=10, tol=0.01)
        >>> s.fit_method
        'newton'
        >>> for k, v in sorted(s.fit_args.items()):
        ...     print k, ':', v
        initial : [0, 0]
        maxit : 10
        precondition : False
        tol : 0.01
        >>> # Omitted arguments will fall back to the method defaults
        >>> s = s.config(method='levmarq', initial=[1, 1])
        >>> for k, v in sorted(s.fit_args.items()):
        ...     print k, ':', v
        dlamb : 2
        initial : [1, 1]
        lamb : 1
        maxit : 30
        maxsteps : 10
        precondition : True
        tol : 1e-05
        >>> # For non-linear gradient solvers, *initial* is required
        >>> s.config(method='newton')
        Traceback (most recent call last):
            ...
        AttributeError: Missing required *initial* argument for 'newton'
        >>> # For ACO-R, *bounds* is required
        >>> s.config(method='acor')
        Traceback (most recent call last):
            ...
        AttributeError: Missing required *bounds* argument for 'acor'
        >>> # fit doesn't support the *iterate* argument
        >>> s.config(method='steepest', iterate=True, initial=[1, 1])
        Traceback (most recent call last):
            ...
        AttributeError: Invalid argument 'iterate'
        >>> # You can only pass arguments for that specific solver
        >>> s.config(method='newton', lamb=10, initial=[1, 1])
        Traceback (most recent call last):
            ...
        AttributeError: Invalid argument 'lamb' for 'newton'

        """
        if method not in self.default_solver_args:
            raise ValueError("Invalid method '%s'" % (method))
        if 'iterate' in kwargs:
            raise AttributeError("Invalid argument 'iterate'")
        if (method in ['newton', 'levmarq', 'steepest'] and
                'initial' not in kwargs):
            raise AttributeError(
                "Missing required *initial* argument for '%s'" % (method))
        if method == 'acor' and 'bounds' not in kwargs:
            raise AttributeError(
                "Missing required *bounds* argument for '%s'" % (method))
        args = self.default_solver_args[method].copy()
        for k in kwargs:
            if k not in args:
                raise AttributeError("Invalid argument '%s' for '%s'"
                                     % (k, method))
            args[k] = kwargs[k]
        self.fit_method = method
        self.fit_args = args
        return self

    @property
    def p_(self):
        """
        The current estimated parameter vector.

        Returns:

        * p : 1d-array or None
            The parameter vector. None, if
            :meth:`~fatiando.inversion.base.Misfit.fit` hasn't been called
            yet.

        """
        if hasattr(self, '_p'):
            return self._p
        else:
            return None

    @property
    def estimate_(self):
        """
        The current estimate.

        .. note::

            May be a formatted version of the parameter vector. It is
            recommened that you use this when accessing the estimate and use
            :meth:`~fatiando.inversion.base.Misfit.p_` when you want the raw
            parameter vector.

        Returns:

        * p : 1d-array or None
            The parameter vector. None, if
            :meth:`~fatiando.inversion.base.Misfit.fit` hasn't been called
            yet.

        """
        if hasattr(self, '_estimate'):
            return self._estimate
        else:
            return None

    def fit(self):
        """
        Solve for the parameter vector that minimizes this objective function.

        Uses the optimization method and parameters defined using the
        :meth:`~fatiando.inversion.base.Misfit.config` method.

        The estimated parameter vector can be accessed through the
        :meth:`~fatiando.inversion.base.Misfit.p_` property. A (possibly)
        formatted version (converted to a more manageble type) of the estimate
        can be accessed through
        :meth:`~fatiando.inversion.base.Misfit.estimate_`.

        """
        self._p = getattr(self, self.fit_method)(**self.fit_args)
        self._estimate = self._p
        return self

    def linear(self, precondition=True):
        """
        Solve for the parameter vector assuming that the problem is linear.

        See :func:`fatiando.inversion.solvers.linear` for more details.

        Parameters:

        * precondition : True or False
            If True, will use Jacobi preconditioning.

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        hessian = self.hessian('null')
        gradient = self.gradient('null')
        p = linear(hessian, gradient, precondition=precondition)
        return p
