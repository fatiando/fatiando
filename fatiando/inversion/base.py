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

        def value(self, p):
            '''
            Calculate the value of the objetive function.

            Parameters:

            * p : 1d-array or None
                The parameter vector.

            Returns:

            * value : float

            '''
            ...

        def hessian(self, p):
            '''
            Calculates the Hessian matrix.

            Parameters:

            * p : 1d-array
                The parameter vector where the Hessian is evaluated

            Returns:

            * hessian : 2d-array

            '''
            ...

        def gradient(self, p):
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

    Parameters:

    * nparams : int
        The number of parameters the objective function takes.
    * islinear : True or False
        Wether the functions is linear with respect to the parameters.

    Operations:

    For joint inversion and regularization, you can add *Objective*
    instances together and multiply them by scalars (i.e., regularization
    parameters):

    >>> a = Objective(10, True)
    >>> a.value = lambda p: 2*p
    >>> a.value(3)
    6
    >>> b = Objective(10, True)
    >>> b.value = lambda p: -3*p
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
        self._components = []

    def __repr__(self):
        return 'Objective instance'

    def value(self, p):
        raise NotImplementedError()

    def gradient(self, p):
        raise NotImplementedError()

    def hessian(self, p):
        raise NotImplementedError()

    def __add__(self, other):
        if self.nparams != other.nparams:
            raise ValueError(
                "Can only add functions with same number of parameters")
        def wrap(name, obj):
            func = getattr(self, name)
            def wrapper(self, p):
                return func(p) + getattr(other, name)(p)
            wrapper.__doc__ = getattr(self, name).__doc__
            setattr(obj, name, types.MethodType(wrapper, obj))
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        tmp = copy.copy(self)
        tmp._components = [other] + other._components
        # Wrap the hessian, gradient and value to be the sums
        wrap('hessian', tmp)
        wrap('gradient', tmp)
        wrap('value', tmp)
        return tmp

    def __mul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Can only multiply a Objective by a float or int')
        def wrap(name, obj):
            func = getattr(self, name)
            def wrapper(self, p):
                return other*func(p)
            wrapper.__doc__ = getattr(self, name).__doc__
            setattr(obj, name, types.MethodType(wrapper, obj))
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        tmp = copy.copy(self)
        # Wrap the hessian, gradient and value to be the products
        wrap('hessian', tmp)
        wrap('gradient', tmp)
        wrap('value', tmp)
        return tmp

    def __rmul__(self, other):
        return self.__mul__(other)

class FitMixin(object):
    """
    A mixin class for the *fit* method that minimizes the Objective function.

    The :meth:`~fatiando.inversion.base.FitMixin.fit` method uses the
    optimization method defined in the *fit_method* attribute.
    When calling the optimization, will pass keyword arguments
    defined in the *fit_args* atttribute.
    Use the :meth:`~fatiando.inversion.base.FitMixin.config` method to set
    these parameters.

    Also specifies methods for the individual solvers in
    :mod:`fatiando.inversion.solvers`:

    * :meth:`~fatiando.inversion.base.FitMixin.linear`
    * :meth:`~fatiando.inversion.base.FitMixin.newton`
    * :meth:`~fatiando.inversion.base.FitMixin.levmarq`
    * :meth:`~fatiando.inversion.base.FitMixin.steepest`
    * :meth:`~fatiando.inversion.base.FitMixin.acor`

    """

    default_solver_args = {
        'linear':{'precondition':True},
        'newton':{'initial':None,
                  'maxit':30,
                  'tol':10**-5,
                  'precondition':True},
        'levmarq':{'initial':None,
                   'maxit':30,
                   'maxsteps':10,
                   'lamb':1,
                   'dlamb':2,
                   'tol':10**-5,
                   'precondition':True},
        'steepest':{'initial':None,
                    'stepsize':0.1,
                    'maxsteps':30,
                    'maxit':1000,
                    'tol':10**-5},
        'acor':{'bounds':None,
                'nants':None,
                'archive_size':None,
                'maxit':1000,
                'diverse':0.5,
                'evap':0.85,
                'seed':None}}

    def config(self, method, **kwargs):
        """
        Configure the optimization method and its parameters.

        This sets the method used by
        :meth:`~fatiando.inversion.base.FitMixin.fit` and the keyword arguments
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

        * :meth:`~fatiando.inversion.base.FitMixin.linear`
        * :meth:`~fatiando.inversion.base.FitMixin.newton`
        * :meth:`~fatiando.inversion.base.FitMixin.levmarq`
        * :meth:`~fatiando.inversion.base.FitMixin.steepest`
        * :meth:`~fatiando.inversion.base.FitMixin.acor`

        .. note::

            The *iterate* keyword is not supported by *fit*.
            Use the individual methods to step through iterations.


        Examples:

        >>> s = FitMixin().config(method='newton', precondition=False,
        ...                       initial=[0, 0], maxit=10, tol=0.01)
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
                raise AttributeError("Invalid argument '%s' for '%s'" % (k,
                    method))
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
            :meth:`~fatiando.inversion.base.FitMixin.fit` hasn't been called
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
            :meth:`~fatiando.inversion.base.FitMixin.p_` when you want the raw
            parameter vector.

        Returns:

        * p : 1d-array or None
            The parameter vector. None, if
            :meth:`~fatiando.inversion.base.FitMixin.fit` hasn't been called
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
        :meth:`~fatiando.inversion.base.FitMixin.config` method.

        The estimated parameter vector can be accessed through the
        :meth:`~fatiando.inversion.base.FitMixin.p_` property. A (possibly)
        formatted version (converted to a more manageble type) of the estimate
        can be accessed through
        :meth:`~fatiando.inversion.base.FitMixin.estimate_`.

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

    def levmarq(self, initial, maxit=30, maxsteps=10, lamb=1, dlamb=2,
                tol=10**-5, precondition=True, iterate=False):
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
                maxit=maxit, maxsteps=maxsteps, lamb=lamb, dlamb=dlamb,
                tol=tol, precondition=precondition)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def newton(self, initial, maxit=30, tol=10**-5, precondition=True,
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
                 tol=10**-5, iterate=False):
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
             diverse=0.5, evap=0.85, seed=None, iterate=True):
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

class Misfit(Objective, FitMixin):
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
        self.hasher = lambda x: hashlib.sha1(x).hexdigest()
        self._cache = {}
        self._cache['predicted'] = {'hash':'', 'array':None}
        self._cache['jacobian'] = {'hash':'', 'array':None}
        self._cache['hessian'] = {'hash':'', 'array':None}
        # Data weights
        self.weights = None
        if weights is not None:
            self.set_weights(weights)
        if islinear:
            self.config(method='linear')

    def _clear_cache(self):
        "Reset the cached matrices"
        self._cache['predicted'] = {'hash':'', 'array':None}
        self._cache['jacobian'] = {'hash':'', 'array':None}
        self._cache['hessian'] = {'hash':'', 'array':None}

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
        self._cache['hessian'] = {'hash':'', 'array':None}
        return self

    def residuals(self, p=None):
        """
        Calculate the residuals vector (observed - predicted data).

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the residuals. If None, will
            use the current estimate stored in ``estimate_``.

        Returns:

        * residuals : 1d-array
            The residual vector

        """
        return self.data - self.predicted(p)

    def predicted(self, p=None):
        """
        Calculate the predicted data for a given parameter vector.

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the predicted data. If None,
            will use the current estimate stored in ``estimate_``.

        Returns:

        * predicted : 1d-array
            The predicted data

        """
        if p == 'null':
            pred = 0
        else:
            if p is None:
                p = self.p_
            hash = self.hasher(p)
            if hash != self._cache['predicted']['hash']:
                self._cache['predicted']['array'] = self._get_predicted(p)
                self._cache['predicted']['hash'] = hash
            pred = self._cache['predicted']['array']
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

    def value(self, p):
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
                )**2/self.ndata
        else:
            return numpy.sum(self.weights*(
                        (self.data - self.predicted(p))**2)
                        )/self.ndata

    def hessian(self, p):
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
                hessian = (2/self.ndata)*safe_dot(jacobian.T, jacobian)
            else:
                hessian = (2/self.ndata)*safe_dot(
                    jacobian.T, self.weights*jacobian)
            if self.islinear:
                self._cache['hessian']['array'] = hessian
        return hessian

    def gradient(self, p):
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
            grad = (-2/self.ndata)*safe_dot(
                jacobian.T, self.data - self.predicted(p))
        else:
            grad = (-2/self.ndata)*safe_dot(
                jacobian.T, self.weights*(self.data - self.predicted(p)))
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break loose
            grad = numpy.array(grad).ravel()
        return grad

    # Overload the add function to make predicted and residuals return a list
    # of all predicted and residual vectors from all the components.
    def __add__(self, other):
        tmp = super(Misfit, self).__add__(other)
        def wrap(name, obj):
            func = getattr(self, name)
            def wrapper(self, p=None):
                if p is None:
                    p = self.p_
                res = [func(p)]
                for o in obj._components:
                    ofunc = getattr(o, name, None)
                    if callable(ofunc):
                        aux = ofunc(p)
                        if isinstance(aux, list):
                            res.extend(aux)
                        else:
                            res.append(aux)
                if len(res) == 1:
                    return res[0]
                else:
                    return res
            wrapper.__doc__ = getattr(self, name).__doc__
            setattr(obj, name, types.MethodType(wrapper, obj))
        wrap('predicted', tmp)
        wrap('residuals', tmp)
        return tmp
