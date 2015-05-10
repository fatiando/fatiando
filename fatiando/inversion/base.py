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
import copy
from abc import ABCMeta, abstractmethod
import six
import numpy
import scipy.sparse

from . import solvers
from ..utils import safe_dot


class OperatorMixin(object):
    """
    Implements the operators + and * for the goal functions classes.

    This class is not meant to be used on its own. Use it as a parent to give
    the child class the + and * operators.

    Used in :class:`~fatiando.inversion.base.Misfit` and the regularization
    classes in :mod:`fatiando.inversion.regularization`.

    .. note::

        Performing ``A + B`` produces a
        :class:`~fatiando.inversion.base.MultiObjetive` with copies of ``A``
        and ``B``.

    .. note::

        Performing ``scalar*A`` produces a copy of ``A`` with ``scalar`` set as
        the ``regul_param`` attribute.


    """

    @property
    def regul_param(self):
        """
        The regularization parameter (scale factor) for the objetive function.

        Defaults to 1.
        """
        return getattr(self, '_regularizing_parameter', 1)

    @regul_param.setter
    def regul_param(self, value):
        """
        Set the value of the regularizing parameter.
        """
        self._regularizing_parameter = value
        for name in ['hessian', 'gradient', 'value']:
            if hasattr(self, name):
                method = getattr(self, name)
                iscached = (isinstance(method, CachedMethodPermanent) or
                            isinstance(method, CachedMethod))
                if iscached:
                    method.hard_reset()

    def copy(self, deep=False):
        """
        Make a copy of me.
        """
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        return obj

    def __add__(self, other):
        """
        Add two objective functions to make a MultiObjective.
        """
        if self.nparams != other.nparams:
            raise ValueError(
                "Can only add functions with same number of parameters")
        # Make a shallow copy of self to return. If returned self, doing
        # 'a = b + c' a and b would reference the same object.
        res = MultiObjective(self.copy(), other.copy())
        return res

    def __mul__(self, other):
        """
        Multiply the objective function by a scallar to set the `regul_param`
        attribute.
        """
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Can only multiply a Objective by a float or int')
        # Make a shallow copy of self to return. If returned self, doing
        # 'a = 10*b' a and b would reference the same object.
        obj = self.copy()
        obj.regul_param = obj.regul_param*other
        return obj

    def __rmul__(self, other):
        return self.__mul__(other)


class OptimizerMixin(six.with_metaclass(ABCMeta)):
    """
    Defines ``fit`` and ``config`` methods plus all the optimization methods.

    This class is not meant to be used on its own. Use it as a parent to give
    the child class the methods it implements.

    Used in :class:`~fatiando.inversion.base.Misfit` and
    :class:`fatiando.inversion.base.MultiObjetive`.

    The :meth:`~fatiando.inversion.base.OptimizerMixin.config` method is used
    to configure the optimization method that will be used.

    The :meth:`~fatiando.inversion.base.OptimizerMixin.fit` method runs the
    optimization method configured and stores the computed parameter vector in
    the ``p_`` attribute.

    The minimum requirement for a class to inherit from ``OptimizerMixin`` is
    that it must define a :meth:`~fatiando.inversion.base.OptimizerMixin.value`
    method.
    """

    @abstractmethod
    def value(self, p):
        """
        Calculates the value of the goal function for a given parameter vector.

        Abstract method that must be implemented when subclassing
        ``OptimizerMixin``.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * result : scalar (float, int or complex)
            The value of the goal function for this parameter vector.

        """
        pass

    def config(self, method, **kwargs):
        """
        Configure the optimization method and its parameters.

        This sets the method used by
        :meth:`~fatiando.inversion.base.Objective.fit` and the keyword
        arguments that are passed to it.

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

        * :meth:`~fatiando.inversion.solver.linear`
        * :meth:`~fatiando.inversion.solver.newton`
        * :meth:`~fatiando.inversion.solver.levmarq`
        * :meth:`~fatiando.inversion.solver.steepest`
        * :meth:`~fatiando.inversion.solver.acor`

        .. note::

            The *iterate* keyword is not supported by *fit*.
            Use the individual methods to step through iterations.

        """
        kwargs = copy.deepcopy(kwargs)
        assert method in ['linear', 'newton', 'levmarq', 'steepest', 'acor'], \
            "Invalid optimization method '{}'".format(method)
        if method in ['newton', 'levmarq', 'steepest']:
            assert 'initial' in kwargs, \
                "Missing required *initial* argument for '{}'".format(method)
        if method == 'acor':
            assert 'bounds' in kwargs, \
                "Missing required *bounds* argument for '{}'".format(method)
        if method == 'acor' and 'nparams' not in kwargs:
            kwargs['nparams'] = self.nparams
        self.fit_method = method
        self.fit_args = kwargs
        return self

    def fit(self):
        """
        Solve for the parameter vector that minimizes this objective function.

        Uses the optimization method and parameters defined using the
        :meth:`~fatiando.inversion.base.OptimizerMixin.config` method.

        The estimated parameter vector can be accessed through the
        ``p_`` attribute. A (possibly) formatted version (converted to a more
        manageable type) of the estimate can be accessed through the property
        ``estimate_``.

        """
        not_configured = (getattr(self, 'fit_method', None) is None
                          or getattr(self, 'fit_args', None) is None)
        if not_configured:
            if self.islinear:
                self.config('linear')
            else:
                self.config('levmarq', initial=numpy.ones(self.nparams))
        optimizer = getattr(solvers, self.fit_method)
        # Make the generators from the optimization function
        if self.fit_method == 'linear':
            solver = optimizer(self.hessian(None), self.gradient(None),
                               **self.fit_args)
        elif self.fit_method in ['newton', 'levmarq']:
            solver = optimizer(self.hessian, self.gradient, self.value,
                               **self.fit_args)
        elif self.fit_method == 'steepest':
            solver = optimizer(self.gradient, self.value, **self.fit_args)
        elif self.fit_method == 'acor':
            solver = optimizer(self.value, **self.fit_args)
        # Run the optimizer to the end
        for i, p, stats in solver:
            continue
        self.p_ = p
        self.stats_ = stats
        return self

    def fmt_estimate(self, p):
        """
        Called when accessing the property ``estimate_``.

        Use this to convert the parameter vector (p) to a more useful form,
        like a geometric object, etc.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * formatted
            Pretty much anything you want.

        """
        return p

    @property
    def estimate_(self):
        """
        A nicely formatted version of the estimate.

        If the class implements a `fmt_estimate` method, this will its results.
        This can be used to convert the parameter vector to a more useful form,
        like a :mod:`fatiando.mesher` object.

        """
        return self.fmt_estimate(self.p_)


class MultiObjective(OptimizerMixin, OperatorMixin):
    """
    An objective (goal) function with more than one component.

    This class is a linear combination of other goal functions (like ``Misfit``
    and regularization classes).

    It is automatically created by adding two goal functions that have the
    :class:`~fatiando.inversion.base.OperatorMixin` as a base class.

    Alternatively, you can create a ``MultiObjetive`` by passing the other
    goals function instances as arguments to the constructor.

    The ``MultiObjetive`` behaves like any other goal function object. It has
    ``fit`` and ``config`` methods and can be added and multiplied by a scalar
    with the same effects.

    Indexing a ``MultiObjetive`` will iterate over the component goal
    functions.
    """

    def __init__(self, *args):
        self._components = self._unpack_components(args)
        self.size = len(self._components)
        self.p_ = None
        nparams = [obj.nparams for obj in self._components]
        assert all(nparams[0] == n for n in nparams[1:])
        self.nparams = nparams[0]
        if all(obj.islinear for obj in self._components):
            self.islinear = True
        else:
            self.islinear = False
        self._i = 0  # Tracker for indexing

    def fit(self):
        super(MultiObjective, self).fit()
        for obj in self:
            obj.p_ = self.p_
        return self

    fit.__doc__ = OptimizerMixin.fit.__doc__

    # Pass along the configuration in case the classes need to change something
    # depending on the optimization method.
    def config(self, *args, **kwargs):
        super(MultiObjective, self).config(*args, **kwargs)
        for obj in self:
            if hasattr(obj, 'config'):
                obj.config(*args, **kwargs)
        return self

    config.__doc__ = OptimizerMixin.config.__doc__

    def _unpack_components(self, args):
        """
        Find all the MultiObjective elements in components and unpack them into
        a single list.

        This is needed so that ``D = A + B + C`` can be indexed as ``D[0] == A,
        D[1] == B, D[2] == C``. Otherwise, ``D[1]`` would be a
        ``MultiObjetive == B + C``.
        """
        components = []
        for comp in args:
            if isinstance(comp, MultiObjective):
                components.extend([c*comp.regul_param for c in comp])
            else:
                components.append(comp)
        return components

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self._components[i]

    def __iter__(self):
        self._i = 0
        return self

    def next(self):
        """
        Used for iterating over the MultiObjetive.
        """
        if self._i >= self.size:
            raise StopIteration
        comp = self.__getitem__(self._i)
        self._i += 1
        return comp

    def fmt_estimate(self, p):
        """
        Format the current estimated parameter vector into a more useful form.

        Will call the ``fmt_estimate`` method of the first component goal
        function (the first term in the addition that created this object).
        """
        return self._components[0].fmt_estimate(p)

    def value(self, p):
        """
        Return the value of the multi-objective function.

        This will be the sum of all goal functions that make up this
        multi-objective.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * result : scalar (float, int, etc)
            The sum of the values of the components.

        """
        return self.regul_param*sum(obj.value(p) for obj in self)

    def gradient(self, p):
        """
        Return the gradient of the multi-objective function.

        This will be the sum of all goal functions that make up this
        multi-objective.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * result : 1d-array
            The sum of the gradients of the components.

        """
        return self.regul_param*sum(obj.gradient(p) for obj in self)

    def hessian(self, p):
        """
        Return the hessian of the multi-objective function.

        This will be the sum of all goal functions that make up this
        multi-objective.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * result : 2d-array
            The sum of the hessians of the components.

        """
        return self.regul_param*sum(obj.hessian(p) for obj in self)


class CachedMethod(object):
    """
    Wrap a method to cache it's output based on the hash of the input array.

    Store the output of calling the method on a numpy array. If the method is
    called in succession with the same input array, the cached result will be
    returned. If the method is called on a different array, the old result will
    be discarded and the new one stored.

    Uses sha1 hashes of the input array to tell if it is the same array.

    Parameters:

    * instance : object
        The instance of the object that has the method you want to cache.
    * meth : string
        The name of the method you want to cache.

    For pickling reasons we can't use the bound method (``obj.method``) and
    need the object and the method name.

    Examples::

    >>> import numpy as np
    >>> class MyClass(object):
    ...     def __init__(self, cached=False):
    ...         if cached:
    ...             self.my_method = CachedMethod(self, 'my_method')
    ...     def my_method(self, p):
    ...         return p**2
    >>> obj = MyClass(cached=False)
    >>> a = obj.my_method(np.arange(0, 5))
    >>> a
    array([ 0,  1,  4,  9, 16])
    >>> b = obj.my_method(np.arange(0, 5))
    >>> a is b
    False
    >>> cached = MyClass(cached=True)
    >>> a = cached.my_method(np.arange(0, 5))
    >>> a
    array([ 0,  1,  4,  9, 16])
    >>> b = cached.my_method(np.arange(0, 5))
    >>> a is b
    True
    >>> cached.my_method.hard_reset()
    >>> b = cached.my_method(np.arange(0, 5))
    >>> a is b
    False
    >>> c = cached.my_method(np.arange(0, 5))
    >>> b is c
    True
    >>> cached.my_method(np.arange(0, 6))
    array([ 0,  1,  4,  9, 16, 25])

    """

    def __init__(self, instance, meth):
        self.array_hash = None
        self.cache = None
        self.instance = instance
        self.meth = meth
        method = getattr(self.instance.__class__, self.meth)
        setattr(self, '__doc__', getattr(method, '__doc__'))

    def hard_reset(self):
        """
        Delete the cached values.
        """
        self.cache = None
        self.array_hash = None

    def __call__(self, p=None):
        if p is None:
            p = getattr(self.instance, 'p_')
        p_hash = hashlib.sha1(p).hexdigest()
        if self.cache is None or self.array_hash != p_hash:
            # Update the cache
            self.array_hash = p_hash
            # Get the method from the class because the instance will overwrite
            # it with the CachedMethod instance.
            method = getattr(self.instance.__class__, self.meth)
            self.cache = method(self.instance, p)
        return self.cache


class CachedMethodPermanent(object):
    """
    Wrap a method to cache it's output and return it whenever the method is
    called..

    This is different from :class:`~fatiando.inversion.base.CachedMethod`
    because it will only run the method once. All other times, the result
    returned will be this first one. This class should be used with methods
    that should return always the same output (like the Jacobian matrix of a
    linear problem).

    Parameters:

    * instance : object
        The instance of the object that has the method you want to cache.
    * meth : string
        The name of the method you want to cache.

    For pickling reasons we can't use the bound method (``obj.method``) and
    need the object and the method name.

    Examples::

    >>> import numpy as np
    >>> class MyClass(object):
    ...     def __init__(self, cached=False):
    ...         if cached:
    ...             self.my_method = CachedMethodPermanent(self, 'my_method')
    ...     def my_method(self, p):
    ...         return p**2
    >>> obj = MyClass(cached=False)
    >>> a = obj.my_method(np.arange(0, 5))
    >>> a
    array([ 0,  1,  4,  9, 16])
    >>> b = obj.my_method(np.arange(0, 5))
    >>> a is b
    False
    >>> cached = MyClass(cached=True)
    >>> a = cached.my_method(np.arange(0, 5))
    >>> a
    array([ 0,  1,  4,  9, 16])
    >>> b = cached.my_method(np.arange(0, 5))
    >>> a is b
    True
    >>> c = cached.my_method(np.arange(10, 15))
    >>> c
    array([ 0,  1,  4,  9, 16])
    >>> a is c
    True

    """
    def __init__(self, instance, meth):
        self.cache = None
        self.instance = instance
        self.meth = meth

    def hard_reset(self):
        """
        Delete the cached values.
        """
        self.cache = None

    def __call__(self, p=None):
        if self.cache is None:
            method = getattr(self.instance.__class__, self.meth)
            self.cache = method(self.instance, p)
        return self.cache


class Misfit(OptimizerMixin, OperatorMixin):
    r"""
    An l2-norm data-misfit function.

    This is a kind of objective function that measures the misfit between
    observed data :math:`\bar{d}^o` and data predicted by a set of model
    parameters :math:`\bar{d} = \bar{f}(\bar{p})`.

    The l2-norm data-misfit is defined as:

    .. math::

        \phi(\bar{p}) = \bar{r}^T\bar{r}}

    where :math:`\bar{r} = \bar{d}^o - \bar{d}` is the residual vector and
    :math:`N` is the number of data.

    When subclassing this class, you must implement the method:

    * ``predicted(self, p)``: calculates the predicted data
      :math:`\bar{d}` for a given parameter vector ``p``

    If you want to use any gradient-based solver (you probably do), you'll need
    to implement the method:

    * ``jacobian(self, p)``: calculates the Jacobian matrix of
      :math:`\bar{f}(\bar{p})` evaluated at ``p``

    If :math:`\bar{f}` is linear, then the Jacobian will be cached in memory so
    that it is only calculated once when using the class multiple times. So
    solving the same problem with different methods or using an iterative
    method doesn't have the penalty of recalculating the Jacobian.

    .. warning::

        When subclassing, be careful not to set the following attributes:
        ``data``, ``nparams``, ``islinear``, ``nparams``, ``ndata``, and
        (most importantly) ``regul_param`` and ``_regularizing_parameter``.
        This could mess with internal behavior and break things in unexpected
        ways.

    .. tip::

        See :mod:`fatiando.inversion` for examples of usage.

    Parameters:

    * data : 1d-array
        The observed data vector :math:`\bar{d}^o`
    * nparams : int
        The number of parameters in parameter vector :math:`\bar{p}`
    * islinear : True or False
        Whether :math:`\bar{f}` is linear or not.
    * weights : 1d-array
        Weights to be applied to the each element in *data* when computing the
        l2-norm. Effectively the diagonal of a matrix :math:`\bar{\bar{W}}`
        such that :math:`\phi = \bar{r}^T\bar{\bar{W}}\bar{r}`
    * cache : True
        Whether or not to cache the output of some methods to avoid recomputing
        matrices and vectors when passed the same input parameter vector.

    """

    def __init__(self, data, nparams, islinear, weights=None, cache=True):
        self.p_ = None
        self.nparams = nparams
        self.islinear = islinear
        self.data = data
        self.ndata = self.data.size
        self.weights = weights
        if cache:
            self.predicted = CachedMethod(self, 'predicted')
            if islinear:
                self.jacobian = CachedMethodPermanent(self, 'jacobian')
                self.hessian = CachedMethodPermanent(self, 'hessian')
            else:
                self.jacobian = CachedMethod(self, 'jacobian')

    def copy(self, deep=False):
        """
        Make a copy of me together with all the cached methods.
        """
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        for name in ['predicted', 'jacobian', 'hessian']:
            meth = getattr(obj, name)
            is_cached = (isinstance(meth, CachedMethod)
                         or isinstance(meth, CachedMethodPermanent))
            if is_cached:
                setattr(obj, name, copy.copy(meth))
                getattr(obj, name).instance = obj
        return obj

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
        self.hessian.hard_reset()
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
        res = self.data - self.predicted(p)
        return res

    @abstractmethod
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
        pass

    def value(self, p):
        r"""
        Calculate the value of the misfit for a given parameter vector.

        The value is given by:

        .. math::

            \phi(\bar{p}) = \bar{r}^T\bar{\bar{W}}\bar{r}


        where :math:`\bar{r}` is the residual vector and :math:`bar{\bar{W}}`
        are optional data weights.

        Parameters:

        * p : 1d-array or None
            The parameter vector.

        Returns:

        * value : float
            The value of the misfit function.

        """
        residuals = self.data - self.predicted(p)
        if self.weights is None:
            val = numpy.linalg.norm(residuals)**2
        else:
            val = numpy.sum(self.weights*(residuals**2))
        return val*self.regul_param

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
        jacobian = self.jacobian(p)
        if self.weights is None:
            hessian = safe_dot(jacobian.T, jacobian)
        else:
            hessian = safe_dot(jacobian.T, self.weights*jacobian)
        hessian *= 2*self.regul_param
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
        if p is None:
            tmp = self.data
        else:
            tmp = self.data - self.predicted(p)
        if self.weights is None:
            grad = safe_dot(jacobian.T, tmp)
        else:
            grad = safe_dot(jacobian.T, self.weights*tmp)
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break loose
            grad = numpy.array(grad).ravel()
        grad *= -2*self.regul_param
        return grad
