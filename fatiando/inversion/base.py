"""
The base classes for inverse problem solving.

See :mod:`fatiando.inversion` for examples, regularization, and more.

This module defines base classes that are used by the rest of the
``inversion`` package:

* :class:`~fatiando.inversion.base.MultiObjective`: A "container" class that
  emulates a sum of different  objective (goal) functions (like
  :class:`~fatiando.inversion.misfit.Misfit` or some form of
  :mod:`~fatiando.inversion.regularization`). When two of those classes are
  added they generate a ``MultiObjective`` object.
* :class:`~fatiando.inversion.base.OperatorMixin`: A mix-in class that defines
  the operators ``+`` and ``*`` (by a scalar). Used to give these properties to
  ``Misfit`` and the regularizing functions. Adding results in a
  ``MultiObjective``. Multiplying sets the ``regul_param`` of the class (like a
  scalar weight factor).
* :class:`~fatiando.inversion.base.OptimizerMixin`: A mix-in class that defines
  the ``fit`` and ``config`` methods for optimizing a ``Misfit`` or
  ``MultiObjective`` and fitting the model to the data.
* :class:`~fatiando.inversion.base.CachedMethod`: A class that wraps a method
  and caches the returned value. When the same argument (an array) is passed
  twice in a row, the class returns the cached value instead of recomputing.
* :class:`~fatiando.inversion.base.CachedMethodPermanent`: Like
  ``CachedMethod`` but always returns the cached value, regardless of the
  input. Effectively calculates only the first time the method is called.
  Useful for caching the Jacobian matrix in a linear problem.

----

"""
from __future__ import division, absolute_import
from future.utils import with_metaclass
from future.builtins import super, object, range, isinstance, zip, map
import hashlib
import copy
from abc import ABCMeta, abstractmethod
import numpy as np

from . import optimization


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
        assert self.nparams == other.nparams, \
            "Can't add goals with different number of parameters:" \
            + ' {}, {}'.format(self.nparams, other.nparams)
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


class OptimizerMixin(with_metaclass(ABCMeta)):
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

    Some stats about the optimization process are stored in the ``stats_``
    attribute as a dictionary.

    The minimum requirement for a class to inherit from ``OptimizerMixin`` is
    that it must define at least a
    :meth:`~fatiando.inversion.base.OptimizerMixin.value` method.
    """

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

        * :meth:`~fatiando.inversion.optimization.linear`
        * :meth:`~fatiando.inversion.optimization.newton`
        * :meth:`~fatiando.inversion.optimization.levmarq`
        * :meth:`~fatiando.inversion.optimization.steepest`
        * :meth:`~fatiando.inversion.optimization.acor`

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
        not_configured = (getattr(self, 'fit_method', None) is None or
                          getattr(self, 'fit_args', None) is None)
        if not_configured:
            if self.islinear:
                self.config('linear')
            else:
                self.config('levmarq', initial=np.ones(self.nparams))
        optimizer = getattr(optimization, self.fit_method)
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
        assert self.p_ is not None, "No estimate found. Run 'fit' first."
        return self.fmt_estimate(self.p_)


class MultiObjective(OptimizerMixin, OperatorMixin):
    """
    An objective (goal) function with more than one component.

    This class is a linear combination of other goal functions (like
    :class:`~fatiando.inversion.misfit.Misfit` and regularization classes).

    It is automatically created by adding two goal functions that have the
    :class:`~fatiando.inversion.base.OperatorMixin` as a base class.

    Alternatively, you can create a ``MultiObjetive`` by passing the other
    goals function instances as arguments to the constructor.

    The ``MultiObjetive`` behaves like any other goal function object. It has
    ``fit`` and ``config`` methods and can be added and multiplied by a scalar
    with the same effects.

    Indexing a ``MultiObjetive`` will iterate over the component goal
    functions.

    Examples:

    To show how this class is generated and works, let's create a simple class
    that subclasses ``OperatorMixin``.

    >>> class MyGoal(OperatorMixin):
    ...     def __init__(self, name, nparams, islinear):
    ...         self.name = name
    ...         self.islinear = islinear
    ...         self.nparams = nparams
    ...     def value(self, p):
    ...         return 1
    ...     def gradient(self, p):
    ...         return 2
    ...     def hessian(self, p):
    ...         return 3
    >>> a = MyGoal('A', 10, True)
    >>> b = MyGoal('B', 10, True)
    >>> c = a + b
    >>> type(c)
    <class 'fatiando.inversion.base.MultiObjective'>
    >>> c.size
    2
    >>> c.nparams
    10
    >>> c.islinear
    True
    >>> c[0].name
    'A'
    >>> c[1].name
    'B'

    Asking for the value, gradient, and Hessian of the ``MultiObjective`` will
    give me the sum of both components.

    >>> c.value(None)
    2
    >>> c.gradient(None)
    4
    >>> c.hessian(None)
    6

    Multiplying the ``MultiObjective`` by a scalar will set the regularization
    parameter for the sum.

    >>> d = 10*c
    >>> d.value(None)
    20
    >>> d.gradient(None)
    40
    >>> d.hessian(None)
    60

    All components must have the same number of parameters. For the moment,
    ``MultiObjetive`` doesn't handle multiple parameter vector (one for each
    objective function).

    >>> e = MyGoal("E", 20, True)
    >>> a + e
    Traceback (most recent call last):
      ...
    AssertionError: Can't add goals with different number of parameters: 10, 20

    The ``MultiObjective`` will automatically detect if the problem remains
    linear or not. For example, adding a non-linear problem to a linear one
    makes the sum non-linear.

    >>> (a + b).islinear
    True
    >>> f = MyGoal('F', 10, False)
    >>> (a + f).islinear
    False
    >>> (f + f).islinear
    False


    """

    def __init__(self, *args):
        self._components = self._unpack_components(args)
        self.size = len(self._components)
        self.p_ = None
        nparams = [obj.nparams for obj in self._components]
        assert all(nparams[0] == n for n in nparams[1:]), \
            "Can't add goals with different number of parameters:" \
            + ' ' + ', '.join(str(n) for n in nparams)
        self.nparams = nparams[0]
        if all(obj.islinear for obj in self._components):
            self.islinear = True
        else:
            self.islinear = False
        self._i = 0  # Tracker for indexing

    def fit(self):
        super().fit()
        for obj in self:
            obj.p_ = self.p_
        return self

    fit.__doc__ = OptimizerMixin.fit.__doc__

    # Pass along the configuration in case the classes need to change something
    # depending on the optimization method.
    def config(self, *args, **kwargs):
        super().config(*args, **kwargs)
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

    def __next__(self):
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

    Uses SHA1 hashes of the input array to tell if it is the same array.

    .. note::

        We need the object instance and method name instead of the bound method
        (like ``obj.method``) because we can't pickle bound methods. We need to
        be able to pickle so that the solvers can be passed between processes
        in parallelization.

    Parameters:

    * instance : object
        The instance of the object that has the method you want to cache.
    * meth : string
        The name of the method you want to cache.

    Examples:

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

    .. note::

        We need the object instance and method name instead of the bound method
        (like ``obj.method``) because we can't pickle bound methods. We need to
        be able to pickle so that the solvers can be passed between processes
        in parallelization.

    Parameters:

    * instance : object
        The instance of the object that has the method you want to cache.
    * meth : string
        The name of the method you want to cache.

    Examples:

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
