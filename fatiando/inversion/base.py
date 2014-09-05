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
import inspect
import copy
from abc import ABCMeta, abstractmethod
import six
import numpy
import scipy.sparse

from . import solvers
from ..utils import safe_dot


class Objective(six.with_metaclass(ABCMeta)):
    """
    An objective function for an inverse problem.

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
    ...         super(MyObjective, self).__init__(1, True)
    ...         self.scale = scale
    ...     def value(self, p):
    ...         return self.scale*p
    >>> a = MyObjective(2)
    >>> a
    MyObjective(scale=2)
    >>> a.value(3)
    6
    >>> b = MyObjective(-3)
    >>> b
    MyObjective(scale=-3)
    >>> b.value(3)
    -9
    >>> c = a + b
    >>> c
    (MyObjective(scale=2) + MyObjective(scale=-3))
    >>> c.value(3)
    -3
    >>> d = 0.5*c
    >>> d
    0.5*(MyObjective(scale=2) + MyObjective(scale=-3))
    >>> d.value(3)
    -1.5
    >>> e = a + 2*b
    >>> e
    (MyObjective(scale=2) + 2*MyObjective(scale=-3))
    >>> e.value(3)
    -12

    """
    # Set default arguments for fit
    default_solver_args = {
        'linear': {'precondition': True},
        'newton': {'initial': None,
                   'maxit': 30,
                   'tol': 1e-5,
                   'precondition': True},
        'levmarq': {'initial': None,
                    'maxit': 30,
                    'maxsteps': 10,
                    'lamb': 1,
                    'dlamb': 2,
                    'tol': 1e-5,
                    'precondition': True},
        'steepest': {'initial': None,
                     'stepsize': 0.1,
                     'maxsteps': 30,
                     'maxit': 1000,
                     'tol': 1e-5},
        'acor': {'bounds': None,
                 'nants': None,
                 'archive_size': None,
                 'maxit': 1000,
                 'diverse': 0.5,
                 'evap': 0.85,
                 'seed': None}
    }

    def __init__(self, nparams, islinear):
        self.nparams = nparams
        self.islinear = islinear
        self.p_ = None
        self.regul_param = 1
        self.fit_method = None
        self.fit_args = None


    def get_basic_repr(self):
        args, varargs, varkw, default = inspect.getargspec(self.__init__)
        # Remove self
        args.pop(0)
        numpy.set_printoptions(precision=5, threshold=5, edgeitems=2)
        args_str = []
        for key in sorted(args):
            val = getattr(self, key)
            arg = '{}={}'.format(key, str(val))
            args_str.append(arg)
        repr_str = '{}({})'.format(self.__class__.__name__, ', '.join(args_str))
        return repr_str


    def __repr__(self):
        repr_str = self.get_basic_repr()
        if self.regul_param != 1:
            repr_str = '*'.join(['{:g}'.format(self.regul_param), repr_str])
        return repr_str


    @abstractmethod
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
        pass


    @abstractmethod
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
        pass


    @abstractmethod
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
        pass



    def copy(self, deep=False):
        """
        Make a copy of me.

        Parameters:

        * deep : True or False
            If True, will make copies of all attributes as well

        Returns:

        * copy
            A copy of me.

        Example:

        >>> class MyObjective(Objective):
        ...     def __init__(self):
        ...         super(MyObjective, self).__init__(1, True)
        ...     def value(self, p):
        ...         return 1
        >>> a = MyObjective()
        >>> b = a
        >>> a is b
        True
        >>> c = a.copy()
        >>> c is a
        False

        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)


    def __add__(self, other):
        """
        Add two Objetives to make a Composite
        """
        if self.nparams != other.nparams:
            raise ValueError(
                "Can only add functions with same number of parameters")
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        res = Composite(self.copy(), other.copy())
        return res


    def __mul__(self, other):
        """
        When multiplied by a scalar, set the `regul_param` attribute.
        """
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Can only multiply a Objective by a float or int')
        # Make a shallow copy of self to return. If returned self, would
        # overwrite the original class and might get recurrence issues
        obj = self.copy()
        obj.regul_param = other
        return obj


    def __rmul__(self, other):
        return self.__mul__(other)


    def fmt_estimate(self, p):
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
        if method not in self.default_solver_args:
            raise ValueError("Invalid method '%s'" % (method))
        if (method in ['newton', 'levmarq', 'steepest'] and 'initial' not in kwargs):
            raise AttributeError("Missing required *initial* argument for '%s'" % (method))
        if method == 'acor' and 'bounds' not in kwargs:
            raise AttributeError("Missing required *bounds* argument for '%s'" % (method))
        args = self.default_solver_args[method].copy()
        for k in kwargs:
            if k not in args:
                raise AttributeError("Invalid argument '%s' for '%s'" % (k, method))
            args[k] = kwargs[k]
        if method == 'acor':
            args['nparams'] = self.nparams
        self.fit_method = method
        self.fit_args = args
        return self


    def fit(self):
        """
        Solve for the parameter vector that minimizes this objective function.

        Uses the optimization method and parameters defined using the
        :meth:`~fatiando.inversion.base.Objective.config` method.

        The estimated parameter vector can be accessed through the
        :meth:`~fatiando.inversion.base.Objective.p_` attribute. A (possibly)
        formatted version (converted to a more manageble type) of the estimate
        can be accessed through
        :meth:`~fatiando.inversion.base.Objective.estimate_`.

        """
        if self.fit_method is None or self.fit_args is None:
            if self.islinear:
                self.config('linear')
            else:
                self.config('levmarq', initial=numpy.ones(self.nparams))
        optimizer = getattr(solvers, self.fit_method)
        if self.fit_method == 'linear':
            p = optimizer(self.hessian(None), self.gradient(None), **self.fit_args)
        elif self.fit_method in ['newton', 'levmarq']:
            for p in optimizer(self.hessian, self.gradient, self.value, **self.fit_args):
                continue
        elif self.fit_method in ['steepest']:
            for p in optimizer(self.gradient, self.value, **self.fit_args):
                continue
        elif self.fit_method in ['acor']:
            for p in optimizer(self.value, **self.fit_args):
                continue
        else:
            raise ValueError('Unsupported solver "{}"'.format(self.fit_method))
        self.p_ = p
        return self

class Composite(Objective):
    def __init__(self, *args):
        self.components = args
        nparams = [obj.nparams for obj in self.components]
        assert all(nparams[0] == n for n in nparams[1:])
        if all(obj.islinear for obj in self.components):
            islinear = True
        else:
            islinear = False
        super(Composite, self).__init__(nparams=nparams[0], islinear=islinear)

    def __getitem__(self, i):
        return self.components[i]

    def fmt_estimate(self, p):
        return self.components[0].fmt_estimate(p)

    def value(self, p):
        return sum(obj.value(p) for obj in self.components)

    def gradient(self, p):
        return sum(obj.gradient(p) for obj in self.components)

    def hessian(self, p):
        return sum(obj.hessian(p) for obj in self.components)

    def get_basic_repr(self):
        expr = ' + '.join(['{}'.format(obj) for obj in self.components])
        return '({})'.format(expr)

    def fit(self):
        super(Composite, self).fit()
        for obj in self.components:
            obj.p_ = self.p_
        return self

class Cached(object):
    def __init__(self, instance, meth):
        self.array_hash = None
        self.cache = None
        self.instance = instance
        self.meth = meth
        setattr(self, '__doc__', getattr(getattr(self.instance.__class__, self.meth), '__doc__'))

    def hard_reset(self):
        self.cache = None
        self.array_hash = None

    def __call__(self, p=None):
        if p is None:
            p = getattr(self.instance, 'p_')
        p_hash = hashlib.sha1(p).hexdigest()
        if self.cache is None or self.array_hash != p_hash:
            self.array_hash = p_hash
            self.cache = getattr(self.instance.__class__, self.meth)(self.instance, p)
        return self.cache

class CachedPermanent(object):
    def __init__(self, instance, meth):
        self.cache = None
        self.instance = instance
        self.meth = meth

    def hard_reset(self):
        self.cache = None

    def __call__(self, p=None):
        if self.cache is None:
            self.cache = getattr(self.instance.__class__, self.meth)(self.instance, p)
        return self.cache

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

    When subclassing this class, you must implement two methods:

    * ``predicted(self, p)``: calculates the predicted data
      :math:`\bar{d}` for a given parameter vector ``p``
    * ``jacobian(self, p)``: calculates the Jacobian matrix of
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


    def __init__(self, data_dict, data_vector, nparams, islinear, weights=None):
        super(Misfit, self).__init__(nparams, islinear)
        self.data_vector = data_vector
        self.data_dict = data_dict
        self.data = data_dict[data_vector]
        self.ndata = self.data.size
        self.set_attributes(data_dict)
        self.weights = weights
        self.predicted = Cached(self, 'predicted')
        if islinear:
            self.jacobian = CachedPermanent(self, 'jacobian')
            self.hessian = CachedPermanent(self, 'hessian')
        else:
            self.jacobian = Cached(self, 'jacobian')


    def set_attributes(self, data_dict):
        for key, val in data_dict.items():
            setattr(self, key, val)


    def copy(self, deep=False):
        obj = super(Misfit, self).copy(deep)
        obj.predicted = copy.copy(obj.predicted)
        obj.predicted.instance = obj
        obj.jacobian = copy.copy(obj.jacobian)
        obj.jacobian.instance = obj
        if self.islinear:
            obj.hessian = copy.copy(obj.hessian)
            obj.hessian.instance = obj
        return obj


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


    @abstractmethod
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
        pass


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
        residuals = self.data - self.predicted(p)
        if self.weights is None:
            val = numpy.linalg.norm(residuals)**2
        else:
            val =  numpy.sum(self.weights*(residuals**2))
        return val*self.regul_param/self.ndata


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
            hessian =  safe_dot(jacobian.T, jacobian)
        else:
            hessian = safe_dot(jacobian.T, self.weights*jacobian)
        hessian *= (2/self.ndata)*self.regul_param
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
        grad *= self.regul_param*(-2/self.ndata)
        return grad
