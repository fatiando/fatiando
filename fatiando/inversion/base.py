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


class OperatorMixin(object):
    """
    Implements the operator overload for + and *.
    """

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
    Defines the `fit` and `config` methods plus all the optmization methods.
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
        not_configured = (getattr(self, 'fit_method', None) is None
                          or getattr(self, 'fit_args', None) is None)
        if not_configured:
            if self.islinear:
                self.config('linear')
            else:
                self.config('levmarq', initial=numpy.ones(self.nparams))
        optimizer = getattr(self, self.fit_method)
        self.p_ = optimizer(**self.fit_args)
        return self

    def linear(self, **kwargs):
        return solvers.linear(self.hessian(None), self.gradient(None),
                              **kwargs)

    def newton(self, **kwargs):
        return solvers.newton(self.hessian, self.gradient, self.value,
                              **kwargs)

    def levmarq(self, **kwargs):
        for p in solvers.levmarq(self.hessian, self.gradient, self.value,
                **kwargs):
            continue
        return p

    def steepest(self, **kwargs):
        return solvers.steepest(self.gradient, self.value, **kwargs)

    def acor(self, **kwargs):
        for p in solvers.acor(self.value, **kwargs):
            continue
        return p

    @abstractmethod
    def value(self, p):
        pass

    def fmt_estimate(self, p):
        """
        Called when accessing the property ``estimate_``. Use this to convert
        the paramter vector (p) to a more useful form, like a geometric object,
        etc.
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
    """

    def __init__(self, *args):
        self._components = self._unpack_components(args)
        self.size = len(self._components)
        self.regul_param = 1
        nparams = [obj.nparams for obj in self._components]
        assert all(nparams[0] == n for n in nparams[1:])
        self.nparams = nparams[0]
        if all(obj.islinear for obj in self._components):
            self.islinear = True
        else:
            self.islinear = False
        self._i = 0  # Tracker for indexing

    def _unpack_components(self, args):
        """
        Find all the MultiObjective elements in components and unpack them into
        a single list.
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
        if self._i >= self.size:
            raise StopIteration
        comp = self.__getitem__(self._i)
        self._i += 1
        return comp

    def copy(self, deep=False):
        """
        Make a copy of me together with all the cached methods.
        """
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        return obj

    def fmt_estimate(self, p):
        return self._components[0].fmt_estimate(p)

    def value(self, p):
        return self.regul_param*sum(obj.value(p) for obj in self)

    def gradient(self, p):
        return self.regul_param*sum(obj.gradient(p) for obj in self)

    def hessian(self, p):
        return self.regul_param*sum(obj.hessian(p) for obj in self)

    def fit(self):
        super(MultiObjective, self).fit()
        for obj in self:
            obj.p_ = self.p_
        return self


class Cached(object):
    def __init__(self, instance, meth):
        self.array_hash = None
        self.cache = None
        self.instance = instance
        self.meth = meth
        method = getattr(self.instance.__class__, self.meth)
        setattr(self, '__doc__', getattr(method, '__doc__'))

    def hard_reset(self):
        self.cache = None
        self.array_hash = None

    def __call__(self, p=None):
        if p is None:
            p = getattr(self.instance, 'p_')
        p_hash = hashlib.sha1(p).hexdigest()
        if self.cache is None or self.array_hash != p_hash:
            self.array_hash = p_hash
            method = getattr(self.instance.__class__, self.meth)
            self.cache = method(self.instance, p)
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

    When subclassing this class, you must implement two methods:

    * ``predicted(self, p)``: calculates the predicted data
      :math:`\bar{d}` for a given parameter vector ``p``
    * ``jacobian(self, p)``: calculates the Jacobian matrix of
      :math:`\bar{f}(\bar{p})` evaluated at ``p``

    ``jacobian`` is only needed for gradient based optimization. If you plan on
    using only heuristic methods to produce an estimate, you choose not to
    implement it.

    If :math:`\bar{f}` is linear, then the Jacobian will be cached in memory so
    that it is only calculated once when using the class multiple times. So
    solving the same problem with different methods or using an iterative
    method doesn't have the penalty of recalculating the Jacobian.

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

    """


    def __init__(self, data, nparams, islinear, weights=None):
        self.p_ = None
        self.regul_param = 1
        self.nparams = nparams
        self.islinear = islinear
        self.data = data
        self.ndata = self.data.size
        self.weights = weights
        self.predicted = Cached(self, 'predicted')
        if islinear:
            self.jacobian = CachedPermanent(self, 'jacobian')
            self.hessian = CachedPermanent(self, 'hessian')
        else:
            self.jacobian = Cached(self, 'jacobian')

    def copy(self, deep=False):
        """
        Make a copy of me together with all the cached methods.
        """
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        obj.predicted = copy.copy(obj.predicted)
        obj.predicted.instance = obj
        obj.jacobian = copy.copy(obj.jacobian)
        obj.jacobian.instance = obj
        if self.islinear:
            obj.hessian = copy.copy(obj.hessian)
            obj.hessian.instance = obj
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
            val =  numpy.sum(self.weights*(residuals**2))
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
            hessian =  safe_dot(jacobian.T, jacobian)
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
