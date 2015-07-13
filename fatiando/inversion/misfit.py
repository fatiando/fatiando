r"""
Defines base classes to represent a data-misfit functions (l2-norm, etc)

These classes can be used to implement parameter estimation problems
(inversions). They automate most of the boiler plate required and provide
direct access to ready-made optimization routines and regularization.

For now, only implements an l2-norm data misfit:

* :class:`~fatiando.inversion.misfit.Misfit`: an l2-norm data-misfit function

Examples
--------

The ``Misfit`` class works by subclassing it. Doing this gives you access to
all optimization functions and possible combinations of regularization. When
subclassing ``Misfit``, you'll need to implement the ``predicted`` method that
calculates a predicted data vector based on an input parameter vector. This is
virtually all that is problem-specific in an inverse problem. If you want to
use gradient-based optimization (or linear problems) you'll need to implement
the ``jacobian``  method as well that calculates the Jacobian (or sensitivity)
matrix.

Linear Regression
+++++++++++++++++

Here is an example of how to implement a simple linear regression using the
:class:`~fatiando.inversion.misfit.Misfit` class.

What we want to do is fit :math:`f(a, b, x) = y = ax + b` to find a and b.
Putting a and b in a parameter vector ``p = [a, b]`` we get:

.. math::

    \mathbf{d} = \mathbf{A} \mathbf{p}

where :math:`\mathbf{d}` is the data vector containing all the values of y
and :math:`\mathbf{A}` is the Jacobian matrix with the values of x in the first
column and 1 in the second column.

All we have to do to implement a solver for this problem is write the
``predicted`` (to calculate y from values of a and b) and ``jacobian`` (to
calculate the Jacobian matrix):

First, I'll load numpy and the ``Misfit`` class.

>>> import numpy as np
>>> from fatiando.inversion import Misfit

Now, I'll make my regression class. Note ``Misfit`` wants a 1D data vector, no
matter what your data is (line, grid, volume).

>>> class Regression(Misfit):
...     "Perform a linear regression"
...     def __init__(self, x, y):
...         # Call the initialization of Misfit
...         super(Regression, self).__init__(data=y, nparams=2, islinear=True)
...         self.x = x  # Store this to use in the other methods
...     def predicted(self, p):
...         a, b = p
...         return a*self.x + b
...     def jacobian(self, p):
...         jac = np.ones((self.ndata, self.nparams))
...         jac[:, 0] = self.x
...         return jac

To test our regression, let's generate some data based on known values of a and
b.

>>> x = np.linspace(0, 5, 6)
>>> x
array([ 0.,  1.,  2.,  3.,  4.,  5.])
>>> y = 2*x + 5
>>> y
array([  5.,   7.,   9.,  11.,  13.,  15.])

We must create a solver object to perform our regression. Fitting the data
(running the optimization) is done by calling the ``fit`` method. The estimate
parameter vector is stored in the ``p_`` attribute. ``Misfit`` also provides a
``estimate_`` attribute that can be a specially formatted version of ``p_``.
It's better to use ``estimate_`` if you're not interested in the parameter
vector as it is.

>>> solver = Regression(x, y).fit()
>>> solver.estimate_
array([ 2.,  5.])

``predicted`` and ``residuals``  will use the cached values based
on ``p_`` if  the parameter vector is omitted.

>>> solver.predicted()
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> np.all(np.abs(solver.residuals()) < 10**10)
True

The Jacobian matrix is cached permanently for linear problems so it is only
calculated once.

>>> A = solver.jacobian(solver.p_)
>>> A
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.]])
>>> B = solver.jacobian(np.array([20, 30]))
>>> B
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.]])
>>> A is B
True

You can also configure the solver to use a different (non-linear) optimization
method:

>>> # Configure solver to use the Levemberg-Marquardt method
>>> solver.config('levmarq', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])
>>> np.all(np.abs(solver.residuals()) < 10**10)
True
>>> # or the Steepest Descent method
>>> solver.config('steepest', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])
>>> # or the Gauss-Newton method
>>> solver.config('newton', initial=[1, 1], maxit=5).fit().estimate_
array([ 2.,  5.])

The ``Misfit`` class keeps track of the optimization process and makes that
data available through the ``stats_`` attribute (a dictionary).

>>> list(sorted(solver.stats_))
['iterations', 'method', 'objective']
>>> solver.stats_['method']
"Newton's method"
>>> solver.stats_['iterations']
5

The ``'objective'`` key holds a list of the objective function value per
iteration of the optimization process.


Re-weighted least squares
+++++++++++++++++++++++++

``Misfit`` allows you to set weights to the data in the form of a weight
matrix or vector (the vector is assumed to be the diagonal of the weight
matrix). We can use this to perform a re-weighted least-squares fit to remove
outliers from our data.

>>> y_outlier = y.copy()
>>> y_outlier[3] += 20
>>> y_outlier
array([  5.,   7.,   9.,  31.,  13.,  15.])

First, we run the regression without any weights.

>>> solver = Regression(x, y_outlier).fit()
>>> print(np.array_repr(solver.estimate_, precision=3))
array([ 2.571,  6.905])

Now, we can use the inverse of the residuals to set the weights for our data.
We repeat this for a few iterations and should have our robust estimate by the
end of it.

>>> for i in range(20):
...     r = np.abs(solver.residuals())
...     # Avoid small residuals because of zero-division errors
...     r[r < 1e-10] = 1
...     _ = solver.set_weights(1/r).fit()
>>> solver.estimate_
array([ 2.,  5.])


Non-linear problems
+++++++++++++++++++

In this example, I want to fit a Gaussian to my data:

.. math::

    f(x) = a\exp(-b(x + c)^{2})

Function *f* is non-linear with respect to inversion parameters *a, b, c*.
Thus, we need to configure the solver and choose an optimization method before
we can call ``fit()``.

First, lets create our solver class based on ``Misfit`` and implement the
``predicted`` and ``jacobian`` methods.

>>> class GaussianFit(Misfit):
...     def __init__(self, x, y):
...         super(GaussianFit, self).__init__(
...             data=y, nparams=3, islinear=False)
...         self.x = x
...     def predicted(self, p):
...         a, b, c = p
...         return a*np.exp(-b*(self.x + c)**2)
...     def jacobian(self, p):
...         a, b, c = p
...         jac = np.empty((self.ndata, self.nparams))
...         var = self.x + c
...         exponential = np.exp(-b*var**2)
...         jac[:, 0] = exponential
...         jac[:, 1] = -a*exponential*(var**2)
...         jac[:, 2] = -a*exponential*2*b*var
...         return jac

Let's create some data to test this.

>>> x = np.linspace(0, 10, 1000)
>>> a, b, c = 100, 0.1, -2
>>> y = a*np.exp(-b*(x + c)**2)
>>> # Non-linear solvers have to be configured. Lets use Levemberg-Marquardt.
>>> solver = GaussianFit(x, y).config('levmarq', initial=[1, 1, 1]).fit()
>>> solver.estimate_
array([ 100. ,    0.1,   -2. ])
>>> np.all(np.abs(solver.residuals()) < 10**-10)
True
>>> # or Ant Colony Optimization
>>> _ = solver.config('acor', bounds=[50, 500, 0, 1, -20, 0], seed=0).fit()
>>> print(np.array_repr(solver.estimate_, precision=3))
array([ 100. ,    0.1,   -2. ])


For non-linear problems, the Jacobian is cached but not permanently. Calling
``jacobian`` twice in a row with the same parameter vector will not
trigger a computation and will return the cached value instead.

>>> A = solver.jacobian(np.array([1, 1, 1]))
>>> B = solver.jacobian(np.array([1, 1, 1]))
>>> A is B
True
>>> C = solver.jacobian(np.array([1, 1, 1.1]))
>>> A is C
False
>>> np.all(A == C)
False

----

"""
from __future__ import division
import copy
from abc import abstractmethod
import numpy as np
import scipy.sparse

from ..utils import safe_dot
from .base import (OptimizerMixin, OperatorMixin, CachedMethod,
                   CachedMethodPermanent)


class Misfit(OptimizerMixin, OperatorMixin):
    r"""
    An l2-norm data-misfit function.

    This is a kind of objective function that measures the misfit between
    observed data :math:`\bar{d}^o` and data predicted by a set of model
    parameters :math:`\bar{d} = \bar{f}(\bar{p})`.

    The l2-norm data-misfit is defined as:

    .. math::

        \phi (\bar{p}) = \bar{r}^T \bar{r}

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

    Parameters:

    * data : 1d-array
        The observed data vector :math:`\bar{d}^o`
    * nparams : int
        The number of parameters in parameter vector :math:`\bar{p}`
    * islinear : True or False
        Whether :math:`\bar{f}` is linear or not.
    * cache : True
        Whether or not to cache the output of some methods to avoid recomputing
        matrices and vectors when passed the same input parameter vector.

    """

    def __init__(self, data, nparams, islinear, cache=True):
        self.p_ = None
        self.nparams = nparams
        self.islinear = islinear
        self.data = data
        self.ndata = self.data.size
        self.weights = None
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
        r"""
        Set the data weights.

        Using weights for the data, the least-squares data-misfit function
        becomes:

        .. math::

            \phi = \bar{r}^T \bar{\bar{W}}\bar{r}

        Parameters:

        * weights : 1d-array or 2d-array or None
            Weights for the data vector.
            If None, will remove any weights that have been set before.
            If it is a 2d-array, it will be interpreted as the weight matrix
            :math:`\bar{\bar{W}}`.
            If it is a 1d-array, it will be interpreted as the diagonal of the
            weight matrix (all off-diagonal elements will default to zero).
            The weight matrix can be a sparse array from ``scipy.sparse``.

        """
        self.weights = weights
        if weights is not None:
            assert len(weights.shape) <= 2, \
                "Invalid weights array with shape {}. ".format(weights.shape) \
                + "Weights array should be 1d or 2d"
        if len(weights.shape) == 1:
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
            val = np.linalg.norm(residuals)**2
        else:
            val = np.sum(self.weights*(residuals**2))
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
            grad = np.array(grad).ravel()
        grad *= -2*self.regul_param
        return grad
