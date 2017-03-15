r"""
Defines base classes to represent a data-misfit functions (l2-norm, etc)

These classes can be used to implement parameter estimation problems
(inversions). They automate most of the boiler plate required and provide
direct access to ready-made optimization routines and regularization.

For now, only implements an l2-norm data misfit:

* :class:`~fatiando.inversion.misfit.Misfit`: an l2-norm data-misfit function

See the documentation for :mod:`fatiando.inversion` for examples of using
``Misfit``.

----

"""
from __future__ import division, absolute_import
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
            is_cached = (isinstance(meth, CachedMethod) or
                         isinstance(meth, CachedMethodPermanent))
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
