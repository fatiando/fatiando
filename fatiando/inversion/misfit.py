r"""
Defines the ``Misfit`` class to represent a data-misfit function.

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
