"""
Data-misfit functions.

* :class:`~fatiando.inversion.misfit.L2Norm`

----

"""
from __future__ import division
import numpy
import scipy.sparse

from .base import Objective
from ..utils import safe_dot


class L2Norm(Objective):
    """
    """

    def __init__(self, data, nparams, weights=None, islinear=False):
        super(L2Norm, self).__init__(nparams, islinear=islinear)
        self.data = data
        self.ndata = len(data)
        self._cache['predicted'] = {'hash':'', 'array':None}
        self._cache['jacobian'] = {'hash':'', 'array':None}
        self._cache['hessian'] = {'hash':'', 'array':None}
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = None

    def _get_predicted(self, p):
        raise NotImplementedError("Predicted data not implemented")

    def _get_jacobian(self, p):
        raise NotImplementedError("Jacobian matrix not implemented")

    def set_data(self, data):
        """
        """
        self.data = data
        return self

    def set_weights(self, weights):
        """
        """
        self.weights = scipy.sparse.diags(weights, 0)
        # Weights change the Hessian
        self._cache['hessian'] = {'hash':'', 'array':None}
        return self

    def residuals(self, p):
        """
        """
        return self.data - self.predicted(p)

    def predicted(self, p):
        """
        """
        if p is None:
            pred = 0
        else:
            hash = self.hasher(p)
            if hash != self._cache['predicted']['hash']:
                self._cache['predicted']['array'] = self._get_predicted(p)
                self._cache['predicted']['hash'] = hash
            pred = self._cache['predicted']['array']
        return pred

    def jacobian(self, p):
        """
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
        """
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
        """
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
        """
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
