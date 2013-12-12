"""
Regularization.

----
"""
from __future__ import division
import numpy
import scipy.sparse

from .base import Objective
from ..utils import safe_dot


class Damping(Objective):
    """

    Examples:

        >>> import numpy
        >>> damp = Damping(3)
        >>> damp
        Damping(nparams=3)
        >>> p = numpy.array([0, 0, 0])
        >>> damp.value(p)
        0.0
        >>> damp.hessian(p).todense()
        matrix([[ 2.,  0.,  0.],
                [ 0.,  2.,  0.],
                [ 0.,  0.,  2.]])
        >>> damp.gradient(p)
        array([ 0.,  0.,  0.])
        >>> p = numpy.array([1, 0, 0])
        >>> damp.value(p)
        1.0
        >>> damp.hessian(p).todense()
        matrix([[ 2.,  0.,  0.],
                [ 0.,  2.,  0.],
                [ 0.,  0.,  2.]])
        >>> damp.gradient(p)
        array([ 2.,  0.,  0.])


    """

    def __init__(self, nparams):
        super(Damping, self).__init__(nparams, islinear=True)

    def __repr__(self):
        return 'Damping(nparams=%d)' % (self.nparams)

    def hessian(self, p):
        # This is cheap so there is no need to cache it
        return 2*scipy.sparse.identity(self.nparams).tocsr()

    def gradient(self, p):
        if p is None:
            grad = 0
        else:
            grad = 2.*p
        return grad

    def value(self, p):
        return numpy.linalg.norm(p)**2

class Smoothness(Objective):
    """
    Smoothness regularization.
    """

    def __init__(self, fdmat):
        super(Smoothness, self).__init__(fdmat.shape[1], islinear=True)
        self._cache['hessian'] = {'hash':'',
                                  'array':2*safe_dot(fdmat.T, fdmat)}

    def hessian(self, p):
        return self._cache['hessian']['array']

    def gradient(self, p):
        if p is None:
            grad = 0
        else:
            grad = safe_dot(self.hessian(p), p)
        return grad

    def value(self, p):
        # Need to divide by 2 because the hessian is 2*R.T*R
        return numpy.sum(p*safe_dot(self.hessian(p), p))/2.

class Smoothness1D(Smoothness):
    def __init__(self, npoints):
        super(Smoothness1D, self).__init__(fd1d(npoints))

class Smoothness2D(Smoothness):
    def __init__(self, shape):
        super(Smoothness2D, self).__init__(fd2d(shape))

class TotalVariation(Objective):
    def __init__(self, beta, fdmat):
        super(TotalVariation, self).__init__(
            nparams=fdmat.shape[1], islinear=False)
        self.beta = beta
        self._fdmat = fdmat

    def value(self, p):
        return numpy.linalg.norm(safe_dot(self._fdmat, p), 1)

    def hessian(self, p):
        derivs = safe_dot(self._fdmat, p)
        q_matrix = scipy.sparse.diags(self.beta/((derivs**2 + self.beta)**1.5),
                                      0).tocsr()
        return safe_dot(self._fdmat.T, q_matrix*self._fdmat)

    def gradient(self, p):
        derivs = safe_dot(self._fdmat, p)
        q = derivs/numpy.sqrt(derivs**2 + self.beta)
        grad = safe_dot(self._fdmat.T, q)
        if len(grad.shape) > 1:
            grad = numpy.array(grad.T).ravel()
        return grad

class TotalVariation1D(TotalVariation):
    def __init__(self, beta, npoints):
        super(TotalVariation1D, self).__init__(beta, fd1d(npoints))

class TotalVariation2D(TotalVariation):
    def __init__(self, beta, shape):
        super(TotalVariation2D, self).__init__(beta, fd2d(shape))

def fd1d(size):
    i = range(size - 1) + range(size - 1)
    j = range(size - 1) + range(1, size)
    v = [1]*(size - 1) + [-1]*(size - 1)
    return scipy.sparse.coo_matrix((v, (i, j)), (size - 1, size)).tocsr()

def fd2d(shape):
    ny, nx = shape
    nderivs = (nx - 1)*ny + (ny - 1)*nx
    I, J, V = [], [], []
    deriv = 0
    param = 0
    for i in xrange(ny):
        for j in xrange(nx - 1):
            I.extend([deriv, deriv])
            J.extend([param, param + 1])
            V.extend([1, -1])
            deriv += 1
            param += 1
        param += 1
    param = 0
    for i in xrange(ny - 1):
        for j in xrange(nx):
            I.extend([deriv, deriv])
            J.extend([param, param + nx])
            V.extend([1, -1])
            deriv += 1
            param += 1
    return scipy.sparse.coo_matrix((V, (I, J)), (nderivs, nx*ny)).tocsr()
