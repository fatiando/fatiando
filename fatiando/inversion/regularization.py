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



