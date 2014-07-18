"""
Epicenter determination in 2D, i.e., assuming a flat Earth.

There are solvers for the following approximations.

**Homogeneous Earth**

Estimates the (x, y) cartesian coordinates of the epicenter based on
travel-time residuals between S and P waves, assuming a homogeneous velocity
distribution.

* :func:`~fatiando.seismic.epic2d.Homogeneous`

----

"""
from __future__ import division
import numpy

from ..inversion.base import Misfit


class Homogeneous(Misfit):

    r"""
    Estimate the epicenter assuming a homogeneous Earth.

    Parameters:

    * ttres : array
        Travel-time residuals between S and P waves
    * recs : list of lists
        List with the (x, y) coordinates of the receivers
    * vp : float
        Assumed velocity of P waves
    * vs : float
        Assumed velocity of S waves

    .. note::

        The recommended solver for this inverse problem is the
        Levemberg-Marquardt method. Since this is a non-linear problem, set the
        desired method and initial solution using the
        :meth:`~fatiando.inversion.base.FitMixin.config` method.
        See the example bellow.

    Examples:

    Using synthetic data.

        >>> from fatiando.mesher import Square
        >>> from fatiando.seismic import ttime2d
        >>> # Generate synthetic travel-time residuals
        >>> area = (0, 10, 0, 10)
        >>> vp = 2
        >>> vs = 1
        >>> model = [Square(area, props={'vp':vp, 'vs':vs})]
        >>> # The true source (epicenter)
        >>> src = (5, 5)
        >>> recs = [(5, 0), (5, 10), (10, 0)]
        >>> srcs = [src, src, src]
        >>> #The travel-time residual between P and S waves
        >>> ptime = ttime2d.straight(model, 'vp', srcs, recs)
        >>> stime = ttime2d.straight(model, 'vs', srcs, recs)
        >>> ttres = stime - ptime
        >>> # Pass the data to the solver class
        >>> solver = Homogeneous(ttres, recs, vp, vs).config('levmarq',
        ...                                                  initial=[1, 1])
        >>> # Estimate the epicenter
        >>> x, y = solver.fit().estimate_
        >>> print "(%.4f, %.4f)" % (x, y)
        (5.0000, 5.0000)

    Notes:

    The travel-time residual measured by the ith receiver is a function of the
    (x, y) coordinates of the epicenter:

    .. math::

        t_{S_i} - t_{P_i} = \Delta t_i (x, y) =
        \left(\frac{1}{V_S} - \frac{1}{V_P} \right)
        \sqrt{(x_i - x)^2 + (y_i - y)^2}

    The elements :math:`G_{i1}` and :math:`G_{i2}` of the Jacobian matrix for
    this data type are

    .. math::

        G_{i1}(x, y) = -\left(\frac{1}{V_S} - \frac{1}{V_P} \right)
        \frac{x_i - x}{\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    .. math::

        G_{i2}(x, y) = -\left(\frac{1}{V_S} - \frac{1}{V_P} \right)
        \frac{y_i - y}{\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    The Hessian matrix is approximated by
    :math:`2\bar{\bar{G}}^T\bar{\bar{G}}` (Gauss-Newton method).

    """

    def __init__(self, ttres, recs, vp, vs):
        super(Homogeneous, self).__init__(
            data=ttres,
            positional=dict(recs=numpy.array(recs)),
            model=dict(vp=vp, vs=vs),
            nparams=2, islinear=False)

    def _get_predicted(self, p):
        x, y = p
        alpha = 1. / self.model['vs'] - 1. / self.model['vp']
        return alpha * numpy.sqrt((self.positional['recs'][:, 0] - x) ** 2 +
                                  (self.positional['recs'][:, 1] - y) ** 2)

    def _get_jacobian(self, p):
        x, y = p
        alpha = 1. / self.model['vs'] - 1. / self.model['vp']
        sqrt = numpy.sqrt((self.positional['recs'][:, 0] - x) ** 2 +
                          (self.positional['recs'][:, 1] - y) ** 2)
        jac = numpy.transpose([
            -alpha * (self.positional['recs'][:, 0] - x) / sqrt,
            -alpha * (self.positional['recs'][:, 1] - y) / sqrt])
        return jac
