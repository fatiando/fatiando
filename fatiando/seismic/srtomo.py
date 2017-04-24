"""
Straight-ray 2D travel-time tomography (i.e., does not consider reflection or
refraction)

.. warning::

    The SRTomo class is meant as a teaching tool and not a **real tomography
    code**. It approximates the seismic rays with straight lines, thus ignoring
    refraction (Snell's Law). Results can be significantly distorted,
    particularly on highly heterogeneous media.


**Solver**

* :class:`~fatiando.seismic.srtomo.SRTomo`: Data misfit class that runs the
  tomography.

**Functions**

* :func:`~fatiando.seismic.srtomo.slowness2vel`: Safely convert slowness to
  velocity (avoids zero division)

----
"""
from __future__ import division, absolute_import
from future.builtins import super
import numpy as np
import scipy.sparse

from ..inversion import Misfit
from ..utils import safe_dot
from . import ttime2d


class SRTomo(Misfit):
    """
    2D travel-time straight-ray tomography.

    Use the :meth:`~fatiando.seismic.srtomo.SRTomo.fit` method to run the
    tomography and produce a velocity estimate. The estimate is stored in the
    ``estimate_`` attribute.

    Generaly requires regularization, like
    :class:`~fatiando.inversion.regularization.Damping` or
    :class:`~fatiando.inversion.regularization.Smoothness2D`.

    Parameters:

    * ttimes : array
        Array with the travel-times of the straight seismic rays.
    * srcs : list of lists
        List of the [x, y] positions of the sources.
    * recs : list of lists
        List of the [x, y] positions of the receivers.
    * mesh : :class:`~fatiando.mesher.SquareMesh` or compatible
        The mesh where the inversion (tomography) will take place.

    The ith travel-time is the time between the ith element in *srcs* and the
    ith element in *recs*.

    """

    def __init__(self, ttimes, srcs, recs, mesh):
        super().__init__(data=ttimes, nparams=mesh.size, islinear=True)
        self.srcs = srcs
        self.recs = recs
        self.mesh = mesh

    def jacobian(self, p):
        """
        Build the Jacobian (sensitivity) matrix.

        The matrix will contain the length of the path takes by the ray inside
        each cell of the mesh.

        Parameters:

        * p : 1d-array
            An estimate of the parameter vector or ``None``.

        Returns:

        * jac : 2d-array (sparse CSR matrix from ``scipy.sparse``)
            The Jacobian

        """
        srcs, recs = self.srcs, self.recs
        i, j, v = [], [], []
        for k, c in enumerate(self.mesh):
            column = ttime2d.straight([c], '', srcs, recs,
                                      velocity=1.)
            nonzero = np.flatnonzero(column)
            i.extend(nonzero)
            j.extend(k*np.ones_like(nonzero))
            v.extend(column[nonzero])
        shape = (self.ndata, self.nparams)
        return scipy.sparse.coo_matrix((v, (i, j)), shape).tocsr()

    def predicted(self, p):
        """
        Calculate the travel time data predicted by a parameter vector.

        Parameters:

        * p : 1d-array
            An estimate of the parameter vector

        Returns:

        * pred : 1d-array
            The predicted travel time data.

        """
        pred = safe_dot(self.jacobian(p), p)
        return pred

    def fmt_estimate(self, p):
        """
        Convert the estimated slowness to velocity.
        """
        return slowness2vel(self.p_, tol=10**-8)


def slowness2vel(slowness, tol=10 ** (-8)):
    """
    Safely convert slowness to velocity.

    Almost 0 slowness is mapped to 0 velocity.

    Parameters:

    * slowness : array
        The slowness values
    * tol : float
        Slowness < tol will be set to 0 velocity

    Returns:

    * velocity : array
        The converted velocities

    Examples:

    >>> import numpy as np
    >>> slow = np.array([1, 2, 0.000001, 4])
    >>> slowness2vel(slow, tol=0.00001)
    array([ 1.  ,  0.5 ,  0.  ,  0.25])

    """
    velocity = np.array(slowness)
    velocity[slowness < tol] = 0
    divide = slowness >= tol
    velocity[divide] = 1. / slowness[divide]
    return velocity
