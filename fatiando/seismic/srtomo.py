"""
Straight-ray 2D travel-time tomography (i.e., does not consider reflection or
refraction)

**Solver**

* :class:`~fatiando.seismic.srtomo.SRTomo`: Data misfit class that runs the
  tomography.

**Functions**

* :func:`~fatiando.seismic.srtomo.slowness2vel`: Safely convert slowness to
  velocity (avoids zero division)

**Examples**


----

"""
from __future__ import division
import numpy
import scipy.sparse

from ..inversion.base import Misfit
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

    Examples:

    Using simple synthetic data:

    >>> from fatiando.mesher import Square, SquareMesh
    >>> from fatiando.seismic import ttime2d
    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
    >>> model = [Square([0, 10, 0, 5], {'vp':2}),
    ...          Square([0, 10, 5, 10], {'vp':5})]
    >>> src = (5, 0)
    >>> srcs = [src, src, src]
    >>> recs = [(0, 0), (5, 10), (10, 0)]
    >>> # Calculate the synthetic travel-times
    >>> ttimes = ttime2d.straight(model, 'vp', srcs, recs)
    >>> print ttimes
    [ 2.5  3.5  2.5]
    >>> # Make a mesh to represent the two blocks
    >>> mesh = SquareMesh((0, 10, 0, 10), shape=(2, 1))
    >>> # Run the tomography
    >>> tomo = SRTomo(ttimes, srcs, recs, mesh)
    >>> tomo.fit().estimate_
    array([ 2.,  5.])

    Using the steepest descent method to solve (no linear systems):

    >>> # Use steepest descent to solve this (requires an initial guess)
    >>> tomo.config(method='steepest', initial=[0, 0]).fit().estimate_
    array([ 2.,  5.])

    .. note::

        A simple way to plot the results is to use the ``addprop`` method of
        the mesh and then pass the mesh to :func:`fatiando.vis.map.squaremesh`.

    """

    def __init__(self, ttimes, srcs, recs, mesh):
        super(SRTomo, self).__init__(
            data=ttimes,
            positional=dict(srcs=srcs, recs=recs),
            model=dict(mesh=mesh),
            nparams=mesh.size, islinear=True)

    def _get_jacobian(self, p):
        """
        Build the Jacobian (sensitivity) matrix using the travel-time data
        stored.
        """
        srcs, recs = self.positional['srcs'], self.positional['recs']
        i, j, v = [], [], []
        for k, c in enumerate(self.model['mesh']):
            column = ttime2d.straight([c], '', srcs, recs,
                                      velocity=1.)
            nonzero = numpy.flatnonzero(column)
            i.extend(nonzero)
            j.extend(k * numpy.ones_like(nonzero))
            v.extend(column[nonzero])
        shape = (self.ndata, self.nparams)
        return scipy.sparse.coo_matrix((v, (i, j)), shape).tocsr()

    def _get_predicted(self, p):
        pred = safe_dot(self.jacobian(p), p)
        if len(pred.shape) > 1:
            pred = numpy.array(pred.T).ravel()
        return pred

    def fit(self):
        """
        Solve the tomography for the velocity of each cell.

        Actually solves for the slowness to make the inverse problem linear.
        The ``estimate_`` attribute holds the estimated velocities and ``p_``
        the respective slownesses.

        See the docstring of :class:`~fatiando.seismic.srtomo.SRTomo` for
        examples.

        """
        super(SRTomo, self).fit()
        self._estimate = slowness2vel(self.p_, tol=10 ** -8)
        return self


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
    velocity = numpy.array(slowness)
    velocity[slowness < tol] = 0
    divide = slowness >= tol
    velocity[divide] = 1. / slowness[divide]
    return velocity
