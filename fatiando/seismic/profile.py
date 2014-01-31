"""
Forward modeling and inversion of vertical seismic profiling (VSP) data.

In this kind of profiling, the wave source is located at the surface on top of
the well. The seismic waves are then measured at different depths along the
well.

**Forward modeling**

* :func:`~fatiando.seismic.profile.layered_straight_ray`: Computes straight-ray
  first-arrival travel-times for a layered model.


**Inversion**

* :class:`~fatiando.seismic.profile.LayeredStraight`: Inverts for the
  slownesses of a layered model assuming straight ray paths.

----

"""
from __future__ import division
import numpy

from . import ttime2d
from .. import utils
from ..mesher import Square
from ..inversion.base import Misfit


def layered_straight_ray(thickness, velocity, zp):
    """
    Calculates straight-ray (no refraction) travel-times in a layered model.

    The source is assumed to be at z = 0 and on the well.
    The z-axis is positive downward.

    Parameters:

    * thickness : list
        The thickness of each layer in order of increasing depth
    * velocity : list
        The velocity of each layer in order of increasing depth
    * zp : list
        The depths of the measurement stations (seismometers)

    Returns:

    * travel_times : array
        The first-arrival travel-times calculated at the measurement stations.

    Examples:

    >>> # Make a 4 layer model
    >>> thicks = [10, 20, 10, 30]
    >>> vels = [2, 4, 10, 5]
    >>> # Set the recording depths
    >>> zs = [10, 30, 40, 70]
    >>> # Calculate the travel-times from a surface source
    >>> layered_straight_ray(thicks, vels, zs)
    array([  5.,  10.,  11.,  17.])

    """
    if len(thickness) != len(velocity):
        raise ValueError, "thickness and velocity must have same length"
    nlayers = len(thickness)
    zmax = sum(thickness)
    z = [sum(thickness[:i]) for i in xrange(nlayers + 1)]
    layers = [Square((0, zmax, z[i], z[i + 1]), props={'vp':velocity[i]})
              for i in xrange(nlayers)]
    srcs = [(0, 0)]*len(zp)
    recs = [(0, z) for z in zp]
    return ttime2d.straight(layers, 'vp', srcs, recs)

class LayeredStraight(Misfit):
    r"""
    Inversion of straight-ray travel-times for the velocity of a layered medium

    Assumes that the source is at the top of the well and that rays follow a
    straight path (no reflection or refraction). Also assumes known
    thicknesses (may be a fine discretization if real thickness is not known).

    Actually solves for the slowness (1/velocity) so that the problem becomes
    linear and more manageable. Use the ``estimate_`` attribute to get the
    estimated velocities. Slowness with stored in the estimated parameter
    vector ``p_``.

    Uses :func:`fatiando.seismic.ttime2d.straight` for forward modeling.

    .. note::

        In most cases requires regularization. The recommended types are
        :class:`~fatiando.inversion.regularization.Damping` and
        :class:`~fatiando.inversion.regularization.Smoothness1D`.


    Parameters:

    * traveltimes : list
        The first-arrival travel-times calculated at the measurement stations
    * zp : list
        The depths of the measurement stations (seismometers)
    * thickness : list
        The thickness of each layer in order of increasing depth

    Notes:

    The ith travel-time :math:`t_i` measured at depth
    :math:`z_i` is a function of the wave velocity :math:`v_j` and distance
    :math:`d_{ij}` that it traveled in each layer

    .. math::

        t_i(z_i) = \sum\limits_{j=1}^M \frac{d_{ij}}{v_j}

    The distance :math:`d_{ij}` is smaller or equal to the thickness of the
    layer :math:`s_j`. Notice that :math:`d_{ij} = 0` if the jth layer is below
    :math:`z_i`, :math:`d_{ij} = s_j` if the jth layer is above :math:`z_i`,
    and :math:`d_{ij} < s_j` if :math:`z_i` is inside the jth layer.

    To make :math:`t_i` linear with respect to :math:`v_j`, we can use the
    *slowness* :math:`w_j = 1/v_j` instead of velocity

    .. math::

        t_i(z_i) = \sum\limits_{j=1}^M d_{ij} w_j

    Thus, the parameters we want to estimate in this inversion are the
    slownesses of each layer.

    From the above equation, we can see that the element :math:`G_{ij}` of the
    Jacobian (sensitivity) matrix is given by

    .. math::

        G_{ij} = d_{ij}


    Examples:

    Using some synthetic data produced by
    :func:`~fatiando.seismic.profile.layered_straight_ray` and assuming that
    the thickness of the layers is known:

    >>> import numpy as np
    >>> # Make a 4 layer model
    >>> thicks = [10, 20, 10, 30]
    >>> vels = [2, 4, 10, 8]
    >>> # Set the recording stations
    >>> zp = range(1, sum(thicks), 5)
    >>> # Calculate the travel-times
    >>> tts = layered_straight_ray(thicks, vels, zp)
    >>> # Solve for the slowness assuming known thicknesses
    >>> solver = LayeredStraight(tts, zp, thicks).fit()
    >>> # The estimated velocities
    >>> solver.estimate_
    array([ 2.,  4., 10.,  8.])
    >>> # and the corresponding slownesses
    >>> solver.p_
    array([ 0.5  ,  0.25 ,  0.1  ,  0.125])
    >>> # Check the fit
    >>> np.all(np.abs(solver.residuals()) < 10**-10)
    True

    See the :ref:`Cookbook <cookbook>` for more complex examples that use
    regularization and unknown thicknesses.

    """

    def __init__(self, traveltimes, zp, thickness):
        super(LayeredStraight, self).__init__(data=traveltimes,
            positional={'zp':zp},
            model={'thickness':thickness},
            nparams=len(thickness),
            islinear=True)

    def _get_predicted(self, p):
        return layered_straight_ray(self.model['thickness'], 1./p,
                                    self.positional['zp'])

    def _get_jacobian(self, p):
        thicks = self.model['thickness']
        nlayers = len(thicks)
        zmax = numpy.sum(thicks)
        z = [numpy.sum(thicks[:i]) for i in xrange(nlayers + 1)]
        layers = [Square((0, zmax, z[i], z[i + 1]), props={'vp':1.})
                  for i in xrange(nlayers)]
        srcs = [(0, 0)]*self.ndata
        recs = numpy.transpose(
            [numpy.zeros(self.ndata), self.positional['zp']])
        jac = numpy.transpose(
            [ttime2d.straight([l], 'vp', srcs, recs) for l in layers])
        return jac

    def fit(self):
        """
        Solve for the velocities of each layer.

        Actually uses slowness instead of velocity to make the problem linear.
        The estimated slowness is stored in the ``p_`` attribute. The
        corresponding velocities are in ``estimate_``.

        See the docstring of :class:`~fatiando.seismic.profile.LayeredStraight`
        for examples.

        """
        super(LayeredStraight, self).fit()
        self._estimate = 1./self.p_
        return self
