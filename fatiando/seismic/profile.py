# Copyright 2012 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Direct modeling and inversion of seismic profiles.

**VERTICAL SEISMIC PROFILING**

* :func:`fatiando.seismic.profile.vertical`
* :func:`fatiando.seismic.profile.invert_vertical`

Model and invert vertical seismic profiling data. In this kind of profiling, the
wave source is located at the surface on top of the well. The travel-times of
first arrivals is then measured at different depths along the well. The
ith travel-time :math:`t_i` measured at depth :math:`z_i` is a function of the
wave velocity :math:`v_j` and distance :math:`d_j` that it traveled in each
layer

.. math::

    t_i(z_i) = \\sum\\limits_{j=1}^M \\frac{d_j}{v_j}

The distance :math:`d_j` is smaller or equal to the thickness of the layer
:math:`s_j`. Notice that :math:`d_j = 0` if the jth layer is bellow :math:`z_i`,
:math:`d_j = s_j` if the jth layer is above :math:`z_i`, and :math:`d_j < s_j`
if :math:`z_i` is inside the jth layer.

To generate synthetic seismic profiling data, use
:func:`fatiando.seismic.profile.vertical` like so::

    >>> from fatiando.seismic import profile
    >>> # Make the synthetic 4 layer model
    >>> thicks = [10, 20, 10, 30]
    >>> vels = [2, 4, 10, 5]
    >>> # Make an array with the z_i
    >>> zs = [10, 30, 40, 70]
    >>> # Calculate the travel-times
    >>> for t in profile.vertical(thicks, vels, zs):    
    ...     print '%.1f' % (t), 
    5.0 10.0 11.0 17.0


To make :math:`t_i` linear with respect to :math:`v_j`, we can use
*slowness* :math:`w_j` instead of velocity

.. math::

    t_i(z_i) = \\sum\\limits_{j=1}^M d_j w_j



----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'

import time

import numpy

from fatiando.seismic import traveltime
from fatiando.mesher.dd import Square
from fatiando import logger, inversion, utils

log = logger.dummy()

def vertical(thickness, velocity, zp):
    """
    Calculates the first-arrival travel-times for given a layered model.
    Simulates a vertical seismic profile.

    The source is assumed to be at z = 0. The z-axis is positive downward.

    Parameters:

    * thickness
        List with the thickness of each layer in order of increasing depth
    * velocity
        List with the velocity of each layer in order of increasing depth
    * zp
        List with the depths of the measurement stations (seismometers)
        
    Returns:

    * travel_times
        List with the first-arrival travel-times calculated at the measurement
        stations.
    
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
    return traveltime.straight_ray_2d(layers, 'vp', srcs, recs)

def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
