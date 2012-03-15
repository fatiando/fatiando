# Copyright 2010 The Fatiando a Terra Development Team
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
Potential field transformations, like upward continuation, derivatives and
total mass.

**Transformations**

* :func:`fatiando.potential.transform.upcontinue`


----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 20-Oct-2010'

import math
import time

import numpy

from fatiando import logger, gridder
from fatiando.potential import _transform


def upcontinue(gz, z0, height, xp, yp, dims):
    """
    Upward continue :math:`g_z` data using numerical integration of the
    analytical formula:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    **Data needs to be on a regular grid.**

    **UNITS**: SI for all coordinates, mGal for :math:`g_z`

    NOTE: be aware of coordinate systems! The *x*, *y*, *z* coordinates are
    x -> North, y -> East and z -> **DOWN**.

    Parameters:
    
    * gz
        Array with the gravity values on the grid points
    * z0
        Original z coordinate of the observations
        (Remember: z is positive downward!)
    * height
        How much higher to move the gravity field (should be POSITIVE!)
        Will be subtracted from z0 to obtain the new z coordinate of the
        continued observations.
    * xp, yp
        Arrays with the x and y coordinates of the grid points
    * dims
        [dy, dx]: the grid spacing in the y and x directions

    Returns:
    
    * gzcont
        Array with the upward continued :math:`g_z`

    """
    log = logger.dummy('fatiando.potential.transform.upcontinue')
    dy, dx = dims
    if len(xp) != len(yp):
        raise ValueError, "xp and yp arrays must have same lengths"
    if height < 0:
        raise ValueError, "'height' should be positive"
    newz = z0 - height
    log.info("Upward continuation:")
    log.info("  original z coordinate: %g m" % (z0))
    log.info("  height increment: %g m" % (height))
    log.info("  new z coordinate: %g m" % (newz))
    start = time.time()
    gzcont = _transform.upcontinue(gz, z0, newz, xp, yp, dx, dy)
    end = time.time()
    log.info("  time: %g s" % (end - start))
    return gzcont
