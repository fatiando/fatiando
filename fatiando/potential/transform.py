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
Potential field transformations.
Ex: upward continuation, derivatives and total mass.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 20-Oct-2010'

import math
import time

import numpy

from fatiando import logger, gridder
from fatiando.potential import _transform

log = logger.dummy()


def upcontinue(height, gz, nodes, dims):
    """
    Upward continue :math:`g_z` data using numerical integration of the
    analytical formula:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    Data needs to be on a regular grid.

    **UNITS**: SI for all coordinates, mGal for :math:`g_z`

    NOTE: be aware of coordinate systems! The *x*, *y*, *z* coordinates are
    x -> North, y -> East and z -> **DOWN**.

    Parameters:
    * height
        How much higher to move the gravity field (should be POSITIVE!)
        Will be summed to the current height of the grid.
    * gz
        Gravity values on the grid points
    * nodes
        [x, y, z]: List of arrays with x, y, z coordinates of the grid points
    * dims
        [dy, dx]: the grid spacing in the y and x directions

    Returns:
    * gzcont
        Upward continued :math:`g_z`

    """
    dy, dx = dims
    xs, ys, zs = nodes
    if len(xs) != len(ys) != len(zs):
        raise ValueError, "nodes has x,y,z coordinates with different lengths"
    if height < 0:
        raise ValueError, "'height' should be positive"
    log.info("Upward continuation:")
    log.info("  height increment: %g m" % (height))
    newzs = zs - height
    start = time.time()
    gzcont = _transform.upcontinue(xs, ys, zs, newzs, gz, dx, dy)
    end = time.time()
    log.info("  time: %g s" % (end - start))
    return gzcont
