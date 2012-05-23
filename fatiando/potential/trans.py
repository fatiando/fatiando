"""
Potential field transformations, like upward continuation, derivatives and
total mass.

**Transformations**

* :func:`~fatiando.potential.trans.upcontinue`

----

"""

import math
import time

import numpy

from fatiando import logger, gridder
from fatiando.potential import _transform


log = logger.dummy('fatiando.potential.transform')

def upcontinue(gz, z0, height, xp, yp, dims):
    """
    Upward continue :math:`g_z` data using numerical integration of the
    analytical formula:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    .. note:: Data needs to be on a regular grid!

    .. note:: Units are SI for all coordinates and mGal for :math:`g_z`

    .. note:: be aware of coordinate systems! The *x*, *y*, *z* coordinates are:
        x -> North, y -> East and z -> **DOWN**.

    Parameters:
    
    * gz : array
        The gravity values on the grid points
    * z0 : float
        Original z coordinate of the observations

        .. note:: Remember that z is positive downward!
        
    * height : float
        How much higher to move the gravity field (should be POSITIVE!)
        Will be subtracted from *z0* to obtain the new z coordinate of the
        continued observations.
    * xp, yp : arrays
        The x and y coordinates of the grid points
    * dims : list = [dy, dx]
        The grid spacing in the y and x directions

    Returns:
    
    * gzcont : array
        The upward continued :math:`g_z`

    """
    if len(xp) != len(yp):
        raise ValueError("xp and yp arrays must have same lengths")
    if height < 0:
        raise ValueError("'height' should be positive")
    dy, dx = dims
    newz = z0 - height
    log.info("Upward continuation using the analytical formula:")
    log.info("  original z coordinate: %g m" % (z0))
    log.info("  height increment: %g m" % (height))
    log.info("  new z coordinate: %g m" % (newz))
    log.info("  grid spacing [dy, dx]: %s m" % (str(dims)))
    start = time.time()
    gzcont = _transform.upcontinue(gz, z0, newz, xp, yp, dx, dy)
    end = time.time()
    log.info("  time to calculate: %g s" % (end - start))
    return gzcont
