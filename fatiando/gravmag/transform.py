"""
Space domain potential field transformations, like upward continuation,
derivatives and total mass.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of the
  vertical component of gravity :math:`g_z` using numerical integration

----

"""
import numpy


def upcontinue(gz, height, xp, yp, dims):
    """
    Upward continue :math:`g_z` data using numerical integration of the
    analytical formula:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    .. note:: Data needs to be on a regular grid!

    .. note:: Units are SI for all coordinates and mGal for :math:`g_z`

    .. note:: be aware of coordinate systems!
        The *x*, *y*, *z* coordinates are:
        x -> North, y -> East and z -> **DOWN**.

    Parameters:

    * gz : array
        The gravity values on the grid points
    * height : float
        How much higher to move the gravity field (should be POSITIVE!)
    * xp, yp : arrays
        The x and y coordinates of the grid points
    * dims : list = [dy, dx]
        The grid spacing in the y and x directions

    Returns:

    * gzcont : array
        The upward continued :math:`g_z`

    """
    if xp.shape != yp.shape:
        raise ValueError("xp and yp arrays must have same shape")
    if height < 0:
        raise ValueError("'height' should be positive")
    dy, dx = dims
    area = dx * dy
    deltaz_sqr = (height) ** 2
    gzcont = numpy.zeros_like(gz)
    for x, y, g in zip(xp, yp, gz):
        gzcont += g * area * \
            ((xp - x) ** 2 + (yp - y) ** 2 + deltaz_sqr) ** (-1.5)
    gzcont *= abs(height) / (2 * numpy.pi)
    return gzcont
