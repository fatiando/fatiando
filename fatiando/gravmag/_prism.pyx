#cython: embedsignature=True
"""
Cython implementation of the gravity and magnetic fields of right rectangular
prisms.
"""
import numpy

from libc.math cimport log, atan2, sqrt
# Import Cython definitions for numpy
cimport numpy
cimport cython

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from fatiando.constants import SI2EOTVOS, SI2MGAL, G, CM, T2NT
from fatiando import utils

__all__ = ['potential', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
           'gzz', 'tf']

def tf(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms,
       double inc, double dec, pmag=None):
    """
    Calculate the total-field anomaly of prisms.

    .. note:: Input units are SI. Output is in nT

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. *prisms* can also be a :class:`~fatiando.mesher.PrismMesh`.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T intensity, pintensity, kernel, r, r_sqr
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    cdef DTYPE_T fx, fy, fz, mx, my, mz, pmx, pmy, pmz
    if len(xp) != len(yp) != len(zp):
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pintensity = pmag
            pmx, pmy, pmz = fx, fy, fz
        else:
            pintensity = numpy.linalg.norm(pmag)
            pmx, pmy, pmz = numpy.array(pmag) / pintensity
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                intensity = mag
                mx, my, mz = fx, fy, fz
            else:
                intensity = numpy.linalg.norm(mag)
                mx, my, mz = numpy.array(mag) / intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        # Calculate on all computation points
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            for k in range(2):
                intensity *= -1.
                for j in range(2):
                    for i in range(2):
                        r_sqr = x[i] ** 2 + y[j] ** 2 + z[k] ** 2
                        r = sqrt(r_sqr)
                        res[l] += ((-1.) ** (i + j)) * intensity * (
                            0.5 * (my * fz + mz * fy) * log((r - x[i]) / (r + x[i]))
                            + 0.5 * (mx * fz + mz * fx) * log((r - y[j]) / (r + y[j]))
                            - (mx * fy + my * fx) * log(r + z[k])
                            - mx * fx * atan2(x[i] * y[j], x[i] ** 2 + z[k] * r + z[k] ** 2)
                            - my * fy * atan2(x[i] * y[j], r_sqr + z[k] * r - x[i] ** 2)
                            + mz * fz * atan2(x[i] * y[j], z[k] * r))
    res *= CM * T2NT
    return res

def potential(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
              numpy.ndarray[DTYPE_T, ndim=1] yp not None,
              numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the gravitational potential.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = (x[i] * y[j] * log(z[k] + r)
                                  + y[j] * z[k] * log(x[i] + r)
                                  + x[i] * z[k] * log(y[j] + r)
                                  - 0.5 * x[i] ** 2 * atan2(z[k] * y[j], x[i] * r)
                                  - 0.5 * y[j] ** 2 * atan2(z[k] * x[i], y[j] * r)
                                  - 0.5 * z[k] ** 2 * atan2(x[i] * y[j], z[k] * r))
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res

def gx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        # Minus because Nagy et al (2000) give the formula for the
                        # gradient of the potential. Gravity is -grad(V)
                        kernel = -(y[j] * log(z[k] + r)
                                   + z[k] * log(y[j] + r)
                                   - x[i] * atan2(z[k] * y[j], x[i] * r))
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G * SI2MGAL
    return res

def gy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        # Minus because Nagy et al (2000) give the formula for the
                        # gradient of the potential. Gravity is -grad(V)
                        kernel = -(z[k] * log(x[i] + r)
                                   + x[i] * log(z[k] + r)
                                   - y[j] * atan2(x[i] * z[k], y[j] * r))
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G * SI2MGAL
    return res

def gz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
       numpy.ndarray[DTYPE_T, ndim=1] yp not None,
       numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        # Minus because Nagy et al (2000) give the formula for the
                        # gradient of the potential. Gravity is -grad(V)
                        kernel = -(x[i] * log(y[j] + r)
                                   + y[j] * log(x[i] + r)
                                   - z[k] * atan2(x[i] * y[j], z[k] * r))
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G * SI2MGAL
    return res

def gxx(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = -atan2(z[k] * y[j], x[i] * r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res

def gxy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = log(z[k] + r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res

def gxz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = log(y[j] + r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res

def gyy(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = -atan2(z[k] * x[i], y[j] * r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res

def gyz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = log(x[i] + r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res

def gzz(numpy.ndarray[DTYPE_T, ndim=1] xp not None,
        numpy.ndarray[DTYPE_T, ndim=1] yp not None,
        numpy.ndarray[DTYPE_T, ndim=1] zp not None, prisms, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    cdef unsigned int l, size, i, j, k
    cdef numpy.ndarray[DTYPE_T, ndim=1] res, x, y, z
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
    x = numpy.zeros(2, dtype=DTYPE)
    y = numpy.zeros(2, dtype=DTYPE)
    z = numpy.zeros(2, dtype=DTYPE)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        for l in xrange(size):
            # First thing to do is make the computation point P the origin of
            # the coordinate system
            x[0] = x2 - xp[l]
            x[1] = x1 - xp[l]
            y[0] = y2 - yp[l]
            y[1] = y1 - yp[l]
            z[0] = z2 - zp[l]
            z[1] = z1 - zp[l]
            # Evaluate the integration limits
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        r = sqrt(x[i] ** 2 + y[j] ** 2 + z[k] ** 2)
                        kernel = -atan2(x[i] * y[j], z[k] * r)
                        res[l] += ((-1.) ** (i + j + k)) * kernel * density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G * SI2EOTVOS
    return res
