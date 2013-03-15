"""
Cython implementation of the potential field effects of right rectangular prisms
"""
import numpy

from libc.math cimport log, atan2, sqrt
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from fatiando.constants import SI2EOTVOS, SI2MGAL, G
from fatiando import utils

__all__ = ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']


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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        # Minus because Nagy et al (2000) give the formula for the
                        # gradient of the potential. Gravity is -grad(V)
                        kernel = -(x[i]*log(y[j] + r)
                                   + y[j]*log(x[i] + r)
                                   - z[k]*atan2(x[i]*y[j], z[k]*r))
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = -atan2(z[k]*y[j], x[i]*r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = log(z[k] + r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = log(y[j] + r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = -atan2(z[k]*x[i], y[j]*r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = log(x[i] + r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
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
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
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
                        r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        kernel = -atan2(x[i]*y[j], z[k]*r)
                        res[l] += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res
