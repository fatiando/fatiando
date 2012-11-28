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


def gz(numpy.ndarray[DTYPE_T, ndim=1] xp,
       numpy.ndarray[DTYPE_T, ndim=1] yp,
       numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = (dx1*log(dy1 + r) + dy1*log(dx1 + r) -
                      dz1*atan2(dx1*dy1, dz1*r))
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += -(dx2*log(dy1 + r) + dy1*log(dx2 + r) -
                        dz1*atan2(dx2*dy1, dz1*r))
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += -(dx1*log(dy2 + r) + dy2*log(dx1 + r) -
                        dz1*atan2(dx1*dy2, dz1*r))
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += (dx2*log(dy2 + r) + dy2*log(dx2 + r) -
                       dz1*atan2(dx2*dy2, dz1*r))
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += -(dx1*log(dy1 + r) + dy1*log(dx1 + r) -
                        dz2*atan2(dx1*dy1, dz2*r))
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += (dx2*log(dy1 + r) + dy1*log(dx2 + r) -
                       dz2*atan2(dx2*dy1, dz2*r))
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += (dx1*log(dy2 + r) + dy2*log(dx1 + r) -
                       dz2*atan2(dx1*dy2, dz2*r))
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += -(dx2*log(dy2 + r) + dy2*log(dx2 + r) -
                        dz2*atan2(dx2*dy2, dz2*r))
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gxx(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = atan2(dy1*dz1, dx1*r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += -atan2(dy1*dz1, dx2*r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += -atan2(dy2*dz1, dx1*r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += atan2(dy2*dz1, dx2*r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += -atan2(dy1*dz2, dx1*r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += atan2(dy1*dz2, dx2*r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += atan2(dy2*dz2, dx1*r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += -atan2(dy2*dz2, dx2*r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxy(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = -log(dz1 + r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += log(dz1 + r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += log(dz1 + r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += -log(dz1 + r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += log(dz2 + r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += -log(dz2 + r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += -log(dz2 + r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += log(dz2 + r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxz(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = -log(dy1 + r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += log(dy1 + r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += log(dy2 + r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += -log(dy2 + r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += log(dy1 + r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += -log(dy1 + r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += -log(dy2 + r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += log(dy2 + r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyy(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = atan2(dz1*dx1, dy1*r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += -atan2(dz1*dx2, dy1*r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += -atan2(dz1*dx1, dy2*r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += atan2(dz1*dx2, dy2*r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += -atan2(dz2*dx1, dy1*r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += atan2(dz2*dx2, dy1*r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += atan2(dz2*dx1, dy2*r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += -atan2(dz2*dx2, dy2*r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyz(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = -log(dx1 + r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += log(dx2 + r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += log(dx1 + r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += -log(dx2 + r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += log(dx1 + r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += -log(dx2 + r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += -log(dx1 + r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += log(dx2 + r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gzz(numpy.ndarray[DTYPE_T, ndim=1] xp,
        numpy.ndarray[DTYPE_T, ndim=1] yp,
        numpy.ndarray[DTYPE_T, ndim=1] zp, prisms, dens=None):
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
    cdef int l, size
    cdef numpy.ndarray[DTYPE_T, ndim=1] res
    cdef DTYPE_T density, kernel, r
    cdef DTYPE_T x1, x2, y1, y2, z1, z2, dx1, dx2, dy1, dy2, dz1, dz2
    size = len(xp)
    res = numpy.zeros(size, dtype=DTYPE)
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
            # Note: doing x1, x2 instead of x2, x1 because this cancels the
            # changed sign of the equations bellow with respect to the formula
            # of Nagy et al (2000)
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1**2 + dy1**2 + dz1**2)
            kernel = atan2(dx1*dy1, dz1*r)
            r = sqrt(dx2**2 + dy1**2 + dz1**2)
            kernel += -atan2(dx2*dy1, dz1*r)
            r = sqrt(dx1**2 + dy2**2 + dz1**2)
            kernel += -atan2(dx1*dy2, dz1*r)
            r = sqrt(dx2**2 + dy2**2 + dz1**2)
            kernel += atan2(dx2*dy2, dz1*r)
            r = sqrt(dx1**2 + dy1**2 + dz2**2)
            kernel += -atan2(dx1*dy1, dz2*r)
            r = sqrt(dx2**2 + dy1**2 + dz2**2)
            kernel += atan2(dx2*dy1, dz2*r)
            r = sqrt(dx1**2 + dy2**2 + dz2**2)
            kernel += atan2(dx1*dy2, dz2*r)
            r = sqrt(dx2**2 + dy2**2 + dz2**2)
            kernel += -atan2(dx2*dy2, dz2*r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res
