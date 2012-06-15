"""
Cython implementation of the potential field effects of right rectangular prisms
"""
__all__ = ['pot', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']

import numpy

from libc.math cimport log, atan2, sqrt
# Import Cython definitions for numpy
cimport numpy

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_T

from fatiando.constants import SI2EOTVOS, SI2MGAL, G


def pot(xp, yp, zp, prisms, dens=None):
    """
    Calculates the gravitational potential.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    cdef int i, j, k
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (-x[i]*y[j]*numpy.log(z[k] + r)
                              - y[j]*z[k]*numpy.log(x[i] + r)
                              - x[i]*z[k]*numpy.log(y[j] + r)
                              + 0.5*x[i]*x[i]*numpy.arctan2(z[k]*y[j], x[i]*r)
                              + 0.5*y[j]*y[j]*numpy.arctan2(z[k]*x[i], y[j]*r)
                              + 0.5*z[k]*z[k]*numpy.arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G
    return res

def gx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    cdef int i, j, k
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (y[j]*numpy.log(z[k] + r) +
                              z[k]*numpy.log(y[j] + r) -
                              x[i]*numpy.arctan2(z[k]*y[j], x[i]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL
    return res

def gy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:
    
    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    cdef int i, j, k
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = numpy.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (z[k]*numpy.log(x[i] + r) +
                              x[i]*numpy.log(z[k] + r) -
                              y[j]*numpy.arctan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL
    return res

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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = (dx1*log(dy1 + r) + dy1*log(dx1 + r) -
                      dz1*atan2(dx1*dy1, dz1*r))
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += -(dx2*log(dy1 + r) + dy1*log(dx2 + r) -
                        dz1*atan2(dx2*dy1, dz1*r))
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += -(dx1*log(dy2 + r) + dy2*log(dx1 + r) -
                        dz1*atan2(dx1*dy2, dz1*r))
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += (dx2*log(dy2 + r) + dy2*log(dx2 + r) -
                       dz1*atan2(dx2*dy2, dz1*r))
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += -(dx1*log(dy1 + r) + dy1*log(dx1 + r) -
                        dz2*atan2(dx1*dy1, dz2*r))
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += (dx2*log(dy1 + r) + dy1*log(dx2 + r) -
                       dz2*atan2(dx2*dy1, dz2*r))
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += (dx1*log(dy2 + r) + dy2*log(dx1 + r) -
                       dz2*atan2(dx1*dy2, dz2*r))
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits 
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = atan2(dy1*dz1, dx1*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += -atan2(dy1*dz1, dx2*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += -atan2(dy2*dz1, dx1*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += atan2(dy2*dz1, dx2*r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += -atan2(dy1*dz2, dx1*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += atan2(dy1*dz2, dx2*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += atan2(dy2*dz2, dx1*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = -log(dz1 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += log(dz1 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += log(dz1 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += -log(dz1 + r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += log(dz2 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += -log(dz2 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += -log(dz2 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits 
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = -log(dy1 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += log(dy1 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += log(dy2 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += -log(dy2 + r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += log(dy1 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += -log(dy1 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += -log(dy2 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits 
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = atan2(dz1*dx1, dy1*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += -atan2(dz1*dx2, dy1*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += -atan2(dz1*dx1, dy2*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += atan2(dz1*dx2, dy2*r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += -atan2(dz2*dx1, dy1*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += atan2(dz2*dx2, dy1*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += atan2(dz2*dx1, dy2*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits 
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = -log(dx1 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += log(dx2 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += log(dx1 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += -log(dx2 + r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += log(dx1 + r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += -log(dx2 + r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += -log(dx1 + r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
            dx1, dx2 = x1 - xp[l], x2 - xp[l]
            dy1, dy2 = y1 - yp[l], y2 - yp[l]
            dz1, dz2 = z1 - zp[l], z2 - zp[l]
            # Evaluate the integration limits 
            r = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            kernel = atan2(dx1*dy1, dz1*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz1*dz1)
            kernel += -atan2(dx2*dy1, dz1*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz1*dz1)
            kernel += -atan2(dx1*dy2, dz1*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz1*dz1)
            kernel += atan2(dx2*dy2, dz1*r)
            r = sqrt(dx1*dx1 + dy1*dy1 + dz2*dz2)
            kernel += -atan2(dx1*dy1, dz2*r)
            r = sqrt(dx2*dx2 + dy1*dy1 + dz2*dz2)
            kernel += atan2(dx2*dy1, dz2*r)
            r = sqrt(dx1*dx1 + dy2*dy2 + dz2*dz2)
            kernel += atan2(dx1*dy2, dz2*r)
            r = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
            kernel += -atan2(dx2*dy2, dz2*r)
            res[l] += kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS
    return res
