"""
.. topic:: Python + Numpy implementation.

    Module :mod:`fatiando.pot.prism` loads all functions from
    ``fatiando.pot._prism``, which contain the Python + Numpy
    implementation. There is also the slightly faster Cython module
    ``fatiando.pot._cprism``. If it is available, then will substitude
    the Python + Numpy functions with its functions. All input and output are
    the same but there is a 15% speed increase.

"""

__all__ = ['pot', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz',
           'tf']

import numpy
from numpy import sqrt, log, arctan2

from fatiando.constants import SI2EOTVOS, SI2MGAL, G, CM, T2NT
from fatiando import utils


def pot(xp, yp, zp, prisms, dens=None):
    """
    Calculates the gravitational potential.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (x[i]*y[j]*log(z[k] + r)
                              + y[j]*z[k]*log(x[i] + r)
                              + x[i]*z[k]*log(y[j] + r)
                              - 0.5*x[i]*x[i]*arctan2(z[k]*y[j], x[i]*r)
                              - 0.5*y[j]*y[j]*arctan2(z[k]*x[i], y[j]*r)
                              - 0.5*z[k]*z[k]*arctan2(x[i]*y[j], z[k]*r))
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
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(y[j]*log(z[k] + r)
                               + z[k]*log(y[j] + r)
                               - x[i]*arctan2(z[k]*y[j], x[i]*r))
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
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(z[k]*log(x[i] + r)
                               + x[i]*log(z[k] + r)
                               - y[j]*arctan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(x[i]*log(y[j] + r)
                               + y[j]*log(x[i] + r)
                               - z[k]*arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gxx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = -arctan2(z[k]*y[j], x[i]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = log(z[k] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = log(y[j] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = -arctan2(z[k]*x[i], y[j]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = log(x[i] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.msh.ddd.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

        .. warning:: Uses this value for **all** prisms! Not only the ones that
            have ``'density'`` as a property.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = -arctan2(x[i]*y[j], z[k]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def tf(xp, yp, zp, prisms, inc, dec, pmag=None, pinc=None, pdec=None):
    """
    Calculate the total-field anomaly of prisms.

    .. note:: Input units are SI. Output is in nT

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.msh.ddd.Prism`
        The model used to calculate the total field anomaly.
        Prisms must have the physical property ``'magnetization'`` will be
        ignored. If the physical properties ``'inclination'`` and
        ``'declination'`` are not present, will use the values of *inc* and
        *dec* instead (regional field).
        *prisms* can also be a :class:`~fatiando.msh.ddd.PrismMesh`.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : float or None
        If not None, will use this value instead of the ``'magnetization'``
        property of the prisms. Use this, e.g., for sensitivity matrix building.
    * pinc : float or None
        If not None, will use this value instead of the ``'inclination'``
        property of the prisms. Use this, e.g., for sensitivity matrix building.
    * pdec : float or None
        If not None, will use this value instead of the ``'declination'``
        property of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                              and pmag is None):
            continue
        if pmag is None:
            magnetization = prism.props['magnetization']
        else:
            magnetization = pmag
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Get the 3 components of the unit vector in the direction of the
        # magnetization from the inclination and declination
        # 1) given by the function
        if pinc is not None and pdec is not None:
            mx, my, mz = utils.dircos(pinc, pdec)
        # 2) given by the prism
        elif 'inclination' in prism.props and 'declination' in prism.props:
            mx, my, mz = utils.dircos(prism.props['inclination'],
                                      prism.props['declination'])
        # 3) Use in the direction of the regional field
        else:
            mx, my, mz = fx, fy, fz
        # Now calculate the total field anomaly
        for k in range(2):
            magnetization *= -1
            z_sqr = z[k]**2
            for j in range(2):
                y_sqr = y[j]**2
                for i in range(2):
                    x_sqr = x[i]**2
                    xy = x[i]*y[j]
                    r_sqr = x_sqr + y_sqr + z_sqr
                    r = sqrt(r_sqr)
                    zr = z[k]*r
                    res += ((-1.)**(i + j))*magnetization*(
                          0.5*(my*fz + mz*fy)*log((r - x[i])/(r + x[i]))
                        + 0.5*(mx*fz + mz*fx)*log((r - y[j])/(r + y[j]))
                        - (mx*fy + my*fx)*log(r + z[k])
                        - mx*fx*arctan2(xy, x_sqr + zr + z_sqr)
                        - my*fy*arctan2(xy, r_sqr + zr - x_sqr)
                        + mz*fz*arctan2(xy, zr))
    res *= CM*T2NT
    return res
