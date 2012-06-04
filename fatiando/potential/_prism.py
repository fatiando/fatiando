"""
.. topic:: Python + Numpy implementation.

    Module :mod:`fatiando.potential.prism` loads all functions from
    ``fatiando.potential._prism``, which contain the Python + Numpy
    implementation. There is also the slightly faster Cython module
    ``fatiando.potentia._cprism``. If it is available, then will substitude
    the Python + Numpy functions with its functions. All input and output are
    the same but there is a 50% speed increase.
    
"""
import numpy
from numpy import sqrt, log, arctan2

SI2EOTVOS = 1000000000.0
SI2MGAL = 100000.0
G = 0.00000000006673 # The gravitational constant (SI)


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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (-x[i]*y[j]*log(z[k] + r)
                              - y[j]*z[k]*log(x[i] + r)
                              - x[i]*z[k]*log(y[j] + r)
                              + 0.5*x[i]*x[i]*arctan2(z[k]*y[j], x[i]*r)
                              + 0.5*y[j]*y[j]*arctan2(z[k]*x[i], y[j]*r)
                              + 0.5*z[k]*z[k]*arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G;
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (y[j]*log(z[k] + r) +
                              z[k]*log(y[j] + r) -
                              x[i]*arctan2(z[k]*y[j], x[i]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (z[k]*log(x[i] + r) +
                              x[i]*log(z[k] + r) -
                              y[j]*arctan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    kernel = (x[i]*log(y[j] + r) +
                              y[j]*log(x[i] + r) -
                              z[k]*arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units 
    res *= G*SI2MGAL;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should be a - sign with arctan
                    # but the tensor components seemed to be with the wrong sign
                    # so I took them out
                    kernel = arctan2(z[k]*y[j], x[i]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should not be a - sign with log
                    # but the tensor components seemed to be with the wrong sign
                    # so I put them in
                    kernel = -log(z[k] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should not be a - sign with log
                    # but the tensor components seemed to be with the wrong sign
                    # so I put them in
                    kernel = -log(y[j] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should be a - sign with arctan
                    # but the tensor components seemed to be with the wrong sign
                    # so I took them out
                    kernel = arctan2(z[k]*x[i], y[j]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should not be a - sign with log
                    # but the tensor components seemed to be with the wrong sign
                    # so I put them in
                    kernel = -log(x[i] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
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
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.
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
        x = [prism.x1 - xp, prism.x2 - xp]
        y = [prism.y1 - yp, prism.y2 - yp]
        z = [prism.z1 - zp, prism.z2 - zp]
        # Evaluate the integration limits 
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])
                    # In Nagy et al (2000) there should be a - sign with arctan
                    # but the tensor components seemed to be with the wrong sign
                    # so I took them out
                    kernel = arctan2(x[i]*y[j], z[k]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units 
    res *= G*SI2EOTVOS;
    return res
