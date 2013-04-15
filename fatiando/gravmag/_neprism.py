"""
Testing numexpr to speedup prism computations
"""
import numpy

from numexpr import evaluate

from fatiando.constants import SI2EOTVOS, SI2MGAL, G, CM, T2NT
from fatiando import utils


def tf(xp, yp, zp, prisms, inc, dec, pmag=None, pinc=None, pdec=None):
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
        Prisms must have the physical property ``'magnetization'`` will be
        ignored. If the physical properties ``'inclination'`` and
        ``'declination'`` are not present, will use the values of *inc* and
        *dec* instead (regional field).
        *prisms* can also be a :class:`~fatiando.mesher.PrismMesh`.
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
    kernel = ''.join([
        'res + ((-1.)**(i + j))*intensity*(',
        '0.5*(my*fz + mz*fy)*log((r - x)/(r + x))',
        ' + 0.5*(mx*fz + mz*fx)*log((r - y)/(r + y))',
        ' - (mx*fy + my*fx)*log(r + z)',
        ' - mx*fx*arctan2(xy, x_sqr + zr + z_sqr)',
        ' - my*fy*arctan2(xy, r_sqr + zr - x_sqr)',
        ' + mz*fz*arctan2(xy, zr))'])
    res = numpy.zeros_like(xp)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        pintensity = numpy.linalg.norm(pmag)
        pmx, pmy, pmz = numpy.array(pmag)/pintensity
    for prism in prisms:
        if (prism is None or
            ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            intensity = numpy.linalg.norm(prism.props['magnetization'])
            mx, my, mz = numpy.array(prism.props['magnetization'])/intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Now calculate the total field anomaly
        for k in range(2):
            intensity *= -1
            z = zs[k]
            z_sqr = evaluate('z**2')
            for j in range(2):
                y = ys[j]
                y_sqr = evaluate('y**2')
                for i in range(2):
                    x = xs[i]
                    x_sqr = evaluate('x**2')
                    xy = evaluate('x*y')
                    r_sqr = evaluate('x_sqr + y_sqr + z_sqr')
                    r = evaluate('sqrt(r_sqr)')
                    zr = evaluate('z*r')
                    res = evaluate(kernel)
    res *= CM*T2NT
    return res

def potential(xp, yp, zp, prisms, dens=None):
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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = ' '.join([
        'x*y*log(z + r) + y*z*log(x + r) + x*z*log(y + r)',
        '- 0.5*(x**2)*arctan2(z*y, x*r)',
        '- 0.5*(y**2)*arctan2(z*x, y*r)',
        '- 0.5*(z**2)*arctan2(x*y, z*r)'])
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res

def gx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Minus because Nagy et al (2000) give the formula for the
    # gradient of the potential. Gravity is -grad(V)
    kernel = '-(y*log(z + r) + z*log(y + r) - x*arctan2(z*y, x*r))'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Minus because Nagy et al (2000) give the formula for the
    # gradient of the potential. Gravity is -grad(V)
    kernel = '-(z*log(x + r) + x*log(z + r) - y*arctan2(x*z, y*r))'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Minus because Nagy et al (2000) give the formula for the
    # gradient of the potential. Gravity is -grad(V)
    kernel = '-(x*log(y + r) + y*log(x + r) - z*arctan2(x*y, z*r))'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res

def gxx(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = '-arctan2(z*y, x*r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = 'log(z + r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = 'log(y + r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = '-arctan2(z*x, y*r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = 'log(x + r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

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
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    kernel = '-arctan2(x*y, z*r)'
    expr = 'res + ((-1.)**(i + j + k))*(%s)*density' % (kernel)
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        xs = [evaluate('x2 - xp'), evaluate('x1 - xp')]
        ys = [evaluate('y2 - yp'), evaluate('y1 - yp')]
        zs = [evaluate('z2 - zp'), evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            z = zs[k]
            for j in range(2):
                y = ys[j]
                for i in range(2):
                    x = xs[i]
                    r = evaluate('sqrt(x**2 + y**2 + z**2)')
                    res = evaluate(expr)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res
