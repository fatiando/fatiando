r"""
Calculate the potential fields of a homogeneous sphere.

**Magnetic**

Calculates the total field anomaly. Uses the formula in Blakely (1995).

* :func:`~fatiando.gravmag.sphere.tf`: calculates the total-field anomaly

Remember that:

The magnetization :math:`\mathbf{M}` and the dipole moment :math:`\mathbf{m}`
are related with the volume V:

.. math::

    \mathbf{M} = \dfrac{\mathbf{m}}{V}.

The total-field anomaly is:

.. math::

    \Delta T = |\mathbf{T}| - |\mathbf{F}|,

where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is a
reference (regional) field. The forward modeling functions
:func:`~fatiando.gravmag.sphere.bx`, :func:`~fatiando.gravmag.sphere.by`,
and :func:`~fatiando.gravmag.sphere.bz` calculate the 3 components of the
field perturbation :math:`\Delta\mathbf{F}`

.. math::

    \Delta\mathbf{F} = \mathbf{T} - \mathbf{F}.

Then the total-field anomaly caused by the sphere is

.. math::

    \Delta T \approx \hat{\mathbf{F}}\cdot\Delta\mathbf{F}.

**Gravity**

Calculates the gravitational acceleration and gravity gradient tensor
components.

* :func:`fatiando.gravmag.sphere.gz`
* :func:`fatiando.gravmag.sphere.gxx`
* :func:`fatiando.gravmag.sphere.gxy`
* :func:`fatiando.gravmag.sphere.gxz`
* :func:`fatiando.gravmag.sphere.gyy`
* :func:`fatiando.gravmag.sphere.gyz`
* :func:`fatiando.gravmag.sphere.gzz`

**References**

Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic Applications,
Cambridge University Press.

----

"""
import numpy

from fatiando.constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from fatiando import utils


def tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    """
    Calculate the total-field anomaly of spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    tf = numpy.zeros_like(xp)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pintensity = pmag
            pmx, pmy, pmz = fx, fy, fz
        else:
            pintensity = numpy.linalg.norm(pmag)
            pmx, pmy, pmz = numpy.array(pmag)/pintensity
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props
                              and pmag is None):
            continue
        radius = sphere.radius
        # Get the intensity and unit vector from the magnetization
        if pmag is None:
            mag = sphere.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                intensity = mag
                mx, my, mz = fx, fy, fz
            else:
                intensity = numpy.linalg.norm(mag)
                mx, my, mz = numpy.array(mag)/intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        # Calculate the 3 components of B
        dotprod = mx*x + my*y + mz*z
        r_sqr = x**2 + y**2 + z**2
        r5 = r_sqr**(2.5)
        moment = intensity*(4.*numpy.pi*(radius**3)/3.)
        bx = moment*(3*dotprod*x - r_sqr*mx)/r5
        by = moment*(3*dotprod*y - r_sqr*my)/r5
        bz = moment*(3*dotprod*z - r_sqr*mz)/r5
        tf += (fx*bx + fy*by + fz*bz)
    tf *= CM*T2NT
    return tf

def gz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in mGal!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_cb = (dx**2 + dy**2 + dz**2)**(1.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*dz/r_cb
    res *= G*SI2MGAL
    return res

def gxx(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_xx` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(((3*dx**2) - r_2)/r_5)
    res *= G*SI2EOTVOS
    return res

def gxy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_xy` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(3*dx*dy)/r_5
    res *= G*SI2EOTVOS
    return res

def gxz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_xz` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(3*dx*dz)/r_5
    res *= G*SI2EOTVOS
    return res

def gyy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_yy` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(((3*dy**2) - r_2)/r_5)
    res *= G*SI2EOTVOS
    return res

def gyz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_yz` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(3*dy*dz)/r_5
    res *= G*SI2EOTVOS
    return res

def gzz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_zz` gravity gradient component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI and output in Eotvos!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. Those
        without will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        mass = density*4.*numpy.pi*(radius**3)/3.
        res += mass*(((3*dz**2) - r_2)/r_5)
    res *= G*SI2EOTVOS
    return res
