r"""
Calculate the potential fields of a homogeneous sphere.

**Magnetic**

Calculates the magnetic effect produced by an sphere. The functions are
based on Blakely (1995).

* :func:`~fatiando.gravmag.sphere.tf`: calculates the total-field anomaly
* :func:`~fatiando.gravmag.sphere.bx`: calculates the x component of the
  induction
* :func:`~fatiando.gravmag.sphere.by`: calculates the y component of the
  induction
* :func:`~fatiando.gravmag.sphere.bz`: calculates the z component of the
  induction

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

* :func:`~fatiando.gravmag.sphere.gz`
* :func:`~fatiando.gravmag.sphere.gxx`
* :func:`~fatiando.gravmag.sphere.gxy`
* :func:`~fatiando.gravmag.sphere.gxz`
* :func:`~fatiando.gravmag.sphere.gyy`
* :func:`~fatiando.gravmag.sphere.gyz`
* :func:`~fatiando.gravmag.sphere.gzz`


**Auxiliary Functions**

Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

with respect to the variables :math:`x`, :math:`y`, and :math:`z`. In
this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2},

and :math:`R` is the radius of a sphere with centre at the Cartesian
coordinates :math:`\nu`, :math:`\eta` and :math:`\zeta`.

These second derivatives are used to calculate the total field magnetic anomaly
and the gravity gradient tensor components.

* :func:`~fatiando.gravmag.sphere.kernelxx`
* :func:`~fatiando.gravmag.sphere.kernelxy`
* :func:`~fatiando.gravmag.sphere.kernelxz`
* :func:`~fatiando.gravmag.sphere.kernelyy`
* :func:`~fatiando.gravmag.sphere.kernelyz`
* :func:`~fatiando.gravmag.sphere.kernelzz`

**References**

Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic Applications,
Cambridge University Press.

----

"""
from __future__ import division

import numpy

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils

try:
    from . import _sphere
except ImportError:
    _sphere = None


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

    Example:

    >>> from fatiando import mesher, gridder, utils
    >>> # Set the inclination and declination of the regional field
    >>> inc, dec = -30, 45
    >>> # Create a sphere model
    >>> model = [
    ...         # One with induced magnetization
    ...         mesher.Sphere(1000, 1000, 600, 500, {'magnetization':5}),
    ...         # and one with remanent
    ...         mesher.Sphere(-1000, -1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(10, 70, -5)})]
    >>> # Create a regular grid at 100m height
    >>> shape = (4, 4)
    >>> area = (-3000, 3000, -3000, 3000)
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the anomaly for a given regional field
    >>> for t in tf(xp, yp, zp, model, inc, dec):
    ...     print '%15.8e' % t
     2.72951375e+01
     3.63637351e+01
     5.35842876e+00
     6.65189557e-02
     6.01998831e+01
    -1.71920499e+03
     3.30025228e+00
    -5.13176612e+00
     1.47871812e+00
    -3.96103758e+01
    -1.85654021e+02
     2.18002960e+01
    -2.82713826e+00
    -1.10341542e+01
     1.93353982e+01
     2.03174254e+01

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pmx, pmy, pmz = pmag * fx, pmag * fy, pmag * fz
        else:
            pmx, pmy, pmz = pmag
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props
                              and pmag is None):
            continue
        # Get the intensity and unit vector from the magnetization
        if pmag is None:
            mag = sphere.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        else:
            mx, my, mz = pmx, pmy, pmz
        _sphere.tf(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, fx, fy, fz, res)
    res *= CM * T2NT
    return res


def bx(xp, yp, zp, spheres):
    """
    Calculates the x component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bx: array
        The x component of the magnetic induction

    Example:

    >>> from fatiando import mesher, gridder, utils
    >>> # Create a model formed by two spheres
    >>> # The magnetization of each sphere is a vector
    >>> model = [
    ...         mesher.Sphere(1000, 1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(13, -10, 28)}),
    ...         mesher.Sphere(-1000, -1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(10, 70, -5)})]
    >>> # Create a regular grid at 100m height
    >>> shape = (4, 4)
    >>> area = (-3000, 3000, -3000, 3000)
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the bx component
    >>> for b in bx(xp, yp, zp, model):
    ...     print '%15.8e' % b
     1.58002397e+01
    -1.76820799e+01
    -1.48049248e+01
    -5.75238567e+00
     9.17572697e+01
    -4.94607307e+02
    -7.92213872e+01
    -4.37781621e+00
     2.97032297e+01
     7.36803996e+01
    -1.73332620e+03
     1.15884125e+02
     4.55847152e+00
    -1.31173236e+01
    -6.42912671e+01
     2.98847909e+01

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.bx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


def by(xp, yp, zp, spheres):
    """
    Calculates the y component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * by: array
        The y component of the magnetic induction

    Example:

    >>> from fatiando import mesher, gridder, utils
    >>> # Create a model formed by two spheres
    >>> # The magnetization of each sphere is a vector
    >>> model = [
    ...         mesher.Sphere(1000, 1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(13, -10, 28)}),
    ...         mesher.Sphere(-1000, -1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(10, 70, -5)})]
    >>> # Create a regular grid at 100m height
    >>> shape = (4, 4)
    >>> area = (-3000, 3000, -3000, 3000)
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the by component
    >>> for b in by(xp, yp, zp, model):
    ...     print '%15.8e' % b
     2.51394441e+01
     5.71383698e+01
     7.46666729e+00
    -4.53730551e+00
     7.44792258e+00
     8.22174414e+01
     4.53451310e+01
    -3.06885735e+01
    -2.49929765e+01
    -8.41961087e+01
    -9.17412395e+02
    -3.18422413e+01
    -1.32728556e+01
    -3.03825859e+01
     6.67990083e+01
     4.21366247e+01

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.by(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


def bz(xp, yp, zp, spheres):
    """
    Calculates the z component of the magnetic induction produced by spheres.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres without ``'magnetization'`` will be
        ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bz: array
        The z component of the magnetic induction

    Example:

    >>> from fatiando import mesher, gridder, utils
    >>> # Create a model formed by two spheres
    >>> # The magnetization of each sphere is a vector
    >>> model = [
    ...         mesher.Sphere(1000, 1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(13, -10, 28)}),
    ...         mesher.Sphere(-1000, -1000, 600, 500,
    ...             {'magnetization':utils.ang2vec(10, 70, -5)})]
    >>> # Create a regular grid at 100m height
    >>> shape = (4, 4)
    >>> area = (-3000, 3000, -3000, 3000)
    >>> xp, yp, zp = gridder.regular(area, shape, z=-100)
    >>> # Calculate the bz component
    >>> for b in bz(xp, yp, zp, model):
    ...     print '%15.8e' % b
    -1.13152279e+01
    -3.24362266e+01
    -1.63235805e+01
    -4.48136597e+00
    -1.27492012e+01
     2.89101261e+03
    -1.30263918e+01
    -9.64182996e+00
    -6.45566985e+00
     3.32987598e+01
    -7.08905624e+02
    -5.55139945e+01
    -1.35745203e+00
     2.91949888e+00
    -2.78345635e+01
    -1.69425703e+01

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('magnetization' not in sphere.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = sphere.props['magnetization']
        _sphere.bz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   mx, my, mz, res)
    res *= CM * T2NT
    return res


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
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                   density, res)
    res *= G * SI2MGAL
    return res


def gxx(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> for g in gxx(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
     1.71893959e-06
    -6.02473678e-06
    -6.02473678e-06
     1.71893959e-06
     1.39192195e-05
     2.76067131e-05
     2.76067131e-05
     1.39192195e-05
     1.39192195e-05
     2.76067131e-05
     2.76067131e-05
     1.39192195e-05
     1.71893959e-06
    -6.02473678e-06
    -6.02473678e-06
     1.71893959e-06

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gxy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the gxy component
    >>> for g in gxy(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
     5.30415646e-06
     7.47898359e-06
    -7.47898359e-06
    -5.30415646e-06
     7.47898359e-06
     1.10426852e-04
    -1.10426852e-04
    -7.47898359e-06
    -7.47898359e-06
    -1.10426852e-04
     1.10426852e-04
     7.47898359e-06
    -5.30415646e-06
    -7.47898359e-06
     7.47898359e-06
     5.30415646e-06

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gxz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the gxz component
    >>> for g in gxz(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
     8.84026077e-07
     1.24649726e-06
    -1.24649726e-06
    -8.84026077e-07
     3.73949179e-06
     5.52134262e-05
    -5.52134262e-05
    -3.73949179e-06
     3.73949179e-06
     5.52134262e-05
    -5.52134262e-05
    -3.73949179e-06
     8.84026077e-07
     1.24649726e-06
    -1.24649726e-06
    -8.84026077e-07

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gxz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gyy(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the gyy component
    >>> for g in gyy(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
     1.71893959e-06
     1.39192195e-05
     1.39192195e-05
     1.71893959e-06
    -6.02473678e-06
     2.76067131e-05
     2.76067131e-05
    -6.02473678e-06
    -6.02473678e-06
     2.76067131e-05
     2.76067131e-05
    -6.02473678e-06
     1.71893959e-06
     1.39192195e-05
     1.39192195e-05
     1.71893959e-06

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gyy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gyz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the gyz component
    >>> for g in gyz(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
     8.84026077e-07
     3.73949179e-06
     3.73949179e-06
     8.84026077e-07
     1.24649726e-06
     5.52134262e-05
     5.52134262e-05
     1.24649726e-06
    -1.24649726e-06
    -5.52134262e-05
    -5.52134262e-05
    -1.24649726e-06
    -8.84026077e-07
    -3.73949179e-06
    -3.73949179e-06
    -8.84026077e-07

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gyz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def gzz(xp, yp, zp, spheres, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient component.

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

    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = [mesher.Sphere(0, 0, 5, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the gzz component
    >>> for g in gzz(xp, yp, zp, sphere):
    ...     print '%15.8e' % g
    -3.43787919e-06
    -7.89448267e-06
    -7.89448267e-06
    -3.43787919e-06
    -7.89448267e-06
    -5.52134262e-05
    -5.52134262e-05
    -7.89448267e-06
    -7.89448267e-06
    -5.52134262e-05
    -5.52134262e-05
    -7.89448267e-06
    -3.43787919e-06
    -7.89448267e-06
    -7.89448267e-06
    -3.43787919e-06

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    for sphere in spheres:
        if sphere is None or ('density' not in sphere.props and dens is None):
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        _sphere.gzz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius,
                    density, res)
    res *= G * SI2EOTVOS
    return res


def kernelxx(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxx = kernelxx(xp, yp, zp, sphere)
    >>> for k in kxx:
    ...     print '%15.8e' % k
     2.57596223e-05
    -9.02852807e-05
    -9.02852807e-05
     2.57596223e-05
     2.08590131e-04
     4.13707675e-04
     4.13707675e-04
     2.08590131e-04
     2.08590131e-04
     4.13707675e-04
     4.13707675e-04
     2.08590131e-04
     2.57596223e-05
    -9.02852807e-05
    -9.02852807e-05
     2.57596223e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxx(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res


def kernelxy(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial y},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxy = kernelxy(xp, yp, zp, sphere)
    >>> for k in kxy:
    ...     print '%15.8e' % k
     7.94868344e-05
     1.12078279e-04
    -1.12078279e-04
    -7.94868344e-05
     1.12078279e-04
     1.65483070e-03
    -1.65483070e-03
    -1.12078279e-04
    -1.12078279e-04
    -1.65483070e-03
     1.65483070e-03
     1.12078279e-04
    -7.94868344e-05
    -1.12078279e-04
     1.12078279e-04
     7.94868344e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res


def kernelxz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial z},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxz = kernelxz(xp, yp, zp, sphere)
    >>> for k in kxz:
    ...     print '%15.8e' % k
     1.32478057e-05
     1.86797132e-05
    -1.86797132e-05
    -1.32478057e-05
     5.60391397e-05
     8.27415349e-04
    -8.27415349e-04
    -5.60391397e-05
     5.60391397e-05
     8.27415349e-04
    -8.27415349e-04
    -5.60391397e-05
     1.32478057e-05
     1.86797132e-05
    -1.86797132e-05
    -1.32478057e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gxz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res


def kernelyy(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kyy = kernelyy(xp, yp, zp, sphere)
    >>> for k in kyy:
    ...     print '%15.8e' % k
     2.57596223e-05
     2.08590131e-04
     2.08590131e-04
     2.57596223e-05
    -9.02852807e-05
     4.13707675e-04
     4.13707675e-04
    -9.02852807e-05
    -9.02852807e-05
     4.13707675e-04
     4.13707675e-04
    -9.02852807e-05
     2.57596223e-05
     2.08590131e-04
     2.08590131e-04
     2.57596223e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gyy(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res


def kernelyz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y \partial z},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kyz = kernelyz(xp, yp, zp, sphere)
    >>> for k in kyz:
    ...     print '%15.8e' % k
     1.32478057e-05
     5.60391397e-05
     5.60391397e-05
     1.32478057e-05
     1.86797132e-05
     8.27415349e-04
     8.27415349e-04
     1.86797132e-05
    -1.86797132e-05
    -8.27415349e-04
    -8.27415349e-04
    -1.86797132e-05
    -1.32478057e-05
    -5.60391397e-05
    -5.60391397e-05
    -1.32478057e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gyz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res


def kernelzz(xp, yp, zp, sphere):
    r"""
    Calculates the function

    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial z^2},

    where

    .. math::

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}

    and

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be
        calculated
    * sphere : object of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp


    Example:

    >>> from fatiando import mesher, gridder
    >>> # Create a sphere model
    >>> sphere = mesher.Sphere(0, 0, 5, 1)
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kzz = kernelzz(xp, yp, zp, sphere)
    >>> for k in kzz:
    ...     print '%15.8e' % k
    -5.15192445e-05
    -1.18304851e-04
    -1.18304851e-04
    -5.15192445e-05
    -1.18304851e-04
    -8.27415349e-04
    -8.27415349e-04
    -1.18304851e-04
    -1.18304851e-04
    -8.27415349e-04
    -8.27415349e-04
    -1.18304851e-04
    -5.15192445e-05
    -1.18304851e-04
    -1.18304851e-04
    -5.15192445e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    _sphere.gzz(xp, yp, zp, sphere.x, sphere.y, sphere.z, sphere.radius, 1,
                res)
    return res
