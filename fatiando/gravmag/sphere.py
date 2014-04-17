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

from __future__ import division

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

def kernelxx(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x^2},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxx = kernelxx(xp, yp, zp, model)
    >>> for k in kxx: print '%15.8e' % k
     1.98341901e-04
    -8.68466354e-04
     1.26372420e-04
     5.16707648e-05
     1.00794299e-03
    -4.31316366e-03
     1.58316258e-05
     1.75903755e-04
     1.12440765e-04
     6.40002709e-04
    -3.34762799e-02
     9.12106340e-04
     3.97886736e-05
     3.64532149e-05
    -4.90391764e-04
     8.75227483e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*(((3*dx**2) - r_2)/r_5)
    return res
    
def kernelxy(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial y},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxy = kernelxy(xp, yp, zp, model)
    >>> for k in kxy: print '%15.8e' % k
     4.85734213e-04
    -4.17597114e-04
    -2.47150351e-04
    -9.41136155e-05
    -4.11598058e-04
     4.85498244e-03
     1.70914015e-04
    -2.40173571e-04
    -1.91994939e-04
     8.11118152e-05
     1.47914349e-04
     4.27990397e-05
    -9.03888851e-05
    -2.48417508e-04
     3.80978795e-05
     2.82555489e-04
 
    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue
        radius = sphere.radius
        dx = sphere.x - xp
        dz = sphere.z - zp
        dy = sphere.y - yp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*((3*dx*dy)/r_5)
    return res

def kernelxz(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial x \partial z},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kxz = kernelxz(xp, yp, zp, model)
    >>> for k in kxz: print '%15.8e' % k
     1.84932340e-04
    -1.82020852e-04
    -9.88601402e-05
    -2.26634399e-05
     5.72913703e-04
    -5.45315568e-03
    -2.05096817e-04
    -8.46149629e-05
     6.96893540e-05
     3.18005004e-04
    -3.54994437e-05
    -3.47743609e-04
     1.84078540e-05
     6.31196073e-05
    -5.07971726e-06
    -6.76713381e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue	
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*((3*dx*dz)/r_5)
    return res

def kernelyy(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y^2},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kyy = kernelyy(xp, yp, zp, model)
    >>> for k in kyy: print '%15.8e' % k
     8.49758255e-05
     1.47922394e-03
     9.74781089e-05
     3.36001724e-05
    -5.21360732e-04
     5.91223500e-04
     6.16669738e-04
     3.21393798e-05
     7.53451559e-05
    -4.36862735e-05
    -3.34515846e-02
    -4.86512204e-04
     3.95559393e-05
     1.66402690e-04
     9.14562674e-04
     8.98363076e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue	
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*(((3*dy**2) - r_2)/r_5)
    return res

def kernelyz(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial y \partial z},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kyz = kernelyz(xp, yp, zp, model)
    >>> for k in kyz: print '%15.8e' % k
     1.63676201e-04
     9.57774279e-04
     8.80845731e-05
     1.91502745e-05
    -1.57688143e-04
    -9.13144605e-03
     2.92885575e-04
     6.19153248e-05
    -6.77629197e-05
    -1.62223630e-04
    -3.85863518e-05
    -5.97195903e-06
    -1.94837189e-05
    -8.16838179e-05
    -3.47410417e-04
    -6.78290807e-05

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue	
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*((3*dy*dz)/r_5)
    return res

def kernelzz(xp, yp, zp, spheres):
    """
    Calculates the function
    
    .. math::

        \frac{\partial^2 \phi(x,y,z)}{\partial z^2},
    
    where
    
    .. math:: 

        \phi(x,y,z) = \frac{4}{3} \pi R^3 \frac{1}{r}
    
    and
    
    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}}.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the function will be 
        calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`

    Returns:

    * res : array
        The function calculated on xp, yp, zp
    
    
    Example:
    
    >>> from fatiando import mesher, gridder
    >>> from fatiando.gravmag import sphere
    >>> # Create a sphere model
    >>> model = [
    ...         mesher.Sphere(10, 10, 5, 1, {'density':1.}),
    ...         mesher.Sphere(-13, -15, 6, 1, {'density':1.})]
    >>> # Create a regular grid at 0m height
    >>> shape = (4, 4)
    >>> area = (-30, 30, -30, 30)
    >>> xp, yp, zp = gridder.regular(area, shape, z=0)
    >>> # Calculate the function
    >>> kzz = kernelzz(xp, yp, zp, model)
    >>> for k in kzz: print '%15.8e' % k
    -2.83317727e-04
    -6.10757583e-04
    -2.23850528e-04
    -8.52709372e-05
    -4.86582256e-04
     3.72194016e-03
    -6.32501364e-04
    -2.08043134e-04
    -1.87785921e-04
    -5.96316436e-04
     6.69278645e-02
    -4.25594136e-04
    -7.93446130e-05
    -2.02855904e-04
    -4.24170910e-04
    -1.77359056e-04

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for sphere in spheres:
        if sphere is None:
            continue	
        radius = sphere.radius
        dx = sphere.x - xp
        dy = sphere.y - yp
        dz = sphere.z - zp
        r_2 = (dx**2 + dy**2 + dz**2)
        r_5 = r_2**(2.5)
        volume = 4.*numpy.pi*(radius**3)/3.
        res += volume*(((3*dz**2) - r_2)/r_5)
    return res

def gzzmod(xp, yp, zp, spheres, dens=None):
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
        res += density*kernelzz(xp, yp, zp, [sphere])
    res *= G*SI2EOTVOS
    return res