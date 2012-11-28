"""
.. topic:: Python + Numpy + Numexpr implementation.

    Trial version to test numexpr to speed up numpy operations.
    On large arrays (250000), got 200% improvement over Cython. In small arrays,
    Cython wins.

"""
import numpy
from numpy import sqrt, log, arctan2
import numexpr

from fatiando.constants import SI2EOTVOS, SI2MGAL, G


def gz(xp, yp, zp, prisms, dens=None):
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
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [numexpr.evaluate('x2 - xp'), numexpr.evaluate('x1 - xp')]
        y = [numexpr.evaluate('y2 - yp'), numexpr.evaluate('y1 - yp')]
        z = [numexpr.evaluate('z2 - zp'), numexpr.evaluate('z1 - zp')]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    xi, yj, zk = x[i], y[j], z[k]
                    r = numexpr.evaluate('sqrt(xi**2 + yj**2 + zk**2)')
                    kernel = numexpr.evaluate(
                        """-(xi*log(yj + r) + \
                             yj*log(xi + r) - \
                             zk*arctan2(xi*yj, zk*r))""")
                    res = numexpr.evaluate(
                        'res + ((-1.)**(i + j + k))*kernel*density')
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res = numexpr.evaluate('res*G*SI2MGAL')
    return res
