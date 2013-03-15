"""
Testing numexpr to speedup prism computations
"""
import numpy

from numexpr import evaluate

from fatiando.constants import SI2EOTVOS, SI2MGAL, G


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
