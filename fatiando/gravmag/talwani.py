"""
Calculate the gravitational attraction of a 2D body with polygonal vertical
cross-section using the formula of Talwani et al. (1959)

Use the :func:`~fatiando.mesher.Polygon` object to create polygons.

.. warning:: the vertices must be given clockwise! If not, the result will have
    an inverted sign.

**Components**

* :func:`~fatiando.gravmag.talwani.gz`

**References**

Talwani, M., J. L. Worzel, and M. Landisman (1959), Rapid Gravity Computations
for Two-Dimensional Bodies with Application to the Mendocino Submarine
Fracture Zone, J. Geophys. Res., 64(1), 49-59, doi:10.1029/JZ064i001p00049.

----

"""
import numpy
from numpy import arctan2, pi, sin, cos, log, tan

from fatiando.constants import G, SI2MGAL


def gz(xp, zp, polygons, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, zp : arrays
        The x and z coordinates of the computation points.
    * polygons : list of :func:`~fatiando.mesher.Polygon`
        The density model used.
        Polygons must have the property ``'density'``. Polygons that don't have
        this property will be ignored in the computations. Elements of
        *polygons* that are None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the polygons. Use this, e.g., for sensitivity matrix building.

        .. note:: The y coordinate of the polygons is used as z!

    Returns:

    * gz : array
        The :math:`g_z` component calculated on the computation points

    """
    if xp.shape != zp.shape:
        raise ValueError("Input arrays xp and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for polygon in polygons:
        if polygon is None or ('density' not in polygon.props
                               and dens is None):
            continue
        if dens is None:
            density = polygon.props['density']
        else:
            density = dens
        x = polygon.x
        z = polygon.y
        nverts = polygon.nverts
        for v in xrange(nverts):
            # Change the coordinates of this vertice
            xv = x[v] - xp
            zv = z[v] - zp
            # The last vertice pairs with the first one
            if v == nverts - 1:
                xvp1 = x[0] - xp
                zvp1 = z[0] - zp
            else:
                xvp1 = x[v + 1] - xp
                zvp1 = z[v + 1] - zp
            # Temporary fix. The analytical conditions for these limits don't
            # work. So if the conditions are breached, sum 0.01 meters to the
            # coodinates and be happy
            xv[xv == 0.] += 0.01
            xv[xv == xvp1] += 0.01
            zv[zv[xv == zv] == 0.] += 0.01
            zv[zv == zvp1] += 0.01
            zvp1[zvp1[xvp1 == zvp1] == 0.] += 0.01
            xvp1[xvp1 == 0.] += 0.01
            # End of fix
            phi_v = arctan2(zvp1 - zv, xvp1 - xv)
            ai = xvp1 + zvp1 * (xvp1 - xv) / (zv - zvp1)
            theta_v = arctan2(zv, xv)
            theta_vp1 = arctan2(zvp1, xvp1)
            theta_v[theta_v < 0] += pi
            theta_vp1[theta_vp1 < 0] += pi
            tmp = ai * sin(phi_v) * cos(phi_v) * (
                theta_v - theta_vp1 + tan(phi_v) * log(
                    (cos(theta_v) * (tan(theta_v) - tan(phi_v))) /
                    (cos(theta_vp1) * (tan(theta_vp1) - tan(phi_v)))))
            tmp[theta_v == theta_vp1] = 0.
            res = res + tmp * density
    res = res * SI2MGAL * 2.0 * G
    return res
