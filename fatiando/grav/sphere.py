# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Calculate the gravitational potential and its first and second derivatives for
a sphere.

Functions:

* :func:`fatiando.grav.sphere.gz`
    Calculates the :math:`g_z` gravity component.

* :func:`fatiando.grav.sphere.gxx`
    Calculates the :math:`g_xx` gravity gradient tensor component.

* :func:`fatiando.grav.sphere.gyy`
    Calculates the :math:`g_yy` gravity gradient tensor component.

* :func:`fatiando.grav.sphere.gzz`
    Calculates the :math:`g_zz` gravity gradient tensor component.

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 30-Nov-2010'


import logging


import fatiando
import fatiando.grav._sphere as sphere_ext


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.sphere')
log.addHandler(fatiando.default_log_handler)


def gz(dens, radius, xc, yc, zc, xp, yp, zp):
    """
    Calculates the :math:`g_z` gravity component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * dens
        Density of the sphere

    * radius
        Radius of the sphere

    * xc, yc, zc
        Coordinates of the center of the sphere

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_z` component calculated at **P**

    """

    res = sphere_ext.sphere_gz(float(dens), float(radius), float(xc), float(yc),
                               float(zc), float(xp), float(yp), float(zp))

    return res


def gxx(dens, radius, xc, yc, zc, xp, yp, zp):
    """
    Calculates the :math:`g_xx` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the sphere

    * radius
        Radius of the sphere

    * xc, yc, zc
        Coordinates of the center of the sphere

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_xx` component calculated at **P**

    """

    res = sphere_ext.sphere_gxx(float(dens), float(radius), float(xc),
                                float(yc), float(zc), float(xp), float(yp),
                                float(zp))

    return res


def gyy(dens, radius, xc, yc, zc, xp, yp, zp):
    """
    Calculates the :math:`g_yy` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the sphere

    * radius
        Radius of the sphere

    * xc, yc, zc
        Coordinates of the center of the sphere

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_yy` component calculated at **P**

    """

    res = sphere_ext.sphere_gyy(float(dens), float(radius), float(xc),
                                float(yc), float(zc), float(xp), float(yp),
                                float(zp))

    return res


def gzz(dens, radius, xc, yc, zc, xp, yp, zp):
    """
    Calculates the :math:`g_zz` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the sphere

    * radius
        Radius of the sphere

    * xc, yc, zc
        Coordinates of the center of the sphere

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_zz` component calculated at **P**

    """

    res = sphere_ext.sphere_gzz(float(dens), float(radius), float(xc),
                                float(yc), float(zc), float(xp), float(yp),
                                float(zp))

    return res