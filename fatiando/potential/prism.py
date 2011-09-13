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
Direct modelling of potential fields using right rectangular prisms.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging


import fatiando
import fatiando.potential._prism as prism_ext


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.prism')
log.addHandler(fatiando.default_log_handler)


def gz(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_z` gravity component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_z` component calculated at **P**
    """

    res = prism_ext.prism_gz(float(dens), float(x1), float(x2), float(y1),
                             float(y2), float(z1), float(z2), float(xp),
                             float(yp), float(zp))

    return res


def gxx(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{xx}` component calculated at **P**
    """

    res = prism_ext.prism_gxx(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res


def gxy(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{xy}` component calculated at **P**
    """

    res = prism_ext.prism_gxy(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res


def gxz(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{xz}` component calculated at **P**
    """

    res = prism_ext.prism_gxz(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res


def gyy(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{yy}` component calculated at **P**
    """

    res = prism_ext.prism_gyy(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res


def gyz(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{yz}` component calculated at **P**
    """

    res = prism_ext.prism_gyz(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res


def gzz(dens, x1, x2, y1, y2, z1, z2, xp, yp, zp):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    The coordinate system of the input parameters is to be x -> North,
    y -> East and z -> **DOWN**.

    **NOTE**: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * dens
        Density of the prism

    * x1, x2, y1, ... z2
        Borders of the prism

    * xp, yp, zp
        Coordinates of the point **P** where the field will be calculated

    Returns:

    * the :math:`g_{zz}` component calculated at **P**
    """

    res = prism_ext.prism_gzz(float(dens), float(x1), float(x2), float(y1),
                              float(y2), float(z1), float(z2), float(xp),
                              float(yp), float(zp))

    return res
