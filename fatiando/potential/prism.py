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
Calculate the potential fields and derivatives of the 3D right rectangular prism 
using the forumla of Nagy et al. (2000)

**Gravity**

* :func:`~fatiando.potential.prism.gz`
* :func:`~fatiando.potential.prism.gxx`
* :func:`~fatiando.potential.prism.gxy`
* :func:`~fatiando.potential.prism.gxz`
* :func:`~fatiando.potential.prism.gyy`
* :func:`~fatiando.potential.prism.gyz`
* :func:`~fatiando.potential.prism.gzz`

**Magnetic**


**References**

Nagy, D., G. Papp, and J. Benedek, 2000, The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.
    
:author: Leonardo Uieda (leouieda@gmail.com)
:date: Created 11-Sep-2010
:license: GNU Lesser General Public License v3 (http://www.gnu.org/licenses/)

----

"""

import logging

import numpy

from fatiando.potential import _prism
from fatiando import logger


log = logger.dummy('fatiando.potential.prism')

def gz(xp, yp, zp, prisms):
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

    Returns:
    
    * gz : array
        The :math:`g_z` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gz(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gxx(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xx}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gxx : array
        The :math:`g_{xx}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gxx(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gxy(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xy}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gxy : array
        The :math:`g_{xy}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gxy(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gxz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{xz}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gxz : array
        The :math:`g_{xz}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gxz(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gyy(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{yy}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gyy : array
        The :math:`g_{yy}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gyy(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gyz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{yz}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gyz : array
        The :math:`g_{yz}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gyz(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res

def gzz(xp, yp, zp, prisms):
    """
    Calculates the :math:`g_{zz}` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be x -> North,
        y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:
    
    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.ddd.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have this
        property will be ignored in the computations. Elements of *prisms* that
        are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.ddd.PrismMesh`.

    Returns:
    
    * gzz : array
        The :math:`g_{zz}` component calculated on *points*

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is not None and 'density' in prism.props:
            x1, x2, y1, y2, z1, z2 = prism.get_bounds()            
            res += _prism.prism_gzz(float(prism.props['density']), x1, x2, y1,
                y2, z1, z2, xp, yp, zp)
    return res
