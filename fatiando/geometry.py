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
Create and operate on data types representing geometric elements.

Functions:

* :func:`fatiando.geometry.prism`
    Create a right rectangular prism.
    
* :func:`fatiando.geometry.sphere`
    Create a sphere.
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'


import logging

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.geometry')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def prism(x1, x2, y1, y2, z1, z2, props={}):
    """
    Create a right rectangular prism.
    
    **NOTE**: Coordinate system is x -> North, y -> East and z -> Down
    
    Parameters:
    
    * x1, x2, y1, y2, z1, z2
        Boundaries of the prism
        
    * props
        Dictionary with additional properties of the prism. 
        Ex: ``{'density':1000, 'P_wave_speed':4000}``
    
    Returns:
    
    * prism
        Dictionary representing the prism::
        
            {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'z1':z1, 'z2':z2}

    Each key-value pair in *props* is also appended to the dictionary.
    
    Raises:
    
    * AssertionError
        If *x1* = *x2*, *y1* = *y2* or *z1* = *z2*
         
    """
    
    assert x1 != x2, ("Can't create prism. 'x1' and 'x2' must be different.")
    assert y1 != y2, ("Can't create prism. 'y1' and 'y2' must be different.")
    assert z1 != z2, ("Can't create prism. 'z1' and 'z2' must be different.")
    
    prism = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'z1':z1, 'z2':z2}
    
    for key in props:
        
        prism[key] = props[key]
        
    return prism


def sphere(xc, yc, zc, radius, props={}):
    """
    Create a sphere.

    **NOTE**: Coordinate system is x -> North, y -> East and z -> Down

    Parameters:

    * xc, yc, zc
        Coordinates of the center of the sphere

    * radius
        radius of the sphere

    * props
        Dictionary with additional properties of the sphere.
        Ex: ``{'density':1000, 'P_wave_speed':4000}``

    Returns:

    * sphere
        Dictionary representing the sphere::

            {'xc':xc, 'yc':yc, 'zc':zc, 'radius':radius}

    Each key-value pair in *props* is also appended to the dictionary.

    Raises:

    * AssertionError
        If *radius* < 0

    """

    assert radius >= 0, "Invalid radius. Must be >= 0"

    sphere = {'xc':xc, 'yc':yc, 'zc':zc, 'radius':radius}

    for key in props:

        sphere[key] = props[key]

    return sphere