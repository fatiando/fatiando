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
Gravity field transformations like upward continuation, derivatives and total
mass.

Functions:

* :func:`fatiando.grav.transform.upcontinue`
    Upward continue a gravity field using the analytical formula. 
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 20-Oct-2010'


import logging

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.transform')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def upcontinue(data, height):
    """
    Upward continue :math:`gz` data using numerical integration of the 
    analytical formula:
    
    .. math::
    
        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'
               
    For now only supports **grid** data on a plain.

    Parameters:
    
    * data
        :math:`gz` data stored in a dictionary (see bellow for explanation).
        
    * height
        How much higher to move the gravity field (should be POSITIVE!)
        
    NOTE: be aware of coordinate systems! The *x*, *y*, *z* coordinates are 
    x -> North, y -> East and z -> **DOWN**.
    
    Returns:
    
    * cont_data
        Upward continued data stored in a dictionary.
        
    The data dictionary should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...],
         'grid':True, 'nx':points_in_x, 'ny':points_in_y}
    
    """
    
    assert height > 0, "'height' should be positive! Can only upward continue."
    assert data['grid'] is True, ("Sorry, for now only supports grid data. " + 
        "Make sure to set the 'grid' key to 'True' in the data dictionary.")    
        
    dx = data['x'][1] - data['x'][0]
    dy = data['y'][1] - data['y'][0]
    
    ndata = data['nx']*data['ny']
    
    # Copy the grid information to the upward continued data dict
    cont_data = {}
    cont_data['x'] = numpy.copy(data['x'])
    cont_data['y'] = numpy.copy(data['y'])
    cont_data['z'] = data['z'] - height
    cont_data['value'] = numpy.zeros_like(data['value'])
    cont_data['error'] = numpy.zeros_like(data['error'])
    cont_data['grid'] = True
    cont_data['nx'] = data['nx']
    cont_data['ny'] = data['ny']
                
    for i in xrange(ndata):
        
        for j in xrange(ndata):
            
            cont_data['value'][i] += 
        
    
        cont_data['value'][i] *= (cont_data['z'][i] - data['z'][i])
    
    
    
    
    
    
    
    