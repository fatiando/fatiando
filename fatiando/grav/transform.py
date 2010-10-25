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
import math
import time

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.transform')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def upcontinue(data, height):
    """
    Upward continue :math:`g_z` data using numerical integration of the 
    analytical formula:
    
    .. math::
    
        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'
               
    For now only supports **grid** data on a plain.
    
    **UNITS**: SI for all coordinates, mGal for :math:`g_z`

    Parameters:
    
    * data
        :math:`g_z(x',y',z_0)` data stored in a dictionary.
        
    * height
        How much higher to move the gravity field (should be POSITIVE!)
        
    NOTE: be aware of coordinate systems! The *x*, *y*, *z* coordinates are 
    x -> North, y -> East and z -> **DOWN**.
    
    Returns:
    
    * cont
        Upward continued data stored in a dictionary.
        
    The data dictionary should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...],
         'grid':True, 'nx':points_in_x, 'ny':points_in_y}
    
    """
    
    assert height > 0, "'height' should be positive! Can only upward continue."
    assert data['grid'] is True, ("Sorry, for now only supports grid data. " + 
        "Make sure to set the 'grid' key to 'True' in the data dictionary.")    
                          
    start = time.time()
        
    dx = data['x'][1] - data['x'][0]
    dy = data['y'][data['nx']] - data['y'][0]
    
    log.info("Upward continuation:")
    log.info("  data: %d points" % (data['nx']*data['ny']))
    log.info("  dx: %g m" % (dx))
    log.info("  dy: %g m" % (dy))
    log.info("  new height: %g m" % (height))
        
    # Copy the grid information to the upward continued data dict
    cont = {}
    cont['x'] = numpy.copy(data['x'])
    cont['y'] = numpy.copy(data['y'])
    cont['z'] = data['z'] - height
    cont['value'] = numpy.zeros_like(data['value'])
    cont['grid'] = True
    cont['nx'] = data['nx']
    cont['ny'] = data['ny']
                
    for i, cont_coords in enumerate(zip(cont['x'], cont['y'], cont['z'])):
        
        x, y, z = cont_coords
        
        for j, coords in enumerate(zip(data['x'], data['y'], data['z'])):
            
            xl, yl, zl = coords
            
            oneover_l = math.pow((x - xl)**2 + (y - yl)**2 + (z - zl)**2, -1.5)
            
            cont['value'][i] += data['value'][j]*oneover_l*dx*dy        
    
        cont['value'][i] *= abs(z - zl)/(2*numpy.pi)
        
    end = time.time()
    
    log.info("  time: %g s" % (end - start))
    
    return cont