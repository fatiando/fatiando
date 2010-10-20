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
Functions to calculate the travel times of seismic waves.

Functions:

* :func:`fatiando.seismo.traveltime.cartesian_straight`
    Calculate the travel time inside a square cell assuming the ray is a 
    straight line.

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging


import fatiando
import fatiando.seismo._traveltime as traveltime_ext


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.prism')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def cartesian_straight(slowness, x1, y1, x2, y2, x_src, y_src, x_rec, y_rec):
    """
    Calculate the travel time inside a square cell assuming the ray is a 
    straight line.
    
    NOTE: Don't care about the units as long they are compatible.
    
    Parameters:
    
    * slowness
        Slowness of the cell (:math:`slowness = \\frac{1}{velocity}`)
        
    * x1, y1
        Coordinates of the lower-left corner of the cell
        
    * x2, y2
        Coordinates of the upper-right corner of the cell
        
    * x_src, y_src
        Coordinates of the wave source
        
    * x_rec, y_rec
        Coordinates of the receiver
    
    Returns:
    
    * Time the ray spent in the cell in compatible units with *slowness*
    
    """
    
    time = traveltime_ext.cartesian_straight(float(slowness), float(x1), 
                                             float(y1), float(x2), float(y2), 
                                             float(x_src), float(y_src), 
                                             float(x_rec), float(y_rec))
    
    return time