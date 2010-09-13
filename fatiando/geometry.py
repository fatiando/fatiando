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
Tools for manipulating geometric elements and creating model space 
discretization meshes.

Functions:
  * square_mesh: divide a region into rectangles
  * copy_mesh: Make a copy of an n-dimensional mesh
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 13-Sep-2010'


import logging

import numpy

import fatiando

log = logging.getLogger('fatiando.geometry')  
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def square_mesh(x1, x2, y1, y2, nx, ny):
    """
    Divide a region into rectangles. 
    
    Parameters:
      
      x1, x2: lower and upper limits of the region in the x direction
      
      y1, y2: lower and upper limits of the region in the y direction
      
      nx, ny: number of cells in the x and y directions
      
    Return:
    
      2D array of cells. Each cell is a dictionary as:
        {'x1':cellx1, 'x2':cellx2, 'y1':celly1, 'y2':celly2}
    """
    
    log.info("Building square mesh:")
    log.info("  Discretization: nx=%d X ny=%d = %d cells" 
             % (nx, ny, nx*ny))
    
    dx = float(x2 - x1)/nx
    dy = float(y2 - y1)/ny
    
    mesh = []
    
    for i, celly1 in enumerate(numpy.arange(y1, y2, dy)):
        
        # To ensure that there are the right number of cells. arange sometimes
        # makes more cells because of floating point rounding
        if i >= ny:
            
            break
        
        line = []
        
        for j, cellx1 in enumerate(numpy.arange(x1, x2, dx)):
            
            if j >= nx:
                
                break
            
            cell = {'x1':cellx1, 'x2':cellx1 + dx, 
                    'y1':celly1, 'y2':celly1 + dy}
            
            line.append(cell)
            
        mesh.append(line)
            
    return numpy.array(mesh)


def copy_mesh(mesh):
    """
    Make a copy of mesh.
    Use this instead of numpy.copy or mesh.copy because they don't make copies
    of the cell dictionaries, so the copied mesh would still refer to the 
    original's cells.
    """
    
    copy = [cell.copy() for cell in mesh.ravel()]
        
    copy = numpy.reshape(copy, mesh.shape)
        
    return copy