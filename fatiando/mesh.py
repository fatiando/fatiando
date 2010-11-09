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
Tools for creating model space discretization meshes.

Functions:

* :func:`fatiando.mesh.prism_mesh`
    Dived a volume into right rectangular prisms

* :func:`fatiando.mesh.square_mesh`
    Divide an area into rectangles

* :func:`fatiando.mesh.line_mesh`
    Divide a line or region into segments. 

* :func:`fatiando.mesh.extract_key`
    Extract the values of key from each of cell of mesh

* :func:`fatiando.mesh.fill`
    Fill the 'key' value of each cell of mesh

* :func:`fatiando.mesh.copy` 
    Make a copy of an n-dimensional mesh
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 13-Sep-2010'


import logging

import numpy

import fatiando
import fatiando.geometry as geometry

log = logging.getLogger('fatiando.mesh')  
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)



def prism_mesh(x1, x2, y1, y2, z1, z2, nx, ny, nz):
    """
    Dived a volume into right rectangular prisms.
    
    The mesh is arranged as a list of matrices (layers with same z). Each
    matrix has x varying with the columns and y with the rows. Therefore the
    shape of the mesh is (*nz*, *ny*, *nx*)
    
    Parameters:
      
    * x1, x2
        Lower and upper limits of the volume in the x direction
    
    * y1, y2
        Lower and upper limits of the volume in the y direction
    
    * z1, z2
        Lower and upper limits of the volume in the z direction
    
    * nx, ny, nz
        Number of prisms in the x, y, and z directions
          
    Returns:
    
    * mesh
        3D array of prisms. (See :func:`fatiando.geometry.prism`)
        
    """
    
    dx = float(x2 - x1)/nx
    dy = float(y2 - y1)/ny
    dz = float(z2 - z1)/nz
        
    log.info("Building prism mesh:")
    log.info("  Discretization: nx=%d X ny=%d X nz=%d = %d prisms" 
             % (nx, ny, nz, nx*ny*nz))
    log.info("  Cell dimensions: dx=%g X dy=%g X dz=%g" % (dx, dy, dz))
    
    mesh = []
    
    for k, cellz1 in enumerate(numpy.arange(z1, z2, dz)):
        
        # To ensure that there are the right number of cells. arange 
        # sometimes makes more cells because of floating point rounding
        if k >= nz:
            
            break
        
        plane = []
    
        for j, celly1 in enumerate(numpy.arange(y1, y2, dy)):
            
            if j >= ny:
                
                break
            
            line = []
            
            for i, cellx1 in enumerate(numpy.arange(x1, x2, dx)):
                
                if i >= nx:
                    
                    break
                
                cell = geometry.prism(cellx1, cellx1 + dx, celly1, celly1 + dy,
                                      cellz1, cellz1 + dz)
                
                line.append(cell)
                
            plane.append(line)
            
        mesh.append(plane)
            
    return numpy.array(mesh)
        

def square_mesh(x1, x2, y1, y2, nx, ny):
    """
    Divide an area into rectangles. 
    
    The mesh is arranged as a matrix with x varying with the columns and y with 
    the rows. Therefore the shape of the mesh is (*ny*, *nx*)
        
    **NOTE**: if the region is in x and z instead of y, simply think of y as z.
    
    Parameters:
      
    * x1, x2
        Lower and upper limits of the region in the x direction
    
    * y1, y2
        Lower and upper limits of the region in the y direction
    
    * nx, ny
        Number of cells in the x and y directions
      
    Return:
    
    * mesh
        2D array of cells. 
        
    Each cell is a dictionary as::
    
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


def line_mesh(x1, x2, nx):
    """
    Divide a line or region into segments. 
    
    Parameters:
      
    * x1, x2
        Lower and upper limits of the region in the x direction
    
    * nx
        Number of cells (segments) in the x
      
    Return:
    
    * mesh
        1D array of cells. 
        
    Each cell is a dictionary as::
    
        {'x1':cellx1, 'x2':cellx2}        
    """
    
    log.info("Building line mesh:")
    log.info("  Discretization: %d cells" % (nx))
    
    dx = float(x2 - x1)/nx
    
    mesh = []
    
    for i, cellx1 in enumerate(numpy.arange(x1, x2, dx)):
                
        # To ensure that there are the right number of cells. arange sometimes
        # makes more cells because of floating point rounding
        if i >= nx:
            
            break
        
        cell = {'x1':cellx1, 'x2':cellx1 + dx}
        
        mesh.append(cell)
    
    return numpy.array(mesh)
    

def extract_key(key, mesh):
    """
    Extract the values of key from each of cell of mesh
    
    Parameters:
    
    * key
        the key whose value will be extracted
    
    * mesh
        an arbitrary mesh
      
    Returns:
      
    * key_array
        ndarray with the same shape as *mesh* filled with the values in *key* of
        each cell
        
    """
    
    res = []
    
    for cell in mesh.ravel():
        
        res.append(cell[key])
        
    res = numpy.reshape(res, mesh.shape)
    
    return res    
        

def fill(estimate, mesh, key='value'):
    """
    Fill the ``key`` of each cell of a mesh with the values in *estimate*
    
    Parameters:
    
    * estimate
        1D array-like vector with the scalar value of each cell (arranged as 
        mesh.ravel())
    
    * mesh
        Mesh to fill
          
    * key
        Key to fill in the *mesh*
        
    """
        
    for value, cell in zip(estimate, mesh.ravel()):
        
        cell[key] = value
    
    
def copy(mesh):
    """
    Make a copy of mesh.
    
    **NOTE**: Use this instead of ``numpy.copy(my_mesh)`` or ``my_mesh.copy()``
    because they don't make copies of the cell dictionaries, so the copied mesh
    would still refer to the original's cells.
    
    Parameters:
        
    * mesh
        Mesh to copy

    Returns:
    
    * copy
        A copy of mesh

    """
    
    copy = [cell.copy() for cell in mesh.ravel()]
        
    copy = numpy.reshape(copy, mesh.shape)
        
    return copy