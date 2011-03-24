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

* :func:`fatiando.mesh.vfilter`
    Remove elements within a given value range from a mesh.
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 13-Sep-2010'


import logging

import numpy
import pylab

import fatiando
import fatiando.geometry as geometry

log = logging.getLogger('fatiando.mesh')  
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def prism_mesh(x1, x2, y1, y2, z1, z2, nx, ny, nz, topo=None):
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
        
    * topo
        Topography data in a dictionary (see :func:`fatiando.grav.io.load_topo`)
        If not ``None``, mesh cells above the topography values will have their
        'value' keys set to ``None`` so that they won't be plotted by
        :func:`fatiando.vis.plot_prism_mesh`
          
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
    mesh = numpy.array(mesh)
    
    if topo is not None:
        # The coordinates of the centers of the cells
        x = numpy.arange(x1, x2, dx) + 0.5*dx
        if len(x) > nx:
            x = x[:-1]
        y = numpy.arange(y1, y2, dy) + 0.5*dy
        if len(y) > ny:
            y = y[:-1]
        X, Y = numpy.meshgrid(x, y)
        # -1 if to transform height into z coordinate
        topo_grid = -1*pylab.griddata(topo['x'], topo['y'], topo['h'], X, Y)
        topo_grid = topo_grid.ravel()
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # bellow a masked topo point (ie, one with no height information)
        if numpy.ma.isMA(topo_grid):
            topo_mask = topo_grid.mask
        else:
            topo_mask = [False]*len(topo_grid)
        for layer in mesh:
            for cell, ztopo, mask in zip(layer.ravel(), topo_grid, topo_mask) :
                if 0.5*(cell['z1'] + cell['z2']) <  ztopo or mask:
                    cell['value'] = None
    return mesh


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
          

def fill(values, mesh, key='value', fillNone=True):
    """
    Fill the ``key`` of each cell of a mesh with *values*
        
    Parameters:
    
    * values
        1D array-like vector with the scalar value of each cell (arranged as 
        mesh.ravel())
    
    * mesh
        Mesh to fill
          
    * key
        Key to fill in the *mesh*
        
    * fillNone
        If ``False``, cells with their *key* already set to ``None`` will not be
        filled. 
        
    """
        
    for value, cell in zip(values, mesh.ravel()):
        
        if not fillNone and key in cell and cell[key] is None:
            
            continue
                
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


def vfilter(mesh, vmin, vmax, vkey='value'):
    """
    Remove elements within a given value range from a mesh.

    Parameters:

    * mesh
        Mesh to copy

    * vmin
        Minimum value

    * vmax
        Maximum value

    * vkey
        The key of the mesh elements whose value will be used to filter

    Returns:

    * elements
        1D list of filtered elements

    """
    filtered = [cell for cell in mesh.ravel() if not cell[vkey] is None and
                cell[vkey] >= vmin and cell[vkey] <= vmax ]
    filtered = numpy.array(filtered)
    return filtered