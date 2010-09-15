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
Collection of plotting functions. 
Uses Matplotlib for 2D and Mayavi2 for 3D.

Functions:
  * plot_src_rec: plot the locations of seismic sources and receivers in a map
  * plot_ray_coverage: Plot the rays between sources and receivers in a map
  * plot_square_mesh: Plot a pcolor map of a 2D mesh made of square cells
  * residuals_histogram: Plot a histogram of the residual vector
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Sep-2010'


import numpy

import pylab

# Do lazy imports of mlab and tvtk to avoid the slow imports when I don't need
# 3D plotting
mlab = None
tvtk = None


def plot_square_mesh(mesh, cmap=pylab.cm.jet, vmin=None, vmax=None):
    """
    Plot a 2D mesh made of square cells. Each cell is a dictionary as:
      {'x1':cellx1, 'x2':cellx2, 'y1':celly1, 'y2':celly2, 'value':value}
    The pseudo color of the plot is key 'value'.
    
    Parameters:
    
      mesh: a list of cells describing the square mesh
      
      cmap: color map to use
      
      vmin, vmax: lower and upper limits for the color scale
    """
    
    xvalues = []
    for cell in mesh[0]:
        
        xvalues.append(cell['x1'])
        
    xvalues.append(mesh[0][-1]['x2'])
        
    yvalues = []
    
    for line in mesh:
        
        yvalues.append(line[0]['y1'])
    
    yvalues.append(mesh[-1][0]['y2'])
    
    X, Y = numpy.meshgrid(xvalues, yvalues)
    
    Z = numpy.zeros_like(X)
    
    for i, line in enumerate(mesh):
        
        for j, cell in enumerate(line):
            
            Z[i][j] = cell['value']
            
    if vmin is None and vmax is None:
        
        pylab.pcolor(X, Y, Z, cmap=cmap)
        
    else:
        
        pylab.pcolor(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)


def residuals_histogram(residuals, nbins=None):
    """
    Plot a histogram of the residual vector.
    
    Parameters:
    
      residuals: 1D array-like vector of residuals
      
      nbins: number of bins (default to len(residuals)/8)
    """
    
    if nbins is None:
    
        nbins = len(residuals)/8
    
    pylab.hist(residuals, bins=nbins, facecolor='gray')
    

def plot_src_rec(sources, receivers, markersize=9):
    """
    Plot the locations of seismic sources and receivers in a map.
    
    Parameters:
          
        sources: list with the x,y coordinates of each source
        
        receivers: list with the x,y coordinates of each receiver
        
        markersize: size of the source and receiver markers
    """
    
    src_x, src_y = numpy.transpose(sources)
    rec_x, rec_y = numpy.transpose(receivers)
    
    pylab.plot(src_x, src_y, 'r*', ms=markersize, label='Source')
    pylab.plot(rec_x, rec_y, 'b^', ms=int(0.78*markersize), label='Receiver')
    
    pylab.legend(numpoints=1, prop={'size':7})
    

def plot_ray_coverage(sources, receivers, linestyle='-k'):
    """
    Plot the rays between sources and receivers in a map.
    
    Parameters:
          
        sources: list with the x,y coordinates of each source
        
        receivers: list with the x,y coordinates of each receiver
        
        linestyle: type of line to display and color. See 'pylab.plot'
                   documentation for more details.
    """
    
    for src, rec in zip(sources, receivers):
        
        pylab.plot([src[0], rec[0]], [src[1], rec[1]], linestyle)
    

def plot_prism_mesh(mesh, style='surface', label='scalar'):
    """
    Plot a 3D prism mesh using Mayavi2.
    
    Parameters:
    
      mesh: 3D array-like prism mesh (see fatiando.geometry.prism_mesh)
      
      style: either 'surface' for solid prisms or 'wireframe' for just the 
             wireframe
             
      label: name of the scalar 'value' of the mesh cells. 
    """
    
    assert style in ['surface', 'wireframe'], "Invalid style '%s'" % (style)
    
    global mlab, tvtk
    
    if mlab is None:        
        
        from enthought.mayavi import mlab
        
    if tvtk is None:
        
        from enthought.tvtk.api import tvtk
        
    points = []
    cells = []
    start = 0   # To mark what index in the points the cell starts
    offsets = []
    offset = 0
        
    for cell in mesh.ravel():
        
        x1, x2 = cell['x1'], cell['x2']
        y1, y2 = cell['y1'], cell['y2']
        z1, z2 = cell['z1'], cell['z2']
        
        points.extend([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                       [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
        
        cells.append(8)
        cells.extend([i for i in xrange(start, start + 8)])
        start += 8
        
        offsets.append(offset)
        offset += 9
                    
    cell_array = tvtk.CellArray()
    cell_array.set_cells(mesh.size, numpy.array(cells))
    cell_types = numpy.array([12]*mesh.size, 'i')
    
    vtkmesh = tvtk.UnstructuredGrid(points=numpy.array(points, 'f'))
    
    vtkmesh.set_cells(cell_types, numpy.array(offsets, 'i'), cell_array)
    
    scalars = []
    
    for cell in mesh.ravel():
        
        if 'value' not in cell.keys():
            
            scalars = numpy.zeros(mesh.size)
            
            break
        
        scalars = numpy.append(scalars, cell['value'])
        
    vtkmesh.cell_data.scalars = scalars
    vtkmesh.cell_data.scalars.name = label
        
    dataset = mlab.pipeline.add_dataset(vtkmesh)                       
        
    if style == 'wireframe':
        
        surf = mlab.pipeline.surface(dataset, vmax=max(scalars), 
                                     vmin=min(scalars))
        surf.actor.property.representation = 'wireframe'
        surf.actor.mapper.scalar_visibility = False
        
    if style == 'surface':
        
        thresh = mlab.pipeline.threshold(dataset)    
        surf = mlab.pipeline.surface(thresh, vmax=max(scalars), 
                                     vmin=min(scalars))
        surf.actor.property.representation = 'surface'        
        mlab.colorbar(surf, title=label, orientation='vertical', nb_labels=10)
        
    return surf