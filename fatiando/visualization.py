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
#from enthought.mayavi import mlab
#from enthought.tvtk.api import tvtk


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
    

def plot_src_rec(sources, receivers):
    """
    Plot the locations of seismic sources and receivers in a map.
    
    Parameters:
          
        sources: list with the x,y coordinates of each source
        
        receivers: list with the x,y coordinates of each receiver
    """
    
    src_x, src_y = numpy.transpose(sources)
    rec_x, rec_y = numpy.transpose(receivers)
    
    pylab.plot(src_x, src_y, 'r*', ms=9, label='Source')
    pylab.plot(rec_x, rec_y, 'b^', ms=7, label='Receiver')
    
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
    

def plot_prism(prism):
        
    vtkprism = tvtk.RectilinearGrid()
    vtkprism.cell_data.scalars = [prism.dens]
    vtkprism.cell_data.scalars.name = 'Density'
    vtkprism.dimensions = (2, 2, 2)
    vtkprism.x_coordinates = [prism.x1, prism.x2]
    vtkprism.y_coordinates = [prism.y1, prism.y2]
    vtkprism.z_coordinates = [prism.z1, prism.z2]    
        
    source = mlab.pipeline.add_dataset(vtkprism)
    outline = mlab.pipeline.outline(source)
    outline.actor.property.line_width = 4
    outline.actor.property.color = (1,1,1)
    
    
    