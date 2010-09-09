"""
Plotting functions.
    Uses Matplotlib for 2D and Mayavi2 for 3D.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Sep-2010'


import pylab
import numpy
from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

import fatiando.utils.geometry


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