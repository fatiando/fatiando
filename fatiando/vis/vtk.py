# Copyright 2012 The Fatiando a Terra Development Team
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
Wrappers for calls to Mayavi2's `mlab` module for plotting
:mod:`fatiando.mesher.ddd` objects and automating common tasks.

**OBJECTS**

* :func:`fatiando.vis.vtk.prisms`

**HELPERS**

* :func:`fatiando.vis.vtk.figure`
* :func:`fatiando.vis.vtk.add_outline`
* :func:`fatiando.vis.vtk.add_axes`
* :func:`fatiando.vis.vtk.wall_north`
* :func:`fatiando.vis.vtk.wall_south`
* :func:`fatiando.vis.vtk.wall_east`
* :func:`fatiando.vis.vtk.wall_west`
* :func:`fatiando.vis.vtk.wall_top`
* :func:`fatiando.vis.vtk.wall_bottom`

----
   
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 30-Jan-2012'

import numpy


# Do lazy imports of mlab and tvtk to avoid the slow imports when I don't need
# 3D plotting
mlab = None
tvtk = None

def _lazy_import_mlab():
    """
    Do the lazy import of mlab
    """
    global mlab
    # For campatibility with versions of Mayavi2 < 4
    if mlab is None:
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

def _lazy_import_tvtk():
    """
    Do the lazy import of tvtk
    """
    global tvtk
    # For campatibility with versions of Mayavi2 < 4
    if tvtk is None:
        try:
            from tvtk.api import tvtk
        except ImportError:
            from enthought.tvtk.api import tvtk

def prisms(prisms, scalars, label='scalars', style='surface', opacity=1,
           edges=True, vmin=None, vmax=None, cmap='blue-red'):
    """
    Plot a list of 3D right rectangular prisms using Mayavi2.

    Will not plot a value None in *prisms*

    Parameters:
    
    * prisms
        List of prisms (see :func:`fatiando.mesher.prism.Prism3D`)
    * scalars
        Array with the scalar value of each prism. Used as the color scale.
    * label
        Label used as the scalar type (like 'density' for example)
    * style
        Either ``'surface'`` for solid prisms or ``'wireframe'`` for just the
        contour
    * opacity
        Decimal percentage of opacity
    * edges
        Wether or not to display the edges of the prisms in black lines. Will
        ignore this is style='wireframe'
    * vmin, vmax
        Min and max values for the color scale of the scalars. If *None* will
        default to min(scalars) or max(scalars).
    * cmap
        Color map to use for the scalar values. See the 'Colors and Legends'
        menu on the Mayavi2 GUI for valid color maps.

    Returns:
    
    * surface: the last element on the pipeline

    """
    if style not in ['surface', 'wireframe']:
        raise ValueError, "Invalid style '%s'" % (style)
    if opacity > 1. or opacity < 0:
        msg = "Invalid opacity %g. Must be in range [1,0]" % (opacity)
        raise ValueError, msg

    # mlab and tvtk are really slow to import
    _lazy_import_mlab()
    _lazy_import_tvtk()

    # VTK parameters
    points = []
    cells = []
    offsets = []
    offset = 0
    mesh_size = 0
    celldata = []
    # To mark what index in the points the cell starts
    start = 0
    for prism, scalar in zip(prisms, scalars):
        if prism is None or scalar is None:
            continue
        x1, x2 = prism['x1'], prism['x2']
        y1, y2 = prism['y1'], prism['y2']
        z1, z2 = prism['z1'], prism['z2']
        points.extend([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                       [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
        cells.append(8)
        cells.extend([i for i in xrange(start, start + 8)])
        start += 8
        offsets.append(offset)
        offset += 9
        celldata.append(scalar)
        mesh_size += 1
    cell_array = tvtk.CellArray()
    cell_array.set_cells(mesh_size, numpy.array(cells))
    cell_types = numpy.array([12]*mesh_size, 'i')
    vtkmesh = tvtk.UnstructuredGrid(points=numpy.array(points, 'f'))
    vtkmesh.set_cells(cell_types, numpy.array(offsets, 'i'), cell_array)
    vtkmesh.cell_data.scalars = numpy.array(celldata)
    vtkmesh.cell_data.scalars.name = label
    dataset = mlab.pipeline.add_dataset(vtkmesh)
    if vmin is None:
        vmin = min(vtkmesh.cell_data.scalars)
    if vmax is None:
        vmax = max(vtkmesh.cell_data.scalars)
    surf = mlab.pipeline.surface(dataset, vmax=vmax, vmin=vmin, colormap=cmap)
    if style == 'wireframe':
        surf.actor.property.representation = 'wireframe'
    if style == 'surface':
        surf.actor.property.representation = 'surface'
        if edges:
            surf.actor.property.edge_visibility = 1
    surf.actor.property.opacity = opacity
    surf.actor.property.backface_culling = 1
    return surf

def figure(size=None):
    """
    Create a default figure in Mayavi with white background and z pointing down

    Parameters:

    * size
        The size of the figure. If ``None`` will use the default size.

    Return:
    
    * fig
        The Mayavi figure object

    """
    _lazy_import_mlab()
    if size is None:
        fig = mlab.figure(bgcolor=(1, 1, 1))
    else:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    fig.scene.camera.view_up = numpy.array([0., 0., -1.])
    fig.scene.camera.elevation(60.)
    fig.scene.camera.azimuth(180.)
    return fig

def add_outline(extent=None, color=(0,0,0), width=2):
    """
    Create a default outline in Mayavi2.

    Parameters:
    
    * extent
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Default if the objects extent.
    * color
        RGB of the color of the axes and text
    * width
        Line width

    Returns:
    
    * outline
        Mayavi outline instace in the pipeline

    """
    _lazy_import_mlab()
    outline = mlab.outline(color=color, line_width=width)
    if extent is not None:
        outline.bounds = extent
    return outline

def add_axes(plot, nlabels=5, extent=None, ranges=None, color=(0,0,0),
             width=2, fmt="%-#.2f"):
    """
    Add an Axes module to a Mayavi2 plot or dataset.

    Parameters:
    
    * plot
        Either the plot (as returned by one of the plotting functions of this
        module) or a TVTK dataset.
    * nlabels
        Number of labels on the axes
    * extent
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``. Default if the objects extent.
    * ranges
        [xmin, xmax, ymin, ymax, zmin, zmax]. What will be display in the axes
        labels. Default is extent
    * color
        RGB of the color of the axes and text
    * width
        Line width
    * fmt
        Label number format

    Returns:
    
    * axes
        The axes object in the pipeline

    """
    _lazy_import_mlab()
    a = mlab.axes(plot, nb_labels=nlabels, color=color)
    a.label_text_property.color = color
    a.title_text_property.color = color
    if extent is not None:
        a.axes.bounds = extent
    if ranges is not None:
        a.axes.ranges = ranges
        a.axes.use_ranges = True
    a.property.line_width = width
    a.axes.label_format = fmt
    a.axes.x_label, a.axes.y_label, a.axes.z_label = "N", "E", "Z"
    return a

def wall_north(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the North side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([n, n, w, e, b, t], color, opacity)

def wall_south(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the South side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([s, s, w, e, b, t], color, opacity)

def wall_east(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the East side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, e, e, b, t], color, opacity)

def wall_west(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the West side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, w, b, t], color, opacity)

def wall_top(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the Top side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, e, t, t], color, opacity)

def wall_bottom(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the Bottom side.

    Remember: x->North, y->East and z->Down

    Parameters:
    
    * bounds
        ``[xmin, xmax, ymin, ymax, zmin, zmax]``
    * color
        RGB of the color of the wall
    * opacity
        Decimal percentage of opacity

    Tip: Use :func:`fatiando.vis.add_axes3d` to create and 'axes' variable and
    get the bounds as 'axes.axes.bounds'

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, e, b, b], color, opacity)

def _wall(bounds, color, opacity):
    """
    Generate a 3D wall in Mayavi
    """
    _lazy_import_mlab()
    p = mlab.pipeline.builtin_surface()
    p.source = 'outline'
    p.data_source.bounds = bounds
    p.data_source.generate_faces = 1
    su = mlab.pipeline.surface(p)
    su.actor.property.color = color
    su.actor.property.opacity = opacity
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
    
