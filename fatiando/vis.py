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
Provides wrappers for `matplotlib` and `mayavi2` functions for easier plotting
of grids, 3D meshes, etc.

Grids are automatically reshaped and interpolated if desired or necessary.

**2D plotting**

* :func:`fatiando.vis.points`
* :func:`fatiando.vis.paths`
* :func:`fatiando.vis.contour`
* :func:`fatiando.vis.contourf`
* :func:`fatiando.vis.pcolor`
* :func:`fatiando.vis.square`
* :func:`fatiando.vis.squaremesh`
* :func:`fatiando.vis.polyprism_contours`

**3D plotting**

* :func:`fatiando.vis.mayavi_figure`
* :func:`fatiando.vis.prisms3D`
* :func:`fatiando.vis.add_outline3d`
* :func:`fatiando.vis.add_axes3d`
* :func:`fatiando.vis.wall_north`
* :func:`fatiando.vis.wall_south`
* :func:`fatiando.vis.wall_east`
* :func:`fatiando.vis.wall_west`
* :func:`fatiando.vis.wall_top`
* :func:`fatiando.vis.wall_bottom`

----
   
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Sep-2010'

import numpy
from matplotlib import pyplot

import fatiando.gridder

# Do lazy imports of mlab and tvtk to avoid the slow imports when I don't need
# 3D plotting
mlab = None
tvtk = None


def set_area(area):
    """
    Set the area of a Matplolib plot using xlim and ylim.

    Parameters:

    * area
        [x1, x2, y1, y2]: coordinates of the top right and bottom left
        corners of the area
         
    """
    x1, x2, y1, y2 = area
    pyplot.xlim(x1, x2)
    pyplot.ylim(y1, y2)
    
def points(pts, style='.k', size=10, label=None):
    """
    Plot a list of points.

    Parameters:

    * pts
        List of (x, y) pairs with the coordinates of the points
    * style
        String with the color and line style (as in matplotlib.pyplot.plot)
    * size
        Size of the plotted points
    * label
        If not None, then the string that will show in the legend

    Returns:
    
    * ``matplitlib.axes`` element of the plot
    
    """
    x, y = numpy.array(pts).T
    kwargs = {}
    if label is not None:
        kwargs['label'] = label
    return pyplot.plot(x, y, style, markersize=size, **kwargs)
    
def paths(pts1, pts2, style='-k', linewidth=1, label=None):
    """
    Plot paths between the two sets of points.

    Parameters:

    * pts1
        List of (x, y) pairs with the coordinates of the points
    * pts2
        List of (x, y) pairs with the coordinates of the points
    * style
        String with the color and line style (as in matplotlib.pyplot.plot)
    * linewidth
        The width of the lines representing the paths
    * label
        If not None, then the string that will show in the legend
    
    """
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    for p1, p2 in zip(pts1, pts2):
        pyplot.plot([p1[0], p2[0]], [p1[1], p2[1]], style, **kwargs)

def contour(x, y, v, shape, levels, interpolate=False, color='k', label=None,
            clabel=True, style='solid', linewidth=1.0):
    """
    Make a contour plot of the data.

    Parameters:

    * x, y
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v
        Array with the scalar value assigned to the grid points.
    * shape
        Shape of the regular grid, ie (ny, nx).
        If interpolation is not False, then will use *shape* to grid the data.
    * levels
        Number of contours to use or a list with the contour values.
    * interpolate
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * color
        Color of the contour lines.
    * label
        String with the label of the contour that would show in a legend.
    * clabel
        Wether or not to print the numerical value of the contour lines
    * style
        The style of the contour lines. Can be ``'dashed'``, ``'solid'`` or
        ``'mixed'`` (solid lines for positive contours and dashed for negative)
    * linewidth
        Width of the contour lines
        
    Returns:

    * levels
        List with the values of the contour levels

    """
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError, "Invalid contour style %s" % (style)
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interpolate:
        X, Y, V = fatiando.gridder.interpolate(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    ct_data = pyplot.contour(X, Y, V, levels, colors=color, picker=True)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
    if style != 'mixed':
        for c in ct_data.collections:
            c.set_linestyle(style)
    for c in ct_data.collections:
        c.set_linewidth(linewidth)
    pyplot.xlim(X.min(), X.max())
    pyplot.ylim(Y.min(), Y.max())
    return ct_data.levels

def contourf(x, y, v, shape, levels, interpolate=False, cmap=pyplot.cm.jet):
    """
    Make a filled contour plot of the data.

    Parameters:

    * x, y
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v
        Array with the scalar value assigned to the grid points.
    * shape
        Shape of the regular grid, ie (ny, nx).
        If interpolation is not False, then will use *shape* to grid the data.
    * levels
        Number of contours to use or a list with the contour values.
    * interpolate
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * cmap
        Color map to be used. (see pyplot.cm module)

    Returns:

    * levels
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interpolate:
        X, Y, V = fatiando.gridder.interpolate(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    ct_data = pyplot.contourf(X, Y, V, levels, cmap=cmap, picker=True)
    pyplot.xlim(X.min(), X.max())
    pyplot.ylim(Y.min(), Y.max())
    return ct_data.levels

def pcolor(x, y, v, shape, interpolate=False, cmap=pyplot.cm.jet, vmin=None,
           vmax=None):
    """
    Make a pseudo-color plot of the data.

    Parameters:

    * x, y
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v
        Array with the scalar value assigned to the grid points.
    * shape
        Shape of the regular grid, ie (ny, nx).
        If interpolation is not False, then will use *shape* to grid the data.
    * interpolate
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * cmap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax
        Saturation values of the colorbar.

    Returns:

    * ``matplitlib.axes`` element of the plot

    """
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interpolate:
        X, Y, V = fatiando.gridder.interpolate(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    plot = pyplot.pcolor(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax, picker=True)
    pyplot.xlim(X.min(), X.max())
    pyplot.ylim(Y.min(), Y.max())
    return plot

def squaremesh(mesh, scalars, cmap=pyplot.cm.jet, vmin=None, vmax=None):
    """
    Make a pseudo-color plot of a mesh of squares
    
    Parameters:

    * mesh
        A :func:`fatiando.mesher.dd.SquareMesh` or other compatible mesh
        (a compatible mesh must implement the methods ``get_xs`` and
        ``get_ys``)
    * scalars
        Array with the scalar value assigned to each square in the mesh
    * cmap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax
        Saturation values of the colorbar.

    Returns:

    * ``matplitlib.axes`` element of the plot

    """
    xs = mesh.get_xs()
    ys = mesh.get_ys()
    X, Y = numpy.meshgrid(xs, ys)
    V = numpy.reshape(scalars, mesh.shape)
    plot = pyplot.pcolor(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax, picker=True)
    pyplot.xlim(xs.min(), xs.max())
    pyplot.ylim(ys.min(), ys.max())
    return plot

def square(area, color='-k', width=1, label=None):
    """
    Plot a square.

    Parameters:

    * area
        (x1, x2, y1, y2): Borders of the square
    * color
        String with the color and line style (as in matplotlib.pyplot.plot)
    * width
        Line width
    * label
        label associated with the square.

    """
    x1, x2, y1, y2 = area
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth':width}
    if label is not None:
        kwargs['label'] = label
    plot, = pyplot.plot(xs, ys, color, **kwargs)
    return plot

def polyprism_contours(prisms, colors=None, labels=None, linewidth=1):
    """
    Plot 2D contours of PolygonalPrism3D objects on a map.

    Parameters:

    * prisms
        List of PolygonalPrism3D
    * colors
        List of color and line style strings, one for each prism (as in
        matplotlib.pyplot.plot)
    * labels
        List of labels (strings) associated with the prisms.

    Returns:

    * lines
        List of line objects corresponding to the prisms plotted

    """
    lines = []
    for i, prism in enumerate(prisms):
        tmpx = [x for x in prism['x']]
        tmpx.append(prism['x'][0])
        tmpy = [y for y in prism['y']]
        tmpy.append(prism['y'][0])
        args = [tmpx, tmpy]
        if colors is not None:
            args.append(colors[i])
        kwargs = {'linewidth':linewidth}
        if labels is not None:
            kwargs['label'] = labels[i]
        line, = pyplot.plot(*args, **kwargs)
        lines.append(line)
    return lines

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

def prisms3D(prisms, scalars, label='scalars', style='surface', opacity=1,
             edges=True, vmin=None, vmax=None, cmap='blue-red'):
    """
    Plot a 3D right rectangular prisms using Mayavi2.

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

def mayavi_figure(size=None):
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

def add_outline3d(extent=None, color=(0,0,0), width=2):
    """
    Create a default outline in Mayavi.

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

def add_axes3d(plot, nlabels=5, extent=None, ranges=None, color=(0,0,0),
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
#
#
#def plot_2d_interface(mesh, key='value', style='-k', linewidth=1, fill=None,
                      #fillcolor='r', fillkey='value', alpha=1, label=''):
    #"""
    #Plot a 2d prism interface mesh.
#
    #Parameters:
#
    #* mesh
        #Model space discretization mesh (see :func:`fatiando.mesh.line_mesh`)
#
    #* key
        #Which key of *mesh* represents the bottom of the prisms
#
    #* style
        #Line and marker style and color (see ``pyplot.plot``)
#
    #* linewidth
        #Width of the line plotted (see ``pyplot.plot``)
#
    #* fill
        #If not ``None``, then another mesh to fill between it and *mesh*
#
    #* fillcolor
        #The color of the fill region
#
    #* fillkey
        #Which key of *fill* represents the bottom of the prisms
#
    #* alpha
        #Opacity of the fill region
#
    #* label
        #Label of the interface line
#
    #"""
#
    #xs = []
    #zs = []
#
    #for cell in mesh:
#
        #xs.append(cell['x1'])
        #xs.append(cell['x2'])
        #zs.append(cell[key])
        #zs.append(cell[key])
#
    #if fill is not None:
#
        #fill_zs = []
#
        #for cell in fill:
#
            #fill_zs.append(cell[fillkey])
            #fill_zs.append(cell[fillkey])
#
        #pyplot.fill_between(xs, fill_zs, zs, facecolor=fillcolor, alpha=alpha)
#
    #plot = pyplot.plot(xs, zs, style, linewidth=linewidth, label=label)
#
    #return plot[0]

