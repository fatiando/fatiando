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
Easy plotting of grids and 3D meshes.
Uses Matplotlib for 2D and Mayavi2 for 3D.
Grids are automatically reshaped and interpolated if desired or necessary.
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


def contour(x, y, v, shape, levels, interpolate=False, color='k', label=None,
            clabel=True):
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
    ct_data = pyplot.contour(X, Y, V, levels, colors=color, picker=True)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
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

def square(area, color='-k', width=1, label=None):
    """
    Plot a square.

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

def prisms3D(prisms, scalars, label='', style='surface', opacity=1,
             invz=True, xy2ne=False):
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
    * invz
        If ``True``, will invert the sign of values in the z-axis so that plot
        doesn't look upside down. (Because z is assume to grow downward)
    * xy2ne
        If ``True``, will change from x,y to North,East. This means exchaging
        the x and y coordinates so that x is pointing North and y East.
    Returns:
    * surface: the last element on the pipeline

    """
    if style not in ['surface', 'wireframe']:
        raise ValueError, "Invalid style '%s'" % (style)
    if opacity > 1. or opacity < 0:
        msg = "Invalid opacity %g. Must be in range [1,0]" % (opacity)
        raise ValueError, msg

    # Do the lazy imports for these slow modules
    global mlab, tvtk
    # For campatibility with versions of Mayavi2 < 4
    if mlab is None:
        try:
            from mayavi import mlab
        except ImportError:            
            from enthought.mayavi import mlab
    if tvtk is None:
        try:
            from tvtk.api import tvtk
        except ImportError:
            from enthought.tvtk.api import tvtk

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
        if prism is None:
            continue
        if xy2ne:
            x1, x2 = prism['y1'], prism['y2']
            y1, y2 = prism['x1'], prism['x2']
        else:
            x1, x2 = prism['x1'], prism['x2']
            y1, y2 = prism['y1'], prism['y2']
        if invz:
            z1, z2 = -prism['z2'], -prism['z1']
        else:
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
    thresh = mlab.pipeline.threshold(dataset)
    surf = mlab.pipeline.surface(thresh, vmax=max(celldata), vmin=min(celldata))
    if style == 'wireframe':
        surf.actor.property.representation = 'wireframe'
    if style == 'surface':
        surf.actor.property.representation = 'surface'
        surf.actor.property.edge_visibility = 1
    surf.actor.property.opacity = opacity
    surf.actor.property.backface_culling = 1
    return surf

def polyprism_contours(prisms, colors=None, labels=None):
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
        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[i]
        line, = pyplot.plot(*args, **kwargs)
        lines.append(line)
    return lines

#
#
#def plot_square_mesh(mesh, cmap=pyplot.cm.jet, vmin=None, vmax=None):
    #"""
    #Plot a 2D mesh made of square cells. Each cell is a dictionary as::
#
        #{'x1':cellx1, 'x2':cellx2, 'y1':celly1, 'y2':celly2, 'value':value}
#
    #The pseudo color of the plot is key 'value'.
#
    #Parameters:
#
    #* mesh
        #A list of cells describing the square mesh
#
    #* cmap
        #Color map to use. See ``pyplot.cm``
#
    #* vmin, vmax
        #Lower and upper limits for the color scale
#
    #Returns:
#
    #* ``matplitlib.axes`` element of the plot
#
    #"""
#
    #xvalues = []
    #for cell in mesh[0]:
#
        #xvalues.append(cell['x1'])
#
    #xvalues.append(mesh[0][-1]['x2'])
#
    #yvalues = []
#
    #for line in mesh:
#
        #yvalues.append(line[0]['y1'])
#
    #yvalues.append(mesh[-1][0]['y2'])
#
    #X, Y = numpy.meshgrid(xvalues, yvalues)
#
    #Z = numpy.zeros_like(X)
#
    #for i, line in enumerate(mesh):
#
        #for j, cell in enumerate(line):
#
            #Z[i][j] = cell['value']
#
    #if vmin is None and vmax is None:
#
        #plot = pyplot.pcolor(X, Y, Z, cmap=cmap)
#
    #else:
#
        #plot = pyplot.pcolor(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
#
    #return plot
#
#
#def residuals_histogram(residuals, nbins=None):
    #"""
    #Plot a histogram of the residual vector.
#
    #Parameters:
#
    #* residuals
        #1D array-like vector of residuals
#
    #* nbins
        #Number of bins (default to ``len(residuals)/8``)
#
    #Returns:
#
    #* ``matplitlib.axes`` element of the plot
#
    #"""
#
    #if nbins is None:
#
        #nbins = len(residuals)/8
#
    #plot = pyplot.hist(residuals, bins=nbins, facecolor='gray')
#
    #return plot[0]
#
#
#def src_rec(sources, receivers, markersize=9):
    #"""
    #Plot the locations of seismic sources and receivers in a map.
#
    #Parameters:
#
    #* sources
        #List with the x,y coordinates of each source
#
    #* receivers
        #List with the x,y coordinates of each receiver
#
    #* markersize
        #Size of the source and receiver markers
#
    #"""
#
    #src_x, src_y = numpy.transpose(sources)
    #rec_x, rec_y = numpy.transpose(receivers)
#
    #pyplot.plot(src_x, src_y, 'r*', ms=markersize, label='Source')
    #pyplot.plot(rec_x, rec_y, 'b^', ms=int(0.78*markersize), label='Receiver')
#
#
#def ray_coverage(sources, receivers, linestyle='-k'):
    #"""
    #Plot the rays between sources and receivers in a map.
#
    #Parameters:
#
    #* sources
        #List with the x,y coordinates of each source
#
    #* receivers
        #List with the x,y coordinates of each receiver
#
    #* linestyle
        #Type of line to display and color. See ``pyplot.plot``
#
    #Returns:
#
    #* ``matplitlib.axes`` element of the plot
#
    #"""
#
    #for src, rec in zip(sources, receivers):
#
        #plot = pyplot.plot([src[0], rec[0]], [src[1], rec[1]], linestyle)
#
    #return plot[0]
#
#
#
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

