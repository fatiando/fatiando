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
Wrappers for `matplotlib` calls to plot grids (from :mod:`fatiando.gridder`),
2D objects (from :mod:`fatiando.mesher.dd`) and more.

**WARNING**: avoid importing this module as ``from fatiando.vis import map``
because it will cause conflicts with Pythons ``map`` function.

**Grids**

* :func:`fatiando.vis.map.contour`
* :func:`fatiando.vis.map.contourf`
* :func:`fatiando.vis.map.pcolor`

Grids are automatically reshaped and interpolated if desired or necessary.

**2D objects**

* :func:`fatiando.vis.map.points`
* :func:`fatiando.vis.map.paths`
* :func:`fatiando.vis.map.square`
* :func:`fatiando.vis.map.squaremesh`
* :func:`fatiando.vis.map.polygon`

----
   
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 30-Jan-2012'

import numpy
from matplotlib import pyplot

import fatiando.gridder


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

def square(area, style='-k', linewidth=1, label=None):
    """
    Plot a square.

    Parameters:

    * area
        (x1, x2, y1, y2): Borders of the square
    * style
        String with the color and line style (as in matplotlib.pyplot.plot)
    * linewidth
        Line width
    * label
        label associated with the square.

    """
    x1, x2, y1, y2 = area
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    plot, = pyplot.plot(xs, ys, style, **kwargs)
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

def polygon(polygon, style='-k', label=None, linewidth=1):
    """
    Plot a polygon.

    Parameters:

    * polygon
        List of :func:`fatiando.mesher.dd.Polygon`
    * style
        Color and line style string (as in matplotlib.pyplot.plot)
    * label
        String with the label identifying the polygon in the legend 

    Returns:

    * lines
        Line objects corresponding to the polygon plotted

    """
    tmpx = [x for x in polygon['x']]
    tmpx.append(polygon['x'][0])
    tmpy = [y for y in polygon['y']]
    tmpy.append(polygon['y'][0])
    args = [tmpx, tmpy, style]
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    line, = pyplot.plot(*args, **kwargs)
    return line

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

