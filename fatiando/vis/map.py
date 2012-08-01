"""
Wrappers for :mod:`matplotlib` calls to plot grids, 2D objects, etc.

.. tip:: Avoid importing this module using ``from fatiando.vis import map``
    because it will cause conflicts with Pythons ``map`` function.

**Grids**

* :func:`~fatiando.vis.map.contour`
* :func:`~fatiando.vis.map.contourf`
* :func:`~fatiando.vis.map.pcolor`

Grids are automatically reshaped and interpolated if desired or necessary.

**2D objects**

* :func:`~fatiando.vis.map.points`
* :func:`~fatiando.vis.map.paths`
* :func:`~fatiando.vis.map.square`
* :func:`~fatiando.vis.map.squaremesh`
* :func:`~fatiando.vis.map.polygon`
* :func:`~fatiando.vis.map.layers`

**Basemap (map projections)**

* :func:`~fatiando.vis.map.basemap`
* :func:`~fatiando.vis.map.draw_geolines`

**Auxiliary**

* :func:`~fatiando.vis.map.set_area`
* :func:`~fatiando.vis.map.m2km`

----

"""

import numpy
from matplotlib import pyplot

from fatiando import gridder, logger

# Dummy variable to laizy import the basemap toolkit (slow)
Basemap = None

__all__ = ['contour', 'contourf', 'pcolor', 'points', 'paths', 'square',
           'squaremesh', 'polygon', 'layers', 'set_area', 'm2km', 'basemap',
           'draw_geolines']

log = logger.dummy('fatiando.vis.map')

def draw_geolines(area, dlon, dlat, basemap, linewidth=1):
    """
    Draw the parallels and meridians on a basemap plot.

    Parameters:

    * area : list
        ``[west, east, south, north]``, i.e., the area where the lines will
        be plotted
    * dlon, dlat : float
        The spacing between the lines in the longitude and latitude directions,
        respectively (in decimal degrees)
    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`~fatiando.vis.map.basemap`)
    * linewidth : float
        The width of the lines

    """
    west, east, south, north = area
    basemap.drawmeridians(numpy.arange(west, east, dlon), labels=[0,0,0,1])
    basemap.drawparallels(numpy.arange(south, north, dlat), labels=[1,0,0,0])

def basemap(area, projection, resolution='c'):
    """
    Make a basemap to use when plotting with map projections.

    Uses the matplotlib basemap toolkit.

    Parameters:

    * area : list
        ``[west, east, south, north]``, i.e., the area of the data that is going
        to be plotted
    * projection : str
        The name of the projection you want to use. Choose from:

        * 'ortho': Orthographic
        * 'geos': Geostationary
        * 'robin': Robinson
        * 'cass': Cassini
        * 'merc': Mercator
        * 'poly': Polyconic
        * 'lcc': Lambert Conformal
        * 'stere': Stereographic

    * resolution : str
        The resolution for the coastlines. Can be 'c' for crude, 'l' for low,
        'i' for intermediate, 'h' for high

    Returns:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap

    """
    if projection not in ['ortho', 'aeqd', 'geos', 'robin', 'cass', 'merc',
        'poly', 'lcc', 'stere']:
        raise ValueError("Unsuported projection '%s'" % (projection))
    global Basemap
    if Basemap is None:
        try:
            from mpl_toolkits.basemap import Basemap
        except ImportError:
            log.error("matplotlib basemap toolkit not found")
            raise
    west, east, south, north = area
    lon_0 = 0.5*(east + west)
    lat_0 = 0.5*(north + south)
    if projection == 'ortho':
        bm = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                     resolution=resolution)
    elif projection == 'geos' or projection == 'robin':
        bm = Basemap(projection=projection,lon_0=lon_0, resolution=resolution)
    elif (projection == 'cass' or
           projection == 'poly'):
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0, lon_0=lon_0,
                     resolution=resolution)
    elif projection == 'merc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_ts=lat_0,
                     resolution=resolution)
    elif projection == 'lcc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0, lon_0=lon_0,
                     rsphere=(6378137.00,6356752.3142), lat_1=lat_0,
                     resolution=resolution)
    elif projection == 'stere':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,lon_0=lon_0,
                     lat_ts=lat_0, resolution=resolution)
    return bm

def m2km(axis=None):
    """
    Convert the x and y tick labels from meters to kilometers.

    Parameters:

    * axis : matplotlib axis instance
        The plot.

    .. tip:: Use ``fatiando.vis.gca()`` to get the current axis. Or the value
        returned by ``fatiando.vis.subplot`` or ``matplotlib.pyplot.subplot``.

    """
    if axis is None:
        axis = pyplot.gca()
    axis.set_xticklabels([str(0.001*l) for l in axis.get_xticks()])
    axis.set_yticklabels([str(0.001*l) for l in axis.get_yticks()])

def set_area(area):
    """
    Set the area of a Matplolib plot using xlim and ylim.

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Coordinates of the top right and bottom left corners of the area

    """
    x1, x2, y1, y2 = area
    pyplot.xlim(x1, x2)
    pyplot.ylim(y1, y2)

def points(pts, style='.k', size=10, label=None):
    """
    Plot a list of points.

    Parameters:

    * pts : list of lists
        List of [x, y] pairs with the coordinates of the points
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * size : int
        Size of the plotted points
    * label : str
        If not None, then the string that will show in the legend

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

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

    * pts1 : list of lists
        List of (x, y) pairs with the coordinates of the points
    * pts2 : list of lists
        List of (x, y) pairs with the coordinates of the points
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * linewidth : float
        The width of the lines representing the paths
    * label : str
        If not None, then the string that will show in the legend

    """
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    for p1, p2 in zip(pts1, pts2):
        pyplot.plot([p1[0], p2[0]], [p1[1], p2[1]], style, **kwargs)

def layers(thickness, values, style='-k', z0=0., linewidth=1, label=None,
    **kwargs):
    """
    Plot a series of layers and values associated to each layer.

    Parameters:

    * thickness : list
        The thickness of each layer in order of increasing depth
    * values : list
        The value associated with each layer in order of increasing
        depth
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * z0 : float
        The depth of the top of the first layer
    * linewidth : float
        Line width
    * label : str
        label associated with the square.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if len(thickness) != len(values):
        raise ValueError, "thickness and values must have same length"
    nlayers = len(thickness)
    interfaces = [z0 + sum(thickness[:i]) for i in xrange(nlayers + 1)]
    ys = [interfaces[0]]
    for y in interfaces[1:-1]:
        ys.append(y)
        ys.append(y)
    ys.append(interfaces[-1])
    xs = []
    for x in values:
        xs.append(x)
        xs.append(x)
    kwargs['linewidth'] = linewidth
    if label is not None:
        kwargs['label'] = label
    plot, = pyplot.plot(xs, ys, style, **kwargs)
    return plot

def square(area, style='-k', linewidth=1, fill=None, alpha=1., label=None):
    """
    Plot a square.

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the square
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * linewidth : float
        Line width
    * fill : str
        A color string used to fill the square. If None, the square is not
        filled
    * alpha : float
        Transparency of the fill (1 >= alpha >= 0). 0 is transparent and 1 is
        opaque
    * label : str
        label associated with the square.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    x1, x2, y1, y2 = area
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    plot, = pyplot.plot(xs, ys, style, **kwargs)
    if fill is not None:
        pyplot.fill(xs, ys, color=fill, alpha=alpha)
    return plot

def squaremesh(mesh, prop, cmap=pyplot.cm.jet, vmin=None, vmax=None):
    """
    Make a pseudo-color plot of a mesh of squares

    Parameters:

    * mesh : :class:`fatiando.mesher.dd.SquareMesh` or compatible
        The mesh (a compatible mesh must implement the methods ``get_xs`` and
        ``get_ys``)
    * prop : str
        The physical property of the squares to use as the color scale.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax : float
        Saturation values of the colorbar.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if prop not in mesh.props:
        raise ValueError("Can't plot because 'mesh' doesn't have property '%s'"
                         % (prop))
    xs = mesh.get_xs()
    ys = mesh.get_ys()
    X, Y = numpy.meshgrid(xs, ys)
    V = numpy.reshape(mesh.props[prop], mesh.shape)
    plot = pyplot.pcolor(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax, picker=True)
    pyplot.xlim(xs.min(), xs.max())
    pyplot.ylim(ys.min(), ys.max())
    return plot

def polygon(polygon, style='-k', linewidth=1, fill=None, alpha=1., label=None,
    xy2ne=False):
    """
    Plot a polygon.

    Parameters:

    * polygon : :class:`fatiando.mesher.dd.Polygon`
        The polygon
    * style : str
        Color and line style string (as in matplotlib.pyplot.plot)
    * linewidth : float
        Line width
    * fill : str
        A color string used to fill the polygon. If None, the polygon is not
        filled
    * alpha : float
        Transparency of the fill (1 >= alpha >= 0). 0 is transparent and 1 is
        opaque
    * label : str
        String with the label identifying the polygon in the legend
    * xy2ne : True or False
        If True, will exchange the x and y axis so that the x coordinates of the
        polygon are north. Use this when drawing on a map viewed from above. If
        the y-axis of the plot is supposed to be z (depth), then use
        ``xy2ne=False``.

    Returns:

    * lines : matplotlib Line object
        Line corresponding to the polygon plotted

    """
    if xy2ne:
        tmpx = [y for y in polygon.y]
        tmpx.append(polygon.y[0])
        tmpy = [x for x in polygon.x]
        tmpy.append(polygon.x[0])
    else:
        tmpx = [x for x in polygon.x]
        tmpx.append(polygon.x[0])
        tmpy = [y for y in polygon.y]
        tmpy.append(polygon.y[0])
    kwargs = {'linewidth':linewidth}
    if label is not None:
        kwargs['label'] = label
    line, = pyplot.plot(tmpx, tmpy, style, **kwargs)
    if fill is not None:
        pyplot.fill(tmpx, tmpy, color=fill, alpha=alpha)
    return line

def contour(x, y, v, shape, levels, interp=False, color='k', label=None,
            clabel=True, style='solid', linewidth=1.0, basemap=None):
    """
    Make a contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * color : str
        Color of the contour lines.
    * label : str
        String with the label of the contour that would show in a legend.
    * clabel : True or False
        Wether or not to print the numerical value of the contour lines
    * style : str
        The style of the contour lines. Can be ``'dashed'``, ``'solid'`` or
        ``'mixed'`` (solid lines for positive contours and dashed for negative)
    * linewidth : float
        Width of the contour lines
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.map.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError, "Invalid contour style %s" % (style)
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interp:
        X, Y, V = gridder.interp(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    if basemap is None:
        ct_data = pyplot.contour(X, Y, V, levels, colors=color, picker=True)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contour(lon, lat, V, levels, colors=color,
                                  picker=True)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
    if style != 'mixed':
        for c in ct_data.collections:
            c.set_linestyle(style)
    for c in ct_data.collections:
        c.set_linewidth(linewidth)
    return ct_data.levels

def contourf(x, y, v, shape, levels, interp=False, cmap=pyplot.cm.jet,
    basemap=None):
    """
    Make a filled contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.map.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interp:
        X, Y, V = gridder.interp(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, cmap=cmap, picker=True)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, cmap=cmap, picker=True)
    return ct_data.levels

def pcolor(x, y, v, shape, interp=False, cmap=pyplot.cm.jet, vmin=None,
           vmax=None, basemap=None):
    """
    Make a pseudo-color plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax
        Saturation values of the colorbar.
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.map.basemap` for creating basemaps)

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if x.shape != y.shape != v.shape:
        raise ValueError, "Input arrays x, y, and v must have same shape!"
    if interp:
        X, Y, V = gridder.interp(x, y, v, shape)
    else:
        X = numpy.reshape(x, shape)
        Y = numpy.reshape(y, shape)
        V = numpy.reshape(v, shape)
    if basemap is None:
        plot = pyplot.pcolor(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax,
                             picker=True)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        plot = basemap.pcolor(lon, lat, V, cmap=cmap, vmin=vmin, vmax=vmax,
                              picker=True)
    return plot

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

