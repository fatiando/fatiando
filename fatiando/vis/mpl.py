"""
Wrappers for :mod:`matplotlib` functions to facilitate plotting grids,
2D objects, etc.

This module loads all functions from :mod:`matplotlib.pyplot`, adds new
functions and overwrites some others (like :func:`~fatiando.vis.mpl.contour`,
:func:`~fatiando.vis.mpl.pcolor`, etc).

**Grids**

* :func:`~fatiando.vis.mpl.contour`
* :func:`~fatiando.vis.mpl.contourf`
* :func:`~fatiando.vis.mpl.pcolor`

Grids are automatically reshaped and interpolated if desired or necessary.

**2D objects**

* :func:`~fatiando.vis.mpl.points`
* :func:`~fatiando.vis.mpl.paths`
* :func:`~fatiando.vis.mpl.square`
* :func:`~fatiando.vis.mpl.squaremesh`
* :func:`~fatiando.vis.mpl.polygon`
* :func:`~fatiando.vis.mpl.layers`
* :func:`~fatiando.vis.mpl.seismic_image`
* :func:`~fatiando.vis.mpl.seismic_wiggle`

**Interactive**

* :func:`~fatiando.vis.mpl.draw_polygon`
* :func:`~fatiando.vis.mpl.draw_layers`
* :func:`~fatiando.vis.mpl.pick_points`

**Basemap (map projections)**

* :func:`~fatiando.vis.mpl.basemap`
* :func:`~fatiando.vis.mpl.draw_geolines`
* :func:`~fatiando.vis.mpl.draw_countries`
* :func:`~fatiando.vis.mpl.draw_coastlines`

**Auxiliary**

* :func:`~fatiando.vis.mpl.set_area`
* :func:`~fatiando.vis.mpl.m2km`

----

"""

import numpy
from matplotlib import pyplot, widgets
# Quick hack so that the docs can build using the mocks for readthedocs
# Ideal would be to log an error message saying that functions from pyplot
# were not imported
try:
    from matplotlib.pyplot import *
except:
    pass

import fatiando.gridder

# Dummy variable to laizy import the basemap toolkit (slow)
Basemap = None


def draw_polygon(area, axes, style='-', marker='o', color='k', width=2,
                 alpha=0.5, xy2ne=False):
    """
    Draw a polygon by clicking with the mouse.

    INSTRUCTIONS:

    * Left click to pick the edges of the polygon;
    * Draw edges CLOCKWISE;
    * Press 'e' to erase the last edge;
    * Right click to close the polygon;
    * Close the figure window to finish;

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the area containing the polygon
    * axes : matplotlib Axes
        The figure to use for drawing the polygon.
        To get an Axes instace, just do::

            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * style : str
        Line style (as in matplotlib.pyplot.plot)
    * marker : str
        Style of the point markers (as in matplotlib.pyplot.plot)
    * color : str
        Line color (as in matplotlib.pyplot.plot)
    * width : float
        The line width (as in matplotlib.pyplot.plot)
    * alpha : float
        Transparency of the fill of the polygon. 0 for transparent, 1 for
        opaque (fills the polygon once done drawing)
    * xy2ne : True or False
        If True, will exchange the x and y axis so that x points north.
        Use this when drawing on a map viewed from above. If the y-axis of the
        plot is supposed to be z (depth), then use ``xy2ne=False``.

    Returns:

    * edges : list of lists
        List of ``[x, y]`` pairs with the edges of the polygon

    """
    axes.set_title("Click to draw polygon. Right click when done.")
    if xy2ne:
        axes.set_xlim(area[2], area[3])
        axes.set_ylim(area[0], area[1])
    else:
        axes.set_xlim(area[0], area[1])
        axes.set_ylim(area[2], area[3])
    # start with an empty line
    line, = axes.plot([], [], marker=marker, linestyle=style, color=color,
                      linewidth=width)
    tmpline, = axes.plot([], [], marker=marker, linestyle=style, color=color,
                         linewidth=width)
    draw = axes.figure.canvas.draw
    x = []
    y = []
    plotx = []
    ploty = []
    # Hack because Python 2 doesn't like nonlocal variables that change value.
    # Lists it doesn't mind.
    picking = [True]

    def draw_guide(px, py):
        if len(x) != 0:
            tmpline.set_data([x[-1], px], [y[-1], py])

    def move(event):
        if event.inaxes != axes:
            return 0
        if picking[0]:
            draw_guide(event.xdata, event.ydata)
            draw()

    def pick(event):
        if event.inaxes != axes:
            return 0
        if event.button == 1 and picking[0]:
            x.append(event.xdata)
            y.append(event.ydata)
            plotx.append(event.xdata)
            ploty.append(event.ydata)
        if event.button == 3 or event.button == 2 and picking[0]:
            if len(x) < 3:
                axes.set_title("Need at least 3 points to make a polygon")
            else:
                picking[0] = False
                axes.set_title("Done! You can close the window now.")
                plotx.append(x[0])
                ploty.append(y[0])
                tmpline.set_data([], [])
                axes.fill(plotx, ploty, color=color, alpha=alpha)
        line.set_data(plotx, ploty)
        draw()

    def erase(event):
        if event.key == 'e' and picking[0]:
            x.pop()
            y.pop()
            plotx.pop()
            ploty.pop()
            line.set_data(plotx, ploty)
            draw_guide(event.xdata, event.ydata)
            draw()
    line.figure.canvas.mpl_connect('button_press_event', pick)
    line.figure.canvas.mpl_connect('key_press_event', erase)
    line.figure.canvas.mpl_connect('motion_notify_event', move)
    pyplot.show()
    if len(x) < 3:
        raise ValueError("Need at least 3 points to make a polygon")
    if xy2ne:
        verts = numpy.transpose([y, x])
    else:
        verts = numpy.transpose([x, y])
    return verts


def pick_points(area, axes, marker='o', color='k', size=8, xy2ne=False):
    """
    Get the coordinates of points by clicking with the mouse.

    INSTRUCTIONS:

    * Left click to pick the points;
    * Press 'e' to erase the last point picked;
    * Close the figure window to finish;

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the area containing the points
    * axes : matplotlib Axes
        The figure to use for drawing the polygon.
        To get an Axes instace, just do::

            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * marker : str
        Style of the point markers (as in matplotlib.pyplot.plot)
    * color : str
        Line color (as in matplotlib.pyplot.plot)
    * size : float
        Marker size (as in matplotlib.pyplot.plot)
    * xy2ne : True or False
        If True, will exchange the x and y axis so that x points north.
        Use this when drawing on a map viewed from above. If the y-axis of the
        plot is supposed to be z (depth), then use ``xy2ne=False``.

    Returns:

    * points : list of lists
        List of ``[x, y]`` coordinates of the points

    """
    axes.set_title("Click to pick points. Close window when done.")
    if xy2ne:
        axes.set_xlim(area[2], area[3])
        axes.set_ylim(area[0], area[1])
    else:
        axes.set_xlim(area[0], area[1])
        axes.set_ylim(area[2], area[3])
    # start with an empty set
    line, = axes.plot([], [], marker=marker, color=color, markersize=size)
    line.figure.canvas.draw()
    x = []
    y = []
    plotx = []
    ploty = []
    # Hack because Python 2 doesn't like nonlocal variables that change value.
    # Lists it doesn't mind.
    picking = [True]

    def pick(event):
        if event.inaxes != axes:
            return 0
        if event.button == 1 and picking[0]:
            x.append(event.xdata)
            y.append(event.ydata)
            plotx.append(event.xdata)
            ploty.append(event.ydata)
            line.set_color(color)
            line.set_marker(marker)
            line.set_markersize(size)
            line.set_linestyle('')
            line.set_data(plotx, ploty)
            line.figure.canvas.draw()

    def erase(event):
        if event.key == 'e' and picking[0]:
            x.pop()
            y.pop()
            plotx.pop()
            ploty.pop()
            line.set_data(plotx, ploty)
            line.figure.canvas.draw()
    line.figure.canvas.mpl_connect('button_press_event', pick)
    line.figure.canvas.mpl_connect('key_press_event', erase)
    pyplot.show()
    if xy2ne:
        points = numpy.transpose([y, x])
    else:
        points = numpy.transpose([x, y])
    return points


def draw_layers(area, axes, style='-', marker='o', color='k', width=2):
    """
    Draw series of horizontal layers by clicking with the mouse.

    The y-axis is assumed to be depth, the x-axis is the physical property of
    each layer.

    INSTRUCTIONS:

    * Click to make a new layer;
    * Press 'e' to erase the last layer;
    * Close the figure window to finish;

    Parameters:

    * area : list = [x1, x2, y1, y2]
        Borders of the area containing the polygon
    * axes : matplotlib Axes
        The figure to use for drawing the polygon.
        To get an Axes instace, just do::

            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * style : str
        Line style (as in matplotlib.pyplot.plot)
    * marker : str
        Style of the point markers (as in matplotlib.pyplot.plot)
    * color : str
        Line color (as in matplotlib.pyplot.plot)
    * width : float
        The line width (as in matplotlib.pyplot.plot)

    Returns:

    * layers : list = [thickness, values]

        * thickness : list
            The thickness of each layer, in order of increasing depth
        * values : list
            The physical property value of each layer, in the same order

    """
    axes.set_title("Click to set a layer. Close the window when done.")
    axes.grid()
    vmin, vmax, zmin, zmax = area
    axes.set_xlim(vmin, vmax)
    axes.set_ylim(zmax, zmin)
    # start with an empty line
    line, = axes.plot([], [], marker=marker, linestyle=style,
                      color=color, linewidth=width)
    midv = 0.5 * (vmax + vmin)
    # this is the line that moves around with the mouse
    tmpline, = axes.plot([midv], [zmin], marker=marker, linestyle='--',
                         color=color, linewidth=width)
    # Make a proxy for drawing
    draw = axes.figure.canvas.draw
    depths = [zmin]
    values = []
    plotv = []
    plotz = []
    tmpz = [zmin]
    # Hack because Python 2 doesn't like nonlocal variables that change value.
    # Lists it doesn't mind.
    picking = [True]

    def draw_guide(v, z):
        if len(values) == 0:
            tmpline.set_data([v, v], [tmpz[0], z])
        else:
            if z > tmpz[0]:
                tmpline.set_data([values[-1], v, v], [tmpz[0], tmpz[0], z])
            else:
                tmpline.set_data([values[-1], v], [tmpz[0], tmpz[0]])

    def move(event):
        if event.inaxes != axes:
            return 0
        v, z = event.xdata, event.ydata
        if picking[0]:
            draw_guide(v, z)
            draw()

    def pick(event):
        if event.inaxes != axes:
            return 0
        if event.button == 1 and picking[0]:
            v, z = event.xdata, event.ydata
            if z > tmpz[0]:
                depths.append(z)
                values.append(v)
                plotz.extend([tmpz[0], z])
                plotv.extend([v, v])
                tmpz[0] = z
                line.set_data(plotv, plotz)
                draw()

    def erase(event):
        if picking[0] and len(values) > 0 and event.key == 'e':
            depths.pop()
            values.pop()
            tmpz[0] = depths[-1]
            plotv.pop()
            plotv.pop()
            plotz.pop()
            plotz.pop()
            line.set_data(plotv, plotz)
            draw_guide(event.xdata, event.ydata)
            draw()
    line.figure.canvas.mpl_connect('button_press_event', pick)
    line.figure.canvas.mpl_connect('key_press_event', erase)
    line.figure.canvas.mpl_connect('motion_notify_event', move)
    pyplot.show()
    thickness = [depths[i + 1] - depths[i] for i in xrange(len(depths) - 1)]
    return thickness, values


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
        The basemap used for plotting (see :func:`~fatiando.vis.mpl.basemap`)
    * linewidth : float
        The width of the lines

    """
    west, east, south, north = area
    meridians = basemap.drawmeridians(numpy.arange(west, east, dlon),
                                      labels=[0, 0, 0, 1], linewidth=linewidth)
    parallels = basemap.drawparallels(numpy.arange(south, north, dlat),
                                      labels=[1, 0, 0, 0], linewidth=linewidth)


def draw_countries(basemap, linewidth=1, style='dashed'):
    """
    Draw the country borders using the given basemap.

    Parameters:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`~fatiando.vis.mpl.basemap`)
    * linewidth : float
        The width of the lines
    * style : str
        The style of the lines. Can be: 'solid', 'dashed', 'dashdot' or
        'dotted'

    """
    lines = basemap.drawcountries(linewidth=linewidth)
    lines.set_linestyles(style)


def draw_coastlines(basemap, linewidth=1, style='solid'):
    """
    Draw the coastlines using the given basemap.

    Parameters:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`~fatiando.vis.mpl.basemap`)
    * linewidth : float
        The width of the lines
    * style : str
        The style of the lines. Can be: 'solid', 'dashed', 'dashdot' or
        'dotted'

    """
    lines = basemap.drawcoastlines(linewidth=linewidth)
    lines.set_linestyles(style)


def basemap(area, projection, resolution='c'):
    """
    Make a basemap to use when plotting with map projections.

    Uses the matplotlib basemap toolkit.

    Parameters:

    * area : list
        ``[west, east, south, north]``, i.e., the area of the data that is
        going to be plotted
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
            raise
    west, east, south, north = area
    lon_0 = 0.5 * (east + west)
    lat_0 = 0.5 * (north + south)
    if projection == 'ortho':
        bm = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                     resolution=resolution)
    elif projection == 'geos' or projection == 'robin':
        bm = Basemap(projection=projection, lon_0=lon_0, resolution=resolution)
    elif (projection == 'cass' or
          projection == 'poly'):
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, resolution=resolution)
    elif projection == 'merc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_ts=lat_0,
                     resolution=resolution)
    elif projection == 'lcc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, rsphere=(6378137.00, 6356752.3142),
                     lat_1=lat_0, resolution=resolution)
    elif projection == 'stere':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, lat_ts=lat_0, resolution=resolution)
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
    axis.set_xticklabels(['%g' % (0.001 * l) for l in axis.get_xticks()])
    axis.set_yticklabels(['%g' % (0.001 * l) for l in axis.get_yticks()])


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


def points(pts, style='.k', size=10, label=None, xy2ne=False):
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
    * xy2ne : True or False
        If True, will exchange the x and y axis so that the x coordinates of
        the polygon are north. Use this when drawing on a map viewed from
        above. If the y-axis of the plot is supposed to be z (depth), then use
        ``xy2ne=False``.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    x, y = numpy.array(pts).T
    if xy2ne:
        x, y = y, x
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
    kwargs = {'linewidth': linewidth}
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
        raise ValueError("thickness and values must have same length")
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


def square(area, style='-k', linewidth=1, fill=None, alpha=1., label=None,
           xy2ne=False):
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
    * xy2ne : True or False
        If True, will exchange the x and y axis so that the x coordinates of
        the polygon are north. Use this when drawing on a map viewed from
        above. If the y-axis of the plot is supposed to be z (depth), then use
        ``xy2ne=False``.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    x1, x2, y1, y2 = area
    if xy2ne:
        x1, x2, y1, y2 = y1, y2, x1, x2
    xs = [x1, x1, x2, x2, x1]
    ys = [y1, y2, y2, y1, y1]
    kwargs = {'linewidth': linewidth}
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

    * mesh : :class:`fatiando.mesher.SquareMesh` or compatible
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
            xy2ne=False, linealpha=1.):
    """
    Plot a polygon.

    Parameters:

    * polygon : :class:`fatiando.mesher.Polygon`
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
    * linealpha : float
        Transparency of the line (1 >= alpha >= 0). 0 is transparent and 1 is
        opaque
    * label : str
        String with the label identifying the polygon in the legend
    * xy2ne : True or False
        If True, will exchange the x and y axis so that the x coordinates of
        the polygon are north. Use this when drawing on a map viewed from
        above. If the y-axis of the plot is supposed to be z (depth), then use
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
    kwargs = {'linewidth': linewidth, 'alpha': linealpha}
    if label is not None:
        kwargs['label'] = label
    line, = pyplot.plot(tmpx, tmpy, style, **kwargs)
    if fill is not None:
        pyplot.fill(tmpx, tmpy, color=fill, alpha=alpha)
    return line


def contour(x, y, v, shape, levels, interp=False, extrapolate=False, color='k',
            label=None, clabel=True, style='solid', linewidth=1.0,
            basemap=None):
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
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
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
        (see :func:`~fatiando.vis.mpl.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError("Invalid contour style %s" % (style))
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = fatiando.gridder.interp(x, y, v, shape,
                                          extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(colors=color, picker=True)
    if basemap is None:
        ct_data = pyplot.contour(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contour(lon, lat, V, levels, **kwargs)
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


def contourf(x, y, v, shape, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap=pyplot.cm.jet, basemap=None):
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
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * vmin, vmax
        Saturation values of the colorbar. If provided, will overwrite what is
        set by *levels*.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.mpl.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = fatiando.gridder.interp(x, y, v, shape,
                                          extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, picker=True)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, **kwargs)
    return ct_data.levels


def pcolor(x, y, v, shape, interp=False, extrapolate=False, cmap=pyplot.cm.jet,
           vmin=None, vmax=None, basemap=None):
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
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax
        Saturation values of the colorbar.
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.mpl.basemap` for creating basemaps)

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    if interp:
        x, y, v = fatiando.gridder.interp(x, y, v, shape,
                                          extrapolate=extrapolate)
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


def seismic_wiggle(section, dt=0.004, ranges=None, scale=1.,
                   color='k', normalize=False):
    """
    Plot a seismic section (numpy 2D array matrix) as wiggles.

    Parameters:

    * section :  2D array
        matrix of traces (first dimension time, second dimension traces)
    * dt : float
        sample rate in seconds (default 4 ms)
    * ranges : (x1, x2)
        min and max horizontal values (default trace number)
    * scale : float
        scale factor multiplied by the section values before plotting
    * color : tuple of strings
        Color for filling the wiggle, positive  and negative lobes.
    * normalize :
        True to normalizes all trace in the section using global max/min
        data will be in the range (-0.5, 0.5) zero centered

    .. warning::
        Slow for more than 200 traces, in this case decimate your
        data or use ``seismic_image``.

    """
    npts, ntraces = section.shape  # time/traces
    if ntraces < 1:
        raise IndexError("Nothing to plot")
    if npts < 1:
        raise IndexError("Nothing to plot")
    t = numpy.linspace(0, dt*npts, npts)
    amp = 1.  # normalization factor
    gmin = 0.  # global minimum
    toffset = 0.  # offset in time to make 0 centered
    if normalize:
        gmax = section.max()
        gmin = section.min()
        amp = (gmax-gmin)
        toffset = 0.5
    pyplot.ylim(max(t), 0)
    if ranges is None:
        ranges = (0, ntraces)
    x0, x1 = ranges
    # horizontal increment
    dx = float((x1-x0)/ntraces)
    pyplot.xlim(x0, x1)
    for i, trace in enumerate(section.transpose()):
        tr = (((trace-gmin)/amp)-toffset)*scale*dx
        x = x0+i*dx  # x positon for this trace
        pyplot.plot(x+tr, t, 'k')
        pyplot.fill_betweenx(t, x+tr, x, tr > 0, color=color)


def seismic_image(section, dt=0.004, ranges=None, cmap=pyplot.cm.gray,
                  aspect=None, vmin=None, vmax=None):
    """
    Plot a seismic section (numpy 2D array matrix) as an image.

    Parameters:

    * section :  2D array
        matrix of traces (first dimension time, second dimension traces)
    * dt : float
        sample rate in seconds (default 4 ms)
    * ranges : (x1, x2)
        min and max horizontal values (default trace number)
    * cmap : colormap
        color map to be used. (see pyplot.cm module)
    * aspect : float
        matplotlib imshow aspect parameter, ratio between axes
    * vmin, vmax : float
        min and max values for imshow

    """
    npts, maxtraces = section.shape  # time/traces
    if maxtraces < 1:
        raise IndexError("Nothing to plot")
    if npts < 1:
        raise IndexError("Nothing to plot")
    t = numpy.linspace(0, dt*npts, npts)
    data = section
    if ranges is None:
        ranges = (0, maxtraces)
    x0, x1 = ranges
    extent = (x0, x1, t[-1:], t[0])
    if aspect is None:  # guarantee a rectangular picture
        aspect = numpy.round((x1-x0)/numpy.max(t))
        aspect -= aspect*0.2
    pyplot.imshow(data, aspect=aspect, cmap=cmap, origin='upper',
                  extent=extent, vmin=vmin, vmax=vmax)
