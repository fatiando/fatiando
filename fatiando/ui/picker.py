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
Pick points of 2D plots.

These functions are meant to make it easy to pick points from matplotlib plots
directly from Python scripts. They all call matplotlib.pyplot.show() to start
an event loop, get the user input, and return the picked values once the plot
window is closed.

**DRAWING GEOMETRIC ELEMENTS AND MODELS**

* :func:`fatiando.ui.picker.draw_polygon`
* :func:`fatiando.ui.picker.draw_layers`

**PICKING POINT COORDINATES**

* :func:`fatiando.ui.picker.points`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Feb-2012'


import numpy
from matplotlib import pyplot, widgets

from fatiando import logger


log = logger.dummy('fatiando.ui.picker')

def draw_polygon(area, axes, style='-', marker='o', color='k', width=2,
    alpha=0.5):
    """
    Draw a polygon by clicking with the mouse.

    INSTRUCTIONS:
    
    * Left click to pick the edges of the polygon;
    * Draw edges CLOCKWISE;
    * Press 'e' to erase the last edge;
    * Right click to close the polygon;
    * Close the figure window to finish;

    Parameters:
    
    * area
        (x1, x2, y1, y2): borders of the area containing the polygon
    * axes
        A matplotlib Axes.
        To get an Axes instace, just do::
        
            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * style
        String with line style (as in matplotlib.pyplot.plot)
    * marker
        String with style of the point markers (as in matplotlib.pyplot.plot)
    * color
        String with line color (as in matplotlib.pyplot.plot)
    * width
        The line width (as in matplotlib.pyplot.plot)
    * alpha
        Transparency of the fill of the polygon. 0 for transparent, 1 for opaque
        (fills the polygon once done drawing)
        
    Returns:
    
    * edges
        List of ``(x, y)`` pairs with the edges of the polygon

    """
    log.info("Drawing polygon...")
    log.info("  INSTRUCTIONS:")
    log.info("  * Left click to pick the edges of the polygon;")
    log.info("  * Draw edges CLOCKWISE;")
    log.info("  * Press 'e' to erase the last edge;")
    log.info("  * Right click to close the polygon;")
    log.info("  * Close the figure window to finish;")
    axes.set_title("Click to draw polygon. Right click when done.")
    axes.set_xlim(area[0], area[1])
    axes.set_ylim(area[2], area[3])
    # start with an empty line
    line, = axes.plot([],[], marker=marker, linestyle=style, color=color,
                      linewidth=width)
    tmpline, = axes.plot([],[], marker=marker, linestyle=style, color=color,
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
            # TODO: Find a way to always plot north on y axis. this would be the
            # other way around if that is the case.
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
        raise ValueError, "Need at least 3 points to make a polygon"
    return numpy.array([x, y]).T

def points(area, axes, marker='o', color='k', size=8):
    """
    Get the coordinates of points by clicking with the mouse.

    INSTRUCTIONS:
    
    * Left click to pick the points;
    * Press 'e' to erase the last point picked;
    * Close the figure window to finish;

    Parameters:
    
    * area
        (x1, x2, y1, y2): borders of the area containing the points
    * axes
        A matplotlib Axes.
        To get an Axes instace, just do::
        
            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * marker
        String with style of the point markers (as in matplotlib.pyplot.plot)
    * color
        String with color of the points (as in matplotlib.pyplot.plot)
    * size
        Marker size (as in matplotlib.pyplot.plot)
        
    Returns:
    
    * points
        List of ``(x, y)`` coordinates of the points

    """
    log.info("Picking points...")
    log.info("  INSTRUCTIONS:")
    log.info("  * Left click to pick the points;")
    log.info("  * Press 'e' to erase the last point picked;")
    log.info("  * Close the figure window to finish;")
    axes.set_title("Click to pick points. Close window when done.")
    axes.set_xlim(area[0], area[1])
    axes.set_ylim(area[2], area[3])
    # start with an empty set
    line, = axes.plot([],[], marker=marker, color=color, markersize=size)
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
            # TODO: Find a way to always plot north on y axis. this would be the
            # other way around if that is the case.
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
    return numpy.array([x, y]).T
    
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
    
    * area
        (v1, v2, z1, z2): lower and upper bounds on physical property value and
        depth, respectively.
    * axes
        A matplotlib Axes.
        To get an Axes instace, just do::
        
            from matplotlib import pyplot
            axes = pyplot.figure().add_subplot(1,1,1)

        You can plot things to ``axes`` before calling this function so that
        they'll appear on the background.
    * style
        String with line style (as in matplotlib.pyplot.plot)
    * marker
        String with style of the point markers (as in matplotlib.pyplot.plot)
    * color
        String with line color (as in matplotlib.pyplot.plot)
    * width
        The line width (as in matplotlib.pyplot.plot)
        
    Returns:
    
    * [thickness, values]

        * thickness
            List with the thickness of each layer, in order of increasing depth
        * values
            List with the physical property value of each layer, in the same
            order        

    """
    log.info("Drawing layers...")
    log.info("  INSTRUCTIONS:")
    log.info("  * Click to make a new layer;")
    log.info("  * Press 'e' to erase the last layer;")
    log.info("  * Close the figure window to finish;")
    axes.set_title("Click to set a layer. Close the window when done.")
    axes.grid()
    vmin, vmax, zmin, zmax = area
    axes.set_xlim(vmin, vmax)
    axes.set_ylim(zmax, zmin)
    # start with an empty line
    line, = axes.plot([], [], marker=marker, linestyle=style,
                       color=color, linewidth=width)
    midv = 0.5*(vmax + vmin)
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
