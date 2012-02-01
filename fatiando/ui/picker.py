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

**DRAWING GEOMETRIC ELEMENTS**

* :func:`fatiando.ui.picker.draw_polygon`

**PICKING POINT COORDINATES**

* :func:`fatiando.ui.picker.points`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Feb-2012'


import numpy
from matplotlib import pyplot

from fatiando import logger
log = logger.dummy()


def draw_polygon(area, axes, style='-', marker='o', color='k', width=1):
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
            line.set_linestyle(style)
            line.set_linewidth(width)
            line.set_marker(marker)
        if event.button == 3 or event.button == 2 and picking[0]:
            if len(x) < 3:
                axes.set_title("Need at least 3 points to make a polygon")
            else:
                picking[0] = False
                axes.set_title("Done! You can close the window now.")
                plotx.append(x[0])
                ploty.append(y[0])
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

def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
