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
Create and operate on grids and profiles.

**Grid generation**

* :func:`fatiando.gridder.regular`
* :func:`fatiando.gridder.scatter`

**Grid I/O**

**Grid operations**

* :func:`fatiando.gridder.cut`
* :func:`fatiando.gridder.interpolate`

**Misc**

* :func:`fatiando.gridder.spacing`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'


import numpy
from matplotlib import pyplot

from fatiando import logger
log = logger.dummy()

def draw_polygon(area, axes):
    """
    Draw a polygon by clicking with the mouse.

    Starts with an empty plot.

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
    pyplot.axis('scaled')
    axes.set_xlim(area[0], area[1])
    axes.set_ylim(area[2], area[3])
    # start with an empty line
    line, = axes.plot([0],[0], '-k')
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
            line.set_color('black')
            line.set_linestyle('-')
            line.set_marker('.')
        if event.button == 3 or event.button == 2 and picking[0]:
            if len(x) < 3:
                axes.set_title("Need at least 3 points to make a polygon")
            else:
                picking[0] = False
                axes.set_title("Done")
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

def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
