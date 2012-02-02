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
Classes for graphical user interfaces (GUIs).

**Interactive matplotlib plots**

* COMING SOON

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Feb-2012'


import numpy
from matplotlib import pyplot, widgets, nxutils

from fatiando.potential import talwani
from fatiando.mesher.dd import Polygon
from fatiando import logger, utils

log = logger.dummy()


class Moulder():
    """
    Interactive potential field direct modeling in 2D using polygons (and module
    :mod:`fatiando.potential.talwani` for computations).

    For the moment only works for the gravity anomaly.

    To run this in a script, use::

        # Define the area of modeling
        area = (0, 1000, 0, 1000)
        # Where the gravity effect is calculated
        xp = range(0, 1000, 10)
        zp = [0]*len(xp)
        # Create the application
        app = Moulder(area, xp, zp)
        # Run it
        app.run()
        # and save the calculated gravity anomaly profile
        app.savedata("mydata.txt")

    Parameters:

    * area
        (xmin, xmax, zmin, zmax): Are of the subsuface to use for modeling.
        Remember, z is positive downward
    * xp, zp
        Array with the x and z coordinates of the computation points
    * gz
        The array with the observed gravity values at the computation points.
        Will be plotted as black points together with the modeled (predicted)
        data. If None, will ignore this.

    "The truth is out there"
    
    """

    instructions = "Left click to close polygon - 'e' to delete"
    name = "Moulder"

    def __init__(self, area, xp, zp, gz=None):
        if len(zp) != len(xp):
            raise ValueError, "xp and zp must have same size"
        # Get the data
        self.area = area
        self.x1, self.x2, z1, z2 = 0.001*numpy.array(area)
        if gz is not None:
            if len(gz) != len(xp):
                raise ValueError, "xp, zp and gz must have same size"
            self.gz = numpy.array(gz)
        else:
            self.gz = gz
        self.xp = numpy.array(xp, dtype='f')
        self.zp = numpy.array(zp, dtype='f')
        # Make the figure
        self.fig = pyplot.figure()
        self.fig.canvas.set_window_title(self.name)
        self.fig.suptitle(self.instructions)
        self.draw = self.fig.canvas.draw
        # Make the data and model canvas
        self.dcanvas = self.fig.add_subplot(2, 1, 1)
        self.dcanvas.set_ylabel("mGal")
        self.dcanvas.set_xlim(self.x1, self.x2)
        self.dcanvas.grid()
        self.mcanvas = self.fig.add_subplot(2, 1, 2)
        self.mcanvas.set_ylabel("Depth (km)")
        self.mcanvas.set_xlabel("x (km)")
        self.mcanvas.set_xlim(self.x1, self.x2)
        self.mcanvas.set_ylim(z2, z1)
        self.mcanvas.grid()
        self.fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.18,
                                 hspace=0.1)
        # Make the sliders
        sliderax = self.fig.add_axes([0.20, 0.08, 0.60, 0.03])
        self.densslider = widgets.Slider(sliderax, 'Density of\nnext polygon',
            -9, 9, valinit=0., valfmt='%1.2f (g/cm3)')
        sliderax = self.fig.add_axes([0.20, 0.03, 0.60, 0.03])
        self.errslider = widgets.Slider(sliderax, 'Error',
            0, 5, valinit=0., valfmt='%1.2f (mGal)')
        # Initialize the data
        self.leg = None
        self.predgz = None
        self.predplot, = self.dcanvas.plot([], [], '-r', linewidth=2)
        if self.gz is not None:
            self.gzplot, = self.dcanvas.plot(xp*0.001, gz, 'ok')
        self.nextdens = 0.
        self.error = 0.
        self.densities = []
        self.polygons = []
        self.nextpoly = []
        self.plotx = []
        self.ploty = []
        self.polyplots = []
        self.polyline, = self.mcanvas.plot([], [], marker='o', linewidth=2)

    def run(self):
        # Connect the event handlers
        self.picking = False
        self.connect()
        self.update()
        pyplot.show()

    def get_data(self):
        return self.predgz

    def savedata(self, fname):
        data = numpy.array([self.xp, self.zp, self.predgz]).T
        numpy.savetxt(fname, data, fmt='%.5f')

    def connect(self):
        self.densslider.on_changed(self.set_density)
        self.errslider.on_changed(self.set_error)
        self.fig.canvas.mpl_connect('button_press_event', self.pick)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.move)

    def update(self):
        if self.polygons:
            polys = []
            for p, d in zip(self.polygons, self.densities):
                polys.append(Polygon(1000.*numpy.array(p), {'density':d}))
            self.predgz = utils.contaminate(talwani.gz(self.xp, self.zp, polys),
                self.error)
        else:
            self.predgz = numpy.zeros_like(self.xp)
        self.predplot.set_data(self.xp*0.001, self.predgz)
        if self.gz is not None:
            ymin = min(self.predgz.min(), self.gz.min())
            ymax = max(self.predgz.max(), self.gz.max())
        else:
            ymin = self.predgz.min()
            ymax = self.predgz.max()
        if ymin != ymax:
            self.dcanvas.set_ylim(ymin, ymax)
        self.draw()

    def set_density(self, value):
        self.nextdens = 1000.*value
        
    def set_error(self, value):
        self.error = value
        self.update()

    def move(self, event):
        pass

    def pick(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        x, y = event.xdata, event.ydata
        #if (event.button == 1 and
            #True not in (nxutils.pnpoly(x, y, p) for p in self.polygons)):
        if (event.button == 1):
            self.picking = True
            self.nextpoly.append([x, y])
            self.plotx.append(x)
            self.ploty.append(y)
            self.polyline.set_data(self.plotx, self.ploty)
            self.draw()
        if event.button == 3 or event.button == 2:
            if len(self.nextpoly) >= 3:
                self.polygons.append(self.nextpoly)
                self.densities.append(float(self.nextdens))
                self.update()
                self.picking = False
                self.plotx.append(self.nextpoly[0][0])
                self.ploty.append(self.nextpoly[0][1])
                self.polyline.set_data(self.plotx, self.ploty)
                fill, = self.mcanvas.fill(self.plotx, self.ploty,
                    color=self.polyline.get_color(), alpha=0.5)
                self.polyline.set_label('%1.2f' % (0.001*self.nextdens))
                self.legend()
                self.draw()
                self.polyplots.append([self.polyline, fill])  
                self.plotx, self.ploty = [], []
                self.nextpoly = []
                self.polyline, = self.mcanvas.plot([], [], marker='o',
                    linewidth=2)

    def legend(self):
        self.leg = self.mcanvas.legend(loc='lower right', numpoints=1,
                                  prop={'size':9})
        self.leg.get_frame().set_alpha(0.5)      
        
    def key_press(self, event):        
        if event.key == 'e':
            if self.picking:
                if len(self.nextpoly) == 0:
                    self.picking = False
                    self.legend()
                    self.draw()
                    return 0
                self.nextpoly.pop()
                self.plotx.pop()
                self.ploty.pop()
                self.polyline.set_data(self.plotx, self.ploty)
            else:
                if len(self.polygons) == 0:
                    return 0
                self.polygons.pop()
                self.densities.pop()
                line, fill = self.polyplots.pop()
                line.remove()
                fill.remove()
                self.update()
            self.draw()

class BasinTrap(Moulder):
    """
    Interactive gravity modeling using a trapezoidal model.

    The trapezoid has two surface nodes with fixed position. The bottom two have
    fixed x coordinates but movable z. The x coordinates for the bottom nodes
    are the same as the ones for the surface nodes. The user can then model by
    controling the depths of the two bottom nodes.

    Example::

        # Define the area of modeling
        area = (0, 1000, 0, 1000)
        # Where the gravity effect is calculated
        xp = range(0, 1000, 10)
        zp = [0]*len(xp)
        # Where the two surface nodes are
        nodes = [[100, 0], [900, 0]]
        # Create the application
        app = ItsATrap(area, nodes, xp, zp)
        # Run it
        app.run()
        # and save the calculated gravity anomaly profile
        app.savedata("mydata.txt")

    Parameters:

    * area
        (xmin, xmax, zmin, zmax): Are of the subsuface to use for modeling.
        Remember, z is positive downward
    * nodes
        [[x1, z1], [x2, z2]]: x and z coordinates of the two top nodes.
        Must be in clockwise order!
    * xp, zp
        Array with the x and z coordinates of the computation points
    * gz
        The array with the observed gravity values at the computation points.
        Will be plotted as black points together with the modeled (predicted)
        data. If None, will ignore this.
        
    """

    instructions = "Click set node depth - Left click to change nodes"
    name = "BasinTrap"

    def __init__(self, area, nodes, xp, zp, gz=None):
        Moulder.__init__(self, area, xp, zp, gz)
        left, right = numpy.array(nodes)*0.001
        z1 = z2 = 0.001*0.5*(area[3] - area[2])
        self.polygons = [[left, right, [right[0], z1], [left[0], z2]]]
        self.densities = [0.]
        self.plotx = [v[0] for v in self.polygons[0]]
        self.plotx.append(left[0])
        self.ploty = [v[1] for v in self.polygons[0]]
        self.ploty.append(left[1])
        self.polyline.set_data(self.plotx, self.ploty)
        self.polyline.set_color('k')
        self.isleft = True
        self.guide, = self.mcanvas.plot([], [], marker='o', linestyle='--',
                 color='red', linewidth=2)

    def draw_guide(self, x, z):
        if self.isleft:
            x0, z0 = self.polygons[0][3]
            x1, z1 = self.polygons[0][2]
        else:
            x0, z0 = self.polygons[0][2]
            x1, z1 = self.polygons[0][3]            
        self.guide.set_data([x0, x0, x1], [z0, z, z1])
                
    def move(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        v, z = event.xdata, event.ydata
        self.draw_guide(v, z)
        self.draw()
        
    def set_density(self, value):
        self.densities[0] = 1000.*value
        self.update()
        self.draw()

    def pick(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        x, y = event.xdata, event.ydata
        if (event.button == 1):
            if self.isleft:
                self.polygons[0][3][1] = y
                self.ploty[3] = y
            else:
                self.polygons[0][2][1] = y
                self.ploty[2] = y
            self.polyline.set_data(self.plotx, self.ploty)
            self.guide.set_data([], [])
            self.update()
            self.draw()
        if event.button == 3 or event.button == 2:
            self.isleft = not self.isleft
            self.draw_guide(x, y)
            self.draw()
        
    def key_press(self, event):
        pass
