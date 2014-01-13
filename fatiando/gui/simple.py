"""
Simple GUIs using the interactive capabilities of :mod:`matplotlib`

**Interactive gravimetric modeling**

* :class:`~fatiando.gui.simple.Moulder`
* :class:`~fatiando.gui.simple.BasinTrap`
* :class:`~fatiando.gui.simple.BasinTri`

**Interactive modeling of layered media**

* :class:`~fatiando.gui.simple.Lasagne`

----

"""
import bisect

import numpy
from matplotlib import pyplot, widgets

from .. import utils
from ..gravmag import talwani
from ..mesher import Polygon
from ..seismic import profile


class Moulder():
    """
    Interactive potential field direct modeling in 2D using polygons.

    Uses module :mod:`~fatiando.gravmag.talwani` for computations.

    For the moment only works for the gravity anomaly.

    To run this in a script, use::

        # Define the area of modeling
        area = (0, 1000, 0, 1000)
        # Where the gravity effect is calculated
        xp = range(0, 1000, 10)
        zp = [0]*len(xp)
        # Create the application
        app = Moulder(area, xp, zp)
        # Run it (close the window to finish)
        app.run()
        # and save the calculated gravity anomaly profile
        app.savedata("mydata.txt")

    Parameters:

    * area : list = [xmin, xmax, zmin, zmax]
        Are of the subsuface to use for modeling. Remember, z is positive
        downward
    * xp, zp : array
        Arrays with the x and z coordinates of the computation points
    * gz : array
        The observed gravity values at the computation points.
        Will be plotted as black points together with the modeled (predicted)
        data. If None, will ignore this.

    "The truth is out there"

    """

    instructions = ("Click to start drawing - Choose density using the slider" +
                    " - Right click to close polygon - 'e' to delete")
    name = "Moulder - Direct gravimetric modeling"

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
        self.fig = pyplot.figure(figsize=(12,8))
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
        self.densslider = widgets.Slider(sliderax, 'Density',
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
        self.nextdens = 1000.
        self.densslider.set_val(self.nextdens*0.001)
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
        # Where the two surface nodes are. Use depth = 1 because direct modeling
        # doesn't like it when the model and computation points coincide
        nodes = [[100, 1], [900, 1]]
        # Create the application
        app = BasinTrap(area, nodes, xp, zp)
        # Run it (close the window to finish)
        app.run()
        # and save the calculated gravity anomaly profile
        app.savedata("mydata.txt")

    Parameters:

    * area : list = [xmin, xmax, zmin, zmax]
        Are of the subsuface to use for modeling. Remember, z is positive
        downward.
    * nodes : list of lists = [[x1, z1], [x2, z2]]
        x and z coordinates of the two top nodes. Must be in clockwise order!
    * xp, zp : array
        Arrays with the x and z coordinates of the computation points
    * gz : array
        The observed gravity values at the computation points.
        Will be plotted as black points together with the modeled (predicted)
        data. If None, will ignore this.

    """

    instructions = "Click to set node depth - Right click to change nodes"
    name = "BasinTrap"

    def __init__(self, area, nodes, xp, zp, gz=None):
        Moulder.__init__(self, area, xp, zp, gz)
        left, right = numpy.array(nodes)*0.001
        z1 = z2 = 0.001*0.5*(area[3] - area[2])
        self.polygons = [[left, right, [right[0], z1], [left[0], z2]]]
        self.nextdens = -1000
        self.densslider.set_val(self.nextdens*0.001)
        self.densities = [self.nextdens]
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
        self.draw_guide(event.xdata, event.ydata)
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

class BasinTri(Moulder):
    """
    Interactive gravity modeling using a triangular model.

    The triangle has two surface nodes with fixed positions. The user can then
    model by controling the bottom node.

    Example::

        # Define the area of modeling
        area = (0, 1000, 0, 1000)
        # Where the gravity effect is calculated
        xp = range(0, 1000, 10)
        zp = [0]*len(xp)
        # Where the two surface nodes are. Use depth = 1 because direct modeling
        # doesn't like it when the model and computation points coincide
        nodes = [[100, 1], [900, 1]]
        # Create the application
        app = BasinTri(area, nodes, xp, zp)
        # Run it (close the window to finish)
        app.run()
        # and save the calculated gravity anomaly profile
        app.savedata("mydata.txt")

    Parameters:

    * area : list = [xmin, xmax, zmin, zmax]
        Are of the subsuface to use for modeling. Remember, z is positive
        downward.
    * nodes : list of lists = [[x1, z1], [x2, z2]]
        x and z coordinates of the two top nodes. Must be in clockwise order!
    * xp, zp : array
        Arrays with the x and z coordinates of the computation points
    * gz : array
        The observed gravity values at the computation points.
        Will be plotted as black points together with the modeled (predicted)
        data. If None, will ignore this.

    """

    instructions = "Click to set node location"
    name = "BasinTri"

    def __init__(self, area, nodes, xp, zp, gz=None):
        Moulder.__init__(self, area, xp, zp, gz)
        left, right = numpy.array(nodes)*0.001
        z = 0.001*0.5*(area[3] - area[2])
        x = 0.5*(right[0] + left[0])
        self.polygons = [[left, right, [x, z]]]
        self.nextdens = -1000
        self.densslider.set_val(self.nextdens*0.001)
        self.densities = [self.nextdens]
        self.plotx = [v[0] for v in self.polygons[0]]
        self.plotx.append(left[0])
        self.ploty = [v[1] for v in self.polygons[0]]
        self.ploty.append(left[1])
        self.polyline.set_data(self.plotx, self.ploty)
        self.polyline.set_color('k')
        self.guide, = self.mcanvas.plot([], [], marker='o', linestyle='--',
                 color='red', linewidth=2)

    def draw_guide(self, x, z):
        x0, z0 = self.polygons[0][0]
        x1, z1 = self.polygons[0][1]
        self.guide.set_data([x0, x, x1], [z0, z, z1])

    def move(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        self.draw_guide(event.xdata, event.ydata)
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
            self.polygons[0][2] = [x, y]
            self.plotx[2] = x
            self.ploty[2] = y
            self.polyline.set_data(self.plotx, self.ploty)
            self.guide.set_data([], [])
            self.update()
            self.draw()

    def key_press(self, event):
        pass

class Lasagne():
    """
    Interactive modeling of vertical seismic profiling for 1D layered media.

    The wave source is assumed to be on the surface of a vertical borehole. The
    receivers are at given depths. What is measured is the travel-time of
    first arrivals.

    Assumes that the thickness of the layers are known. The user then only needs
    to choose the velocities.

    Example::

        # Define the thickness of the layers
        thickness = [10, 20, 5, 10]
        # Define the measuring points along the well
        zp = range(1, sum(thickness), 1)
        # Define the velocity range
        vmin, vmax = 1, 10000
        # Run the application
        app = Lasagne(thickness, zp, vmin, vmax)
        app.run()
        # Save the modeled data
        app.savedata("mydata.txt")

    Parameters:

    * thickness : list
        The thickness of each layer in order of increasing depth
    * zp : list
        The depths of the measurement stations (seismometers)
    * vmin, vmax : float
        Range of velocities to allow
    * tts : array
        The observed travel-time values at the measurement stations. Will be
        plotted as black points together with the modeled (predicted) data.
        If None, will ignore this.

    """

    instructions = "Click to set the velocity of the layers"
    name = "Lasagne - Vertical seismic profiling for 1D layered media"

    def __init__(self, thickness, zp, vmin, vmax, tts=None):
        if tts is not None:
            if len(tts) != len(zp):
                raise ValueError("zp and tts must have same size")
        if vmin <= 0. or vmax <= 0.:
            raise ValueError("Can't have velocity vmin or vmax <= 0")
        self.tts = tts
        self.zp = zp
        self.thickness = thickness
        # Make the figure
        self.fig = pyplot.figure(figsize=(14,8))
        self.fig.canvas.set_window_title(self.name)
        self.fig.suptitle(self.instructions)
        self.draw = self.fig.canvas.draw
        # Make the data and model canvas
        self.dcanvas = self.fig.add_subplot(1, 2, 1)
        self.dcanvas.set_ylabel("Depth (m)")
        self.dcanvas.set_xlabel("Travel-time (s)")
        self.dcanvas.set_ylim(sum(thickness), 0)
        self.dcanvas.grid()
        self.dcanvas.set_ylim(sum(thickness), 0)
        self.mcanvas = self.fig.add_subplot(1, 2, 2)
        self.mcanvas.set_ylabel("Depth (m)")
        self.mcanvas.set_xlabel("Velocity (m/s2)")
        self.mcanvas.set_xlim(vmin, vmax)
        self.mcanvas.set_ylim(sum(thickness), 0)
        self.mcanvas.grid()
        self.fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.15,
                                 hspace=0.1)
        # Make the sliders
        sliderax = self.fig.add_axes([0.20, 0.03, 0.60, 0.03])
        self.errslider = widgets.Slider(sliderax, 'Error',
            0, 10, valinit=0., valfmt='%2.1f (percent)')
        # Initialize the data
        self.error = 0.
        self.velocity = vmin*numpy.ones_like(thickness)
        self.predtts = profile.layered_straight_ray(thickness, self.velocity,
                                                      zp)
        self.layers = [sum(thickness[:i]) for i in xrange(len(thickness) + 1)]
        self.predplot, = self.dcanvas.plot(self.predtts, zp, '-r', linewidth=2)
        if self.tts is not None:
            self.ttsplot, = self.dcanvas.plot(self.tts, self.zp, 'ok')
        self.ploty = [self.layers[0]]
        for y in self.layers[1:-1]:
            self.ploty.append(y)
            self.ploty.append(y)
        self.ploty.append(self.layers[-1])
        self.plotx = numpy.zeros_like(self.ploty)
        self.layerplot, = self.mcanvas.plot(self.plotx, self.ploty, 'o-k',
            linewidth=2)
        self.guide, = self.mcanvas.plot([], [], marker='o', linestyle='--',
                 color='red', linewidth=2)

    def run(self):
        self.connect()
        pyplot.show()

    def get_data(self):
        return self.predtts

    def savedata(self, fname):
        data = numpy.array([self.zp, self.predtts]).T
        numpy.savetxt(fname, data, fmt='%.5f')

    def connect(self):
        self.errslider.on_changed(self.set_error)
        self.fig.canvas.mpl_connect('button_press_event', self.pick)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.move)

    def set_error(self, value):
        self.error = 0.01*value
        self.update()
        self.draw()

    def update(self):
        self.predtts = utils.contaminate(
            profile.layered_straight_ray(self.thickness, self.velocity,
                self.zp),
            self.error, percent=True)
        self.predplot.set_data(self.predtts, self.zp)
        if self.tts is not None:
            xmin = min(self.predtts.min(), self.tts.min())
            xmax = max(self.predtts.max(), self.tts.max())
        else:
            xmin = self.predtts.min()
            xmax = self.predtts.max()
        if xmin != xmax:
            self.dcanvas.set_xlim(xmin, xmax)

    def draw_guide(self, x, z):
        i = bisect.bisect(self.layers, z)
        if i > 0:
            z1 = self.layers[i - 1]
            z2 = self.layers[i]
            x1 = self.velocity[i - 1]
            self.guide.set_data([x1, x, x, x1], [z1, z1, z2, z2])

    def move(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        self.draw_guide(event.xdata, event.ydata)
        self.draw()

    def pick(self, event):
        if event.inaxes != self.mcanvas:
            return 0
        x, z = event.xdata, event.ydata
        if (event.button == 1):
            i = bisect.bisect(self.layers, z) - 1
            self.velocity[i] = x
            self.plotx[2*i] = x
            self.plotx[2*i + 1] = x
            self.layerplot.set_data(self.plotx, self.ploty)
            self.guide.set_data([], [])
            self.update()
            self.draw()

    def key_press(self, event):
        pass
