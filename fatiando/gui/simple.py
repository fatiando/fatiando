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
from __future__ import division
import bisect
import cPickle as pickle

import numpy
from matplotlib import pyplot, widgets, patches
from matplotlib.lines import Line2D
from IPython.core.pylabtools import print_figure
from IPython.display import Image

from .. import utils
from ..gravmag import talwani
from ..mesher import Polygon
from ..seismic import profile


class Moulder(object):

    epsilon = 5
    instructions = 'n: New polygon | d: delete | click: select/move | esc: cancel'

    def __init__(self, area, x, z, data=None, density_range=[-2000, 2000], **kwargs):
        self.area = area
        self.x, self.z = numpy.asarray(x), numpy.asarray(z)
        self.density_range = density_range
        self.data = data
        if data is None:
            self.dmin, self.dmax = 0, 0
        else:
            self.dmin, self.dmax = data.min(), data.max()
        self.predicted = kwargs.get('predicted', numpy.zeros_like(x))
        self.polygons = kwargs.get('polygons', [])
        self.lines = kwargs.get('lines', [])
        self.densities = kwargs.get('densities', [])
        self.error = kwargs.get('error', 0)
        self.cmap = kwargs.get('cmap', pyplot.cm.RdBu_r)
        self.line_args = dict(
            linewidth=2, linestyle='-', color='k', marker='o',
            markerfacecolor='k', markersize=5, animated=False, alpha=0.6)

    def save_predicted(self, fname):
        numpy.savetxt(fname, numpy.transpose([self.x, self.z, self.predicted]))

    def save(self, fname):
        """
        Save the application state into a pickle file.
        """
        with open(fname, 'w') as f:
            state = dict(area=self.area, x=self.x,
                         z=self.z, data=self.data,
                         density_range=self.density_range,
                         cmap=self.cmap,
                         predicted=self.predicted,
                         polygons=self.polygons,
                         lines=self.lines,
                         densities=self.densities,
                         error=self.error)
            pickle.dump(state, f)

    @classmethod
    def load(cls, fname):
        with open(fname) as f:
            state = pickle.load(f)
        app = cls(**state)
        return app

    @property
    def model(self):
        m = [Polygon(p.xy, {'density': d})
             for p, d in zip(self.polygons, self.densities)]
        return m

    def run(self):
        fig = self.figure_setup()
        # Sliders to control the density and the error in the data
        self.density_slider = widgets.Slider(
            fig.add_axes([0.10, 0.01, 0.30, 0.02]), 'Density',
            self.density_range[0], self.density_range[1], valinit=0.,
            valfmt='%6.0f kg/m3')
        self.error_slider = widgets.Slider(
            fig.add_axes([0.60, 0.01, 0.30, 0.02]), 'Error',
            0, 5, valinit=self.error, valfmt='%1.2f mGal')
        # Put instructions on figure title
        self.dataax.set_title(self.instructions)
        # Markers for mouse click events
        self._ivert = None
        self._ipoly = None
        self._lastevent = None
        self._drawing = False
        self._xy = []
        self._drawing_plot = None
        # Used to blit the model plot and make
        # rendering faster
        self.background = None
        # Connect event callbacks
        self.connect()
        self.update_data()
        self.update_data_plot()
        self.canvas.draw()
        pyplot.show()
        for line, poly in zip(self.lines, self.polygons):
            poly.set_animated(False)
            line.set_animated(False)
            line.set_color([0, 0, 0, 0])

    def connect(self):
        # Make the proper callback connections
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_move_callback)
        self.density_slider.on_changed(self.set_density_callback)
        self.error_slider.on_changed(self.set_error_callback)

    def plot(self, figsize=(10, 8), dpi=70):
        """
        Return an IPython compatible figure of the app.

        Use for embedding in notebooks.
        """
        fig = self.figure_setup(figsize=figsize, facecolor='white')
        fig.canvas.draw()
        self.update_data_plot()
        pyplot.close(fig)
        data = print_figure(fig, dpi=dpi)
        return Image(data=data)

    def figure_setup(self, **kwargs):
        fig, axes = pyplot.subplots(2, 1, **kwargs)
        ax1, ax2 = axes
        self.predicted_line, = ax1.plot(self.x, self.predicted, '-r')
        if self.data is not None:
            self.data_line, = ax1.plot(self.x, self.data, '.k')
        ax1.set_ylabel('Gravity anomaly (mGal)')
        ax1.set_xlabel('x (m)', labelpad=-10)
        ax1.set_xlim(self.area[:2])
        ax1.set_ylim((-200, 200))
        ax1.grid()
        tmp = ax2.pcolor(numpy.array([self.density_range]), cmap=self.cmap)
        tmp.set_visible(False)
        pyplot.colorbar(tmp, orientation='horizontal',
                     pad=0.08, aspect=80).set_label(r'Density (kg/cm3)')
        for poly, line in zip(self.polygons, self.lines):
            line.set_color([0, 0, 0, 0])
            poly.set_animated(False)
            line.set_animated(False)
            ax2.add_patch(poly)
            ax2.add_line(line)
        ax2.set_xlim(self.area[:2])
        ax2.set_ylim(self.area[2:])
        ax2.grid()
        ax2.invert_yaxis()
        ax2.set_ylabel('z (m)')
        fig.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.06,
                            hspace=0.1)
        self.canvas = fig.canvas
        self.dataax = axes[0]
        self.modelax = axes[1]
        return fig

    def density2color(self, density):
        dmin, dmax = self.density_range
        return self.cmap((density - dmin)/(dmax - dmin))

    def make_polygon(self, vertices, density):
        poly = patches.Polygon(vertices, animated=False, alpha=0.9,
                               color=self.density2color(density))
        x, y = zip(*poly.xy)
        line = Line2D(x, y, **self.line_args)
        return poly, line

    def update_data(self):
        self.predicted = talwani.gz(self.x, self.z, self.model)
        if self.error > 0:
            self.predicted = utils.contaminate(self.predicted, self.error)

    def update_data_plot(self):
        self.predicted_line.set_ydata(self.predicted)
        vmin = 1.2*min(self.predicted.min(), self.dmin)
        vmax = 1.2*max(self.predicted.max(), self.dmax)
        self.dataax.set_ylim(vmin, vmax)
        self.canvas.draw()

    def set_error_callback(self, value):
        self.error = value
        self.update_data()
        self.update_data_plot()

    def set_density_callback(self, value):
        if self._ipoly is not None:
            self.densities[self._ipoly] = value
            self.polygons[self._ipoly].set_color(self.density2color(value))
            self.update_data()
            self.update_data_plot()
            self.canvas.draw()

    def get_polygon_vertice_id(self, event):
        """
        Find out which vertice of which polygon the event
        happened in.

        If the distance from the event to nearest vertice is
        larger than Moulder.epsilon, returns None.
        """
        distances = []
        indices = []
        for poly in self.polygons:
            x, y = poly.get_transform().transform(poly.xy).T
            d = numpy.sqrt((x - event.x)**2 + (y - event.y)**2)
            distances.append(d.min())
            indices.append(numpy.argmin(d))
        p = numpy.argmin(distances)
        if distances[p] >= self.epsilon:
            # Check if the event was inside a polygon
            x, y = event.x, event.y
            p, v = None, None
            for i, poly in enumerate(self.polygons):
                if poly.contains_point([x, y]):
                    p = i
                    break
        else:
            v = indices[p]
            last = len(self.polygons[p].xy) - 1
            if v == 0 or v == last:
                v = [0, last]
        return p, v

    def button_press_callback(self, event):
        """
        What actions to perform when a mouse button is clicked
        """
        if event.inaxes != self.modelax:
            return
        if event.button == 1 and not self._drawing and self.polygons:
            self._lastevent = event
            for line, poly in zip(self.lines, self.polygons):
                poly.set_animated(False)
                line.set_animated(False)
                line.set_color([0, 0, 0, 0])
            self.canvas.draw()
            # Find out if a click happened on a vertice
            # and which vertice of which polygon
            self._ipoly, self._ivert = self.get_polygon_vertice_id(event)
            if self._ipoly is not None:
                self.density_slider.set_val(self.densities[self._ipoly])
                self.polygons[self._ipoly].set_animated(True)
                self.lines[self._ipoly].set_animated(True)
                self.lines[self._ipoly].set_color([0, 1, 0, 0])
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.modelax.bbox)
                self.modelax.draw_artist(self.polygons[self._ipoly])
                self.modelax.draw_artist(self.lines[self._ipoly])
                self.canvas.blit(self.modelax.bbox)
        elif self._drawing:
            if event.button == 1:
                self._xy.append([event.xdata, event.ydata])
                self._drawing_plot.set_data(zip(*self._xy))
                self.canvas.restore_region(self.background)
                self.modelax.draw_artist(self._drawing_plot)
                self.canvas.blit(self.modelax.bbox)
            elif event.button == 3:
                if len(self._xy) >= 3:
                    density = self.density_slider.val
                    poly, line = self.make_polygon(self._xy, density)
                    self.polygons.append(poly)
                    self.lines.append(line)
                    self.densities.append(density)
                    self.modelax.add_patch(poly)
                    self.modelax.add_line(line)
                    self._drawing_plot.remove()
                    self._drawing_plot = None
                    self._xy = None
                    self._drawing = False
                    self._ipoly = len(self.polygons) - 1
                    self.lines[self._ipoly].set_color([0, 1, 0, 0])
                    self.dataax.set_title(self.instructions)
                    self.canvas.draw()
                    self.update_data()
                    self.update_data_plot()

    def button_release_callback(self, event):
        """
        Reset place markers on mouse button release
        """
        if event.inaxes != self.modelax:
            return
        if event.button != 1:
            return
        if self._ivert is None and self._ipoly is None:
            return
        self.background = None
        for line, poly in zip(self.lines, self.polygons):
            poly.set_animated(False)
            line.set_animated(False)
        self.canvas.draw()
        self._ivert = None
        # self._ipoly is only released when clicking outside
        # the polygons
        self._lastevent = None
        self.update_data()
        self.update_data_plot()

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if event.inaxes is None:
            return
        if event.key == 'd':
            if self._drawing and self._xy:
                self._xy.pop()
                if self._xy:
                    self._drawing_plot.set_data(zip(*self._xy))
                else:
                    self._drawing_plot.set_data([], [])
                self.canvas.restore_region(self.background)
                self.modelax.draw_artist(self._drawing_plot)
                self.canvas.blit(self.modelax.bbox)
            elif self._ivert is not None:
                poly = self.polygons[self._ipoly]
                line = self.lines[self._ipoly]
                if len(poly.xy) > 4:
                    verts = numpy.atleast_1d(self._ivert)
                    poly.xy = numpy.array([xy for i, xy in enumerate(poly.xy)
                                        if i not in verts])
                    line.set_data(zip(*poly.xy))
                    self.update_data()
                    self.update_data_plot()
                    self.canvas.restore_region(self.background)
                    self.modelax.draw_artist(poly)
                    self.modelax.draw_artist(line)
                    self.canvas.blit(self.modelax.bbox)
                    self._ivert = None
            elif self._ipoly is not None:
                self.polygons[self._ipoly].remove()
                self.lines[self._ipoly].remove()
                self.polygons.pop(self._ipoly)
                self.lines.pop(self._ipoly)
                self.densities.pop(self._ipoly)
                self._ipoly = None
                self.canvas.draw()
                self.update_data()
                self.update_data_plot()
        elif event.key == 'n':
            self._ivert = None
            self._ipoly = None
            for line, poly in zip(self.lines, self.polygons):
                poly.set_animated(False)
                line.set_animated(False)
                line.set_color([0, 0, 0, 0])
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.modelax.bbox)
            self._drawing = True
            self._xy = []
            self._drawing_plot = Line2D([], [], **self.line_args)
            self._drawing_plot.set_animated(True)
            self.modelax.add_line(self._drawing_plot)
            self.dataax.set_title('left click: set vertice | right click: finish | esc: cancel')
            self.canvas.draw()
        elif event.key == 'escape':
            self._drawing = False
            self._xy = []
            if self._drawing_plot is not None:
                self._drawing_plot.remove()
                self._drawing_plot = None
            for line, poly in zip(self.lines, self.polygons):
                poly.set_animated(False)
                line.set_animated(False)
                line.set_color([0, 0, 0, 0])
            self.canvas.draw()

    def mouse_move_callback(self, event):
        """
        Handle things when the mouse move.
        """
        if event.inaxes != self.modelax:
            return
        if event.button != 1:
            return
        if self._ivert is None and self._ipoly is None:
            return
        x, y = event.xdata, event.ydata
        p = self._ipoly
        v = self._ivert
        if self._ivert is not None:
            self.polygons[p].xy[v] = x, y
        else:
            dx = x - self._lastevent.xdata
            dy = y - self._lastevent.ydata
            self.polygons[p].xy[:, 0] += dx
            self.polygons[p].xy[:, 1] += dy
        self.lines[p].set_data(zip(*self.polygons[p].xy))
        self._lastevent = event
        self.canvas.restore_region(self.background)
        self.modelax.draw_artist(self.polygons[p])
        self.modelax.draw_artist(self.lines[p])
        self.canvas.blit(self.modelax.bbox)


class BasinTrap(Moulder):
    """
    Interactive gravity modeling using a trapezoidal model.

    The trapezoid has two surface nodes with fixed position. The bottom two
    have fixed x coordinates but movable z. The x coordinates for the bottom
    nodes are the same as the ones for the surface nodes. The user can then
    model by controling the depths of the two bottom nodes.

    Example::

        # Define the area of modeling
        area = (0, 1000, 0, 1000)
        # Where the gravity effect is calculated
        xp = range(0, 1000, 10)
        zp = [0]*len(xp)
        # Where the two surface nodes are. Use depth = 1 because direct
        # modeling doesn't like it when the model and computation points
        # coincide
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
        left, right = numpy.array(nodes) * 0.001
        z1 = z2 = 0.001 * 0.5 * (area[3] - area[2])
        self.polygons = [[left, right, [right[0], z1], [left[0], z2]]]
        self.nextdens = -1000
        self.densslider.set_val(self.nextdens * 0.001)
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
        self.densities[0] = 1000. * value
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
        # Where the two surface nodes are. Use depth = 1 because direct
        # modeling doesn't like it when the model and computation points
        # coincide
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
        left, right = numpy.array(nodes) * 0.001
        z = 0.001 * 0.5 * (area[3] - area[2])
        x = 0.5 * (right[0] + left[0])
        self.polygons = [[left, right, [x, z]]]
        self.nextdens = -1000
        self.densslider.set_val(self.nextdens * 0.001)
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
        self.densities[0] = 1000. * value
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

    Assumes that the thickness of the layers are known. The user then only
    needs to choose the velocities.

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
        self.fig = pyplot.figure(figsize=(14, 8))
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
                                        0, 10, valinit=0.,
                                        valfmt='%2.1f (percent)')
        # Initialize the data
        self.error = 0.
        self.velocity = vmin * numpy.ones_like(thickness)
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
        self.error = 0.01 * value
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
            self.plotx[2 * i] = x
            self.plotx[2 * i + 1] = x
            self.layerplot.set_data(self.plotx, self.ploty)
            self.guide.set_data([], [])
            self.update()
            self.draw()

    def key_press(self, event):
        pass
