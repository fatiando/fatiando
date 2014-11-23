"""
Interactivity functions and classes using matplotlib and IPython widgets

**Gravity forward modeling**

* :class:`~fatiando.gravmag.interactive.Moulder`: a matplitlib GUI for 2D
  forward modeling using polygons


----

"""
from __future__ import division
import cPickle as pickle

import numpy
from matplotlib import pyplot, widgets, patches
from matplotlib.lines import Line2D
from IPython.core.pylabtools import print_figure
from IPython.display import Image

from .. import utils
from . import talwani
from ..mesher import Polygon


class Moulder(object):
    """
    Interactive 2D forward modeling using polygons.

    A matplotlib GUI application. Allows drawing and manipulating polygons and
    computes their predicted data automatically. Also permits contaminating the
    data with gaussian pseudo-random error for producing synthetic data sets.

    Uses :mod:`fatiando.gravmag.talwani` for computations.

    *Moulder* objects can be persisted to Python pickle files using the
    :meth:`~fatiando.gravmag.interactive.Moulder.save` method and later
    restored using :meth:`~fatiando.gravmag.interactive.Moulder.load`.

    .. warning::

        Cannot be used with ``%matplotlib inline`` on IPython notebooks because
        the app uses the matplotlib plot window. You can still embed the
        generated model and data figure on notebooks using the
        :meth:`~fatiando.gravmag.interactive.Moulder.plot` method.

    Parameters:

    * area : list = (x1, x2, z1, z2)
        The limits of the model drawing area, in meters.
    * x, z : 1d-arrays
        The x- and z-coordinates of the computation points (places where
        predicted data will be computed). In meters.
    * data : None or 1d-array
        Observed data measured at *x* and *z*. Will plot this with black dots
        along the predicted data.
    * density_range : list = [min, max]
        The minimum and maximum values allowed for the density. Determines the
        limits of the density slider of the application. In kg.m^-3. Defaults
        to [-2000, 2000].
    * kwargs : dict
        Other keyword arguments used to restore the state of the application.
        Used by the :meth:`~fatiando.gravmag.interactive.Moulder.load` method.
        Not intended for general use.

    Examples:

    Make the Moulder object and start the app::

        import numpy as np
        area = (0, 10e3, 0, 5e3)
        # Calculate on 100 points
        x = np.linspace(area[0], area[1], 100)
        z = np.zeros_like(x)
        app = Moulder(area, x, z)
        app.run()
        # This will pop-up a window with the application (like the screenshot
        # below). Start drawing (follow the instruction in the figure title).
        # When satisfied, close the window to resume execution.

    .. image:: ../_static/Moulder-screenshot.png
        :alt: Screenshot of the Moulder GUI


    After closing the plot window, you can access the model and data from the
    *Moulder* object::

        app.model  # The drawn model as fatiando.mesher.Polygon
        app.predicted  # 1d-array with the data predicted by the model
        # You can save the predicted data to use later
        app.save_predicted('data.txt')
        # You can also save the application and resume it later
        app.save('application.pkl')
        # Close this session/IPython notebook/etc.
        # To resume drawing later:
        app = Moulder.load('application.pkl')
        app.run()

    """

    # The tolerance range for mouse clicks on vertices. In pixels.
    epsilon = 5
    # App instructions printed in the figure suptitle
    instructions = ' | '.join([
        'n: New polygon', 'd: delete', 'click: select/move', 'esc: cancel'])

    def __init__(self, area, x, z, data=None, density_range=[-2000, 2000],
                 **kwargs):
        self.area = area
        self.x, self.z = numpy.asarray(x), numpy.asarray(z)
        self.density_range = density_range
        self.data = data
        # Used to set the ylims for the data axes.
        if data is None:
            self.dmin, self.dmax = 0, 0
        else:
            self.dmin, self.dmax = data.min(), data.max()
        self.predicted = kwargs.get('predicted', numpy.zeros_like(x))
        self.error = kwargs.get('error', 0)
        self.cmap = kwargs.get('cmap', pyplot.cm.RdBu_r)
        self.line_args = dict(
            linewidth=2, linestyle='-', color='k', marker='o',
            markerfacecolor='k', markersize=5, animated=False, alpha=0.6)
        self.polygons = []
        self.lines = []
        self.densities = kwargs.get('densities', [])
        vertices = kwargs.get('vertices', [])
        for xy, dens in zip(vertices, self.densities):
            poly, line = self._make_polygon(xy, dens)
            self.polygons.append(poly)
            self.lines.append(line)

    def save_predicted(self, fname):
        """
        Save the predicted data to a text file.

        Data will be saved in 3 columns separated by spaces: x  z  data

        Parameters:

        * fname : string or file-like object
            The name of the output file or an open file-like object.

        """
        numpy.savetxt(fname, numpy.transpose([self.x, self.z, self.predicted]))

    def save(self, fname):
        """
        Save the application state into a pickle file.

        Use this to persist the application. You can later reload the entire
        object, with the drawn model and data, using the
        :meth:`~fatiando.gravmag.interactive.Moulder.load` method.

        Parameters:

        * fname : string
            The name of the file to save the application. The extension doesn't
            matter (use ``.pkl`` if in doubt).

        """
        with open(fname, 'w') as f:
            vertices = [numpy.asarray(p.xy) for p in self.polygons]
            state = dict(area=self.area, x=self.x,
                         z=self.z, data=self.data,
                         density_range=self.density_range,
                         cmap=self.cmap,
                         predicted=self.predicted,
                         vertices=vertices,
                         densities=self.densities,
                         error=self.error)
            pickle.dump(state, f)

    @classmethod
    def load(cls, fname):
        """
        Restore an application from a pickle file.

        The pickle file should have been generated by  the
        :meth:`~fatiando.gravmag.interactive.Moulder.save` method.

        Parameters:

        * fname : string
            The name of the file.

        Returns:

        * app : Moulder object
            The restored application. You can continue using it as if nothing
            had happened.

        """
        with open(fname) as f:
            state = pickle.load(f)
        app = cls(**state)
        return app

    @property
    def model(self):
        """
        The polygon model drawn as :class:`fatiando.mesher.Polygon` objects.
        """
        m = [Polygon(p.xy, {'density': d})
             for p, d in zip(self.polygons, self.densities)]
        return m

    def run(self):
        """
        Start the application for drawing.

        Will pop-up a window with a place for drawing the model (below) and a
        place with the predicted (and, optionally, observed) data (top).

        Follow the instruction on the figure title.

        When done, close the window to resume program execution.
        """
        fig = self._figure_setup()
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
        self._connect()
        self._update_data()
        self._update_data_plot()
        self.canvas.draw()
        pyplot.show()

    def _connect(self):
        """
        Connect the matplotlib events to their callback methods.
        """
        # Make the proper callback connections
        self.canvas.mpl_connect('button_press_event',
                                self._button_press_callback)
        self.canvas.mpl_connect('key_press_event',
                                self._key_press_callback)
        self.canvas.mpl_connect('button_release_event',
                                self._button_release_callback)
        self.canvas.mpl_connect('motion_notify_event',
                                self._mouse_move_callback)
        self.density_slider.on_changed(self._set_density_callback)
        self.error_slider.on_changed(self._set_error_callback)

    def plot(self, figsize=(10, 8), dpi=70):
        """
        Make a plot of the data and model for embedding in IPython notebooks

        Doesn't require ``%matplotlib inline`` to embed the plot (as that would
        not allow the app to run).

        Parameters:

        * figsize : list = (width, height)
            The figure size in inches.
        * dpi : float
            The number of dots-per-inch for the figure resolution.

        """
        fig = self._figure_setup(figsize=figsize, facecolor='white')
        self._update_data_plot()
        pyplot.close(fig)
        data = print_figure(fig, dpi=dpi)
        return Image(data=data)

    def _figure_setup(self, **kwargs):
        """
        Setup the plot figure with labels, titles, ticks, etc.

        Sets the *canvas*, *dataax*, *modelax*, *polygons* and *lines*
        attributes.

        Parameters:

        * kwargs : dict
            Keyword arguments passed to ``pyplot.subplots``.

        Returns:

        * fig : matplotlib figure object
            The created figure

        """
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
        # Remake the polygons and lines to make sure they belong to the right
        # axis coordinates
        vertices = [p.xy for p in self.polygons]
        newpolygons, newlines = [], []
        for xy, dens in zip(vertices, self.densities):
            poly, line = self._make_polygon(xy, dens)
            newpolygons.append(poly)
            newlines.append(line)
            ax2.add_patch(poly)
            ax2.add_line(line)
        self.polygons = newpolygons
        self.lines = newlines
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
        fig.canvas.draw()
        return fig

    def _density2color(self, density):
        """
        Map density values to colors using the given *cmap* attribute.

        Parameters:

        * density : 1d-array
            The density values of the model polygons

        Returns

        * colors : 1d-array
            The colors mapped to each density value (returned by a matplotlib
            colormap object.

        """
        dmin, dmax = self.density_range
        return self.cmap((density - dmin)/(dmax - dmin))

    def _make_polygon(self, vertices, density):
        """
        Create a polygon for drawing.

        Polygons are matplitlib.patches.Polygon objects for the fill and
        matplotlib.lines.Line2D for the contour.

        Parameters:

        * vertices : list of [x, z]
            List of the [x, z]  coordinate pairs of each vertex of the polygon
        * density : float
            The density of the polygon (used to set the color)

        Returns:

        * polygon, line
            The matplotlib Polygon and Line2D objects

        """
        poly = patches.Polygon(vertices, animated=False, alpha=0.9,
                               color=self._density2color(density))
        x, y = zip(*poly.xy)
        line = Line2D(x, y, **self.line_args)
        return poly, line

    def _update_data(self):
        """
        Recalculate the predicted data (optionally with random error)
        """
        self.predicted = talwani.gz(self.x, self.z, self.model)
        if self.error > 0:
            self.predicted = utils.contaminate(self.predicted, self.error)

    def _update_data_plot(self):
        """
        Update the predicted data plot in the *dataax*.

        Adjusts the xlim of the axes to fit the data.
        """
        self.predicted_line.set_ydata(self.predicted)
        vmin = 1.2*min(self.predicted.min(), self.dmin)
        vmax = 1.2*max(self.predicted.max(), self.dmax)
        self.dataax.set_ylim(vmin, vmax)
        self.canvas.draw()

    def _set_error_callback(self, value):
        """
        Callback when error slider is edited
        """
        self.error = value
        self._update_data()
        self._update_data_plot()

    def _set_density_callback(self, value):
        """
        Callback when density slider is edited
        """
        if self._ipoly is not None:
            self.densities[self._ipoly] = value
            self.polygons[self._ipoly].set_color(self._density2color(value))
            self._update_data()
            self._update_data_plot()
            self.canvas.draw()

    def _get_polygon_vertice_id(self, event):
        """
        Find out which vertex of which polygon the event happened in.

        If the click was inside a polygon (not on a vertex), identify that
        polygon.

        Returns:

        * p, v : int, int
            p: the index of the polygon the event happened in or None if
            outside all polygons.
            v: the index of the polygon vertex that was clicked or None if the
            click was not on a vertex.

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

    def _button_press_callback(self, event):
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
            self._ipoly, self._ivert = self._get_polygon_vertice_id(event)
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
                    poly, line = self._make_polygon(self._xy, density)
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
                    self._update_data()
                    self._update_data_plot()

    def _button_release_callback(self, event):
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
        self._update_data()
        self._update_data_plot()

    def _key_press_callback(self, event):
        """
        What to do when a key is pressed on the keyboard.
        """
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
                    self._update_data()
                    self._update_data_plot()
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
                self._update_data()
                self._update_data_plot()
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
            self.dataax.set_title(' | '.join([
                'left click: set vertice', 'right click: finish',
                'esc: cancel']))
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

    def _mouse_move_callback(self, event):
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
