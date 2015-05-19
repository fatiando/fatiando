"""
Create and operate on grids and profiles.

All the main functionality of this module (and more) is implemented in the
:class:`~fatiando.gridder.Grid` class.
Use it to generate grids of points, load from file, plot data, interpolate,
and work with map projections.

You can still use the functions below to achieve some of this functionality.

**Grid generation**

* :func:`~fatiando.gridder.regular`
* :func:`~fatiando.gridder.scatter`

**Grid operations**

* :func:`~fatiando.gridder.cut`
* :func:`~fatiando.gridder.profile`

**Interpolation**

* :func:`~fatiando.gridder.interp`
* :func:`~fatiando.gridder.interp_at`
* :func:`~fatiando.gridder.extrapolate_nans`

**Input/Output**

* :func:`~fatiando.gridder.load_surfer`: Read a Surfer grid file and return
  three 1d numpy arrays and the grid shape

**Misc**

* :func:`~fatiando.gridder.spacing`

----

"""
from __future__ import division
import six
import copy as pycopy
import numpy
import scipy.interpolate
import matplotlib.mlab
from matplotlib import pyplot


class Grid(object):
    """
    Grid data generation, manipulation, I/O and plotting.
    """

    _coord_names = ['x', 'y', 'lon', 'long', 'longitude', 'lat', 'latitude']
    _special_attributes = ['shape', 'projection', 'latlon', 'metadata']

    def __init__(self, **kwargs):
        kwargs = dict((k.lower(), v) for k, v in kwargs.iteritems())
        added = []
        for k, v in kwargs.iteritems():
            if k not in self._special_attributes:
                v = numpy.atleast_1d(v).ravel()
            if k in ['lon', 'long', 'longitude']:
                k = 'lon'
                if 'y' not in kwargs:
                    setattr(self, 'y', v)
                    added.append('y')
            elif k in ['lat', 'latitude']:
                k = 'lat'
                if 'x' not in kwargs:
                    setattr(self, 'x', v)
                    added.append('x')
            added.append(k)
            setattr(self, k, v)
        if 'shape' not in added:
            self.shape = None
        if 'projection' not in added:
            self.projection = None
        if 'x' not in kwargs or 'y' not in kwargs:
            self.latlon = True
        else:
            self.latlon = False
        not_attributes = set(self._coord_names + self._special_attributes)
        self._attributes = list(set(added).difference(not_attributes))
        # Check all attributes have the same size
        sizes = [getattr(self, k).size for k in self._attributes + ['x', 'y']]
        assert numpy.all(numpy.asarray(sizes) == sizes[0]), \
            "All attributes and coordinates must have same size"
        self.size = sizes[0]

    def add_attribute(self, name, value):
        """
        Add a column to the grid.
        """
        value = numpy.atleast_1d(value).ravel()
        assert value.size == self.size, "Attribute must have same size as grid"
        setattr(self, name.lower(), value)
        self._attributes.append(name)
        return self

    def __setitem__(self, name, value):
        assert isinstance(name, six.string_types), \
            "Invalid attribute '{}'. Must be a string.".format(name)
        self.add_attribute(name, value)

    def __getitem__(self, item):
        # Indexing with a string leads to getting an attribute...
        if isinstance(item, six.string_types):
            assert item in self.attributes, \
                "Grid doesn't have an attribute '{}'".format(item)
            return getattr(self, item)
        # ... while indexing with an integer or slice leads to slicing the
        # attributes and returning a subgrid
        else:
            args = dict((k, getattr(self, k)[item])
                        for k in self._attributes + ['x', 'y'])
            if self.latlon:
                args['latlon'] = True
            if self.projection is not None:
                args['projection'] = self.projection
            args['shape'] = None
            return Grid(**args)

    def cut(self, area):
        """
        Return a smaller grid with the points that fall inside an area.

        If the grid is regular (i.e., has a ``shape`` attribute), the output
        will also be a regular grid.

        Parameters:

        * area : list = [x1, x2, y1, y2]
            The area that contains the desired points

        Returns:

        * subgrid : :class:`~fatiando.gridder.Grid`
            A ``Grid`` object with points that fall inside *area* (with all the
            attributes of the original grid)

        """
        x1, x2, y1, y2 = area
        if self.shape is not None:
            shape = self.shape
            x = numpy.linspace(self.area[0], self.area[1], shape[1])
            y = numpy.linspace(self.area[2], self.area[3], shape[0])
            jmin = numpy.abs(x - x1).argmin()
            jmax = numpy.abs(x - x2).argmin() + 1
            imin = numpy.abs(y - y1).argmin()
            imax = numpy.abs(y - y2).argmin() + 1
            args = dict(
                (k, getattr(self, k).reshape(shape)[imin:imax, jmin:jmax])
                for k in self._attributes + ['x', 'y'])
            args['shape'] = (imax - imin, jmax - jmin)
            if self.latlon:
                args['latlon'] = True
            if self.projection is not None:
                args['projection'] = self.projection
            return Grid(**args)
        else:
            inside = ((self.x >= x1) & (self.x <= x2) &
                      (self.y >= y1) & (self.y <= y2))
            return self[inside]

    @staticmethod
    def regular(area, shape, z=None, latlon=False, projection=None):
        """
        Generate a regular grid
        """
        x1, x2, y1, y2 = area
        ny, nx = shape
        x, y = numpy.meshgrid(numpy.linspace(x1, x2, nx),
                              numpy.linspace(y1, y2, ny))
        args = dict(shape=shape, projection=projection)
        if latlon:
            args['lon'] = y
            args['lat'] = x
        else:
            args['y'] = y
            args['x'] = x
        if z is not None:
            args['z'] = z*numpy.ones(shape[0]*shape[1])
        return Grid(**args)

    @staticmethod
    def scatter(area, n, z=None, latlon=False, projection=None, seed=None):
        """
        Generate a random scatter of points.
        """
        coords = scatter(area, n, z, seed)
        x, y = coords[0], coords[1]
        args = dict(shape=None, projection=projection)
        if len(coords) == 3:
            args['z'] = coords[2]
        if latlon:
            args['lon'] = y
            args['lat'] = x
        else:
            args['y'] = y
            args['x'] = x
        return Grid(**args)

    @staticmethod
    def load(fname, **kwargs):
        """
        Load data from a file.

        Will try to guess the file type by the extension.
        If it's an open file, will pass it to
        :meth:`~fatiando.gridder.Grid.load_numpy`.
        """
        grid = None
        if isinstance(fname, str):
            # Try to guess the file type from the extension
            ext = fname.strip().split('.')[-1]
            if ext == 'gdf':
                grid = Grid.load_gdf(fname, **kwargs)
            elif ext == 'csv':
                grid = Grid.load_csv(fname, **kwargs)
            elif ext == 'grd':
                grid = Grid.load_surfer(fname, **kwargs)
        if grid is None:
            if 'binary' not in kwargs:
                try:
                    grid = Grid.load_numpy(fname, binary=False, **kwargs)
                except:
                    grid = Grid.load_numpy(fname, binary=True, **kwargs)
            else:
                grid = Grid.load_numpy(fname, **kwargs)
        return grid

    @staticmethod
    def load_csv(fname, **kwargs):
        """
        Load data from a CSV file.
        """
        column_names = kwargs.get('column_names', None)
        usecols = kwargs.get('usecols', None)
        if 'delimiter' not in kwargs:
            kwargs['delimiter'] = ';'
        with open(fname) as f:
            # Check if first line contains the column names
            first = f.readline().strip().split(';')
            columns = [s.strip() for s in first if s.strip()]
            numbers = False
            for item in columns:
                try:
                    float(item)
                    numbers = True
                except ValueError:
                    pass
            if numbers:
                # Rewind the file to the origin to read later
                f.seek(0)
            else:
                if column_names is None:
                    if usecols is None:
                        column_names = columns
                    else:
                        column_names = [columns[c] for c in usecols]
                kwargs['column_names'] = column_names
            grid = Grid.load_numpy(f, **kwargs)
        return grid

    @staticmethod
    def load_numpy(fname, binary=False, column_names=None, **kwargs):
        """
        Load data from a numpy saved text or binary file.
        """
        if binary:
            data = numpy.load(fname, **kwargs).T
        else:
            kwargs['unpack'] = True
            data = numpy.loadtxt(fname, **kwargs)
        if column_names is None:
            nattr = len(data) - 2
            attributes = ['data{:d}'.format(i + 1) for i in range(nattr)]
            column_names = ['x', 'y'] + attributes
        assert len(column_names) == len(data), "Insuffient column names."
        args = dict([k, v] for k, v in zip(column_names, data))
        return Grid(**args)

    @staticmethod
    def load_surfer(fname, attribute_name='data', latlon=False, tradexy=False):
        """
        Load data from an ASCII Surfer grid file.
        """
        x, y, value, shape = load_surfer(fname, fmt='ascii')
        args = dict(shape=shape, latlon=latlon)
        if tradexy:
            args['x'] = y
            args['y'] = x
        else:
            args['x'] = x
            args['y'] = y
        args[attribute_name] = value
        return Grid(**args)

    @staticmethod
    def load_gdf(fname, column_names=None, usecols=None, **kwargs):
        """
        Load data from an ICGEM .gdf file.
        """
        with open(fname) as f:
            # Read the header and extract metadata
            metadata = []
            shape = [None, None]
            size = None
            height = None
            attributes = None
            attr_line = False
            for line in f:
                if line.strip()[:11] == 'end_of_head':
                    break
                metadata.append(line)
                if not line.strip():
                    attr_line = True
                    continue
                if not attr_line:
                    parts = line.strip().split()
                    if parts[0] == 'height_over_ell':
                        height = float(parts[1])
                    elif parts[0] == 'latitude_parallels':
                        shape[1] = int(parts[1])
                    elif parts[0] == 'longitude_parallels':
                        shape[0] = int(parts[1])
                    elif parts[0] == 'number_of_gridpoints':
                        size = int(parts[1])
                else:
                    attributes = line.strip().split()
                    attr_line = False
            # Read the numerical values
            kwargs['unpack'] = True
            kwargs['usecols'] = usecols
            kwargs['ndmin'] = 2
            data = numpy.loadtxt(f, **kwargs)
        # Sanity checks
        if any(n is None for n in shape):
            shape = None
        else:
            shape = tuple(shape)
            if size is not None and shape[0]*shape[1] != size:
                raise ValueError(
                    "Grid shape {} and size read ({})".format(shape, size)
                    + " are incompatible.")
        if column_names is not None:
            attributes = column_names
        elif attributes is None:
            raise ValueError("Couldn't read column names.")
        elif usecols is not None:
            attributes = [attributes[i] for i in usecols]
        if len(attributes) != data.shape[0]:
            raise ValueError(
                "Number of columns names ({})".format(len(attributes))
                + " doesn't match columns read ({}).".format(data.shape[0]))
        # Make the argument list for the grid
        args = dict(shape=shape, metadata=''.join(metadata))
        if (height is not None) and ('height' not in attributes):
            if shape is None:
                if size is not None:
                    n = size
                else:
                    n = data[0].size
            else:
                n = shape[0]*shape[1]
            args['height'] = height*numpy.ones(n)
        for attr, value in zip(attributes, data):
            # Do all this because the ICGEM grids vary lon first, then lat
            # But for all else here we need lat first (x), then lon (y)
            # reshapes further on will rely on this, so fix it when reading
            args[attr] = value.reshape((shape[1], shape[0])).T.ravel()
        return Grid(**args)

    def dump_csv(self, fname, **kwargs):
        """
        Save the data to a CSV file.
        """
        if isinstance(fname, str):
            f = open(fname, 'w')
        else:
            f = fname
        if self.latlon:
            column_names = ['latitude', 'longitude']
        else:
            column_names = ['x', 'y']
        column_names.extend(self.attributes)
        f.write('; '.join(column_names))
        f.write('\n')
        data = (getattr(self, i) for i in ['x', 'y'] + self.attributes)
        if 'delimiter' not in kwargs:
            kwargs['delimiter'] = ';'
        if 'fmt' not in kwargs:
            kwargs['fmt'] = '%.5f'
        numpy.savetxt(f, numpy.hstack(c.reshape((self.size, 1)) for c in data),
                      **kwargs)
        if isinstance(fname, str):
            f.close()

    @property
    def area(self):
        """
        Return the [xmin, xmax, ymin, ymax] limits of the data
        """
        return (self.x.min(), self.x.max(), self.y.min(), self.y.max())

    @property
    def spacing(self):
        """
        If the grid is regular (has a ``shape`` attribute), this will be the
        grid spacing.
        """
        if self.shape is None:
            raise ValueError("Can't calculate spacing for irregular grids."
                             + " Grid must have a shape attribute.")
        x1, x2, y1, y2 = self.area
        ny, nx = self.shape
        dx = (x2 - x1)/(nx - 1)
        dy = (y2 - y1)/(ny - 1)
        return dx, dy

    @property
    def attributes(self):
        """
        Return a list of the attributes (column names) of the grid
        """
        return list(self._attributes)

    def interp_at(self, x, y, attribute=None, algorithm='cubic',
                  extrapolate=False):
        """
        Interpolate the grid data on the x, y points given.
        """
        if attribute is None:
            attribute = self._attributes
        elif isinstance(attribute, str):
            attribute = [attribute]
        args = dict(x=x, y=y, projection=self.projection)
        for k in attribute:
            args[k] = interp_at(self.x, self.y, getattr(self, k), x, y,
                                algorithm, extrapolate)
        return self.__class__(**args)

    def interp(self, shape, attribute=None, area=None, algorithm='cubic',
               extrapolate=False):
        """
        Interpolate the grid data on a regular grid.
        """
        if area is None:
            area = self.area
        grid = Grid.regular(area, shape, z=None, projection=self.projection,
                            latlon=self.latlon)
        if attribute is None:
            attribute = self._attributes
        elif isinstance(attribute, str):
            attribute = [attribute]
        for k in attribute:
            values = interp_at(self.x, self.y, getattr(self, k),
                               grid.x, grid.y, algorithm, extrapolate)
            grid.add_attribute(k,  values)
        return grid

    def interp_nans(self, dummy=None, algorithm='cubic', copy=False):
        """
        Fill the nan values in the attributes by interpolation.
        """
        if copy:
            grid = pycopy.deepcopy(self)
        else:
            grid = pycopy.copy(self)
        for attribute in grid._attributes:
            v = getattr(grid, attribute)
            if dummy is not None:
                where = v == dummy
            else:
                where = ~numpy.isfinite(v)
            nans = grid[where]
            if nans.size > 0:
                notnans = grid[~where]
                fill = interp_at(notnans.x, notnans.y,
                                 getattr(notnans, attribute), nans.x, nans.y,
                                 algorithm, extrapolate=True)
                getattr(grid, attribute)[where] = fill
        return grid

    def profile(self, point1, point2, npoints, attribute=None,
                extrapolate=False, projection=None):
        """
        Extract a profile between point 1 and point 2.
        """
        if attribute is None:
            attribute = self._attributes
        elif isinstance(attribute, str):
            attribute = [attribute]
        if projection is not None:
            x, y = numpy.transpose([point1, point2])
            y, x = self.projection.transform_points(projection, y, x).T[:2]
            point1, point2 = numpy.transpose([x, y])
        args = dict(shape=None, projection=self.projection)
        for k in attribute:
            x, y, dist, args[k] = profile(self.x, self.y, getattr(self, k),
                                          point1, point2, npoints,
                                          extrapolate=extrapolate)
        return self.__class__(x=x, y=y, distance=dist, **args)

    def set_shape(self, shape):
        """
        Replace the shape attribute.
        """
        self.shape = shape
        return self

    def set_projection(self, proj):
        """
        Specify the Cartopy projection object that corresponds to the
        projection of the data.
        """
        self.projection = proj
        return self

    def reproject(self, target_proj):
        """
        Transform the coordinates from the current projection to the given
        projection.
        """
        if self.projection is None:
            raise AttributeError(
                'Call .set_projection to specify the grid projection first.')
        coords = target_proj.transform_points(self.projection, self.y, self.x)
        y, x = coords.T[:2]
        args = dict((k, getattr(self, k)) for k in self._attributes)
        grid = self.__class__(x=x, y=y, projection=target_proj,
                              shape=self.shape, **args)
        return grid

    def plot(self, style='.k', ax=None, **kwargs):
        """
        Plot the data points on a map.
        """
        if ax is None:
            if self.projection is None:
                ax = pyplot.gca()
            else:
                ax = pyplot.axes(projection=self.projection)
        if self.projection is not None:
            kwargs['transform'] = self.projection
        ax.plot(self.y, self.x, style, **kwargs)
        return ax

    def pcolor(self, attribute, autoranges=True, colorbar='vertical', ax=None,
               basemap=None, **kwargs):
        """
        Make a pseudo-color map of the given attribute.
        """
        assert self.shape is not None, \
            "'shape' not set. Can only plot regular grids. " + \
            "Try method 'interp' to produce a regular grid"
        if isinstance(attribute, str):
            attribute = getattr(self, attribute)
        values = numpy.ma.masked_where(
            numpy.isnan(attribute), attribute).reshape(self.shape)
        _autocmap(values, autoranges, kwargs)
        if ax is None:
            if self.projection is None:
                ax = pyplot.gca()
            else:
                ax = pyplot.axes(projection=self.projection)
        X, Y = self.y.reshape(self.shape), self.x.reshape(self.shape)
        if basemap is not None:
            X, Y = basemap(X, Y)
            plot = basemap.pcolormesh(X, Y, values, **kwargs)
        else:
            if 'transform' not in kwargs and self.projection is not None:
                kwargs['transform'] = self.projection
            plot = ax.pcolormesh(X, Y, values, **kwargs)
        if colorbar is not None:
            _add_colorbar(plot, colorbar, ax)
        x1, x2, y1, y2 = self.area
        ax.set_xlim(y1, y2)
        ax.set_ylim(x1, x2)
        return ax

    def contourf(self, attribute, levels=10, autoranges=True,
                 colorbar='vertical', ax=None, basemap=None, **kwargs):
        """
        Make a filled contour map of the given attribute.
        """
        assert self.shape is not None, \
            "'shape' not set. Can only plot regular grids. " + \
            "Try method 'interp' to produce a regular grid"
        if isinstance(attribute, str):
            attribute = getattr(self, attribute)
        values = numpy.ma.masked_where(
            numpy.isnan(attribute), attribute).reshape(self.shape)
        _autocmap(values, autoranges, kwargs)
        if ax is None:
            if self.projection is None:
                ax = pyplot.gca()
            else:
                ax = pyplot.axes(projection=self.projection)
        X, Y = self.y.reshape(self.shape), self.x.reshape(self.shape)
        if basemap is not None:
            X, Y = basemap(X, Y)
            plot = basemap.contourf(X, Y, values, levels, **kwargs)
        else:
            if 'transform' not in kwargs and self.projection is not None:
                kwargs['transform'] = self.projection
            plot = ax.contourf(X, Y, values, levels, **kwargs)
        if colorbar is not None:
            _add_colorbar(plot, colorbar, ax)
        x1, x2, y1, y2 = self.area
        ax.set_xlim(y1, y2)
        ax.set_ylim(x1, x2)
        return ax

    def contour(self, attribute, levels=10, autoranges=True, ax=None,
                basemap=None, label=None, clabel=True, style='solid',
                linewidth=1.0, **kwargs):
        """
        Make a contour plot of the given attribute
        """
        assert self.shape is not None, \
            "'shape' not set. Can only plot regular grids. " + \
            "Try method 'interp' to produce a regular grid"
        if isinstance(attribute, str):
            attribute = getattr(self, attribute)
        values = numpy.ma.masked_where(
            numpy.isnan(attribute), attribute).reshape(self.shape)
        if 'colors' not in kwargs:
            _autocmap(values, autoranges, kwargs)
        if ax is None:
            if self.projection is None:
                ax = pyplot.gca()
            else:
                ax = pyplot.axes(projection=self.projection)
        X, Y = self.y.reshape(self.shape), self.x.reshape(self.shape)
        if basemap is not None:
            X, Y = basemap(X, Y)
            ct = basemap.contour(X, Y, values, levels, **kwargs)
        else:
            if 'transform' not in kwargs and self.projection is not None:
                kwargs['transform'] = self.projection
            ct = ax.contour(X, Y, values, levels, **kwargs)
        if clabel:
            ct.clabel(fmt='%g')
        if label is not None:
            ct.collections[0].set_label(label)
        if style != 'mixed':
            for c in ct.collections:
                c.set_linestyle(style)
        for c in ct.collections:
            c.set_linewidth(linewidth)
        x1, x2, y1, y2 = self.area
        ax.set_xlim(y1, y2)
        ax.set_ylim(x1, x2)
        return ax


def _autocmap(values, autoranges, args):
    """
    Automatically choose a color map for the given data.
    Also adjust the value range for diverging colormaps so that 0 is the middle
    color.
    """
    vmax, vmin = numpy.nanmax(values), numpy.nanmin(values)
    if vmax <= 0 or vmin >= 0:
        diverging = False
        if vmax <= 0:
            sign = 'negative'
        else:
            sign = 'positive'
    else:
        diverging = True
    has_vmin_vmax = any(k in args for k in ['vmin', 'vmax'])
    if autoranges and diverging and not has_vmin_vmax:
        absmax = numpy.abs([vmax, vmin]).max()
        args['vmin'] = -absmax
        args['vmax'] = absmax
    if 'cmap' not in args:
        if diverging:
            args['cmap'] = pyplot.cm.RdYlBu_r
        else:
            if sign == 'positive':
                args['cmap'] = pyplot.cm.Reds
            elif sign == 'negative':
                args['cmap'] = pyplot.cm.Blues_r


def _add_colorbar(mappable, orientation, ax):
    """
    Add a color bar to the given mappable object (returned by a matplotlib
    plot command).
    """
    cbargs = dict(orientation=orientation)
    if orientation == 'horizontal':
        cbargs['pad'] = 0.02
        cbargs['aspect'] = 50
        ax.xaxis.tick_top()
    else:
        cbargs['pad'] = 0
        cbargs['aspect'] = 20
    cb = ax.get_figure().colorbar(mappable, ax=ax, **cbargs)
    return cb


def load_surfer(fname, fmt='ascii'):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape

    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    According to Surfer structure, x and y are horizontal and vertical
    screen-based coordinates respectively. If the grid is in geographic
    coordinates, x will be longitude and y latitude. If the coordinates
    are cartesian, x will be the easting and y the norting coordinates.

    WARNING: This is opposite to the convention used for Fatiando.
    See io_surfer.py in cookbook.

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'

    Returns:

    * x : 1d-array
        Value of the horizontal coordinate of each grid point.
    * y : 1d-array
        Value of the vertical coordinate of each grid point.
    * grd : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : tuple = (ny, nx)
        The number of points in the vertical and horizontal grid dimensions,
        respectively

    """
    assert fmt in ['ascii', 'binary'], "Invalid grid format '%s'. Should be \
        'ascii' or 'binary'." % (fmt)
    if fmt == 'ascii':
        # Surfer ASCII grid structure
        # DSAA            Surfer ASCII GRD ID
        # nCols nRows     number of columns and rows
        # xMin xMax       X min max
        # yMin yMax       Y min max
        # zMin zMax       Z min max
        # z11 z21 z31 ... List of Z values
        with open(fname) as ftext:
            # DSAA is a Surfer ASCII GRD ID
            id = ftext.readline()
            # Read the number of columns (nx) and rows (ny)
            nx, ny = [int(s) for s in ftext.readline().split()]
            # Read the min/max value of x (columns/longitue)
            xmin, xmax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of  y(rows/latitude)
            ymin, ymax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of grd
            zmin, zmax = [float(s) for s in ftext.readline().split()]
            grd = numpy.fromiter((float(i) for line in ftext for i in
                                 line.split()), dtype=numpy.float)
            grd[grd >= 1.70141e+38] = numpy.nan
        # Create x and y numpy arrays
        x = numpy.linspace(xmin, xmax, nx)
        y = numpy.linspace(ymin, ymax, ny)
        x, y = [tmp.ravel() for tmp in numpy.meshgrid(x, y)]
    if fmt == 'binary':
        raise NotImplementedError(
            "Binary file support is not implemented yet.")
    return x, y, grd, (ny, nx)


def regular(area, shape, z=None):
    """
    Create a regular grid. Order of the output grid is x varies first, then y.

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(ny, nx)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[xcoords, ycoords]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[xcoords, ycoords, zcoords]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    """
    ny, nx = shape
    x1, x2, y1, y2 = area
    dy, dx = spacing(area, shape)
    x_range = numpy.arange(x1, x2, dx)
    y_range = numpy.arange(y1, y2, dy)
    # Need to make sure that the number of points in the grid is correct
    # because of rounding errors in arange. Sometimes x2 and y2 are included,
    # sometimes not
    if len(x_range) < nx:
        x_range = numpy.append(x_range, x2)
    if len(y_range) < ny:
        y_range = numpy.append(y_range, y2)
    assert len(x_range) == nx, "Failed! x_range doesn't have nx points"
    assert len(y_range) == ny, "Failed! y_range doesn't have ny points"
    xcoords, ycoords = [mat.ravel()
                        for mat in numpy.meshgrid(x_range, y_range)]
    if z is not None:
        zcoords = z * numpy.ones_like(xcoords)
        return [xcoords, ycoords, zcoords]
    else:
        return [xcoords, ycoords]


def scatter(area, n, z=None, seed=None):
    """
    Create an irregular grid with a random scattering of points.

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * n
        Number of points
    * z
        Optional. z coordinate of the points. If given, will return an
        array with the value *z*.
    * seed : None or int
        Seed used to generate the pseudo-random numbers. If `None`, will use a
        different seed every time. Use the same seed to generate the same
        random points.

    Returns:

    * ``[xcoords, ycoords]``
        Numpy arrays with the x and y coordinates of the points
    * ``[xcoords, ycoords, zcoords]``
        If *z* given. Arrays with the x, y, and z coordinates of the points

    """
    x1, x2, y1, y2 = area
    numpy.random.seed(seed)
    xcoords = numpy.random.uniform(x1, x2, n)
    ycoords = numpy.random.uniform(y1, y2, n)
    numpy.random.seed()
    if z is not None:
        zcoords = z * numpy.ones(n)
        return [xcoords, ycoords, zcoords]
    else:
        return [xcoords, ycoords]


def spacing(area, shape):
    """
    Returns the spacing between grid nodes

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(ny, nx)``.

    Returns:

    * ``[dy, dx]``
        Spacing the y and x directions

    """
    x1, x2, y1, y2 = area
    ny, nx = shape
    dx = (x2 - x1)/(nx - 1)
    dy = (y2 - y1)/(ny - 1)
    return [dy, dx]


def interp(x, y, v, shape, area=None, algorithm='cubic', extrapolate=False):
    """
    Interpolate data onto a regular grid.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * shape : tuple = (ny, nx)
        Shape of the interpolated regular grid, ie (ny, nx).
    * area : tuple = (x1, x2, y1, y2)
        The are where the data will be interpolated. If None, then will get the
        area from *x* and *y*.
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata).
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * ``[x, y, v]``
        Three 1D arrays with the interpolated x, y, and v

    """
    if algorithm not in ['cubic', 'linear', 'nearest']:
        raise ValueError("Invalid interpolation algorithm: " + str(algorithm))
    ny, nx = shape
    if area is None:
        area = (x.min(), x.max(), y.min(), y.max())
    x1, x2, y1, y2 = area
    xs = numpy.linspace(x1, x2, nx)
    ys = numpy.linspace(y1, y2, ny)
    xp, yp = [i.ravel() for i in numpy.meshgrid(xs, ys)]
    grid = interp_at(x, y, v, xp, yp, algorithm=algorithm,
                     extrapolate=extrapolate)
    return [xp, yp, grid]


def interp_at(x, y, v, xp, yp, algorithm='cubic', extrapolate=False):
    """
    Interpolate data onto the specified points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * xp, yp : 1D arrays
        Points where the data values will be interpolated
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata)
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * v : 1D array
        1D array with the interpolated v values.

    """
    if algorithm not in ['cubic', 'linear', 'nearest']:
        raise ValueError("Invalid interpolation algorithm: " + str(algorithm))
    grid = scipy.interpolate.griddata((x, y), v, (xp, yp),
                                      method=algorithm).ravel()
    if extrapolate and algorithm != 'nearest' and numpy.any(numpy.isnan(grid)):
        grid = extrapolate_nans(xp, yp, grid)
    return grid


def profile(x, y, v, point1, point2, size, extrapolate=False):
    """
    Extract a data profile between 2 points.

    Uses interpolation to calculate the data values at the profile points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * point1, point2 : lists = [x, y]
        Lists the x, y coordinates of the 2 points between which the profile
        will be extracted.
    * size : int
        Number of points along the profile.
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * [xp, yp, distances, vp] : 1d arrays
        ``xp`` and ``yp`` are the x, y coordinates of the points along the
        profile.
        ``distances`` are the distances of the profile points to ``point1``
        ``vp`` are the data points along the profile.

    """
    x1, y1 = point1
    x2, y2 = point2
    maxdist = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    distances = numpy.linspace(0, maxdist, size)
    angle = numpy.arctan2(y2 - y1, x2 - x1)
    xp = x1 + distances * numpy.cos(angle)
    yp = y1 + distances * numpy.sin(angle)
    vp = interp_at(x, y, v, xp, yp, algorithm='cubic', extrapolate=extrapolate)
    return xp, yp, distances, vp


def extrapolate_nans(x, y, v):
    """"
    Extrapolate the NaNs or masked values in a grid INPLACE using nearest
    value.

    .. warning:: Replaces the NaN or masked values of the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.

    Returns:

    * v : 1D array
        The array with NaNs or masked values extrapolated.

    """
    if numpy.ma.is_masked(v):
        nans = v.mask
    else:
        nans = numpy.isnan(v)
    notnans = numpy.logical_not(nans)
    v[nans] = scipy.interpolate.griddata((x[notnans], y[notnans]), v[notnans],
                                         (x[nans], y[nans]),
                                         method='nearest').ravel()
    return v


def cut(x, y, scalars, area):
    """
    Return a subsection of a grid.

    The returned subsection is not a copy! In technical terms, returns a slice
    of the numpy arrays. So changes made to the subsection reflect on the
    original grid. Use numpy.copy to make copies of the subsections and avoid
    this.

    Parameters:

    * x, y
        Arrays with the x and y coordinates of the data points.
    * scalars
        List of arrays with the scalar values assigned to the grid points.
    * area
        ``(x1, x2, y1, y2)``: Borders of the subsection

    Returns:

    * ``[subx, suby, subscalars]``
        Arrays with x and y coordinates and scalar values of the subsection.

    """
    xmin, xmax, ymin, ymax = area
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    inside = [i for i in xrange(len(x))
              if x[i] >= xmin and x[i] <= xmax
              and y[i] >= ymin and y[i] <= ymax]
    return [x[inside], y[inside], [s[inside] for s in scalars]]
