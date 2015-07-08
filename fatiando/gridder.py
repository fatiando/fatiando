"""
Create and operate on grids and profiles.

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
import numpy
import scipy.interpolate


def load_surfer(fname, fmt='ascii'):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape

    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'

    Returns:

    * x : 1d-array
        Value of the North-South coordinate of each grid point.
    * y : 1d-array
        Value of the East-West coordinate of each grid point.
    * data : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : tuple = (nx, ny)
        The number of points in the x and y grid dimensions, respectively

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
            # Read the number of columns (ny) and rows (nx)
            ny, nx = [int(s) for s in ftext.readline().split()]
            shape = (nx, ny)
            # Read the min/max value of columns/longitue (y direction)
            ymin, ymax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of rows/latitude (x direction)
            xmin, xmax = [float(s) for s in ftext.readline().split()]
            area = (xmin, xmax, ymin, ymax)
            # Read the min/max value of grid values
            datamin, datamax = [float(s) for s in ftext.readline().split()]
            data = numpy.fromiter((float(i) for line in ftext for i in
                                   line.split()), dtype='f')
            data = numpy.ma.masked_greater_equal(data, 1.70141e+38)
            assert numpy.allclose(datamin, data.min()) \
                and numpy.allclose(datamax, data.max()), \
                "Min and max values of grid don't match ones read from file." \
                + "Read: ({}, {})  Actual: ({}, {})".format(
                    datamin, datamax, data.min(), data.max())
        # Create x and y coordinate numpy arrays
        x, y = regular(area, shape)
    if fmt == 'binary':
        raise NotImplementedError(
            "Binary file support is not implemented yet.")
    return x, y, data, shape


def regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples::

        >>> x, y = regular((0, 10, 0, 5), (5, 3))
        >>> x
        array([  0. ,   0. ,   0. ,   2.5,   2.5,   2.5,   5. ,   5. ,   5. ,
                 7.5,   7.5,   7.5,  10. ,  10. ,  10. ])
        >>> x.reshape((5, 3))
        array([[  0. ,   0. ,   0. ],
               [  2.5,   2.5,   2.5],
               [  5. ,   5. ,   5. ],
               [  7.5,   7.5,   7.5],
               [ 10. ,  10. ,  10. ]])
        >>> y.reshape((5, 3))
        array([[ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ]])
        >>> x, y = regular((0, 0, 0, 5), (1, 3))
        >>> x.reshape((1, 3))
        array([[ 0.,  0.,  0.]])
        >>> y.reshape((1, 3))
        array([[ 0. ,  2.5,  5. ]])
        >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
        >>> z.reshape((5, 3))
        array([[-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.]])


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)
    xs = numpy.linspace(x1, x2, nx)
    ys = numpy.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = numpy.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*numpy.ones(nx*ny, dtype=numpy.float))
    return [i.ravel() for i in arrays]


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

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the points
    * ``[x, y, z]``
        If *z* given. Arrays with the x, y, and z coordinates of the points

    Examples::

        >>> x, y = scatter((0, 10, 0, 2), 4, seed=0)
        >>> x
        array([ 5.48813504,  7.15189366,  6.02763376,  5.44883183])
        >>> y
        array([ 0.8473096 ,  1.29178823,  0.87517442,  1.783546  ])

    """
    x1, x2, y1, y2 = area
    numpy.random.seed(seed)
    arrays = [numpy.random.uniform(x1, x2, n), numpy.random.uniform(y1, y2, n)]
    if z is not None:
        arrays.append(z*numpy.ones(n))
    return arrays


def spacing(area, shape):
    """
    Returns the spacing between grid nodes

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.

    Returns:

    * ``[dx, dy]``
        Spacing the y and x directions

    Examples::

        >>> spacing((0, 10, 0, 20), (11, 11))
        [1.0, 2.0]
        >>> spacing((0, 10, 0, 20), (11, 21))
        [1.0, 1.0]
        >>> spacing((0, 10, 0, 20), (5, 21))
        [2.5, 1.0]
        >>> spacing((0, 10, 0, 20), (21, 21))
        [0.5, 1.0]

    """
    x1, x2, y1, y2 = area
    nx, ny = shape
    dx = (x2 - x1)/(nx - 1)
    dy = (y2 - y1)/(ny - 1)
    return [dx, dy]


def interp(x, y, v, shape, area=None, algorithm='cubic', extrapolate=False):
    """
    Interpolate data onto a regular grid.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * shape : tuple = (nx, ny)
        Shape of the interpolated regular grid, ie (nx, ny).
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
    nx, ny = shape
    if area is None:
        area = (x.min(), x.max(), y.min(), y.max())
    x1, x2, y1, y2 = area
    xp, yp = regular(area, shape)
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
