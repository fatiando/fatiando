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

**Padding**

* :func:`~fatiando.gridder.pad_array`
* :func:`~fatiando.gridder.unpad_array`

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
            # Read the min/max value of columns/longitude (y direction)
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

def pad_array(a, xy=None, npd=None, padtype='OddReflectionTaper'):
    """
    Return a padded array of arbitrary dimension.

    The function takes an array of arbitrary dimension and pads it either to 
    the dimensions given by the tuple npd, or to the next power of 2 if npd is
    not given.  New coordinate vectors are computed for each dimension, if
    provided.

    An odd reflection with a cosine taper is the author's preferred method of
    padding for Fourier operations.  The odd reflection optimally preserves
    the frequency content while adding minimal sharp inflections.  The cosine
    taper is a smooth function which also adds few sharp inflection points.

    .. note:: Requires gridded data of the same dimension as desired
        (i.e. no flattened arrays; use reshape).

    .. note:: This function returns a deep copy of the original array.

    .. warning:: This function returns the coordinate vectors as a list of 
        arrays.  This means that even for a 1D input array, the returned 
        vector will be in a list of one element.

    Parameters:

    * a : numpy array
        numpy array (N-D) to be padded
    * xy : N-D array (optional)
        [MxN] array where M is the number of observation points and N is the 
        dimension.  This is effectively a concatenated xp,yp, etc...
    * npd : tuple (optional)
        Optional tuple containing the total number of desired elements in each
        dimension
    * padtype : string (optional)
        String describing what to pad the new values with. Options:

        [ OddReflectionTaper | OddReflection | Reflection | value | LinTaper
        | edge | mean ]
            OddReflectionTaper - Generates odd reflection then tapers to the 
            mean using a cosine function (Default)

            OddReflection - Pads with the odd reflection, with no taper

            Reflection - Pads with simple reflection

            LinTaper - Linearly tapers to the mean

            value - numeric value

            edge - uses the edge value as a constant pad

            mean - uses the mean of the vector along each axis

    Returns:

    * ap : numpy array
        Padded array. The array core is a deep copy of the original array
    * cp : list
        List of coordinate arrays containing the extrapolated coordinate values
        Returns None if xy not provided
    * nps : list
        List of tuples containing the number of elements padded onto each 
        dimension.

    """

    # Test to make sure padtype is valid
    padopts = ['','oddreflectiontaper','oddreflection', 'reflection',
            'lintaper', 'edge', 'value', 'mean']
    #API Note:
    # If one wishes to add more options to the padding, use the following
    # checklist to make sure all the sections are consistent.
    #  [ ] Docstring (2 spots: the options, and the option descriptions
    #  [ ] padopts - Make sure to add to avoid raising a ValueError
    #  [ ] if statements below - Finally add an elif to the if statements
    #                            below. Add a descriptive comment.
    if padtype.lower() not in padopts and not _is_number(padtype):
        raise ValueError('gridder.pad_array: Pad Type not understood')
    # If npd is not provided, populate with next power of 2
    npt = []
    nd = a.ndim
    if npd == None:
        for ii in range(0, nd):
            if nd == 1:
                npt.append(_nextpow2(len(a)))
            else:
                npt.append(_nextpow2(a.shape[ii]))
    else:
        et = 'gridder.pad_array: pad dimensions do not match array dims'
        if nd == 1:
            if _is_integer(npd):
                npt.append(npd)
            else:
                if len(npd) != 1:
                    raise ValueError(et)
                else:
                    npt.append(npd)
        else:
            if _is_integer(npd):
                raise ValueError(et)
            elif len(npd) != a.ndim:
                raise ValueError(et)
            else:
                npt = npd
        for ii in numpy.arange(len(npt)):
            if npt[ii] <= a.shape[ii]:
                # How are you supposed to properly assign a multi-line string?
                et1 = 'gridder.pad_array: Desired padding is less than array '
                et2 = 'length along dimension '+str(ii)
                et = et1+et2
                raise ValueError(et)
    # Compute numbers to pad on the left and right side of the array along
    # each dimension
    nps = []
    for ii in range(0, nd):
        nps.append((int(numpy.ceil((npt[ii] - a.shape[ii])/2.)),
            int(numpy.floor((npt[ii] - a.shape[ii])/2.))))

    # If it will be needed, compute the mean
    meanneeded = ['lintaper', 'oddreflectiontaper']
    if padtype.lower() in meanneeded:
        m = numpy.mean(a)
    # Use numpy's padding routines where possible
    if _is_number(padtype):
        # Pad with value
        ap = numpy.pad(numpy.copy(a), nps, mode='constant',
            constant_values = float(padtype))
    elif padtype.lower() == 'mean':
        # Pad with the mean
        ap = numpy.pad(numpy.copy(a), nps, mode='mean')
    elif padtype.lower() == 'lintaper':
        # Linearly taper to the mean
        ap = numpy.pad(numpy.copy(a), nps, mode='linear_ramp', end_values=m)
    elif padtype.lower() == 'edge':
        # Pad with edge values
        ap = numpy.pad(numpy.copy(a), nps, mode='edge')
    elif padtype.lower() == 'reflection':
        # Pad with even reflection 
        ap = numpy.pad(numpy.copy(a), nps, mode='reflect', 
            reflect_type='even')
    elif padtype.lower() == 'oddreflection':
        # Pad with odd reflection
        ap = numpy.pad(numpy.copy(a), nps, mode='reflect', 
            reflect_type='odd')
    elif padtype.lower() == 'oddreflectiontaper':
        # Pad with odd reflection and a cosine taper to mean
        ap = (numpy.pad(numpy.copy(a), nps, mode='reflect', 
            reflect_type='odd') - m)
        for ii in numpy.arange(nd):
            ap = numpy.apply_along_axis(_costaper, ii, ap, lp=nps[ii][0],
                rp=nps[ii][1])
        ap += m
    if xy is not None:
        cp = _padcoords(xy, a.shape, nps)
    else:
        # What is the preferred way to do this?
        cp = None

    return ap, cp, nps

def unpad_array(a, nps):
    '''
    Unpads an array using the outputs from pad_array.

    This function takes a padded array and removes the padding from both. 
    Effectively, this is a complement to gridder.cut for when you already 
    know the number of elements to remove.

    .. note: Unlike pad_array, this returns a slice of the input array.
    Therefore, any changes to the padded array will be reflected in the 
    unpadded array.

    Parameters:

    * a : N-D array
        Array to be un-padded.  Can be of arbitrary dimension.
    * nps : list
        List of tuples giving the min and max indices for the cutoff
        Identical to nps returned by pad_array

    Returns:

    * b : N-D array
        Array of same dimension as a, with padding removed
    '''

    # xkcd.com/1597
    # Remove padding from the n-d array
    o = []
    for ii in numpy.arange(a.ndim):
        o.append(slice(nps[ii][0],a.shape[ii] - nps[ii][1]))
    b = a[o]

    return b

def _padcoords(xy, s, nps):
    # Define vector for coordinates for each dimension
    coords = []
    d = []
    coordspad = []
    for ii in numpy.arange(len(s)):
        if len(s) <= 1:
            coords.append(xy)
        else:
            coords.append(xy[:,ii].reshape(s).transpose().take(0,axis=ii))
        d.append(coords[ii][1]-coords[ii][0])
        coordspad.append(_padcvec(coords[ii], nps[ii], d[ii]))

    return coordspad

def _padcvec(x,n,dx):
    # Used by _padcoords to pad an individual vector based on the number
    # of points on either side and the point spacing
    xp = numpy.zeros(len(x) + n[0] + n[1])
    xp[n[0]:n[0]+len(x)] = x[:]
    for ii,jj in enumerate(numpy.arange(n[0])[::-1]):
        xp[ii] = x[0] - ((jj + 1) * dx)
    for ii,jj in enumerate(numpy.arange(len(x)+n[0], len(xp))):
        xp[jj] = x[-1] + (dx * (ii + 1))
    return xp

def _unpadcvec(x,n):
    # Takes a vector, x, and a tuple, n, and removes n[0] elements from the 
    # left and n[1] to the right  
    return x[n[0]:len(x)-n[1]]
            
def _costaper(a,lp,rp):
    # This takes an array and applies a cosine taper to each end.
    # The array has already been deep copied above.  This is by reference only.
    a[0:lp] = a[0:lp] * _calccostaper(lp)[::-1]
    a[-rp:] = a[-rp:] * _calccostaper(rp)
    return a

def _calccostaper(ntp):
    # Used by _costaper to compute a cosine taper from 1 to zero over
    # ntp points
    tp = numpy.zeros(ntp)
    for ii in range(1,ntp + 1):
        tp[ii-1] = (1.0 + numpy.cos((ii*numpy.pi) / float(ntp))/2.) - 0.5
    return tp

def _nextpow2(ii):
    # Computes the next power of two
    buf = numpy.ceil(numpy.log(ii) / numpy.log(2))
    return int(2**buf)

def _is_number(s):
    # Returns true if s can be cast as a float, false otherwise
    try:
        float(s)
        return True
    except ValueError:
        return False

def _is_integer(s):
    # Returns true if s is an integer. Used for testing int/array
    try:
        int(s)
        return True
    except TypeError:
        return False




