"""
Miscellaneous utility functions.
"""
from __future__ import absolute_import, division
from future.builtins import range
import math

import numpy
import scipy.sparse
import scipy.sparse.linalg
import scipy.misc

from . import constants, gridder


def safe_inverse(matrix):
    """
    Calculate the inverse of a matrix using an apropriate algorithm.

    Uses the standard :func:`numpy.linalg.inv` if *matrix* is dense.
    If it is sparse (from :mod:`scipy.sparse`) then will use
    :func:`scipy.sparse.linalg.inv`.

    Parameters:

    * matrix : 2d-array
        The matrix

    Returns:

    * inverse : 2d-array
        The inverse of *matrix*

    """
    if scipy.sparse.issparse(matrix):
        return scipy.sparse.linalg.inv(matrix)
    else:
        return numpy.linalg.inv(matrix)


def safe_solve(matrix, vector):
    """
    Solve a linear system using an apropriate algorithm.

    Uses the standard :func:`numpy.linalg.solve` if both *matrix* and *vector*
    are dense.

    If any of the two is sparse (from :mod:`scipy.sparse`) then will use the
    Conjugate Gradient Method (:func:`scipy.sparse.cgs`).

    Parameters:

    * matrix : 2d-array
        The matrix defining the linear system
    * vector : 1d or 2d-array
        The right-side vector of the system

    Returns:

    * solution : 1d or 2d-array
        The solution of the linear system


    """
    if scipy.sparse.issparse(matrix) or scipy.sparse.issparse(vector):
        estimate, status = scipy.sparse.linalg.cgs(matrix, vector)
        if status >= 0:
            return estimate
        else:
            raise ValueError('CGS exited with input error')
    else:
        return numpy.linalg.solve(matrix, vector)


def safe_dot(a, b):
    """
    Make the dot product using the appropriate method.

    If *a* and *b* are dense, will use :func:`numpy.dot`. If either is sparse
    (from :mod:`scipy.sparse`) will use the multiplication operator (i.e., \*).

    Parameters:

    * a, b : array or matrix
        The vectors/matrices to take the dot product of.

    Returns:

    * prod : array or matrix
        The dot product of *a* and *b*

    """
    if scipy.sparse.issparse(a) or scipy.sparse.issparse(b):
        return a * b
    else:
        return numpy.dot(a, b)


def safe_diagonal(matrix):
    """
    Get the diagonal of a matrix using the appropriate method.

    Parameters:

    * matrix : 2d-array, matrix, sparse matrix
        The matrix...

    Returns:

    * diag : 1d-array
        A numpy array with the diagonal of the matrix

    """
    if scipy.sparse.issparse(matrix):
        return numpy.array(matrix.diagonal())
    else:
        return numpy.diagonal(matrix).copy()


def sph2cart(lon, lat, height):
    """
    Convert spherical coordinates to Cartesian geocentric coordinates.

    Parameters:

    * lon, lat, height : floats
        Spherical coordinates. lon and lat in degrees, height in meters. height
        is the height above mean Earth radius.

    Returns:

    * x, y, z : floats
        Converted Cartesian coordinates

    """
    d2r = numpy.pi / 180.0
    radius = constants.MEAN_EARTH_RADIUS + height
    x = numpy.cos(d2r * lat) * numpy.cos(d2r * lon) * radius
    y = numpy.cos(d2r * lat) * numpy.sin(d2r * lon) * radius
    z = numpy.sin(d2r * lat) * radius
    return x, y, z


def si2nt(value):
    """
    Convert a value from SI units to nanoTesla.

    Parameters:

    * value : number or array
        The value in SI

    Returns:

    * value : number or array
        The value in nanoTesla

    """
    return value * constants.T2NT


def nt2si(value):
    """
    Convert a value from nanoTesla to SI units.

    Parameters:

    * value : number or array
        The value in nanoTesla

    Returns:

    * value : number or array
        The value in SI

    """
    return value / constants.T2NT


def si2eotvos(value):
    """
    Convert a value from SI units to Eotvos.

    Parameters:

    * value : number or array
        The value in SI

    Returns:

    * value : number or array
        The value in Eotvos

    """
    return value * constants.SI2EOTVOS


def eotvos2si(value):
    """
    Convert a value from Eotvos to SI units.

    Parameters:

    * value : number or array
        The value in Eotvos

    Returns:

    * value : number or array
        The value in SI

    """
    return value / constants.SI2EOTVOS


def si2mgal(value):
    """
    Convert a value from SI units to mGal.

    Parameters:

    * value : number or array
        The value in SI

    Returns:

    * value : number or array
        The value in mGal

    """
    return value * constants.SI2MGAL


def mgal2si(value):
    """
    Convert a value from mGal to SI units.

    Parameters:

    * value : number or array
        The value in mGal

    Returns:

    * value : number or array
        The value in SI

    """
    return value / constants.SI2MGAL


def vec2ang(vector):
    """
    Convert a 3-component vector to intensity, inclination and declination.

    .. note:: Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Parameter:

    * vector : array = [x, y, z]
        The vector

    Returns:

    * [intensity, inclination, declination] : floats
        The intensity, inclination and declination (in degrees)

    Examples::

        >>> s = vec2ang([1.5, 1.5, 2.121320343559643])
        >>> print "%.3f %.3f %.3f" % tuple(s)
        3.000 45.000 45.000

    """
    intensity = numpy.linalg.norm(vector)
    r2d = 180. / numpy.pi
    x, y, z = vector
    declination = r2d * numpy.arctan2(y, x)
    inclination = r2d * numpy.arcsin(z / intensity)
    return [intensity, inclination, declination]


def ang2vec(intensity, inc, dec):
    """
    Convert intensity, inclination and  declination to a 3-component vector

    .. note:: Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Parameter:

    * intensity : float or array
        The intensity (norm) of the vector
    * inc : float
        The inclination of the vector (in degrees)
    * dec : float
        The declination of the vector (in degrees)

    Returns:

    * vec : array = [x, y, z]
        The vector

    Examples::

        >>> import numpy
        >>> print ang2vec(3, 45, 45)
        [ 1.5         1.5         2.12132034]
        >>> print ang2vec(numpy.arange(4), 45, 45)
        [[ 0.          0.          0.        ]
         [ 0.5         0.5         0.70710678]
         [ 1.          1.          1.41421356]
         [ 1.5         1.5         2.12132034]]

    """
    return numpy.transpose([intensity * i for i in dircos(inc, dec)])


def dircos(inc, dec):
    """
    Returns the 3 coordinates of a unit vector given its inclination and
    declination.

    .. note:: Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Parameter:

    * inc : float
        The inclination of the vector (in degrees)
    * dec : float
        The declination of the vector (in degrees)

    Returns:

    * vect : list = [x, y, z]
        The unit vector

    """
    d2r = numpy.pi / 180.
    vect = [numpy.cos(d2r * inc) * numpy.cos(d2r * dec),
            numpy.cos(d2r * inc) * numpy.sin(d2r * dec),
            numpy.sin(d2r * inc)]
    return vect


class SparseList(object):

    """
    Store only non-zero elements on an immutable list.

    Can iterate over and access elements just like if it were a list.

    Parameters:

    * size : int
        Size of the list.
    * elements : dict
        Dictionary used to initialize the list. Keys are the index of the
        elements and values are their respective values.

    Example::

        >>> l = SparseList(5)
        >>> l[3] = 42.0
        >>> print len(l)
        5
        >>> print l[1], l[3]
        0.0 42.0
        >>> l[1] += 3.0
        >>> for i in l:
        ...     print i,
        0.0 3.0 0.0 42.0 0.0
        >>> l2 = SparseList(4, elements={1:3.2, 3:2.8})
        >>> for i in l2:
        ...     print i,
        0.0 3.2 0.0 2.8

    """

    def __init__(self, size, elements=None):
        self.size = size
        self.i = 0
        if elements is None:
            self.elements = {}
        else:
            self.elements = elements

    def __str__(self):
        return str(self.elements)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, index):
        if index < 0:
            index = self.size + index
        if index >= self.size or index < 0:
            raise IndexError('index out of range')
        return self.elements.get(index, 0.)

    def __setitem__(self, key, value):
        if key >= self.size:
            raise IndexError('index out of range')
        self.elements[key] = value

    def next(self):
        if self.i == self.size:
            raise StopIteration()
        res = self.__getitem__(self.i)
        self.i += 1
        return res


def contaminate(data, stddev, percent=False, return_stddev=False, seed=None):
    r"""
    Add pseudorandom gaussian noise to an array.

    Noise added is normally distributed with zero mean.

    Parameters:

    * data : array or list of arrays
        Data to contaminate
    * stddev : float or list of floats
        Standard deviation of the Gaussian noise that will be added to *data*
    * percent : True or False
        If ``True``, will consider *stddev* as a decimal percentage and the
        standard deviation of the Gaussian noise will be this percentage of
        the maximum absolute value of *data*
    * return_stddev : True or False
        If ``True``, will return also the standard deviation used to
        contaminate *data*
    * seed : None or int
        Seed used to generate the pseudo-random numbers. If `None`, will use a
        different seed every time. Use the same seed to generate the same
        random sequence to contaminate the data.

    Returns:

    if *return_stddev* is ``False``:

    * contam : array or list of arrays
        The contaminated data array

    else:

    * results : list = [contam, stddev]
        The contaminated data array and the standard deviation used to
        contaminate it.

    Examples:

    >>> import numpy as np
    >>> data = np.ones(5)
    >>> noisy = contaminate(data, 0.1, seed=0)
    >>> print noisy
    [ 1.03137726  0.89498775  0.95284582  1.07906135  1.04172782]
    >>> noisy, std = contaminate(data, 0.05, seed=0, percent=True,
    ...                          return_stddev=True)
    >>> print std
    0.05
    >>> print noisy
    [ 1.01568863  0.94749387  0.97642291  1.03953067  1.02086391]
    >>> data = [np.zeros(5), np.ones(3)]
    >>> noisy = contaminate(data, [0.1, 0.2], seed=0)
    >>> print noisy[0]
    [ 0.03137726 -0.10501225 -0.04715418  0.07906135  0.04172782]
    >>> print noisy[1]
    [ 0.81644754  1.20192079  0.98163167]

    """
    numpy.random.seed(seed)
    # Check if dealing with an array or list of arrays
    if not isinstance(stddev, list):
        stddev = [stddev]
        data = [data]
    contam = []
    for i in range(len(stddev)):
        if stddev[i] == 0.:
            contam.append(data[i])
            continue
        if percent:
            stddev[i] = stddev[i] * max(abs(data[i]))
        noise = numpy.random.normal(scale=stddev[i], size=len(data[i]))
        # Subtract the mean so that the noise doesn't introduce a systematic
        # shift in the data
        noise -= noise.mean()
        contam.append(numpy.array(data[i]) + noise)
    numpy.random.seed()
    if len(contam) == 1:
        contam = contam[0]
        stddev = stddev[0]
    if return_stddev:
        return [contam, stddev]
    else:
        return contam


def gaussian(x, mean, std):
    """
    Non-normalized Gaussian function

    .. math::

        G(x,\\bar{x},\sigma) = \exp\\left(-\\frac{(x-\\bar{x})^2}{\sigma^2}
        \\right)

    Parameters:

    * x : float or array
        Values at which to calculate the Gaussian function
    * mean : float
        The mean of the distribution :math:`\\bar{x}`
    * std : float
        The standard deviation of the distribution :math:`\sigma`

    Returns:

    * gauss : array
        Gaussian function evaluated at *x*

    """
    return numpy.exp(-1 * ((mean - x) / std) ** 2)


def gaussian2d(x, y, sigma_x, sigma_y, x0=0, y0=0, angle=0.0):
    """
    Non-normalized 2D Gaussian function

    Parameters:

    * x, y : float or arrays
        Coordinates at which to calculate the Gaussian function
    * sigma_x, sigma_y : float
        Standard deviation in the x and y directions
    * x0, y0 : float
        Coordinates of the center of the distribution
    * angle : float
        Rotation angle of the gaussian measure from the x axis (north) growing
        positive to the east (positive y axis)

    Returns:

    * gauss : array
        Gaussian function evaluated at *x*, *y*

    """
    theta = -1 * angle * numpy.pi / 180.
    tmpx = 1. / sigma_x ** 2
    tmpy = 1. / sigma_y ** 2
    sintheta = numpy.sin(theta)
    costheta = numpy.cos(theta)
    a = tmpx * costheta + tmpy * sintheta ** 2
    b = (tmpy - tmpx) * costheta * sintheta
    c = tmpx * sintheta ** 2 + tmpy * costheta ** 2
    xhat = x - x0
    yhat = y - y0
    return numpy.exp(-(a * xhat ** 2 + 2. * b * xhat * yhat + c * yhat ** 2))
