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
Miscellaneous utility functions and classes.

Mathematical functions
^^^^^^^^^^^^^^^^^^^^^^

* :func:`fatiando.utils.normal`
* :func:`fatiando.utils.gaussian`
* :func:`fatiando.utils.gaussian2d`

Point scatter generation
^^^^^^^^^^^^^^^^^^^^^^^^^

* :func:`fatiando.utils.random_points`
* :func:`fatiando.utils.circular_points`
* :func:`fatiando.utils.connect_points`

Others
^^^^^^

* :func:`fatiando.utils.sec2hms`
    Convert seconds to hours, minutes, and seconds
* :func:`fatiando.utils.contaminate`
    Contaminate a vector with pseudo-random Gaussian noise
* :class:`fatiando.utils.SparseList`
    Store only non-zero elements on an immutable list

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'

import math

import numpy

from fatiando import logger

log = logger.dummy()

  
class SparseList(object):
    """
    Store only non-zero elements on an immutable list.

    Can iterate over and access elements just like if it were a list.

    Parameters:
    
    * size
        Size of the list.

    Example::

        >>> l = SparseList(5)
        >>> l[3] = 42.0
        >>> print len(l)
        5
        >>> print l[1], l[3]
        0.0 42.0
        >>> for i in l:
        ...     print i
        0.0
        0.0
        0.0
        42.0
        0.0
        
    """

    def __init__(self, size):
        self.size = size
        self.i = 0
        self.elements = {}

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
        if index not in self.elements:
            return 0.0
        return self.elements[index]

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

def sec2hms(seconds):
    """
    Convert seconds into a string with hours, minutes and seconds.

    Parameters:
    
    * seconds
        Time in seconds

    Returns:
    
    * string
        String in the format ``'%dh %dm %2.5fs'``

    Example::
    
        >>> print sec2hms(62.2)
        0h 1m 2.20000s
        >>> print sec2hms(3862.12345678)
        1h 4m 22.12346s
    
    """
    h = int(seconds/3600)
    m = int((seconds - h*3600)/60)
    s = seconds - h*3600 - m*60
    return '%dh %dm %2.5fs' % (h, m, s)

def contaminate(data, stddev, percent=False, return_stddev=False):
    """
    Add pseudorandom gaussian noise to an array.

    Noise added is normally distributed.

    Parameters:
    
    * data
        1D array-like data to contaminate
    * stddev
        Standard deviation of the Gaussian noise that will be added to *data*
    * percent
        If ``True``, will consider *stddev* as a decimal percentage and the
        standard deviation of the Gaussian noise will be this percentage of
        the maximum absolute value of *data*
    * return_stddev
        If ``True``, will return also the standard deviation used to contaminate
        *data*
        
    Returns:
    
    * [contam_data, stddev if *return_stddev* is ``True``]
        The contaminated data array

    """
    log.info("Contaminating data with Gaussian noise:")
    if percent:
        log.info("  noise level = %g percent" % (100*stddev))
        stddev = stddev*max(abs(data))
    log.info("  noise stddev = %s" % (str(stddev)))
    contam = numpy.array([numpy.random.normal(v, stddev) for v in data])
    if return_stddev:
        return [contam, stddev]
    else:
        return contam

def normal(x, mean, std):
    """
    Normal distribution.

    .. math::

        N(x,\\bar{x},\sigma) = \\frac{1}{\sigma\sqrt{2 \pi}}
        \exp(-\\frac{(x-\\bar{x})^2}{\sigma^2})

    Parameters:
    
    * x
        Value at which to calculate the normal distribution
    * mean
        The mean of the distribution :math:`\\bar{x}`
    * std
        The standard deviation of the distribution :math:`\sigma`
        
    Returns:
    
    * normal distribution evaluated at *x*

    """
    return numpy.exp(-1*((mean - x)/std)**2)/(std*numpy.sqrt(2*numpy.pi))

def gaussian(x, mean, std):
    """
    Non-normalized Gaussian function

    .. math::

        G(x,\\bar{x},\sigma) = \exp(-\\frac{(x-\\bar{x})^2}{\sigma^2})

    Parameters:
    
    * x
        Value at which to calculate the Gaussian function
    * mean
        The mean of the distribution :math:`\\bar{x}`
    * std
        The standard deviation of the distribution :math:`\sigma`
        
    Returns:
    
    * Gaussian function evaluated at *x*

    """
    return numpy.exp(-1*((mean - x)/std)**2)

def gaussian2d(x, y, sigma_x, sigma_y, x0=0, y0=0, angle=0.0):
    """
    Non-normalized 2D Gaussian function

    Parameters:
    
    * x, y
        Coordinates at which to calculate the Gaussian function
    * sigma_x, sigma_y
        Standard deviation in the x and y directions
    * x0, y0
        Coordinates of the center of the distribution
    * angle
        Rotation angle of the gaussian measure from the x axis (north) growing
        positive to the east (positive y axis)
        
    Returns:
    
    * Gaussian function evaluated at *x*, *y*

    """
    theta = -1*angle*numpy.pi/180.
    tmpx = 1./sigma_x**2
    tmpy = 1./sigma_y**2
    sintheta = numpy.sin(theta)
    costheta = numpy.cos(theta)
    a = tmpx*costheta + tmpy*sintheta**2
    b = (tmpy - tmpx)*costheta*sintheta
    c = tmpx*sintheta**2 + tmpy*costheta**2
    xhat = x - x0
    yhat = y - y0
    return numpy.exp(-(a*xhat**2 + 2.*b*xhat*yhat + c*yhat**2))

def random_points(area, n):
    """
    Generate a set of n random points.
 
    Parameters:

    * area
        [x1, x2, y1, y2]: area inside of which the points are contained
    * n
        Number of points

    Result:

    * points
        List of (x, y) coordinates of the points
        
    """
    x1, x2, y1, y2 = area
    xs = numpy.random.uniform(x1, x2, n)
    ys = numpy.random.uniform(y1, y2, n)
    return numpy.array([xs, ys]).T

def circular_points(area, n, random=False):
    """
    Generate a set of n points positioned in a circular array.
    
    The diameter of the circle is equal to the smallest dimension of the area
 
    Parameters:

    * area
        [x1, x2, y1, y2]: area inside of which the points are contained
    * n
        Number of points
    * random
        If True, positions of the points on the circle will be chosen at random

    Result:

    * points
        List of (x, y) coordinates of the points
        
    """
    x1, x2, y1, y2 = area
    radius = 0.5*min(x2 - x1, y2 - y1)
    if random:
        angles = numpy.random.uniform(0, 2*math.pi, n)
    else:
        da = 2.*math.pi/float(n)
        angles = numpy.arange(0., 2.*math.pi, da)
    xs = 0.5*(x1 + x2) + radius*numpy.cos(angles)
    ys = 0.5*(y1 + y2) + radius*numpy.sin(angles)
    return numpy.array([xs, ys]).T

def connect_points(pts1, pts2):
    """
    Connects each point in the first list with all points in the second.
    If the first list has N points and the second has M, the result are 2 lists
    with N*M points each, representing the connections.

    Parameters:

    * pts1
        List of (x, y) coordinates of the points.
    * pts2
        List of (x, y) coordinates of the points.

    Returns:

    * [connect1, connect2]
        2 lists with the connected points
        
    """
    connect1 = []
    append1 = connect1.append
    connect2 = []
    append2 = connect2.append
    for p1 in pts1:
        for p2 in pts2:
            append1(p1)
            append2(p2)
    return [connect1, connect2]

def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
