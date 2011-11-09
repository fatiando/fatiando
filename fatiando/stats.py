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
Utility mathematical functions and contaminate data with noise.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'

import math

import numpy

from fatiando import logger

log = logger.dummy()


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
