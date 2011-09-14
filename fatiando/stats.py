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
Perform statistical analysis on inversion results.

Functions:

* :func:`fatiando.stats.mean`
    Calculate the mean (per parameter) of several parameter vectors

* :func:`fatiando.stats.stddev`
    Calculate the standard deviation (per parameter) of several parameter 
    vectors
        
* :func:`fatiando.stats.normal`
    Normal distribution
    
* :func:`fatiando.stats.gaussian`
    Non-normalized Gaussian function
    
* :func:`fatiando.stats.gaussian2d`
    Non-normalized 2D Gaussian function
         
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'

import logging

import numpy

import fatiando

log = logging.getLogger('fatiando.stats')
log.addHandler(fatiando.default_log_handler)


def mean(estimates):
    """
    Calculate the mean (per parameter) of several parameter vectors.
    
    Parameters:
    
    * estimates
        List of parameter vectors. Each estimate should be array-like 1D.
                 
    Returns:
    
    * mean
        array-like 1D mean parameter vector.
        
    """
    
    mean_estimate = []
    
    for parameter in numpy.transpose(estimates):
        
        mean_estimate.append(parameter.mean())
        
    return numpy.array(mean_estimate)


def stddev(estimates):
    """    
    Calculate the standard deviation (per parameter) of several parameter 
    vectors.
    
    Parameters:
    
    * estimates
        List of parameter vectors. Each estimate should be array-like 1D.
                 
    Return:
    
    * stddev
        array-like 1D parameter standard deviation vector.
        
    """       
    
    stddev_estimate = []
    
    for parameter in numpy.transpose(estimates):
        
        stddev_estimate.append(parameter.std())
        
    return numpy.array(stddev_estimate)


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


def gaussian2d(x, y, xmean, ymean, cov):
    """
    Non-normalized 2D Gaussian function
        
    Parameters:
    
    * x, y
        Coordinates at which to calculate the Gaussian function
        
    * xmean, ymean
        Coordinates of the center of the distribution
        
    * cov
        Covariance matrix of the distribution. 2 x 2 array.
        
    Returns:
    
    * Gaussian function evaluated at *x*, *y*
    
    """
   
    cov_inv = numpy.linalg.inv(numpy.array(cov))

    value = (cov_inv[0][0]*(x - xmean)**2 + 
             2*cov_inv[0][1]*(x - xmean)*(y - ymean) +
             cov_inv[1][1]*(y - ymean)**2)
    
    return numpy.exp(-value)