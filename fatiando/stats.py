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
  * mean: calculate the mean (per parameter) of several parameter vectors
  * stddev: calculate the standard deviation (per parameter) of several 
            parameter vectors
  * chisquare: perform the chi square test on a population
  * outliers: test a population for outliers
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import numpy


def mean(estimates):
    """
    Calculate the mean (per parameter) of several parameter vectors.
    
    Parameters:
    
      estimates: list of parameter vector estimates. Each estimate should be
                 an array-like 1D parameter vector.
                 
    Return:
    
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
    
      estimates: list of parameter vector estimates. Each estimate should be
                 an array-like 1D parameter vector.
                 
    Return:
    
      array-like 1D parameter standard deviation vector.
    """       
    
    stddev_estimate = []
    
    for parameter in numpy.transpose(estimates):
        
        stddev_estimate.append(parameter.std())
        
    return numpy.array(stddev_estimate)  