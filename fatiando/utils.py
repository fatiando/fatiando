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
Set of misc utility functions.

Functions:
  * contaminate: Add random noise to a data array
  * extract_matrices: Extract value and x and y coordinate matrices from grid
  * get_logger: Get a logger to stderr
  * set_logfile: Enable logging to a file.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-Mar-2010'


import logging

import numpy


def get_logger(level=logging.DEBUG):
    """
    Get a logger to stderr
    
    Parameters:
    
      level: the logging level. Default: logging.DEBUG. See logging module
      
    Returns:
    
      the root logger
    """
    
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter())
    logger.addHandler(handler)    
    logger.setLevel(level)

    return logger


def set_logfile(fname, level=logging.DEBUG):
    """
    Enable logging to a file.
    (Adds a file handler to the root logger)
    
    Parameters:
    
      fname: log file name
      
      level: the logging level. Default: logging.DEBUG. See logging module
      
    Returns:
    
      the root logger 
    """
    
    logger = logging.getLogger()
    fhandler = logging.FileHandler(fname, 'w')
    fhandler.setFormatter(logging.Formatter())
    logger.addHandler(fhandler)
    logger.setLevel(level)
    
    return logger


def contaminate(data, stddev, percent=True, return_stddev=False):
    """
    Contaminate a given data array (1D) with a normally distributed error of
    standard deviation stddev. If percent=True, then stddev is assumed to be a
    percentage of the maximum value in data (0 < stddev <= 1).
    If return_stddev=True, the calculated stddev will be returned with the 
    contaminated data.
    """

    if percent:

        maximum = abs(numpy.array(data)).max()

        stddev = stddev*maximum

    cont_data = []
    
    for value in data:

        cont_data.append(numpy.random.normal(value, stddev))

    if return_stddev:
        
        return [cont_data, stddev]
    
    else:
        
        return cont_data
    
    
def extract_matrices(grid):
    """
    Extract value and x and y coordinate matrices from grid. Use to plot with
    matplotlib. 
    
    If the data is not a regular grid, it will be gridded.
    
    Parameters:
    
      grid: data to contour. Should be a dictionay with the keys:
            {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...]
             'value':[data1, data2, ...], 'error':[error1, error2, ...],
             'grid':True or False, 'nx':points_in_x, 'ny':points_in_y} 
            the keys 'nx' and 'ny' are only given if 'grid' is True
            
    Return:
        
      X, Y, V matrices
      
    """

    assert grid['grid'] is True, "Only regular grids supported at the moment"
    assert 'nx' in grid.keys() and 'ny' in grid.keys(), \
        "Need nx and ny values in the grid (number of points in x and y)"
    
    X = numpy.reshape(grid['x'], (grid['ny'], grid['nx']))
    Y = numpy.reshape(grid['y'], (grid['ny'], grid['nx']))
    V = numpy.reshape(grid['value'], (grid['ny'], grid['nx']))
    
    return X, Y, V
