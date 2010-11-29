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
Set of utility functions.

Functions:

* :func:`fatiando.utils.contaminate`
    Add random noise to a data array

* :func:`fatiando.utils.extract_matrices`
    Extract value and x and y coordinate matrices from grid

* :func:`fatiando.utils.get_logger`
    Get a logger to ``stderr``

* :func:`fatiando.utils.set_logfile`
    Enable logging to a file.
    
* :func:`fatiando.utils.log_header`
    Generate a header message with the current version and changeset information

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-Mar-2010'


import logging
import time

import numpy

import fatiando
import fatiando.csinfo as csinfo


def get_logger(level=logging.DEBUG):
    """
    Get a logger to ``stderr``.
    (Adds a stream handler to the base logger ``'fatiando'``)
    
    Parameters:
    
    * level
        The logging level. Default to ``logging.DEBUG``. See ``logging`` module
      
    Returns:
    
    * a logger object
    
    """
    
    logger = logging.getLogger('fatiando')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter())
    logger.addHandler(handler)    
    logger.setLevel(level)

    return logger


def set_logfile(fname, level=logging.DEBUG):
    """
    Enable logging to a file.
    (Adds a file handler to the base logger ``'fatiando'``)
    
    Parameters:
    
    * fname 
        Log file name
    
    * level
        The logging level. Default to ``logging.DEBUG``. See ``logging`` module
      
    Returns:
    
    * a logger object
     
    """
    
    logger = logging.getLogger('fatiando')
    fhandler = logging.FileHandler(fname, 'w')
    fhandler.setFormatter(logging.Formatter())
    logger.addHandler(fhandler)
    logger.setLevel(level)
    
    return logger


def header(comment=''):
    """    
    Generate a header message with the current version and changeset information
    and current date.
                   
    Parameters:
    
    * comment
        Character inserted at the beginning of each line. Use this to make a
        message that can be inserted in source code files as comments.
                
    Returns:
    
    * msg
        string with the header message
                
    """
    
    lines = ["%sFatiando a Terra:\n" % (comment),
             "%s  version: %s\n" % (comment, fatiando.__version__),
             "%s  date: %s\n" % (comment, time.asctime()),
             "%s  version control info:\n" % (comment)             
            ]
    
    for line in csinfo.csinfo:
        
        newline = ''.join(['%s    ' % (comment), line])
        
        lines.append(newline)
        
    msg = ''.join(lines)
    
    return msg    
    

def contaminate(data, stddev, percent=True, return_stddev=False):
    """
    Add random noise to a data array.
    
    Noise added is normally distributed.
    
    Parameters:
    
    * data
        1D array-like data to contaminate
        
    * stddev
        Standard deviation of the Gaussian noise that will be added to *data*
        
    * percent
        If ``True``, will consider *stddev* as a decimal percentage and the 
        Standard deviation of the Gaussian noise will be this percentage of
        the maximum absolute value of *data*
        
    * return_stddev
        If ``True``, will return also the standard deviation used to contaminate
        *data*

    Returns:
    
    * [contam_data, stddev if *return_stddev* is ``True``]
        The contaminated data array        
        
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
    Extract value and x and y coordinate matrices from a grid dictionary. 
    
    Use to plot using matplotlib. 
    
    If the data is not a regular grid, it will be gridded.
    
    Parameters:
    
    * grid
        Data grid stored in a dictionary
            
    Return:
        
    * [X, Y, V]
        Matrices with the values of x, y and ``value`` at each grid point
        
    The data dictionary should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...],
         'grid':True, 'nx':points_in_x, 'ny':points_in_y}
               
    """

    assert grid['grid'] is True, "Only regular grids supported at the moment"
    assert 'nx' in grid.keys() and 'ny' in grid.keys(), \
        "Need nx and ny values in the grid (number of points in x and y)"
    
    X = numpy.reshape(grid['x'], (grid['ny'], grid['nx']))
    Y = numpy.reshape(grid['y'], (grid['ny'], grid['nx']))
    V = numpy.reshape(grid['value'], (grid['ny'], grid['nx']))
    
    return X, Y, V
