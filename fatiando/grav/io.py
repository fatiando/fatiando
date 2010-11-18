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
Input and output of gravity field data.

Functions:

* :func:`fatiando.grav.io.dump`
    Save gravity field data to a file.

* :func:`fatiando.grav.io.load`
    Load gravity field data from a file.

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.io')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def dump(fname, data, fmt='ascii'):
    """
    Save gravity field data to a file.
    
    Note: lines that start with # will be considered comments
    
    Column format:
    
    ``x  y  z value  error``
      
    If the data is a grid, the first line of the file will contain the number
    of points in the x and y directions separated by a space.
    
    Grids will be stored with x values varying first, then y. 
    
    Parameters:
    
    * fname
        Either string with file name or file object
    
    * data
        Gravity field data stored in a dictionary
              
    * fmt
        File format. For now only supports ASCII
        
    The data dictionary must be such as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...],
         'grid':True or False, 'nx':points_in_x, 'ny':points_in_y}
    
    the keys ``nx`` and ``ny`` are only required if ``grid`` is ``True``
    """
    
    if isinstance(fname, file):
        
        output = fname
        
    else:
        
        output = open(fname, 'w')
        
    log.info("Saving gravity field data to file '%s'" % (output.name))
        
    output.write("# File structure:\n")
    if data['grid']:        
        output.write("# nx  ny\n")
    output.write("# x  y  z  value  error\n")
    
    if data['grid']:        
        output.write("%d %d\n" % (data['nx'], data['ny']))
        
    for x, y, z, value, error in zip(data['x'], data['y'], data['z'], 
                                     data['value'], data['error']):
        
        output.write("%f %f %f %f %f\n" % (x, y, z, value, error))
        
    output.close()
    
    
def load(fname, fmt='ascii'):
    """
    Load gravity field data from a file.
    
    Note: lines that start with # will be considered comments
    
    Column format:
    
    ``x  y  z  value  error``
      
    If the file contains grid data, the first line should contain the number
    of points in the x and y directions separated by a space.
    
    Grids should be stored with x values varying first, then y. 
    
    Parameters:
    
    * fname
        Either string with file name or file object
                  
    * fmt
        File format. For now only supports ASCII
      
    Return:
    
    * data
        Gravity field data stored in a dictionary.
        
    The data dictionary will be such as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...],
         'grid':True or False, 'nx':points_in_x, 'ny':points_in_y}
    
    the keys ``nx`` and ``ny`` are only required if ``grid`` is ``True``        
    """
    
    if isinstance(fname, file):
        
        input = fname
        
    else:
        
        input = open(fname, 'r')
        
    log.info("Loading gravity field data from file '%s'" % (input.name))
       
    data = {'grid':False}
        
    xs = []
    ys = []
    zs = []
    values = []
    errors = []
    
    for l, line in enumerate(input):
        
        if line[0] == '#':
            
            continue
        
        args = line.strip().split(" ")
        
        if len(args) != 5:
            
            if len(args) == 2 and not values:
                
                data['grid'] = True
                data['nx'] = int(args[0])
                data['ny'] = int(args[1])
                
            else:
                
                log.warning("  Wrong number of values in line %d." % (l + 1) + 
                            " Ignoring it.")
            
            continue
        
        x, y, z, value, error = args
        
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
        values.append(float(value))
        errors.append(float(error))
        
    data['x'] = numpy.array(xs)
    data['y'] = numpy.array(ys)
    data['z'] = numpy.array(zs)
    data['value'] = numpy.array(values)
    data['error'] = numpy.array(errors)
        
    log.info("  data loaded=%d" % (len(data['value'])))
    
    return data


def dump_topo(fname, data, fmt='ascii'):
    """
    Save topography data to a file.
    
    Note: lines that start with # will be considered comments
    
    Column format:
    
    ``x  y  height``
      
    If the data is a grid, the first line of the file will contain the number
    of points in the x and y directions separated by a space.
    
    Grids will be stored with x values varying first, then y. 
    
    Parameters:
    
    * fname
        Either string with file name or file object
    
    * data
        Topography data stored in a dictionary
              
    * fmt
        File format. For now only supports ASCII
        
    The data dictionary must be such as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'h':[h1, h2, ...],
         'grid':True or False, 'nx':points_in_x, 'ny':points_in_y}
    
    the keys ``nx`` and ``ny`` are only required if ``grid`` is ``True``
    """
    
    if isinstance(fname, file):
        
        output = fname
        
    else:
        
        output = open(fname, 'w')
        
    log.info("Saving topography data to file '%s'" % (output.name))
        
    output.write("# File structure:\n")
    if data['grid']:        
        output.write("# nx  ny\n")
    output.write("# x  y  height\n")
    
    if data['grid']:        
        output.write("%d %d\n" % (data['nx'], data['ny']))
        
    for x, y, h in zip(data['x'], data['y'], data['h']):
        
        output.write("%f %f %f\n" % (x, y, h))
        
    output.close()
    
    
def load_topo(fname, fmt='ascii'):
    """
    Load topography data from a file.
    
    Note: lines that start with # will be considered comments
    
    Column format:
    
    ``x  y  height``
      
    If the file contains grid data, the first line should contain the number
    of points in the x and y directions separated by a space.
    
    Grids should be stored with x values varying first, then y. 
    
    Parameters:
    
    * fname
        Either string with file name or file object
                  
    * fmt
        File format. For now only supports ASCII
      
    Return:
    
    * data
        Topography data stored in a dictionary.
        
    The data dictionary will be such as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'h':[h1, h2, ...],
         'grid':True or False, 'nx':points_in_x, 'ny':points_in_y}
    
    the keys ``nx`` and ``ny`` are only required if ``grid`` is ``True``        
    """
    
    if isinstance(fname, file):
        
        input = fname
        
    else:
        
        input = open(fname, 'r')
        
    log.info("Loading topography data from file '%s'" % (input.name))
       
    data = {'grid':False}
        
    xs = []
    ys = []
    hs = []
    
    for l, line in enumerate(input):
        
        if line[0] == '#':
            
            continue
        
        args = line.strip().split(" ")
        
        if len(args) != 3:
            
            if len(args) == 2 and not hs:
                
                data['grid'] = True
                data['nx'] = int(args[0])
                data['ny'] = int(args[1])
                
            else:
                
                log.warning("  Wrong number of values in line %d." % (l + 1) + 
                            " Ignoring it.")
            
            continue
        
        x, y, h = args
        
        xs.append(float(x))
        ys.append(float(y))
        hs.append(float(h))
        
    data['x'] = numpy.array(xs)
    data['y'] = numpy.array(ys)
    data['h'] = numpy.array(hs)
        
    log.info("  data loaded=%d" % (len(data['h'])))
    
    return data