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
Create synthetic gravity data from various types of model.

Functions:
  * from_prisms: Create synthetic data from a prism model
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging

import numpy

import fatiando
import fatiando.grav.prism


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grav.synthetic')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def from_prisms(prisms, x1, x2, y1, y2, nx, ny, height, field='gz', 
                grid='regular'):
    """
    Create synthetic gravity data from a prism model.
    
    Note: to make a profile along x, set y1=y2 and ny=1
    
    Parameters:
      
      prisms: list of dictionaries representing each prism in the model. 
              Required keys are {'x1':, 'x2':, 'y1':, 'y2':, 'z1':, 'z2':, 
              'value':density}
              
      x1, x2: limits in the x direction of the region where the data will be 
              computed
              
      y1, y2: limits in the y direction of the region where the data will be 
              computed
              
      nx, ny: number of data points in the x and y directions
      
      height: height at which the data will be computed
      
      field: what component of the gravity field to calculate. Can be any one of
             'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'
            
      grid: distribution of the data along the selected region. Can be either
            'regular' or 'random'
            
    Return:
        
        dictionary with the data values and coordinates of each point.
        keys: {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...]
               'value':[data1, data2, ...]} 
    """
    
    assert grid in ['regular', 'random'], "Invalid grid type '%s'" % (grid)
    
    fields = {'gz':fatiando.grav.prism.gz, 
              'gxx':fatiando.grav.prism.gxx, 
              'gxy':fatiando.grav.prism.gxy, 
              'gxz':fatiando.grav.prism.gxz, 
              'gyy':fatiando.grav.prism.gyy, 
              'gyz':fatiando.grav.prism.gyz, 
              'gzz':fatiando.grav.prism.gzz}    
    
    assert field in fields.keys(), "Invalid gravity field '%s'" % (field)
    
    data = {'z':-height*numpy.ones(nx*ny), 'value':numpy.zeros(nx*ny), 
            'error':numpy.zeros(nx*ny)}
    
    log.info("Generating synthetic %s data:" % (field))
    
    if grid == 'regular':
        
        log.info("  grid type: regular")
        
        dx = float(x2 - x1)/(nx - 1)
        x_range = numpy.arange(x1, x2, dx)
        
        if ny > 1:
            
            dy = float(y2 - y1)/(ny - 1)
            y_range = numpy.arange(y1, y2, dy)
            
        else:
            
            dy = 0
            y_range = [y1]
        
        if len(x_range) < nx:
            
            x_range = numpy.append(x_range, x2)
        
        if len(y_range) < ny:
            
            y_range = numpy.append(y_range, y2)
            
        xs = []
        ys = []
        
        for y in y_range:
            
            for x in x_range:
                
                xs.append(x)
                ys.append(y)
                
        xs = numpy.array(xs)
        ys = numpy.array(ys)
        
        data['grid'] = True
        data['nx'] = nx
        data['ny'] = ny
            
    if grid == 'random':
        
        log.info("  grid type: random")
        
        xs = numpy.random.uniform(x1, x2, nx*ny)
        
        if ny > 1:
            
            ys = numpy.random.uniform(y1, y2, nx*ny)
            
        else:
            
            ys = [y1]
        
        data['grid'] = False
        
    data['x'] = xs
    data['y'] = ys
    
    function = fields[field]
    value = data['value']
    
    for i, coordinates in enumerate(zip(data['x'], data['y'], data['z'])):
        
        x, y, z = coordinates
        
        for prism in prisms:        
                        
            value[i] += function(prism['value'], 
                                 prism['x1'], prism['x2'], 
                                 prism['y1'], prism['y2'], 
                                 prism['z1'], prism['z2'], 
                                 float(x), float(y), float(z))
            
    log.info("  data points = %d" % (len(data['value'])))
    
    return data