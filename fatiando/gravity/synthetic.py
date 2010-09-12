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
Create, load and dump synthetic gravity data.

Functions:
  * from_prisms: create synthetic data from a prism model
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import numpy


def from_prisms(prisms, x1, x2, y1, y2, nx, ny, height, type='gz', dist='grid'):
    """
    Create synthetic gravity data from a prism model.
    
    Parameters:
      
      prisms: list of dictionaries representing each prism in the model. 
              Required keys are {'x1':, 'x2':, 'y1':, 'y2':, 'z1':, 'z2':, 
              'density':}
              
      x1, x2: limits in the x direction of the region where the data will be 
              computed
              
      y1, y2: limits in the y direction of the region where the data will be 
              computed
              
      nx, ny: number of data points in the x and y directions
      
      height: height at which the data will be computed
      
      type: what component of the gravity field to calculate. Can be any one of
            'potential', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'
            
      dist: distribution of the data along the selected region. Can be either
            'grid' or 'random'
            
    Return:
        
        dictionary with the data values and coordinates of each point.
        keys: {'x':[x1, x2, x3, ...], 'y':[y1, y2, y3, ...], 
               type:[data1, data2, data3, ...]} 
    """
        
    raise NotImplementedError(
          "from_prisms was called before being implemented")