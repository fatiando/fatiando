# Copyright 2010 Leonardo Uieda
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

  * contaminate: add random noise to a data array
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-Mar-2010'


import numpy


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

