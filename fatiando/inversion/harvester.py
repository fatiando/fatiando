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
3D iterative inversion using a planting algorithm.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 17-Nov-2010'

from fatiando import potential


class DataModule(object):
    """
    """

    def __init__(self, norm='l2'):
        pass

    def residuals(self):
        pass

    def misfit(self):
        pass
    
        


def sow(mesh, point, props):
    """
    Return seed
    """
    pass

def grow(seeds, mesh, dmods, compact=0, thresh=0.0001):
    """
    Yield one accretion at a time
    """
    pass

def harvest(seeds, mesh, dmods, compact=0, thresh=0.0001):
    """
    Perform all accretions. Return mesh with special props in it.
    special props don't store 0s, return 0 if try to access element not in dict
    """
    pass


