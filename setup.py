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
Build extention modules and package Fatiando for release.
"""

import os

from numpy.distutils.extension import Extension
from numpy.distutils.core import setup


# Define the paths
c_dir = os.path.join('src', 'c')
wrap_dir = os.path.join('src', 'wrap')
fortran_dir = os.path.join('src', 'fortran')

# Define the extention modules
head_diffusionfd = Extension('fatiando.heat._diffusionfd',
                    sources=[os.path.join(fortran_dir, 'heat_diffusionfd.f95')])

grav_prism = Extension('fatiando.grav._prism', 
                       sources=[os.path.join(c_dir, 'grav_prism.c'),
                                os.path.join(wrap_dir, 'grav_prism.pyf')])

seismo_traveltime = Extension('fatiando.seismo._traveltime', 
                    sources=[os.path.join(c_dir, 'seismo_traveltime.c'),
                             os.path.join(wrap_dir, 'seismo_traveltime.pyf')])

ext_modules = []
ext_modules.append(head_diffusionfd)
ext_modules.append(grav_prism)
ext_modules.append(seismo_traveltime)


if __name__ == '__main__':

    setup(name='fatiando',
          ext_modules=ext_modules
         )