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

import fatiando


# Define the paths
c_dir = os.path.join('src', 'c')
wrap_dir = os.path.join('src', 'wrap')
f_dir = os.path.join('src', 'fortran')

# Define the extention modules
heat_diffusionfd = Extension('fatiando.heat._diffusionfd1d',
                    sources=[os.path.join(f_dir, 'heat_diffusionfd1d.f95')])

grav_prism = Extension('fatiando.grav._prism', 
                       sources=[os.path.join(c_dir, 'grav_prism.c'),
                                os.path.join(wrap_dir, 'grav_prism.pyf')])

grav_sphere = Extension('fatiando.grav._sphere',
                        sources=[os.path.join(c_dir, 'grav_sphere.c'),
                                 os.path.join(wrap_dir, 'grav_sphere.pyf')])
                                
seismo_traveltime = Extension('fatiando.seismo._traveltime', 
                    sources=[os.path.join(c_dir, 'seismo_traveltime.c'),
                             os.path.join(wrap_dir, 'seismo_traveltime.pyf')])

ext_mods = [heat_diffusionfd,
            grav_prism,
            grav_sphere,
            seismo_traveltime]
            

# Define the setup tags
name = 'fatiando'
fullname = 'Fatiando a Terra'
description = "Geophysical direct and inverse modeling"
long_description = \
"""
Fatiando a Terra is a software package containing various kinds of geophysical
modeling utilities for both direct and inverse problems. It serves as a sandbox
for rapidly prototyping of modeling ideas and algorithms. We hope that one day 
it will serve as a teaching tool for inverse problems in geophysics. 

Fatiando is being developed by a group of geophysics graduates and 
undergraduates from the Universidade de Sao Paulo and the Observatorio Nacional 
in Brazil.
"""
version = fatiando.__version__
author = "Leonardo Uieda, "
author_email = 'leouieda at gmail.com'
license = 'GNU LGPL'
url = 'http://code.google.com/p/fatiando/'
platforms = "Any"
scripts = []
py_modules = []
packages = ['fatiando',
            'fatiando.grav',
            'fatiando.grav.tests',
            'fatiando.heat',
            'fatiando.heat.tests',
            'fatiando.inv',
            'fatiando.inv.tests',
            'fatiando.seismo',
            'fatiando.seismo.tests',
            'fatiando.tests']
ext_modules = ext_mods
data_files = []

# Write the changeset information to file fatiando/csinfo.py
pipe = os.popen('hg parents')
csinfo = pipe.readlines()
csfile = open(os.path.join('fatiando', 'csinfo.py'), 'w')
csfile.write("csinfo = ")
csfile.write(str(csinfo[:-1]))
csfile.close()


if __name__ == '__main__':

    setup(name=name,
          fullname=fullname,
          description=description,
          long_description=long_description,
          version=version,
          author=author,
          author_email=author_email,
          license=license,
          url=url,
          platforms=platforms,
          scripts=scripts,
          py_modules=py_modules,
          packages=packages,
          ext_modules=ext_modules,
          data_files=data_files
         )