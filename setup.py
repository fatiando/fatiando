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
Build extention modules, package and install Fatiando.
Uses the numpy's extension of distutils to build the f2py extension modules
"""
from os.path import join
from numpy.distutils.core import setup, Extension

import fatiando

# Base paths for extention modules
potdir = join('src', 'potential')
seisdir = join('src', 'seismic')
heatdir = join('src', 'heat')

potential_prism = Extension('fatiando.potential._prism',
                            sources=[join(potdir, 'prism.c'),
                                     join(potdir, 'prism.pyf')],
                            extra_compile_args=['-fno-strict-aliasing'])
potential_talwani = Extension('fatiando.potential._talwani',
                            sources=[join(potdir, 'talwani.c'),
                                     join(potdir, 'talwani.pyf')],
                            extra_compile_args=['-fno-strict-aliasing'])
potential_polyprism = Extension('fatiando.potential._polyprism',
                                sources=[join(potdir, 'polyprism.c'),
                                         join(potdir, 'polyprism.pyf')],
                                extra_compile_args=['-fno-strict-aliasing'])
potential_transform = Extension('fatiando.potential._transform',
                                sources=[join(potdir, 'transform.c'),
                                         join(potdir, 'transform.pyf')],
                                extra_compile_args=['-fno-strict-aliasing'])
seismic_traveltime = Extension('fatiando.seismic._traveltime',
                                sources=[join(seisdir, 'traveltime.c'),
                                         join(seisdir, 'traveltime.pyf')],
                                extra_compile_args=['-fno-strict-aliasing'])
heat_climatesignal = Extension('fatiando.heat._climatesignal',
                                sources=[join(heatdir, 'climatesignal.c'),
                                         join(heatdir, 'climatesignal.pyf')],
                                extra_compile_args=['-fno-strict-aliasing'])

extmods = [potential_prism,
           potential_talwani,
           potential_polyprism,
           potential_transform,
           seismic_traveltime,
           heat_climatesignal]

packages = ['fatiando',
            'fatiando.potential',
            'fatiando.seismic',
            'fatiando.heat',
            'fatiando.vis',
            'fatiando.mesher',
            'fatiando.inversion',
            'fatiando.tests']

with open("README.txt") as f:
    long_description = f.readlines()

if __name__ == '__main__':

    setup(name='fatiando',
          fullname="Fatiando a Terra",
          description="Geophysical direct and inverse modeling",
          long_description=long_description,
          version=fatiando.__version__,
          author="Leonardo Uieda",
          author_email='leouieda at gmail.com',
          license='GNU LGPL',
          url="www.fatiando.org",
          platforms="Linux",
          scripts=[],
          packages=packages,
          ext_modules=extmods
         )
