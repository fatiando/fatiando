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
import subprocess
from numpy.distutils.core import setup, Extension

import fatiando

# Base paths for extention modules
potdir = join('src', 'potential')

potential_prism = Extension('fatiando.potential._prism',
                            sources=[join(potdir, 'prism.c'),
                                     join(potdir, 'prism.pyf')])

extmods = [potential_prism]

packages = ['fatiando',
            'fatiando.potential',
            'fatiando.seismic',
            'fatiando.inversion',
            'fatiando.tests']

with open("README.txt") as f:
    long_description = f.readlines()

# Get the changeset information from Mercurial and save it to module
# fatiando.changeset
with open(join('fatiando','changeset.py'), 'w') as csmod:
    proc = subprocess.Popen('hg parents', shell=True, stdout=subprocess.PIPE)
    csmod.write('"""\nInformation on the latest changeset packaged\n"""\n')
    for line in proc.stdout.readlines():
        entries = [e.strip() for e in line.split(':')]
        if len(entries) == 1:
            continue
        key = entries[0]
        value = ':'.join(entries[1:])
        csmod.write('%s = "%s"\n' % (key, value))

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
