"""
Build extention modules, package and install Fatiando.
Uses the numpy's extension of distutils to build the f2py extension modules
"""
import sys
import subprocess
import os
from os.path import join
try:
    from numpy.distutils.core import setup, Extension
except ImportError:
    print ("Sorry, Numpy <http://numpy.org/> and a C compiler are needed to " +
           "build Fatiando.")
    sys.exit()

# Base paths for extention modules
potdir = join('src', 'potential')
seisdir = join('src', 'seismic')
heatdir = join('src', 'heat')

extmods = [
    Extension('fatiando.potential._prism', sources=[join(potdir, 'prism.c'),
        join(potdir, 'prism.pyf')]),
    Extension('fatiando.potential._talwani', sources=[join(potdir, 'talwani.c'),
        join(potdir, 'talwani.pyf')]),
    Extension('fatiando.potential._polyprism',
        sources=[join(potdir, 'polyprism.c'), join(potdir, 'polyprism.pyf')]),
    Extension('fatiando.potential._transform',
        sources=[join(potdir, 'transform.c'), join(potdir, 'transform.pyf')]),
    Extension('fatiando.seismic._traveltime',
        sources=[join(seisdir, 'traveltime.c'),
                 join(seisdir, 'traveltime.pyf')]),
    Extension('fatiando.heat._climatesignal',
        sources=[join(heatdir, 'climatesignal.c'),
                 join(heatdir, 'climatesignal.pyf')])
    ]

packages = ['fatiando',
            'fatiando.potential',
            'fatiando.seismic',
            'fatiando.heat',
            'fatiando.vis',
            'fatiando.ui',
            'fatiando.mesher',
            'fatiando.inversion']

with open("README.txt") as f:
    long_description = ''.join(f.readlines())

def setrevison():
    # Check if the script is building/packaging or if this is a src dist
    if os.path.exists('.hg'):
        with open(join('fatiando','changeset.txt'), 'w') as versionfile:
            proc = subprocess.Popen('hg tip', shell=True,
                                    stdout=subprocess.PIPE)
            csline, bline = [l.strip() for l in proc.stdout.readlines()[0:2]]
            changeset = csline.split(':')[-1].strip()
            branch = bline.split(':')[-1].strip()
            if branch == 'tip':
                branch = 'default'
            versionfile.write("%s" % (changeset))

if __name__ == '__main__':
    version = '0.1.dev'
    setrevison()
    setup(name='fatiando',
          fullname="Fatiando a Terra",
          description="Fatiando a Terra - Geophysical modeling and inversion",
          long_description=long_description,
          version=version,
          author="Leonardo Uieda",
          author_email='leouieda@gmail.com',
          license='BSD License',
          url="http://www.fatiando.org",
          platforms="Any",
          scripts=[],
          packages=packages,
          ext_modules=extmods,
          classifiers=["Intended Audience :: Science/Research",
                       "Intended Audience :: Developers",
                       "Intended Audience :: Education",
                       "Programming Language :: C",
                       "Programming Language :: Python",
                       "Topic :: Scientific/Engineering"]
         )
