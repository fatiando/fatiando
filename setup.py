"""
Build extention modules, package and install Fatiando.
Uses the numpy's extension of distutils to build the f2py extension modules
"""
import sys
import os
from distutils.core import setup
from distutils.extension import Extension


NAME = 'fatiando'
FULLNAME = 'Fatiando a Terra'
DESCRIPTION = "Geophysical modeling and inversion"
VERSION = '0.3'

if os.path.isfile("README.rst"):
    with open("README.rst") as f:
        LONG_DESCRIPTION = ''.join(f.readlines())
elif os.path.isfile("README.txt"):
    with open("README.txt") as f:
        LONG_DESCRIPTION = ''.join(f.readlines())
else:
    LONG_DESCRIPTION = ''

PACKAGES = ['fatiando',
            'fatiando.gravmag',
            'fatiando.seismic',
            'fatiando.geothermal',
            'fatiando.vis',
            'fatiando.gui',
            'fatiando.inversion']
AUTHOR = "Leonardo Uieda"
AUTHOR_EMAIL = 'leouieda@gmail.com'
LICENSE = "BSD License"
URL = "http://www.fatiando.org"
PLATFORMS = "Any"
SCRIPTS = []
CLASSIFIERS = ["Intended Audience :: End Users/Desktop",
               "Intended Audience :: Science/Research",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Topic :: Scientific/Engineering",
               "Topic :: Software Development :: Libraries",
               "Environment :: Console",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Cython",
               "License :: OSI Approved :: BSD License",
               "Development Status :: 3 - Alpha",
               "Natural Language :: English"]


# The runing setup.py with --cython, then set things up to generate the Cython
# .c files. If not, then compile the pre-converted C files.
USE_CYTHON = True if '--cython' in sys.argv else False
ext = '.pyx' if USE_CYTHON else '.c'

libs = []
if os.name == 'posix':
    libs.append('m')


def get_include_dirs(include_dirs=[]):
    try:
        import numpy

        include_dirs = [numpy.get_include()]
    except ImportError:
        print "numpy couldn't be imported"
    return include_dirs


extensions = [
    Extension('.'.join(e), [os.path.join(*e) + ext],
              libraries=libs,
              include_dirs=get_include_dirs())
    for e in [
        ['fatiando', 'gravmag', '_prism'],
        ['fatiando', 'gravmag', '_tesseroid'],
        ['fatiando', 'seismic', '_ttime2d'],
        ['fatiando', 'seismic', '_wavefd']
    ]
]
if USE_CYTHON:
    sys.argv.remove('--cython')
    try:
        from Cython.Build import cythonize

        extensions = cythonize(extensions)
    except ImportError:
        exit("Cython couldn't be imported; try without `--cython`")

if __name__ == '__main__':
    setup(name=NAME,
          fullname=FULLNAME,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          url=URL,
          platforms=PLATFORMS,
          scripts=SCRIPTS,
          packages=PACKAGES,
          ext_modules=extensions,
          classifiers=CLASSIFIERS)
