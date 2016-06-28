"""
Build extension modules, package and install Fatiando.
"""
import sys
import os
from distutils.core import setup
from distutils.extension import Extension
import numpy

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'fatiando/_version.py'
versioneer.versionfile_build = 'fatiando/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

NAME = 'fatiando'
FULLNAME = 'Fatiando a Terra'
DESCRIPTION = "Modeling and inversion for geophysics"
VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()
with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())
PACKAGES = ['fatiando',
            'fatiando.tests',
            'fatiando.tests.gravmag',
            'fatiando.tests.gridder',
            'fatiando.tests.seismic',
            'fatiando.gravmag',
            'fatiando.seismic',
            'fatiando.geothermal',
            'fatiando.vis',
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
KEYWORDS = 'geophysics modeling inversion gravimetry seismic magnetometry'

# The running setup.py with --cython, then set things up to generate the Cython
# .c files. If not, then compile the pre-converted C files.
USE_CYTHON = True if '--cython' in sys.argv else False
ext = '.pyx' if USE_CYTHON else '.c'
libs = []
if os.name == 'posix':
    libs.append('m')
C_EXT = [[['fatiando', 'seismic', '_ttime2d'], {}],
         [['fatiando', 'seismic', '_wavefd'], {}],
         [['fatiando', 'gravmag', '_polyprism'], {}],
         [['fatiando', 'gravmag', '_sphere'], {}],
         [['fatiando', 'gravmag', '_prism'], {}],
         ]
extensions = []
for e, extra_args in C_EXT:
    extensions.append(
        Extension('.'.join(e), [os.path.join(*e) + ext],
                  libraries=libs,
                  include_dirs=[numpy.get_include()],
                  **extra_args))
if USE_CYTHON:
    sys.argv.remove('--cython')
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

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
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          cmdclass=CMDCLASS)
