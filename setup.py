"""
Build extension modules, package and install Fatiando.
"""
import sys
import os
from setuptools import setup, Extension, find_packages
import numpy

# Get the version number and setup versioneer
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'fatiando/_version.py'
versioneer.versionfile_build = 'fatiando/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

NAME = 'fatiando'
FULLNAME = 'Fatiando a Terra'
DESCRIPTION = "Modeling and inversion for geophysics"
AUTHOR = "Leonardo Uieda"
AUTHOR_EMAIL = 'leouieda@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()
with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())
PACKAGES = find_packages(exclude=['doc', 'ci', 'cookbook', 'gallery'])
LICENSE = "BSD 3-clause"
URL = "http://www.fatiando.org"
PLATFORMS = "Any"
SCRIPTS = []
PACKAGE_DATA = {'fatiando': [os.path.join('data', '*')]}
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 2.7",
    "License :: OSI Approved :: {}".format(LICENSE),
]
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
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          license=LICENSE,
          url=URL,
          platforms=PLATFORMS,
          scripts=SCRIPTS,
          packages=PACKAGES,
          ext_modules=extensions,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          cmdclass=CMDCLASS)
