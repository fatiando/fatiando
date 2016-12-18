"""
Build extension modules, package and install Fatiando.
"""
import sys
import os
from setuptools import setup, Extension, find_packages
import numpy
import versioneer

# VERSIONEER SETUP
# #############################################################################
versioneer.VCS = 'git'
versioneer.versionfile_source = 'fatiando/_version.py'
versioneer.versionfile_build = 'fatiando/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

# PACKAGE METADATA
# #############################################################################
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
PACKAGE_DATA = {
    'fatiando.datasets': ['data/*'],
    'fatiando.datasets.tests': ['data/*'],
    'fatiando.gravmag.tests': ['data/*'],
}
LICENSE = "BSD License"
URL = "http://www.fatiando.org"
PLATFORMS = "Any"
SCRIPTS = []
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

# DEPENDENCIES
# #############################################################################
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'numba',
    'future',
    'matplotlib',
    'pillow',
    'jupyter',
]

# C EXTENSIONS
# #############################################################################
# The running setup.py with --cython, then set things up to generate the Cython
# .c files. If not, then compile the pre-converted C files.
use_cython = True if '--cython' in sys.argv else False
# Build the module name and the path name for the extension modules
ext = '.pyx' if use_cython else '.c'
ext_parts = [
    ['fatiando', 'seismic', '_ttime2d'],
    ['fatiando', 'seismic', '_wavefd'],
    ['fatiando', 'gravmag', '_polyprism'],
    ['fatiando', 'gravmag', '_prism'],
]
extensions = [('.'.join(parts), os.path.join(*parts) + ext)
              for parts in ext_parts]
libs = []
if os.name == 'posix':
    libs.append('m')
ext_args = dict(libraries=libs, include_dirs=[numpy.get_include()])
EXT_MODULES = [Extension(name, [path], **ext_args)
               for name, path in extensions]
# Cythonize the .pyx modules if --cython is used
if use_cython:
    sys.argv.remove('--cython')
    from Cython.Build import cythonize
    EXT_MODULES = cythonize(EXT_MODULES)

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
          ext_modules=EXT_MODULES,
          package_data=PACKAGE_DATA,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          cmdclass=CMDCLASS,
          install_requires=INSTALL_REQUIRES)
