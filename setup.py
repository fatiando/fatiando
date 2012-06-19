"""
Build extention modules, package and install Fatiando.
Uses the numpy's extension of distutils to build the f2py extension modules
"""
import sys
import subprocess
import os
from os.path import join
from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    ext_modules = [
        Extension("fatiando.potential._cprism",
                  [join('fatiando', join('potential', '_cprism.pyx'))],
                  libraries=['m'],
                  extra_compile_args=['-O3'])]
    CYTHON = True
except ImportError:
    print ("Couldn't find Cython to build C extension.\n" +
        "Don't panic! Will use Python alternatives instead.")
    CYTHON = False

NAME = 'fatiando'
FULLNAME = 'Fatiando a Terra'
DESCRIPTION = "Fatiando a Terra - Geophysical modeling and inversion"
VERSION = '0.1.dev'
with open("README.txt") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())
PACKAGES = ['fatiando',
            'fatiando.potential',
            'fatiando.seismic',
            'fatiando.heat',
            'fatiando.vis',
            'fatiando.ui',
            'fatiando.mesher',
            'fatiando.inversion']
AUTHOR = "Leonardo Uieda"
AUTHOR_EMAIL = 'leouieda@gmail.com'
LICENSE = 'BSD License'
URL = "http://www.fatiando.org/software/fatiando"
PLATFORMS = "Any"
SCRIPTS = ['scripts/harvester']
CLASSIFIERS = ["Intended Audience :: Science/Research",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

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
    setrevison()
    if CYTHON:
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
              ext_modules=ext_modules,
              cmdclass = {'build_ext': build_ext},
              classifiers=CLASSIFIERS)
    else:
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
              classifiers=CLASSIFIERS)

