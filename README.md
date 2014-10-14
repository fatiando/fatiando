# ![Fatiando a Terra](https://raw.githubusercontent.com/fatiando/logo/master/fatiando-banner-long.png)

[Website](http://www.fatiando.org) |
[Docs](http://fatiando.readthedocs.org) |
[Mailing list](https://groups.google.com/d/forum/fatiando) |
[Google+](https://plus.google.com/+FatiandoOrg)

A Python package for modeling and inversion in geophysics.

[![Latest PyPI version | ](http://img.shields.io/pypi/v/fatiando.svg?style=flat)](https://crate.io/packages/fatiando)
[![Number of PyPI downloads | ](http://img.shields.io/pypi/dm/fatiando.svg?style=flat)](https://crate.io/packages/fatiando/)
[![BSD 3 clause license | ](http://img.shields.io/badge/license-BSD-lightgrey.svg?style=flat)](https://github.com/fatiando/fatiando/blob/master/LICENSE.txt)
[![Travis CI build status | ](http://img.shields.io/travis/fatiando/fatiando.svg?style=flat)](https://travis-ci.org/fatiando/fatiando)
[![Test coverage status | ](http://img.shields.io/coveralls/fatiando/fatiando.svg?style=flat)](https://coveralls.io/r/fatiando/fatiando?branch=master)
[![doi:10.6084/m9.figshare.1115194](http://img.shields.io/badge/doi-10.6084/m9.figshare.1115194-blue.svg?style=flat)](http://dx.doi.org/10.6084/m9.figshare.1115194)

## Dependencies

For the moment, Fatiando runs and is tested in Python 2.7.
To install and run Fatiando, you'll need the following packages:

* numpy >= 1.8
* scipy >= 0.14
* matplotlib >= 1.3
* mayavi >= 4.3
* PIL >= 1.1.7
* basemap >= 1.0.7
* gcc >= 4.8.2

You can get all of these on Linux, Mac, and Windows through
the [Anaconda distribution](http://continuum.io/downloads).
See file `requirements.txt` and the
[documentation](http://fatiando.readthedocs.org).

## Installing

Download and install the latest release of Fatiando from
[PyPI](https://pypi.python.org/pypi/fatiando):

    pip install fatiando

or get the latest development version from Github:

    pip install --upgrade https://github.com/fatiando/fatiando/archive/master.zip

**Note**: `fatiando.__version__` has the current version number. If you install
from PyPI, this will be something like `'0.2'`. If you installed from Github,
this will be the latest commit hash. This way you can track exactly what
version of Fatiando generated your results.

## Citing

Fatiando is research software. If you use it in your research,
please **cite it** in your publications as:

    Uieda, L, Oliveira Jr, V C, Ferreira, A, Santos, H B; Caparica Jr, J F (2014),
    Fatiando a Terra: a Python package for modeling and inversion in geophysics.
    figshare. doi: 10.6084/m9.figshare.1115194

Some of the methods implemented here are also **original research** by some of
the developers. Please **also cite the method papers**.
See the
[CITATION.rst](https://github.com/fatiando/fatiando/blob/master/CITATION.rst)
file for more information.

Read [this blog post by Robin Wilson](http://www.software.ac.uk/blog/2013-09-02-encouraging-citation-software-introducing-citation-files)
if you haven't heard of CITATION files.

## Documentation

The latest documentation is available at ReadTheDocs. The docs reflects the
*master* branch on Github.

http://fatiando.readthedocs.org


License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the **BSD License**. A copy of this license is provided in
`LICENSE.txt`.
