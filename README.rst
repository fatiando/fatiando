|title|
=======

.. |title| image:: https://raw.githubusercontent.com/fatiando/logo/master/fatiando-banner-long.png
    :alt: Fatiando a Terra

`Website <http://www.fatiando.org>`__ |
`Docs <http://fatiando.github.io/docs.html>`__ |
`Mailing list <https://groups.google.com/d/forum/fatiando>`__ |
`Google+ <https://plus.google.com/+FatiandoOrg>`__

A Python package for modeling and inversion in geophysics.

.. image:: http://img.shields.io/pypi/v/fatiando.svg?style=flat
    :alt: Latest PyPI version
    :target: https://crate.io/packages/fatiando
.. image:: http://img.shields.io/pypi/dm/fatiando.svg?style=flat
    :alt: Number of PyPI downloads
    :target:  https://crate.io/packages/fatiando/
.. image:: http://img.shields.io/badge/license-BSD-lightgrey.svg?style=flat
    :alt: BSD 3 clause license
    :target: https://github.com/fatiando/fatiando/blob/master/LICENSE.txt
.. image:: http://img.shields.io/travis/fatiando/fatiando/master.svg?style=flat
    :alt: Travis CI build status
    :target: https://travis-ci.org/fatiando/fatiando
.. image:: http://img.shields.io/coveralls/fatiando/fatiando/master.svg?style=flat
    :alt: Test coverage status
    :target: https://coveralls.io/r/fatiando/fatiando?branch=master
.. image:: http://img.shields.io/badge/doi-10.6084/m9.figshare.1115194-blue.svg?style=flat
    :alt: doi:10.6084/m9.figshare.1115194
    :target: http://dx.doi.org/10.6084/m9.figshare.1115194

Dependencies
------------

For the moment, Fatiando runs and is tested in Python 2.7.
To install and run Fatiando, you'll need the following packages:

* numpy >= 1.8
* scipy >= 0.14
* matplotlib >= 1.3
* IPython >= 2.0.0
* mayavi >= 4.3
* PIL >= 1.1.7
* basemap >= 1.0.7
* gcc >= 4.8.2
* numba >= 0.17

You can get all of these on Linux, Mac, and Windows through
the `Anaconda distribution <http://continuum.io/downloads>`__.
See file ``requirements.txt``.

**Windows users**: See the
`install docs <http://fatiando.github.io/install.html#extra-dependencies-for-windows-users>`__
for special instructions on getting the dependencies installed.

Installing
----------

Download and install the latest release of Fatiando from
`PyPI <https://pypi.python.org/pypi/fatiando>`__::

    pip install fatiando

or get the latest development version from Github::

    pip install --upgrade https://github.com/fatiando/fatiando/archive/master.zip

**Note**: ``fatiando.__version__`` has the current version number. If you install
from PyPI, this will be something like ``'0.2'``. If you installed from Github,
this will be the latest commit hash. This way you can track exactly what
version of Fatiando generated your results.

Citing
------

Fatiando is research software. If you use it in your research,
please **cite it** in your publications as::

    Uieda, L, Oliveira Jr, V C, Ferreira, A, Santos, H B; Caparica Jr, J F (2014),
    Fatiando a Terra: a Python package for modeling and inversion in geophysics.
    figshare. doi: 10.6084/m9.figshare.1115194

Some of the methods implemented here are also **original research** by some of
the developers. Please **also cite the method papers**.
References are available in the documentation of each module.
See the
`CITATION.rst <https://github.com/fatiando/fatiando/blob/master/CITATION.rst>`__
file or the `documentation <http://fatiando.github.io/cite.html>`__
for more information.

Read `this blog post by Robin Wilson
<http://www.software.ac.uk/blog/2013-09-02-encouraging-citation-software-introducing-citation-files>`__
if you haven't heard of CITATION files.


License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the **BSD License**. A copy of this license is provided in
`LICENSE.txt`.
