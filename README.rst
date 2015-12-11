.. image:: https://raw.githubusercontent.com/fatiando/logo/master/fatiando-banner-long.png
    :alt: Fatiando a Terra

`Website <http://www.fatiando.org>`__ |
`Docs <http://www.fatiando.org/docs.html>`__ |
`Mailing list <https://groups.google.com/d/forum/fatiando>`__

An open-source Python library for modeling and inversion in geophysics.

.. image:: http://img.shields.io/pypi/v/fatiando.svg?style=flat-square
    :alt: Latest PyPI version
    :target: https://crate.io/packages/fatiando
.. image:: http://img.shields.io/pypi/dm/fatiando.svg?style=flat-square
    :alt: Number of PyPI downloads
    :target:  https://crate.io/packages/fatiando/
.. image:: http://img.shields.io/travis/fatiando/fatiando/master.svg?style=flat-square
    :alt: Travis CI build status
    :target: https://travis-ci.org/fatiando/fatiando
.. image:: http://img.shields.io/coveralls/fatiando/fatiando/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://coveralls.io/r/fatiando/fatiando?branch=master
.. image:: https://landscape.io/github/fatiando/fatiando/master/landscape.svg?style=flat-square
   :target: https://landscape.io/github/fatiando/fatiando/master
   :alt: Code Health from landscape.io
.. image:: http://img.shields.io/badge/doi-10.5281/zenodo.16205-blue.svg?style=flat-square
    :alt: doi:10.5281/zenodo.16205
    :target: http://dx.doi.org/10.5281/zenodo.16205
.. image:: http://img.shields.io/badge/GITTER-JOIN_CHAT-brightgreen.svg?style=flat-square
    :alt: gitter chat room at https://gitter.im/fatiando/fatiando
    :target: https://gitter.im/fatiando/fatiando

Overview
--------

Our goal is provide a comprehensive and extensible framework
for geophysical data analysis and the development of new methodologies.

**Research:** Fatiando allows you to write Python scripts to
perform your data analysis and generate figures in a reproducible way.

**Development:** Designed for extensibility, Fatiando offers tools for users to
build upon the existing infrastructure and develop new inversion methods.
We take care of the boilerplate.

**Teaching:** Fatiando can be combined with the `Jupyter notebook
<https://jupyter.org/>`__ to make rich, interactive documents. Great for
teaching fundamental concepts of geophysics.

Getting started
---------------

Take a look at the `Documentation <http://www.fatiando.org/docs.html>`__ for a
detailed tour of the library.  You can also browse the `Cookbook
<http://www.fatiando.org/cookbook.html>`__ for examples of what Fatiando can
do.

Dependencies
------------

For the moment, Fatiando runs and is tested in **Python 2.7**.
To install and run Fatiando, you'll need the following Python packages:
``numpy``, ``scipy``, ``matplotlib``, ``ipython``, ``pillow``,
``basemap``, ``numba``, ``future``, ``mayavi``.
You'll also need a C compiler, preferably ``gcc``.

You can get all of these on Linux, Mac, and Windows through
the `Anaconda distribution <http://continuum.io/downloads>`__.
See file ``requirements.txt``.

Installing
----------

Download and install the latest release of Fatiando from
`PyPI <https://pypi.python.org/pypi/fatiando>`__::

    pip install fatiando

or get the latest development version from Github::

    pip install --upgrade https://github.com/fatiando/fatiando/archive/master.zip

**Note**: ``fatiando.__version__`` has the current version number. If you
install from PyPI, this will be something like ``'0.2'``. If you installed from
Github, this will be the latest commit hash. This way you can track exactly
what version of Fatiando generated your results.

See the `documentation <http://www.fatiando.org/docs.html>`__ for detailed
instructions.

Citing
------

If you use it in your research, please cite Fatiando in your publications as:

    Uieda, L., V. C. Oliveira Jr, and V. C. F. Barbosa (2013), Modeling the
    Earth with Fatiando a Terra, Proceedings of the 12th Python in Science
    Conference, pp. 91 - 98.

Please **also cite the method papers** of individual functions/classes.
References are available in the documentation of each module.

See the `CITATION.rst
<https://github.com/fatiando/fatiando/blob/master/CITATION.rst>`__ file or the
`Citing section <http://www.fatiando.org/cite.html>`__ of the docs for more
information.

Read `this blog post by Robin Wilson
<http://www.software.ac.uk/blog/2013-09-02-encouraging-citation-software-introducing-citation-files>`__
if you haven't heard of CITATION files.

Getting help
------------

Here are a few option to get in touch with us:

* `Open an issue on Github <https://github.com/fatiando/fatiando/issues>`__
* `Write to the mailing list <https://groups.google.com/d/forum/fatiando>`__
* `Ask on the Gitter chat room <https://gitter.im/fatiando/fatiando>`__

License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the **BSD 3-clause License**. A copy of this license is
provided in `LICENSE.txt`.
