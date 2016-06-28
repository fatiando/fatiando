.. image:: https://raw.githubusercontent.com/fatiando/logo/master/fatiando-banner-homepage.png
    :alt: Fatiando a Terra
    :target: http://www.fatiando.org

----

**Website and documentation**: `http://www.fatiando.org <http://www.fatiando.org>`__

**Development documentation**: `http://www.fatiando.org/dev <http://www.fatiando.org/dev>`__

**Mailing list (get updates and ask for help)**: `https://groups.google.com/d/forum/fatiando <https://groups.google.com/d/forum/fatiando>`__


.. image:: http://img.shields.io/pypi/v/fatiando.svg?style=flat-square
    :alt: Latest PyPI version
    :target: https://pypi.python.org/pypi/fatiando/
.. image:: http://img.shields.io/travis/fatiando/fatiando/master.svg?style=flat-square
    :alt: Travis CI build status
    :target: https://travis-ci.org/fatiando/fatiando
.. image:: http://img.shields.io/coveralls/fatiando/fatiando/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://coveralls.io/r/fatiando/fatiando?branch=master
.. image:: http://img.shields.io/badge/doi-10.5281/zenodo.49087-blue.svg?style=flat-square
    :alt: doi:10.5281/zenodo.49087
    :target: http://dx.doi.org/10.5281/zenodo.49087


Overview
--------

Our goal is provide a comprehensive and extensible framework
for geophysical data analysis and the development of new methodologies.

**Research:** Fatiando allows you to write Python scripts to
perform your data analysis and generate figures in a **reproducible** way.

**Development:** Designed for extensibility, Fatiando offers tools for users to
build upon the existing infrastructure and **develop new methods**.
We take care of the boilerplate.

**Teaching:** Fatiando can be combined with the `Jupyter notebook
<https://jupyter.org/>`__ to make rich, **interactive documents**. Great for
teaching fundamental concepts of geophysics.


Getting started
---------------

Take a look at the `Documentation <http://www.fatiando.org>`__ for a
detailed tour of the library and install instructions.  You can also browse the
`Gallery
<http://www.fatiando.org/dev/gallery/index.html>`__
and
`Cookbook
<http://www.fatiando.org/cookbook.html>`__ for examples of what Fatiando can
do.


Participating and asking for help
---------------------------------

**Subscribe** to our Google Groups mailing list to stay informed and ask for
help:

`groups.google.com/d/forum/fatiando <https://groups.google.com/d/forum/fatiando>`__

We'll post updates about new releases and features, events, and future plans
for the project. **Get involved** to help us shape the project and make it even
better!

Another option for reaching out and reporting bugs is to
`open an issue on Github <https://github.com/fatiando/fatiando/issues>`__.


Dependencies
------------

For the moment, Fatiando runs and is tested in **Python 2.7**.
To install and run Fatiando, you'll need the following Python packages:
``numpy``, ``scipy``, ``matplotlib``, ``jupyter``, ``pillow``,
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

See the `Installing documentation <http://www.fatiando.org/install.html>`__ for
detailed instructions.


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


License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the **BSD 3-clause License**. A copy of this license is
provided in `LICENSE.txt`.
