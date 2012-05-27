================
Fatiando a Terra
================


Fatiando a Terra is an open-source Python package for geophysical modeling and
inversion.

For more information visit the http://www.fatiando.org

Documentation is available at http://www.fatiando.readthedocs.org

Downloading and Installing
--------------------------

Fatiando requires the following packages:

* numpy
* scipy
* matplotlib
* PIL

Optional packages:

* mayavi (for 3D plots)

All of these can be found on most GNU/Linux distros. On Windows, we recommend
downloading PythonXY_. It comes with Python, all of our dependencies, plus a
whole bunch of useful stuff! Trust me, it's better than installing things
separately. You will also need the Python header files (called ``python-dev``
in Debian and Ubuntu) and a C compiler (GCC comes with almost every distro).

After you've installed the dependencies you can proceed to install Fatiando
using pip_ (remember to install pip before if don't have it)::

    pip install fatiando

That's it! Alternatively, you can download a source distribution from PyPI_,
unpack it, and run the ``setup.py`` script::

    python setup.py install


.. _PythonXY: http://code.google.com/p/pythonxy/
.. _pip: http://www.pip-installer.org
.. _PyPI: http://pypi.python.org/pypi/fatiando

The authors
-----------

Fatiando is developed by working geophysicists. Work done here is part of some
Masters and Phd projects. See a list of `people involved`_.

.. _people involved: http://readthedocs.org/docs/fatiando/en/latest/contributors.html

License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the BSD License. A copy of this license is provided in file
LICENSE.txt and at http://readthedocs.org/docs/fatiando/en/latest/license.html

