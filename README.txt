================
Fatiando a Terra
================


Fatiando a Terra is an API for geophysical modeling and inversion.

For more information, documentation, download, and installing, visit the
`official site <http://www.fatiando.org/>`_

Downloading and Installing
--------------------------

Fatiando requires the following packages:

* numpy
* scipy
* matplotlib
* PIL
* mayavi

All of these can be found on most GNU/Linux distros. On Windows, we recommend
downloading `PythonXY <http://code.google.com/p/pythonxy/>`_. It comes with
Python, all of our dependencies, plus a whole bunch of useful stuff! Trust me,
it's better than installing things separately.
You will also need the Python header files
(called python-dev in Debian and Ubuntu)
and a C compiler (GCC comes with almost every distro).

After you've installed the dependencies
you can proceed to install Fatiando using pip
(remember to install pip before if don't have it)::

    pip install fatiando

That's it!
Alternatively, you can run the setup.py script::

    python setup.py install


Testing the build
-----------------

Fatiando uses `nose <https://github.com/nose-devs/nose>`_ to run the unit tests
and doctests. To run the tests, install nose, go to the directory with the
``fatiando`` package and run::

    nosetests fatiando


The team
--------

Fatiando is being developed by a group of geophysics graduates from the
Universidade de Sao Paulo and the Observatorio Nacional in Brazil. Work done
here is part of some Masters and Phd projects.

See a list of `people involved <http://www.fatiando.org/people>`_.

License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the BSD License.
A copy of this license is provided in file LICENSE.txt

