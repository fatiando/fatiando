================
Fatiando a Terra
================


Fatiando a Terra provides an API for a variety of methods of geophysical
modeling and inversion.

For more information, documentation, download, and installing, visit the
`official site <http://www.fatiando.org>`_

Dependencies
------------

Fatiando requires the following packages:

* numpy
* scipy
* matplotlib
* mayavi
* PIL
* nose (for running the tests)

All of these can be found on most GNU/Linux distros. On Windows, we recommend
downloading `PythonXY <http://code.google.com/p/pythonxy/>`_. It comes with
Python, all of our dependencies, plus a whole bunch of useful stuff! Trust me,
it's better than installing things separately.

If compiling from source, you'll also need:

* Python development files (usually called ``python-dev`` on GNU/Linux)
* A C compiler

If building in Windows using the MinGW compiler, see
`this post <http://boodebr.org/main/python/build-windows-extensions>`_ (and good
luck).


Installing
-----------

At the moment, there are few options for installing Fatiando from source on
GNU/Linux (sorry Windows users, we're working on it):

0 - For all options, download the source from `http://www.fatiando.org`_
1 - Using ``pip``:

    * Install `pip <http://pypi.python.org/pypi/pip>`_
    * Run::
    
        pip install fatiando-X.X.X.tar.gz
      
    * To uninstall, run::

        pip uninstall fatiando
        
2 - Using the setup.py script:

    * Unpack the .tar.gz file anywhere (say, ``~/src/fatiando``)
    * Go to where you unpacked it and run::

        python setup.py install

.. note:: that using ``pip`` is the preferred option since it's the more modern
way (see `The Guide <http://guide.python-distribute.org/index.html>`_ for more
information about this). Also, using setup.py doesn't give you an uninstall
option.


The team
--------

Fatiando is being developed by a group of geophysics graduates from the
Universidade de Sao Paulo and the Observatorio Nacional in Brazil. Work done
here is part of some Masters and Phd projects.

See a list of `people involved <http://www.fatiando.org/people>`_.


License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
A copy of this license is provided in file LICENSE.txt


Testing
--------

Fatiando uses ``nose`` to run the unit test suite as well as the doctests. To
run the tests, go to the directory with the ``fatiando`` package and run::

    nosetests
