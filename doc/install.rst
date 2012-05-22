.. _install:

Installing Fatiando
===================

Bellow you'll find instructions on
how to install Fatiando and
how to compile it from source.

Dependencies
------------

Fatiando requires the following packages:

* numpy
* PIL
* scipy
* matplotlib

Optional:

* mayavi (for 3D visualizations)

All of these can be found on the repositories of most GNU/Linux distros.
On Windows,
we recommend downloading PythonXY_.
It comes with Python,
all of our dependencies,
plus a whole bunch of useful stuff!
Trust me,
it's better than installing things separately.

If compiling from source, you'll also need:

* Python development files (usually called ``python-dev`` on GNU/Linux)
* A C compiler (I recommend GCC)

If building in Windows using the MinGW_ compiler,
see `this post`_ (and good luck).

After you've installed all dependencies,
you can proceed to installing Fatiando.


.. _PythonXY: http://code.google.com/p/pythonxy/
.. _MinGW: http://mingw.org/
.. _this post: http://boodebr.org/main/python/build-windows-extensions


Installing with pip
-------------------

At the moment,
the easiest (only) option is building Fatiando from source.
GNU/Linux users should use pip_ to avoid headaches.
Windows users can try using the ``setup.py`` script.

Just follow these steps:

1. Download the source from `the official site`_
2. Using ``pip``:

    * Install pip_
    * Run::
    
        pip install fatiando-X.X.X.tar.gz
      
    * To uninstall, run::

        pip uninstall fatiando
        
3. If don't want to install ``pip``, you can use the ``setup.py`` script:

    * Unpack the .tar.gz or .zip file anywhere (say, ``~/src/fatiando``)
    * Go to where you unpacked it and run::

        python setup.py install


.. note:: Using ``pip`` is the preferred option since it's the most modern way
    (see `The Guide`_ for more information about this).
    Using ``setup.py`` doesn't give you an uninstall option.

.. _the official site: http://www.fatiando.org
.. _pip: http://pypi.python.org/pypi/pip
.. _The Guide: http://guide.python-distribute.org/index.html
