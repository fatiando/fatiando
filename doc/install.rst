.. _install:

Installing Fatiando
===================

Bellow you'll find instructions on
how to install Fatiando and
how to compile it from source.

Dependencies
------------

Fatiando requires the following packages:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_
* `PIL <http://www.pythonware.com/products/pil/>`_

Optional packages:

* `mayavi <http://code.enthought.com/projects/mayavi/>`_ (for 3D plots)
* `Cython <http://cython.org/>`_ (to build faster replacements to a few modules)

All of these can be found on most GNU/Linux distros. On Windows, we recommend
downloading PythonXY_. It comes with Python, all of our dependencies, plus a
whole bunch of useful stuff! Trust me, it's better than installing things
separately.

.. _PythonXY: http://code.google.com/p/pythonxy/

Installing from PyPI
--------------------

After you've installed the dependencies you can proceed to install Fatiando
using pip_ (remember to install pip before if don't have it)::

    pip install fatiando

That's it! If you already have Fatiando installed and want to upgrade to a newer
version, use::

    pip install fatiando --upgrade

To uninstall simply run::

    pip uninstall fatiando

Alternatively, you can download a source distribution from PyPI_,
unpack it, and run the ``setup.py`` script::

    python setup.py install

.. note:: **Using pip is the preferred option** since it's the most modern way
    (see `the packaging guide`_ for more information about this). Using
    ``setup.py`` doesn't give you an uninstall option.

.. _pip: http://www.pip-installer.org
.. _PyPI: http://pypi.python.org/pypi/fatiando
.. _the packaging guide: http://guide.python-distribute.org/index.html
