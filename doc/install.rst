.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please write to the
    `mailing list`_ or to `Leonardo Uieda`_. This will help us make
    Fatiando better!

.. _mailing list: https://groups.google.com/forum/#!forum/fatiando
.. _Leonardo Uieda: http://fatiando.org/people/uieda/

Install the dependencies
------------------------

Fatiando requires the following packages:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_
* `PIL <http://www.pythonware.com/products/pil/>`_
* `mayavi <http://code.enthought.com/projects/mayavi/>`_
* `Cython <http://cython.org/>`_: to compile faster modules in C. Needed only
  when installing from source (or using ``pip``).

All of these can be found on most **GNU/Linux** distros.
If you're on the latest (or close to) **Ubuntu**, you can run::

    sudo apt-get install python-dev python-numpy python-matplotlib python-scipy\
    python-mpltoolkits.basemap python-imaging mayavi2 cython python-pip

You'll need ``python-dev`` to build the optimized Cython modules and
``python-pip`` to install fatiando.
All of these can also be found in the Software Center, Synaptic, etc.

.. note:: The '2' in ``mayavi2`` is not a typo. It really is called that.

On **Windows**, I recommend downloading PythonXY_.
It comes with Python, all of our dependencies,
plus a whole bunch of useful stuff!
Trust me, it's better than installing things separately.
**Warning**: If you already have Python installed,
you should uninstall it before installing PythonXY.
When installing PythonXY,
make sure the following are selected
(or go with a full install to be sure):

* numpy
* scipy
* matplotlib
* PIL
* ETS (for mayavi)
* VTK (for mayavi)

.. _PythonXY: http://code.google.com/p/pythonxy/

Installing on Linux
-------------------

After you've installed the dependencies you can proceed to install Fatiando
using pip_::

    sudo pip install fatiando

That's it!
If you already have Fatiando installed and want to **upgrade** to a newer
version, use::

    sudo pip install fatiando --upgrade

To uninstall simply run::

    sudo pip uninstall fatiando

If you don't have root access (no ``sudo`` for you),
you can install Fatiando on a virtualenv_.
If you don't know what that means,
`read this`_.

.. _pip: http://www.pip-installer.org
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _read this: http://jontourage.com/2011/02/09/virtualenv-pip-basics/

Installing on Windows
---------------------

After you've installed PythonXY (or similar)
with all the dependencies,
download the latest Windows installer from PyPI_.
Just click through the installer and you should be done!

.. _PyPI: http://pypi.python.org/pypi/fatiando

Testing the install
-------------------

Try running one of the recipes from the :ref:`Cookbook <cookbook>`.
If you get an error message,
please post to the `mailing list`_.



