.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please write to the
    `mailing list`_ or to `Leonardo Uieda`_. This will help us make
    Fatiando better!

.. _mailing list: https://groups.google.com/forum/#!forum/fatiando
.. _Leonardo Uieda: http://fatiando.org/people/uieda/

Dependencies
------------

Fatiando requires the following packages:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_
* `PIL <http://www.pythonware.com/products/pil/>`_

And these are optional (though recommended):

* `mayavi <http://code.enthought.com/projects/mayavi/>`_: for 3D plotting
* `Cython <http://cython.org/>`_: to compile faster modules in C. Needed only
  when installing from source (or using ``pip``).

All of these can be found on most **GNU/Linux** distros.
If you're on **Ubuntu**, you can run::

    sudo apt-get install python-numpy python-matplotlib python-scipy \
    python-imaging

And if you want the optional packages as well
(and you should!)::

    sudo apt-get install mayavi2 cython python-dev

You'll need ``python-dev`` to build the optimized Cython modules.
All of these can be found in the `Software Center`_.

.. note:: The '2' in ``mayavi2`` is not a typo. It really is called that.

On **Windows**, we recommend downloading PythonXY_.
It comes with Python, all of our dependencies,
plus a whole bunch of useful stuff!
Trust me, it's better than installing things separately.
If you already have Python installed,
you should uninstall it before installing PythonXY.
When installing PythonXY,
make sure the following are selected:

* numpy
* scipy
* matplotlib
* PIL
* ETS (for mayavi)
* VTK (for mayavi)

.. _PythonXY: http://code.google.com/p/pythonxy/
.. _Software Center: http://www.ubuntu.com/ubuntu/features/ubuntu-software-centre

Installing from PyPI
--------------------

.. note:: This is the preffered method of installing in **GNU/Linux**.


After you've installed the dependencies you can proceed to install Fatiando
using pip_.
Make sure you have pip installed!
To install it on Ubuntu (or Debian)::

    sudo apt-get install python-pip

When you have pip installed, simply run::

    sudo pip install fatiando

That's it! If you already have Fatiando installed and want to upgrade to a newer
version, use::

    sudo pip install fatiando --upgrade

To uninstall simply run::

    sudo pip uninstall fatiando

If you don't have root access (no ``sudo`` for you),
you can install Fatiando on a virtualenv_.
If you don't know what that means,
`read this`_.

Alternatively, you can download a source distribution from PyPI_,
unpack it, and run the ``setup.py`` script::

    python setup.py install

.. note:: **Using pip is the preferred option** since it's the most modern way
    (see `the packaging guide`_ for more information about this). Using
    ``setup.py`` doesn't give you an uninstall option.

.. _pip: http://www.pip-installer.org
.. _PyPI: http://pypi.python.org/pypi/fatiando
.. _the packaging guide: http://guide.python-distribute.org/index.html
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _read this: http://jontourage.com/2011/02/09/virtualenv-pip-basics/

Using the Windows installer
---------------------------

On **Windows**, make sure you installed PythonXY and all the dependencies.
After that, you can just click through the installer.
It should be straight forward.

