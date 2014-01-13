.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please write to the
    `mailing list`_ or to `Leonardo Uieda`_. This will help us make
    Fatiando better!

Installing the dependencies
---------------------------

Fatiando requires the following packages:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_
* `PIL <http://www.pythonware.com/products/pil/>`_
* `mayavi <http://code.enthought.com/projects/mayavi/>`_
* A C compiler (preferably GCC or MinGW_ on Windows)

The easiest and **preferred** way to get all dependencies in the latest
version is using the Anaconda_ Python distribution by `Continuum Analytics`_.
It does not require administrative rights to your computer and doesn't
interfere with the Python installed in your system.
For Windows users, it even comes with MinGW_ so you don't have to worry about
the many, many, many issues of compiling under Windows.

Once you have downloaded and installed Anaconda_,
open a terminal (or ``cmd.exe`` on Windows) and run::

    conda install numpy scipy matplotlib basemap imaging mayavi pip

And you're done!

Installing Fatiando
-------------------

After you've installed the dependencies you can proceed to install Fatiando
using pip_.
Open a terminal (or ``cmd.exe`` on Windows) and run::

    pip install fatiando

and that's it!

If you already have Fatiando installed and want to **upgrade** to a newer
version, use::

    pip install fatiando --upgrade

To uninstall simply run::

    pip uninstall fatiando


.. note::

    The Windows installer from older versions is no longer supported.


Testing the install
-------------------

Try running one of the recipes from the :ref:`Cookbook <cookbook>`.
If you get an error message or weird result,
please post to the `mailing list`_.


.. _pip: http://www.pip-installer.org
.. _MinGW: http://www.mingw.org/
.. _mailing list: https://groups.google.com/forum/#!forum/fatiando
.. _Leonardo Uieda: http://fatiando.org/people/uieda/
.. _Continuum Analytics: http://continuum.io/
.. _Anaconda: http://continuum.io/downloads
