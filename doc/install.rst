.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please write to the
    `mailing list`_ or to `Leonardo Uieda`_. This will help us make
    Fatiando better!

Which Python?
-------------

There are many versions of the Python_ language in
`use today <https://wiki.python.org/moin/Python2orPython3>`__.
The main ones are Python 2.7 and Python 3.x.
Most, if not all, of the scientific Python packages that Fatiando relies on
support both versions of the language.
However, while it is possible (and not that difficult) to
`support both versions simultaneously
<http://docs.python.org/3.4/howto/pyporting.html>`__,
it does take work.
And work takes time.
And my time is quite limited at the moment.
For the time being, I choose to spend my time improving the functionality of
Fatiando.
In the future, the need to support both versions might arise and I hope I have
help to do that when the time comes.

So, for the moment, **Fatiando is tested and works on Python 2.7**.

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

Installing the latest development version
-----------------------------------------

If you want the very latest code and features,
you can install Fatiando directly from Github_.
We try to maintain the *master* branch stable and
`passing all tests <https://travis-ci.org/leouieda/fatiando/branches>`__,
so it should be safe to use.

First, you'll need to `install git`_.
Then, open a terminal and run::

    git clone --depth=50 --branch=master https://github.com/leouieda/fatiando.git

This will fetch the source code from Github_
and place it in a folder called ``fatiando`` in the directory where you ran the
command.
Then, just ``cd`` into the directory and run ``pip``::

    cd fatiando
    pip install --upgrade .


Testing the install
-------------------

Try running one of the recipes from the :ref:`Cookbook <cookbook>`.
If you get an error message or weird result,
please write to the `mailing list`_.
To make it easier for us to debug you problem, please include the following
information:

* Operating system
* Python distribution (Anaconda_, PythonXY_, `ETS/Canopy`_, own install)
* Python version (2.6, 2.7, 3.3, 3.4, etc)
* The script you ran (and gave you an error/weird result)
* The error message (the part that says ``Traceback: ...``) or result (figure,
  numbers, etc)


.. _install git: http://git-scm.com/
.. _Github: https://github.com/leouieda/fatiando
.. _Python: http://www.python.org/
.. _pip: http://www.pip-installer.org
.. _MinGW: http://www.mingw.org/
.. _mailing list: https://groups.google.com/forum/#!forum/fatiando
.. _Leonardo Uieda: http://fatiando.org/people/uieda/
.. _Continuum Analytics: http://continuum.io/
.. _Anaconda: http://continuum.io/downloads
.. _PythonXY: http://code.google.com/p/pythonxy/
.. _ETS/Canopy: http://code.enthought.com/projects/index.php
