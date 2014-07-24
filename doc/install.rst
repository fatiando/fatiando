.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please
    `submit a bug report on Github`_
    or write to the `mailing list`_.

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
Unfortunately, my time is quite limited at the moment and I can't work as much
as I'd like on Fatiando.
For the time being, I choose to spend my time improving the functionality of
Fatiando first.
In the future, the need to support both versions might arise and, hopefully,
I'll have help to do that when the time comes.

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

Extra dependencies for Windows users
++++++++++++++++++++++++++++++++++++

Unfortunately, the ``gcc`` compiler included in Anaconda MinGW_
doesn't have OpenMP_ support. This is required to compile
some extension modules in Fatiando that have multi-threaded parallel code.

You'll have download and install TDM-GCC_
**after** you've installed Anaconda and **before** you install Fatiando.
Don't forget to mark the ``openmp`` option in the "Choose Components" part of
the installation. See this `excellent documentation for Windows users`_
(they even have screenshots!).

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

To install the latest version from Github::

    pip install --upgrade https://github.com/leouieda/fatiando/archive/master.zip

or if you have git installed and want to see the code::

    git clone https://github.com/leouieda/fatiando.git
    cd fatiando
    pip install --upgrade .

.. note::

    ``fatiando.__version__`` has the current version number. If you install
    from PyPI, this will be something like ``'0.2'``. If you installed from
    Github, this will be the latest commit hash. This way you can track exactly
    what version of Fatiando generated your results.


Testing the install
-------------------

Try running one of the recipes from the :ref:`Cookbook <cookbook>`.
If you get an error message or weird result,
please write to the `mailing list`_.
To make it easier for us to debug you problem, please include the following
information:

* Operating system
* Version of Fatiando you installed
* Python distribution (Anaconda_, PythonXY_, `ETS/Canopy`_, own install)
* Python version (2.6, 2.7, 3.3, 3.4, etc)
* The script you ran (and gave you an error/weird result)
* The error message (the part that says ``Traceback: ...``) or result (figure,
  numbers, etc)

.. _submit a bug report on Github: https://github.com/leouieda/fatiando/issues
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
.. _OpenMP: http://openmp.org/
.. _TDM-GCC: http://tdm-gcc.tdragon.net/
.. _excellent documentation for Windows users: http://docs-windows.readthedocs.org/en/latest/devel.html#mingw-with-openmp-support
