.. _install:

Installing Fatiando
===================

.. note:: If you have any trouble installing please write to the
    `mailing list`_ or to `Leonardo Uieda`_. This will help us make
    Fatiando better!

.. _mailing list: https://groups.google.com/forum/#!forum/fatiando
.. _Leonardo Uieda: http://fatiando.org/people/uieda/


Python version
--------------

Fatiando requires `Python 2.7x`_.

.. _Python 2.7x: http://www.python.org/download/


Install the dependencies
------------------------

Fatiando requires the following packages:

* `Numpy <http://numpy.scipy.org/>`_
* `Scipy <http://scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net/>`_
* `PIL <http://www.pythonware.com/products/pil/>`_
* `Mayavi <http://code.enthought.com/projects/mayavi/>`_
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

* Numpy
* Scipy
* matplotlib
* PIL
* ETS (for Mayavi)
* VTK (for Mayavi)

.. _PythonXY: http://code.google.com/p/pythonxy/

Some times, **Mac OS X** install can be a challenge.
I recommend downloading the full `Enthought Python Distribution (EPD)`_, 
it will provide Python, Numpy, Scipy, matplotlib, PIL, Mayavi, 
Cython and many others 3rd-party libraries. 

.. _Enthought Python Distribution (EPD): https://www.enthought.com/


In the same way, the new EPD's product, Canopy_, includes many Python packages, 
but unfortunately, Maiavi and Cython are not present in the free_ version. 
If possible, I recommend to request the `academic license`_.

.. _Canopy: https://www.enthought.com/products/canopy/
.. _free: https://www.enthought.com/products/epd/free/
.. _academic license: https://www.enthought.com/products/canopy/academic/

**Warning**: There are a problem_ with VTK and consequently Mayavi on 
Mac OS X and unresolved problems with graphics that depends which Mac 
EPD installations (`32 or 64 bit`_) has been using. In order to preserve 
uniformity between 32 or 64 bit installations of EPD, I recommend to use 
**PyQt** instead of **wxPython**. To this, the environment variable ``ETS_TOOLKIT`` 
has to be set to ``qt4`` in your ``.profile`` or ``.bash_profile``::

    export ETS_TOOLKIT = qt4

If for some reason you prefer install the dependencies by hand, 
you can search for binary installers or try to install using homebrew_ or macports_.

.. _problem: https://support.enthought.com/entries/23419888-VTK-Mayavi-on-Mac-OS-X
.. _32 or 64 bit: https://support.enthought.com/entries/22961778-EPD-for-Mac-OS-X-32-and-64-bit-versions
.. _homebrew: http://mxcl.github.io/homebrew/
.. _macports: http://www.macports.org/

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

Installing on Mac OS X
----------------------

Theoretically, the installation on a Mac OS X is equal that on Linux (see above).
However, if you are using EPD or “other” virtualenv_, will not be possible to install Fatiando using ``pip``. The easiest way is to download Fatiando from its repository_ and run::

    make build

.. _repository: https://github.com/leouieda/fatiando

Do not forget to set the ``PYTHONPATH`` variable in your ``.profile`` or ``.bash_profile``::

    export PYTHONPATH=/Path_to_your_directory/fatiando:$PYTHONPATH

where ``Path_to_your_directory`` is the complete path to the directory `fatiando`. Finally, to test the install use::

    make test

Testing the install
-------------------

Try running one of the recipes from the :ref:`Cookbook <cookbook>`.
If you get an error message,
please post to the `mailing list`_.



