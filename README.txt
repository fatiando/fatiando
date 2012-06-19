================
Fatiando a Terra
================

*Fatiando a Terra* is an **open-source** Python package for geophysical
**modeling and inversion**.

For more **information** visit http://www.fatiando.org/software/fatiando

The most recent **documentation** is available at
http://fatiando.readthedocs.org

Here is a quick example of using Fatiando to generate synthetic gravity data
on random points, contaminate it with gaussian noise, and plot it::

    import fatiando as ft
    # Create the prism model
    prisms = [
        ft.msh.ddd.Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density':1000}),
        ft.msh.ddd.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density':-1000}),
        ft.msh.ddd.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density':1000})]
    # Generate 500 random observation points at 100m height
    xp, yp, zp = ft.grd.scatter((-5000, 5000, -5000, 5000), 500, z=-100)
    # Calculate their gravitational effect and contaminate it with 0.1 mGal
    gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.1)
    # Plot the result
    ft.vis.contourf(xp, yp, gz, (100, 100), 12, interp=True)
    ft.vis.colorbar()
    ft.vis.plot(xp, yp, '.k')
    ft.vis.show()

Source code
-----------

The source code of Fatiando is hosted on several online repositories:

* `fatiando on Bitbucket`_ with the stable version (latest release)
* `fatiando-dev on Bitbucket`_ with the development version (this is where
  development happens)
* `fatiando on GoogleCode`_ with a mirror of the stable version

.. _fatiando on Bitbucket: https://bitbucket.org/fatiando/fatiando
.. _fatiando-dev on Bitbucket: https://bitbucket.org/fatiando/fatiando-dev
.. _fatiando on GoogleCode: http://code.google.com/p/fatiando/

Downloading and Installing
--------------------------

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

.. _PythonXY: http://code.google.com/p/pythonxy/
.. _pip: http://www.pip-installer.org
.. _PyPI: http://pypi.python.org/pypi/fatiando
.. _the packaging guide: http://guide.python-distribute.org/index.html

The authors
-----------

Fatiando is developed by working (or studying) geophysicists. Work done here is
part of some Masters and Phd projects. See a list of `people involved`_.

.. _people involved: http://readthedocs.org/docs/fatiando/en/latest/contributors.html

License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the BSD License. A copy of this license is provided in file
LICENSE.txt and at http://readthedocs.org/docs/fatiando/en/latest/license.html

