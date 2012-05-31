Fatiando a Terra: Geophysical modeling and inversion
====================================================

.. topic:: An open source toolkit for geophysical modeling and inversion

    Fatiando provides an easy and flexible way to perform common tasks like:
    generating synthetic data, forward modeling, inversion, 3D visualization,
    and more! All from inside the powerfull Python_ language.

For more information visit the http://www.fatiando.org

The source code of Fatiando is hosted on several online repositories:

* `fatiando on Bitbucket`_ with the stable version (latest release)
* `fatiando-dev on Bitbucket`_ with the development version (this is where
  development happens)
* `fatiando on GoogleCode`_ with a mirror of the stable version
  
.. _fatiando on Bitbucket: https://bitbucket.org/fatiando/fatiando
.. _fatiando-dev on Bitbucket: https://bitbucket.org/fatiando/fatiando-dev
.. _fatiando on GoogleCode: http://code.google.com/p/fatiando/

**License**: Fatiando is licensed under the **BSD license**.
This means that it can be reused and remixed
with few restrictions.
See the :ref:`license text <license>` for more information.

The main interface for Fatiando
is provided by a Python_ package called ``fatiando``.
This means that commands in Fatiando
are executed using Python scripts,
instead of the command line or shell scripts.
The advantage of this is that
Python is a very feature rich and powerful programming language.
So, you can easily combine different commands together
and even extend the functionality of Fatiando with your own code!

.. _Python: http://www.python.org

As an example, this is how easy it is to create synthetic noise-corrupted
gravity data on random points from a 3D prism model:

.. doctest::

    >>> import fatiando as ft
    >>> # Create the prism model
    >>> prisms = [
    ...     ft.msh.ddd.Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density':1000}),
    ...     ft.msh.ddd.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density':-1000}),
    ...     ft.msh.ddd.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density':1000})]
    >>> # Generate 500 random observation points at 100m height
    >>> xp, yp, zp = ft.grd.scatter((-5000, 5000, -5000, 5000), 500, z=-100)
    >>> # Calculate their gravitational effect and contaminate it with 0.1 mGal
    >>> # gaussian noise
    >>> gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.1)
    >>> # Plot the result
    >>> ft.vis.contourf(xp, yp, gz, (100, 100), 12, interp=True)
    >>> cb = ft.vis.colorbar()
    >>> cb.set_label('mGal')
    >>> ft.vis.plot(xp, yp, '.k')
    >>> ft.vis.show()

which results in something like this:

.. image:: _static/sample.png
    :align: center

If you want to find out more,
have a look at the rest of this documentation.

.. raw:: html

    <h1>Contents:</h1>

.. toctree::
    :maxdepth: 3

    overview.rst
    contributors.rst
    license.rst
    changelog.rst
    install.rst
    using.rst
    tutorial.rst
    cookbook.rst
    api/fatiando.rst


