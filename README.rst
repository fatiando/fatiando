================
Fatiando a Terra
================

*Fatiando a Terra* is an **open-source** Python package for geophysical
**modeling and inversion**.

For more **information** visit http://www.fatiando.org

Here is a quick example of using Fatiando to generate synthetic gravity data
on random points, contaminate it with gaussian noise, and plot it::

    >>> import fatiando as ft
    >>> # Create the prism model
    >>> prisms = [
    ...     ft.mesher.Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density':1000}),
    ...     ft.mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density':-1000}),
    ...     ft.mesher.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density':1000})]
    >>> # Generate 500 random observation points at 100m height
    >>> xp, yp, zp = ft.gridder.scatter((-5000, 5000, -5000, 5000), 500, z=-100)
    >>> # Calculate their gravitational effect and contaminate it with 0.1 mGal
    >>> # gaussian noise
    >>> gz = ft.utils.contaminate(ft.gravmag.prism.gz(xp, yp, zp, prisms), 0.1)
    >>> # Plot the result
    >>> ft.vis.contourf(xp, yp, gz, (100, 100), 12, interp=True)
    >>> cb = ft.vis.colorbar()
    >>> cb.set_label('mGal')
    >>> ft.vis.plot(xp, yp, '.k')
    >>> ft.vis.show()

Documentation
-------------

The latest documentation is available at ReadTheDocs:

http://fatiando.readthedocs.org

The documentation reflects the *master* branch on Github_.


Source code
-----------

The source code of Fatiando is hosted on several online repositories:

* Github_: stable version and where development takes place
* Bitbucket_: stable version (latest release)
* GoogleCode_: mirror of the stable version

.. _Github: https://github.com/leouieda/fatiando
.. _Bitbucket: https://bitbucket.org/fatiando/fatiando
.. _GoogleCode: http://code.google.com/p/fatiando/

The authors
-----------

Fatiando is developed by working geophysicists. Work done here is
part of some Masters and Phd projects. See a list of `people involved`_.

.. _people involved: http://readthedocs.org/docs/fatiando/en/latest/contributors.html

License
-------

Fatiando a Terra is free software: you can redistribute it and/or modify it
under the terms of the **BSD License**. A copy of this license is provided in
LICENSE.txt and at http://readthedocs.org/docs/fatiando/en/latest/license.html

