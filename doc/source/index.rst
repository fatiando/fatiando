Overview
---------

Fatiando a Terra is an open source toolkit for geophysical modeling and
inversion. Fatiando provides an easy and flexible way to perform common
tasks like: generating synthetic data, manipulating data sets, plotting
non-gridded data, direct modeling, inversion, etc.

The main interface for Fatiando is provided by a
`Python <http://www.python.org>`_ package (:ref:`the API <api>`). This means that
commands in Fatiando are in executed using Python scripts, instead of the
command line or shell scripts. The advantage of this is that Python is a very
feature rich and powerful programming language. So, you can easily extend the
functionality of Fatiando with your own code!

To show off our API, this is how easy it is to create synthetic noise-corrupted
gravity data on random points from a 3D prism model::

    >>> # Get the needed components
    >>> from fatiando import potential, gridder, utils
    >>> from fatiando.mesher.ddd import Prism
    >>> # Create the prism model
    >>> prisms = [Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density':1000}),
    ...           Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density':-1000}),
    ...           Prism(2000, 4000, 3000, 4000, 0, 2000, {'density':1000})]
    >>> # Generate 1000 random observation points at 100m height
    >>> xp, yp, zp = gridder.scatter((-5000, 5000, -5000, 5000), 1000, z=-100)
    >>> # Calculate their gravitational effect and contaminate it with 0.1 mGal
    >>> # gaussian noise
    >>> gz = utils.contaminate(potential.prism.gz(xp, yp, zp, prisms), 0.1)
    
Contents of this documentation:

.. toctree::
    :maxdepth: 1
    
    api/index.rst
    cookbook/index.rst
    the-team.rst

Developing
-----------

:ref:`the-team` behind Fatiando a Terra.

