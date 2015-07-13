"""
Everything you need to solve inverse problems!

This package provides the basic building blocks to implement inverse problem
solvers. The main class for this is :class:`~fatiando.inversion.misfit.Misfit`.
It represents a data-misfit function and contains various tools to fit a
model to some data. All you have to do is implement methods to calculate the
predicted (or modeled) data and (optionally) the Jacobian (or sensitivity)
matrix. With only that, you have access to a range of optimization methods,
regularization, joint inversion, etc.

Modules
-------

* :mod:`~fatiando.inversion.misfit`: Defines the data-misfit classes. Used to
  implement new inverse problems.
* :mod:`~fatiando.inversion.regularization`: Classes for common regularizing
  functions and base classes for building new ones
* :mod:`~fatiando.inversion.hyper_param`: Classes hyper parameter optimization
  (e.g., estimating the regularization parameter through an L-curve).
* :mod:`~fatiando.inversion.optimization`: Functions for several optimization
  methods (used by :class:`~fatiando.inversion.misfit.Misfit`)
* :mod:`~fatiando.inversion.base`: Base classes used internally to define
  common operations and method.

You can import the ``Misfit``, regularization, and hyper parameter optimization
classes directly from the ``fatiando.inversion`` namespace:

>>> from fatiando.inversion import Misfit, LCurve, Damping, Smoothness

The :class:`~fatiando.inversion.misfit.Misfit` class is used internally in
Fatiando to implement all of our inversion algorithms. Take a look at the
modules below for some examples:

* :mod:`fatiando.seismic.srtomo`
* :mod:`fatiando.gravmag.basin2d`
* :mod:`fatiando.gravmag.eqlayer`
* :mod:`fatiando.gravmag.euler`
* :mod:`fatiando.gravmag.magdir`
* :mod:`fatiando.seismic.profile`

----

"""
from .misfit import *
from .regularization import *
from .hyper_param import *
