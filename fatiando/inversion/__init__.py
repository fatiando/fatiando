"""
Everything you need to solve inverse problems!

This package provides the basic building blocks to implement inverse problem
solvers. The main class for this is :class:`~fatiando.inversion.base.Misfit`.
It represents a data-misfit function and contains various tools to fit a
model to some data. All you have to do is implement methods to calculate the
predicted (or modeled) data and (optionally) the Jacobian (or sensitivity)
matrix. With only that, you have access to a range of optimization methods,
regularization, joint inversion, etc.

Modules
-------

* :mod:`~fatiando.inversion.base`: Base classes for building inverse problem
  solvers
* :mod:`~fatiando.inversion.regularization`: Classes for common regularizing
  functions and base classes for building new ones
* :mod:`~fatiando.inversion.solvers`: Functions for several optimization
  methods (used by :class:`~fatiando.inversion.base.Misfit`)

Have a look at the examples bellow on how to use the package. More geophysical
examples include :mod:`fatiando.seismic.srtomo`,
:mod:`fatiando.seismic.profile`,
:mod:`fatiando.gravmag.basin2d`,
:mod:`fatiando.gravmag.eqlayer`,
and
:mod:`fatiando.gravmag.euler`.

----

"""
from .misfit import *
from .regularization import *
from .hyper_param import *
