"""
Gravity and magnetics forward modeling, inversion, transformations and
utilities.

Inversion
---------

The inversion modules use the forward modeling models and the
:mod:`fatiando.inversion` package to solve potential field inverse problems:

* :mod:`~fatiando.gravmag.basin2d`: 2D inversion of the shape of sedimentary
  basins and other outcropping bodies
* :mod:`~fatiando.gravmag.harvester`: 3D inversion of compact bodies by
  planting anomalous densities
* :mod:`~fatiando.gravmag.euler`: 3D Euler deconvolution methods to estimate
  source location
* :mod:`~fatiando.gravmag.magdir`: Inversion methods to estimate the total
  magnetization vector of multiple sources.

Processing
----------

The processing modules offer tools to prepare potential field data before or
after modeling.

* :mod:`~fatiando.gravmag.normal_gravity`: Compute normal gravity and
  reductions.
* :mod:`~fatiando.gravmag.eqlayer`: Equivalent layer processing
* :mod:`~fatiando.gravmag.transform`: Potential field transformations,
  like upward continuation, derivatives, etc
* :mod:`~fatiando.gravmag.imaging`: Imaging methods for potential fields for
  estimating physical property distributions
* :mod:`~fatiando.gravmag.tensor`: Utilities for operating on the gradient
  tensor

Interactivity
-------------

Module :mod:`~fatiando.gravmag.interactive` implements matplotlib GUIs and
IPython HTML widgets for interacting with the modeling and processing
functions.

----

"""
from __future__ import absolute_import
from .euler import EulerDeconv, EulerDeconvMW, EulerDeconvEW
