"""
The ``fatiando`` package contains all the subpackages and modules required for
most tasks.

Modules for each geophysical method are group in subpackages:

* :mod:`gravmag <fatiando.gravmag>`:
  Gravity and magnetics (i.e., potential fields)
* :mod:`seismic <fatiando.seismic>`:
  Seismics and seismology
* :mod:`geothermal <fatiando.geothermal>`:
  Geothermal heat transfer modeling

Modules for gridding, meshing, visualization, user interface, input/output etc:

* :mod:`mesher <fatiando.mesher>`:
  Mesh generation and definition of geometric elements
* :mod:`gridder <fatiando.gridder>`:
  Grid generation and operations (e.g., interpolation)
* :mod:`vis <fatiando.vis>`:
  Plotting utilities for 2D (using matplotlib) and 3D (using mayavi)
* :mod:`datasets <fatiando.datasets>`:
  Fetch and load datasets and models from web repositories
* :mod:`gui <fatiando.gui>`:
  Graphical user interfaces (still very primitive)
* :mod:`utils <fatiando.utils>`:
  Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`~fatiando.constants`:
  Physical constants and unit conversions

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find:

* :mod:`~fatiando.inversion.regularization`: Common regularizing functions and
  base classes for building custom ones
* :mod:`~fatiando.inversion.base`: Base classes to implement your inverse
  problem. They do most of the heavy lifting for you!
* :mod:`~fatiando.inversion.solvers`: Functions for optimization (used by
  :mod:`~fatiando.inversion.base` classes)

Inversion methods in Fatiando leverage :mod:`fatiando.inversion`, providing a
common interface and usage patters. For examples, see modules
:mod:`fatiando.seismic.epic2d`,
:mod:`fatiando.seismic.srtomo`,
:mod:`fatiando.gravmag.basin2d`,
:mod:`fatiando.gravmag.euler`,
:mod:`fatiando.gravmag.eqlayer`,
etc.

The design of :mod:`fatiando.inversion` was inspired by `scikit-learn`_, an
amazing machine-learning library.

.. _scikit-learn: http://scikit-learn.org

----

"""

version = '0.2'
