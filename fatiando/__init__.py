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
* :mod:`io <fatiando.io>`:
  Input/Output of models, data sets, etc (fetch from web repositories)
* :mod:`gui <fatiando.gui>`:
  Graphical user interfaces (still very primitive)
* :mod:`utils <fatiando.utils>`:
  Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`~fatiando.constants`:
  Physical constants and unit conversions

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find common regularizing functions,
linear inverse problem solvers, and non-linear gradient solvers. This package
is generaly only used from inside Fatiando itself, not when using Fatiando in
scripts. For usage examples, see the source of modules
:mod:`fatiando.seismic.epic2d` and :mod:`fatiando.gravmag.basin2d`.

See the documentation for each module to find out more about what they do and
how to use them.

----

"""

version = '0.2'
