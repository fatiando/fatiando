"""
**Fatiando a Terra**:

Tools for geophysical modeling and inversion.

Contains utilities for gridding, meshing, and visualization:

* :mod:`fatiando.mesher`
    Mesh generation and definition of geometric elements
* :mod:`fatiando.gridder`
    Grid generation and operations (like interpolation)
* :mod:`fatiando.vis`
    Plotting utilities for maps (using matplotlib) and 3D (using mayavi)
* :mod:`fatiando.ui`
    User interfaces, like map picking and interactive drawing
* :mod:`fatiando.utils`
    Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`fatiando.logger`
    An interface to Pythons :mod:`logging` module for easy printing to log files

Modules for specific geophysical methods are divided into subpackages:

* :mod:`fatiando.potential`
    Potential fields
* :mod:`fatiando.seismic`
    Seismic method and seismology
* :mod:`fatiando.heat`
    Geothermics

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find common regularizing functions,
linear inverse problem solvers, and non-linear gradient solvers.

----

"""

version = '0.1.dev'

# Import all the modules and subpackages so that they are accessible just by
# importing fatiando
from fatiando import logger, gridder, utils
from fatiando import heat, mesher, potential, seismic, ui, vis
from fatiando import inversion
