"""
**Fatiando a Terra**:

Tools for geophysical modeling and inversion.

Contains utilities for gridding, meshing, and visualization.

Some modules have "nicknames" for easier access when importing ``fatiando``
directly. To import each module separately, use the full names:

* :mod:`msh <fatiando.mesher>`:
  Mesh generation and definition of geometric elements
* :mod:`grd <fatiando.gridder>`:
  Grid generation and operations (like interpolation)
* :mod:`vis <fatiando.vis>`:
  Plotting utilities for maps (using matplotlib) and 3D (using mayavi)
* :mod:`ui <fatiando.ui>`:
  User interfaces, like map picking and interactive drawing
* :mod:`utils <fatiando.utils>`:
  Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`log <fatiando.logger>`:
  An interface to Pythons :mod:`logging` module for easy printing to log files

Modules for specific geophysical methods are divided into subpackages:

* :mod:`pot <fatiando.potential>`:
  Potential fields
* :mod:`seis <fatiando.seismic>`:
  Seismic method and seismology
* :mod:`heat <fatiando.heat>`:
  Geothermics

See the documentation for each module to find out more about what they do and
how to use them.

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find common regularizing functions,
linear inverse problem solvers, and non-linear gradient solvers. This package
is generaly only used from inside Fatiando itself, not when using Fatiando in
scripts.

----

"""

version = '0.1.dev'

# Import all the modules and subpackages so that they are accessible just by
# importing fatiando
from fatiando import logger as log
from fatiando import gridder as grd
from fatiando import mesher as msh
from fatiando import potential as pot
from fatiando import seismic as seis
from fatiando import utils
from fatiando import heat
from fatiando import ui
from fatiando import vis
from fatiando import inversion


__all__ = ['log', 'grd', 'msh', 'pot', 'seis', 'utils', 'heat', 'ui', 'vis',
           'inversion']
