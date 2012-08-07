"""
The ``fatiando`` package contains all the subpackages and modules required for
most tasks.

Modules for each geophysical method are group in subpackages:

* :mod:`pot <fatiando.pot>`:
  Potential fields
* :mod:`seis <fatiando.seis>`:
  Seismics and seismology
* :mod:`heat <fatiando.heat>`:
  Geothermics

Modules for gridding, meshing, visualization, user interface, etc:

* :mod:`msh <fatiando.msh>`:
  Mesh generation and definition of geometric elements
* :mod:`grd <fatiando.grd>`:
  Grid generation and operations (like interpolation)
* :mod:`vis <fatiando.vis>`:
  Plotting utilities for maps (using matplotlib) and 3D (using mayavi)
* :mod:`ui <fatiando.ui>`:
  User interfaces, like map picking and interactive drawing
* :mod:`utils <fatiando.utils>`:
  Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`log <fatiando.log>`:
  A simpler interface to the Python :mod:`logging` module for log files
* :mod:`~fatiando.constants`:
  Physical constants and unit conversions

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find common regularizing functions,
linear inverse problem solvers, and non-linear gradient solvers. This package
is generaly only used from inside Fatiando itself, not when using Fatiando in
scripts. For usage examples, see the source of modules
:mod:`fatiando.seis.epic2d` and :mod:`fatiando.pot.basin2d`.

See the documentation for each module to find out more about what they do and
how to use them.

----

"""

version = '0.1.dev'

# Import all the modules and subpackages so that they are accessible just by
# importing fatiando
from fatiando import log
from fatiando import grd
from fatiando import msh
from fatiando import pot
from fatiando import seis
from fatiando import utils
from fatiando import heat
from fatiando import ui
from fatiando import vis
from fatiando import inversion


__all__ = ['log', 'grd', 'msh', 'pot', 'seis', 'utils', 'heat', 'ui', 'vis',
           'inversion']
