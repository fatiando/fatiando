"""
The ``fatiando`` package contains all the subpackages and modules required for
most tasks. 

Modules for each geophysical method are group in subpackages:

* :mod:`pot <fatiando.potential>`:
  Potential fields
* :mod:`seis <fatiando.seismic>`:
  Seismics and seismology
* :mod:`heat <fatiando.heat>`:
  Geothermics

Modules for gridding, meshing, visualization, user interface, etc:

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
  A simpler interface to the Python :mod:`logging` module for log files

See the documentation for each module to find out more about what they do and
how to use them.

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find common regularizing functions,
linear inverse problem solvers, and non-linear gradient solvers. This package
is generaly only used from inside Fatiando itself, not when using Fatiando in
scripts. For usage examples, see the source of modules
:mod:`fatiando.seismic.epic2d` and :mod:`fatiando.potential.basin2d`.

.. note:: Some modules have "nicknames" for easier access when importing
    the ``fatiando`` package directly. To import each module separately,
    use the full names, e.g., ``from fatiando.mesher.ddd import PrismMesh`` or
    ``from fatiando.potential import talwani``.

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
