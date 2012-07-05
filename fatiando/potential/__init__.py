"""
Potential field direct modeling, inversion, transformations and utilities.

**Forward modeling**

The forward modeling modules provide ways to calculate the gravitational and
magnetic field of various types of geometric objects:

* :mod:`~fatiando.potential.prism`: 3D right rectangular prisms
* :mod:`~fatiando.potential.polyprism`: 3D prisms with polygonal horizontal
  cross-sections
* :mod:`~fatiando.potential.sphere`: Homogeneous spheres
* :mod:`~fatiando.potential.talwani`: 2D bodies with polygonal vertical
  cross-sections

**Inversion**

The inversion modules use the forward modeling models and the
:mod:`fatiando.inversion` package to solve potential field inverse problems:

* :mod:`~fatiando.potential.basin2d`: 2D inversion of the shape of sedimentary
  basins and other outcropping bodies
* :mod:`~fatiando.potential.harvester`: 3D inversion of compact bodies by
  planting anomalous densities

**Processing**

The processing modules offer tools to prepare potential field data before or
after modeling.

* :mod:`~fatiando.potential.trans`: Analytical potential field transformations,
  like upward continuation
* :mod:`~fatiando.potential.fourier`: Potential field transformations using the
  FFT
* :mod:`~fatiando.potential.tensor`: Utilities for operating on the gradient
  tensor

----

"""

from fatiando.potential import (basin2d,
                                polyprism,
                                prism,
                                talwani,
                                trans,
                                harvester,
                                sphere,
                                tensor,
                                fourier)
