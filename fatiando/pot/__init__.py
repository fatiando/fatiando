"""
Potential field direct modeling, inversion, transformations and utilities.

**Forward modeling**

The forward modeling modules provide ways to calculate the gravitational and
magnetic field of various types of geometric objects:

* :mod:`~fatiando.pot.prism`: 3D right rectangular prisms
* :mod:`~fatiando.pot.polyprism`: 3D prisms with polygonal horizontal
  cross-sections
* :mod:`~fatiando.pot.sphere`: Homogeneous spheres
* :mod:`~fatiando.pot.talwani`: 2D bodies with polygonal vertical
  cross-sections

**Inversion**

The inversion modules use the forward modeling models and the
:mod:`fatiando.inversion` package to solve potential field inverse problems:

* :mod:`~fatiando.pot.basin2d`: 2D inversion of the shape of sedimentary
  basins and other outcropping bodies
* :mod:`~fatiando.pot.harvester`: 3D inversion of compact bodies by
  planting anomalous densities

**Processing**

The processing modules offer tools to prepare potential field data before or
after modeling.

* :mod:`~fatiando.pot.trans`: Analytical potential field transformations,
  like upward continuation
* :mod:`~fatiando.pot.fourier`: Potential field transformations using the
  FFT
* :mod:`~fatiando.pot.imaging`: Imaging methods for potential fields for
  estimating physical property distributions
* :mod:`~fatiando.pot.tensor`: Utilities for operating on the gradient
  tensor

----

"""

from fatiando.pot import (basin2d, polyprism, prism, talwani, trans, harvester,
                           sphere, tensor, fourier, imaging, euler)
