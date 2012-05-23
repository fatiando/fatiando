"""
Potential field direct modeling, inversion, transformations and utilities.

**Forward modeling**

The forward modeling modules provide ways to calculate the gravitational and
magnetic field of various types of geometric objects:

* :mod:`~fatiando.potential.prism`
    3D right rectangular prisms
* :mod:`~fatiando.potential.polyprism`
    3D prisms with polygonal horizontal cross-sections
* :mod:`~fatiando.potential.talwani`
    2D bodies with polygonal vertical cross-sections

**Inversion**

The inversion modules use the forward modeling models and the
:mod:`fatiando.inversion` package to solve potential field inverse problems:

* :mod:`~fatiando.potential.basin2d`
    2D inversion of the shape of sedimentary basins and other outcropping bodies

**Processing**

The processing modules offer tools to prepare potential field data before or
after modeling.

* :mod:`~fatiando.potential.transform`
    Potential field transformations, like upward continuation

----

"""

from fatiando.potential import prism, polyprism, transform, talwani
