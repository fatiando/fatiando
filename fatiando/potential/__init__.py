"""
Potential field direct modeling, inversion, transformations and utilities.

**Direct modelling**

* :mod:`~fatiando.potential.prism`
* :mod:`~fatiando.potential.polyprism`
* :mod:`~fatiando.potential.talwani`

The direct modeling modules provide ways to calculate the gravitational and
magnetic field of various types of geometric objects. For 3D right rectangular
prisms, use :mod:`~fatiando.potential.prism`. For 2D bodies with polygonal
vertical cross-sections, use :mod:`~fatiando.potential.talwani`. For 3D bodies
with polygonal horizontal cross-sections, use
:mod:`~fatiando.potential.polyprism`.

**Inversion**

* :mod:`~fatiando.potential.basin2d`

The inverse modeling modules use the direct models and the
:mod:`~fatiando.inversion` package to solve potential field inverse problems.

**Processing**

* :mod:`~fatiando.potential.transform`

The processing modules offer tools to prepare potential field data before or
after modeling.

----

"""

from fatiando.potential import prism, polyprism, transform, talwani
