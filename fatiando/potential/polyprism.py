"""
Calculate the potential fields and derivatives of the 3D prism with polygonal
crossection using the forumla of Plouff (1976).

**Gravity**

* :func:`~fatiando.potential._polyprism.gz`

**References**

Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
applications to magnetic terrain corrections, Geophysics, 41(4), 727-741.

----

"""

from fatiando.potential._polyprism import *

try:
    from fatiando.potential._cpolyprism import *
except ImportError:
    pass
