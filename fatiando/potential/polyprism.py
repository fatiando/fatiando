"""
Calculate the potential fields of the 3D prism with polygonal crossection using
the formula of Plouff (1976).

**Gravity**

First and second derivatives of the gravitational potential:

* :func:`~fatiando.potential._polyprism.gz`
* :func:`~fatiando.potential._polyprism.gxx`
* :func:`~fatiando.potential._polyprism.gxy`
* :func:`~fatiando.potential._polyprism.gxz`
* :func:`~fatiando.potential._polyprism.gyy`
* :func:`~fatiando.potential._polyprism.gyz`
* :func:`~fatiando.potential._polyprism.gzz`

**Magnetic**

The Total Field magnetic anomaly:

* :func:`~fatiando.potential._polyprism.tf`

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
