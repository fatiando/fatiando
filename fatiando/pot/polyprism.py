"""
Calculate the potential fields of the 3D prism with polygonal crossection using
the formula of Plouff (1976).

**Gravity**

First and second derivatives of the gravitational potential:

* :func:`~fatiando.pot._polyprism.gz`
* :func:`~fatiando.pot._polyprism.gxx`
* :func:`~fatiando.pot._polyprism.gxy`
* :func:`~fatiando.pot._polyprism.gxz`
* :func:`~fatiando.pot._polyprism.gyy`
* :func:`~fatiando.pot._polyprism.gyz`
* :func:`~fatiando.pot._polyprism.gzz`

**Magnetic**

The Total Field magnetic anomaly:

* :func:`~fatiando.pot._polyprism.tf`

**References**

Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
applications to magnetic terrain corrections, Geophysics, 41(4), 727-741.

----

"""

from fatiando.pot._polyprism import *

try:
    from fatiando.pot._cpolyprism import *
except ImportError:
    pass
