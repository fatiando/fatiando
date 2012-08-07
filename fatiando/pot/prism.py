"""
Calculate the potential fields of the 3D right rectangular prism.

**Gravity**

The gravitational fields are calculated using the forumla of Nagy et al. (2000)

* :func:`~fatiando.pot._prism.pot`
* :func:`~fatiando.pot._prism.gx`
* :func:`~fatiando.pot._prism.gy`
* :func:`~fatiando.pot._prism.gz`
* :func:`~fatiando.pot._prism.gxx`
* :func:`~fatiando.pot._prism.gxy`
* :func:`~fatiando.pot._prism.gxz`
* :func:`~fatiando.pot._prism.gyy`
* :func:`~fatiando.pot._prism.gyz`
* :func:`~fatiando.pot._prism.gzz`

**Magnetic**

The Total Field anomaly is calculated using the formula of Bhattacharyya (1964).

* :func:`~fatiando.pot._prism.tf`

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

----

"""

from fatiando.pot._prism import *

try:
    from fatiando.pot._cprism import *
except ImportError:
    pass
