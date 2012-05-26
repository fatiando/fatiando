"""
Calculate the potential fields of the 3D right rectangular prism.

**Gravity**
 
The gravitational fields are calculated using the forumla of Nagy et al. (2000)

* :func:`~fatiando.potential.prism.pot`
* :func:`~fatiando.potential.prism.gx`
* :func:`~fatiando.potential.prism.gy`
* :func:`~fatiando.potential.prism.gz`
* :func:`~fatiando.potential.prism.gxx`
* :func:`~fatiando.potential.prism.gxy`
* :func:`~fatiando.potential.prism.gxz`
* :func:`~fatiando.potential.prism.gyy`
* :func:`~fatiando.potential.prism.gyz`
* :func:`~fatiando.potential.prism.gzz`

**Magnetic**


**References**

Nagy, D., G. Papp, and J. Benedek, 2000, The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.
    
----

"""
from fatiando import logger

log = logger.dummy('fatiando.potential.prism')

try:
    from _cprism import *
except ImportError:
    from _prism import *


