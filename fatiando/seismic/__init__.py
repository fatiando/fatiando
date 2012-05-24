"""
Various tools for seismic and seismology, like direct modeling, inversion
(tomography), epicenter determination, etc.

**Forward modeling and inversion**

* :mod:`~fatiando.seismic.traveltime`
    2D seismic ray travel-time modeling
* :mod:`~fatiando.seismic.epic2d`
    2D epicenter determination
* :mod:`~fatiando.seismic.profile`
    Modeling and inversion of seismic profiling

**Tomography**

* :mod:`~fatiando.seismic.srtomo`
    2D straight-ray tomography problem

----

"""

from fatiando.seismic import epic2d, traveltime, srtomo, profile
