"""
Various tools for seismic and seismology, like direct modeling, inversion
(tomography), epicenter determination, etc.

**Forward modeling and inversion**

* :mod:`~fatiando.seis.ttime2d`: 2D seismic ray travel-time modeling
* :mod:`~fatiando.seis.epic2d`: 2D epicenter determination
* :mod:`~fatiando.seis.profile`: Modeling and inversion of seismic profiling
* :mod:`~fatiando.seis.wavefd`: Finite difference solution of the 2D wave
  equation

**Tomography**

* :mod:`~fatiando.seis.srtomo`: 2D straight-ray tomography problem

----

"""

from fatiando.seis import epic2d, ttime2d, srtomo, profile, wavefd
