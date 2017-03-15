"""
Tools for seismic and seismology, like convolutional modeling and wave
propagation.

Forward modeling and inversion
------------------------------

* :mod:`~fatiando.seismic.ttime2d`: 2D seismic ray travel-time modeling
* :mod:`~fatiando.seismic.epic2d`: 2D epicenter determination
* :mod:`~fatiando.seismic.profile`: Modeling and inversion of seismic profiling
* :mod:`~fatiando.seismic.wavefd`: Finite difference solution of the 2D elastic
  wave equation
* :mod:`~fatiando.seismic.conv`: Convolutional seismic modeling

Tomography
----------

* :mod:`~fatiando.seismic.srtomo`: 2D straight-ray tomography

----

"""
from __future__ import absolute_import
from .elastic_moduli import lame_lambda, lame_mu
from .wavelets import RickerWavelet
