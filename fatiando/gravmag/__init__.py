"""
Gravity and magnetics forward modeling, inversion, transformations and
utilities.

**Forward modeling**

The forward modeling modules provide ways to calculate the gravitational and
magnetic field of various types of geometric objects:

* :mod:`~fatiando.gravmag.prism`: 3D right rectangular prisms
* :mod:`~fatiando.gravmag.polyprism`: 3D prisms with polygonal horizontal
  cross-sections
* :mod:`~fatiando.gravmag.sphere`: Spheres in Cartesian coordinates
* :mod:`~fatiando.gravmag.tesseroid`: Tesseroids (spherical prisms) for
  modeling in spherical coordinates
* :mod:`~fatiando.gravmag.talwani`: 2D bodies with polygonal vertical
  cross-sections
* :mod:`~fatiando.gravmag.half_sph_shell`: Gravity fields of half a spherical
  shell. Useful for benchmarking and testing.

**Inversion**

The inversion modules use the forward modeling models and the
:mod:`fatiando.inversion` package to solve potential field inverse problems:

* :mod:`~fatiando.gravmag.basin2d`: 2D inversion of the shape of sedimentary
  basins and other outcropping bodies
* :mod:`~fatiando.gravmag.harvester`: 3D inversion of compact bodies by
  planting anomalous densities
* :mod:`~fatiando.gravmag.euler`: 3D Euler deconvolution methods to estimate
  source location
* :mod:`~fatiando.gravmag.magdir`: Inversion methods to estimate the total
  magnetization vector of multiple sources.

**Processing**

The processing modules offer tools to prepare potential field data before or
after modeling.

* :mod:`~fatiando.gravmag.eqlayer`: Equivalent layer processing
* :mod:`~fatiando.gravmag.transform`: Space domain potential field
  transformations, like upward continuation
* :mod:`~fatiando.gravmag.fourier`: Potential field transformations using the
  FFT
* :mod:`~fatiando.gravmag.imaging`: Imaging methods for potential fields for
  estimating physical property distributions
* :mod:`~fatiando.gravmag.tensor`: Utilities for operating on the gradient
  tensor

----

"""
