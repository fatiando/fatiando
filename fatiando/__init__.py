"""
The ``fatiando`` package contains all the subpackages and modules required for
most tasks.

Modules for each geophysical method are group in subpackages:

* :mod:`gravmag <fatiando.gravmag>`:
  Gravity and magnetics (i.e., potential fields)
* :mod:`seismic <fatiando.seismic>`:
  Seismics and seismology
* :mod:`geothermal <fatiando.geothermal>`:
  Geothermal heat transfer modeling

Modules for gridding, meshing, visualization, etc:

* :mod:`mesher <fatiando.mesher>`:
  Mesh generation and definition of geometric elements
* :mod:`gridder <fatiando.gridder>`:
  Grid generation and operations (e.g., interpolation)
* :mod:`vis <fatiando.vis>`:
  Plotting utilities for 2D (using matplotlib) and 3D (using mayavi)
* :mod:`datasets <fatiando.datasets>`:
  Fetch and load datasets and models from web repositories
* :mod:`utils <fatiando.utils>`:
  Miscelaneous utilities, like mathematical functions, unit conversion, etc
* :mod:`~fatiando.constants`:
  Physical constants and unit conversions

Also included is the :mod:`fatiando.inversion` package with utilities for
implementing inverse problems. There you'll find ready to use regularization,
optimization methods, and templates to implement new inversion methods.

Inversions implemented in Fatiando leverage :mod:`fatiando.inversion`,
providing a common interface and usage patters. For examples, see modules
:mod:`fatiando.seismic.epic2d`,
:mod:`fatiando.seismic.srtomo`,
:mod:`fatiando.gravmag.basin2d`,
:mod:`fatiando.gravmag.euler`,
:mod:`fatiando.gravmag.eqlayer`,
etc.

The design of :mod:`fatiando.inversion` was inspired by `scikit-learn`_, an
amazing machine-learning library.

.. _scikit-learn: http://scikit-learn.org

----

"""
from ._version import get_versions
__version__ = get_versions()['version']
__commit__ = get_versions()['full']
del get_versions


def test(doctest=True, verbose=False, coverage=False):
    """
    Run the test suite for Fatiando a Terra.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests. If you
    haven't already, you can install it with `conda
    <http://conda.pydata.org/>`__ or `pip <https://pip.pypa.io>`__.

    Parameters:

    * doctest : bool
        If ``True``, will run the doctests as well (code examples that start
        with a ``>>>`` in the docs).
    * verbose : bool
        If ``True``, will print extra information during the test run.
    * coverage : bool
        If ``True``, will run test coverage analysis on the code as well.
        Requires ``pytest-cov``.

    Returns:

    * exit_code : int
        The exit code for the test run. If ``0``, then all tests pass.

    """
    import pytest
    args = []
    if verbose:
        args.append('-v')
    if coverage:
        args.append('--cov-report term-missing --cov=fatiando')
    if doctest:
        args.append('--doctest-modules')
    args.append('--pyargs fatiando')
    return pytest.main(' '.join(args))
