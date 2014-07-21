.. _changelog:

Changelog
=========

Version (future)
----------------

**Release date**:

**Changes**:

* Using `versioneer <https://github.com/warner/python-versioneer>`__ to manage
  version numbers. Access the version number + git commit hash from
  ``fatiando.__version__``.
  (`PR 117 <https://github.com/leouieda/fatiando/pull/117>`_)
* **BUG FIX**: :ref:`fatiando.gravmag.prism <fatiando_gravmag_prism>`
  gravitational field functions give correct results in all sides of the prism.
  There were singularities due to log(0) and weird results because of arctan2.
  (`PR 113 <https://github.com/leouieda/fatiando/pull/113>`_)
* `PEP8 <http://www.python.org/dev/peps/pep-0008/>`__ compliance (started by
  @SamuelMarks).
  (`PR 115 <https://github.com/leouieda/fatiando/pull/115>`_)
* Multithreaded parallelism with OpenMP in
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>`,
  :ref:`fatiando.gravmag.polyprism <fatiando_gravmag_polyprism>` and
  :ref:`fatiando.gravmag.prism <fatiando_gravmag_prism>`.
  Speedups are range from practically none to over 3x.
  Works automatically.
  **Windows users will have to install an extra dependency!**
  See the :ref:`install instructions <install>`.
  (`PR 106 <https://github.com/leouieda/fatiando/pull/106>`_)
* Faster Cython implementations of
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>` and
  :ref:`fatiando.gravmag.polyprism <fatiando_gravmag_polyprism>`.
  Also separated gravmag forward modeling functions into "kernels" for gravity
  tensor components. This allows them to be reused in the magnetic field
  computations.
  (`PR 105 <https://github.com/leouieda/fatiando/pull/105>`_)
* Added ``xy2ne`` flag for ``square`` and ``points`` functions in
  :ref:`fatiando.vis.mpl <fatiando_vis_mpl>`.
  (`PR 94 <https://github.com/leouieda/fatiando/pull/94>`_)
* **New** class ``LCurve`` in :ref:`fatiando.inversion.regularization
  <fatiando_inversion_regularization>` for estimating the regularization
  parameter using an L-curve criterion.
  (`PR 90 <https://github.com/leouieda/fatiando/pull/90>`_)
* Added support for ``vmin`` and ``vmax`` arguments in
  :ref:`fatiando.vis.mpl.contourf <fatiando_vis_mpl>`.
  (`PR 89 <https://github.com/leouieda/fatiando/pull/89>`_)
* **New** module :ref:`fatiando.gravmag.magdir <fatiando_gravmag_magdir>` for
  estimating the total magnetization vector of multiple sources.
  (`PR 87 <https://github.com/leouieda/fatiando/pull/87>`_)

Version 0.2
-----------

**Release date**: 2014-01-15

**Changes**:

* Complete re-implementation of :ref:`fatiando.inversion <fatiando_inversion>`
  and all modules that depended on it. Inversion routines now have a standard
  interface. (`PR 72 <https://github.com/leouieda/fatiando/pull/72>`_)
* Added moving window solution for Euler deconvolution in
  :ref:`fatiando.gravmag.euler <fatiando_gravmag_euler>`.
  (`PR 85 <https://github.com/leouieda/fatiando/pull/85>`_)
* Renamed the ``fatiando.io`` module to
  :ref:`fatiando.datasets <fatiando_datasets>`
  (`PR 82 <https://github.com/leouieda/fatiando/pull/82>`_)
* :ref:`fatiando.utils.contaminate <fatiando_utils>` can now take multiple data
  vectors and stddevs
* 2x speed-up of :ref:`fatiando.gravmag.talwani <fatiando_gravmag_talwani>`
  with smarter numpy array usage. (`PR 57
  <https://github.com/leouieda/fatiando/pull/57>`_)
* 300x speed-up of :ref:`fatiando.seismic.ttime2d <fatiando_seismic_ttime2d>`
  with new Cython code. (`PR 62
  <https://github.com/leouieda/fatiando/pull/62>`_)
* Speed-up of :ref:`fatiando.gravmag.tesseroid <fatiando_gravmag_tesseroid>`
  with better Cython code. (`PR 58
  <https://github.com/leouieda/fatiando/pull/58>`_)
* Various tweaks to :ref:`fatiando.vis.myv <fatiando_vis_myv>`. (`PR 56
  <https://github.com/leouieda/fatiando/pull/56>`_ and `PR 60
  <https://github.com/leouieda/fatiando/pull/60>`_)
* **New** gravity gradient tensor modeling with spheres in
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>`. (`PR 55
  <https://github.com/leouieda/fatiando/pull/55>`_ and `PR 24
  <https://github.com/leouieda/fatiando/pull/24>`_, the first one by
  `Vanderlei <http://fatiando.org/people/oliveira-jr/>`__)
* **New** function :ref:`fatiando.gridder.profile <fatiando_gridder>` to
  extract a profile (cross-section) from map data. (`PR 46
  <https://github.com/leouieda/fatiando/pull/46>`_)
* Better support for random numbers. ``contaminate`` function now guaranteed to
  use errors with zero mean. Can now control the random seed used in all
  functions relying on random numbers. (`PR 41
  <https://github.com/leouieda/fatiando/pull/41>`_)
* **New** scalar wave 2D finite differences modeling in
  :ref:`fatiando.seismic.wavefd <fatiando_seismic_wavefd>`. (`PR 38
  <https://github.com/leouieda/fatiando/pull/38>`_ the first by `Andre
  <http://www.fatiando.org/people/ferreira/>`__!)
* **New** algorithms in :ref:`fatiando.seismic.wavefd
  <fatiando_seismic_wavefd>` for elastic waves and a new scalar wave solver!
  Using staggered grid finite
  differences makes elastic wave methods are more stable. (`PR 52
  <https://github.com/leouieda/fatiando/pull/52>`_)
* **New** ``extrapolate_nans`` function in
  :ref:`fatiando.gridder <fatiando_gridder>` to fill NaNs and masked
  values in arrays using the nearest data point.
* ``interp`` function of :ref:`fatiando.gridder <fatiando_gridder>` has option
  to extrapolate values outside the convex hull of the data (enabled by
  default). Uses better cubic interpolation by default and returns
  1D arrays like the rest of fatiando, instead of 2D. (`PR 44
  <https://github.com/leouieda/fatiando/pull/44>`_ and `PR 42
  <https://github.com/leouieda/fatiando/pull/42>`_)
* **New** function to load a grid in Surfer format. (`PR
  <https://github.com/leouieda/fatiando/pull/33>`_ the first by `Henrique
  <http://fatiando.org/people/santos/>`__!)
* **New** module :ref:`fatiando.gravmag.eqlayer <fatiando_gravmag_eqlayer>` for
  equivalent layer processing of potential fields.
* Refactored all magnetic modeling and inversion to use either scalar or vector
  magnetization.
* ``Seed`` class of
  :ref:`fatiando.gravmag.harvester <fatiando_gravmag_harvester>` can now be
  used as a ``Prism`` object.
* :ref:`fatiando.gravmag.harvester <fatiando_gravmag_harvester>` now supports
  data weights and magnetic data inversion.
* Removed module ``fatiando.logger``. (`PR 30
  <https://github.com/leouieda/fatiando/pull/30>`_)

Version 0.1
-----------

**Release date**: 2013-04-12

**Changes**:

* Change license to BSD (see the :ref:`license text <license>`).
* The API is now fully accessible by only importing ``fatiando``
* Added a :ref:`Cookbook <cookbook>` section to the documentation with all the
  sample scripts from the cookbook folder.
* Implemented "Robust 3D gravity gradient inversion by planting anomalous
  densities" by Uieda and Barbosa (2012) in
  :ref:`fatiando.gravmag.harvester <fatiando_gravmag_harvester>`
* Added harvester command line program that runs this new inversion
* Added magnetic total field anomaly function to
  :ref:`fatiando.gravmag.prism <fatiando_gravmag_prism>`
* Added :ref:`fatiando.vis.myv.savefig3d <fatiando_vis_myv>` to save a Mayavi
  scene
* Added :ref:`fatiando.vis.myv.polyprisms <fatiando_vis_myv>` 3D plotter
  function for PolygonalPrism
* Added :ref:`fatiando.vis.myv.points3d <fatiando_vis_myv>` 3D plotter
  function for points
* Added gravity gradient tensor components and magnetic total field anomaly to
  :ref:`fatiando.gravmag.polyprism <fatiando_gravmag_polyprism>`
* Added option to control the line width to `prisms` and `polyprisms` in
  :ref:`fatiando.vis.myv <fatiando_vis_myv>`
* Added module :ref:`fatiando.gravmag.tensor <fatiando_gravmag_tensor>` for
  processing gradient tensor data. Includes eigenvalues and eigenvectors,
  tensor invariants, center of mass estimation, etc.
* Added module :ref:`fatiando.gravmag.imaging <fatiando_gravmag_imaging>` with
  imaging methods for potential fields
* Added module :ref:`fatiando.gravmag.euler <fatiando_gravmag_euler>` with Euler
  deconvolution methods for potential field data
* Added module :ref:`fatiando.seismic.wavefd <fatiando_seismic_wavefd>` with 2D
  Finite Difference simulations of elastic seismic waves
* Added unit conversion functions to :ref:`fatiando.utils <fatiando_utils>`
* Added tesseroids forward modeling :ref:`fatiando.gravmag.tesseroid
  <fatiando_gravmag_tesseroid>`, meshing and plotting with Mayavi
* New :ref:`fatiando.io <fatiando_io>` module to fetch models and data from the
  web and convert them to useful formats (for now supports the CRUST2.0 global
  curstal model)
* If building inplace or packaging, the setup script puts the Mercurial
  changeset hash in a file. Then fatiando.logger.header
  loads the hash from file and put a "Unknown" if it can't read.
  This way importing fatiando won't fail if the there is no changeset
  information available.
* :ref:`fatiando.mesher.PrismMesh.dump <fatiando_mesher>`: takes a mesh
  file, a property file and a property name. Saves the output to these files.
* Transformed all geometric elements (like Prism, Polygon, etc) into classes
* Ported all C extensions to Python + Numpy. This way compiling is not a
  prerequisite to installing
* Using `Cython <http://www.cython.org>`_ for optional extension modules. If
  they exist, they are loaded to replace the Python + Numpy versions. This all
  happens at runtime.
* Move all physical constants used in ``fatiando`` to module
  :ref:`fatiando.constants <fatiando_constants>`
* Data modules hidden inside functions in
  :ref:`fatiando.gravmag.basin2d <fatiando_gravmag_basin2d>`
* Functions in :ref:`fatiando.gravmag.basin2d <fatiando_gravmag_basin2d>` spit
  out Polygons instead of the vertices estimated. Now you don't have to build
  the polygons by hand.
