.. _changelog:

Changelog
=========

Version 0.2
-----------

**Release date**:

**Changes**:

* Complete re-implementation of :ref:`fatiando.inversion <fatiando_inversion>`
  and all modules that depended on it. Inversion routines now have a standard
  interface.
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
