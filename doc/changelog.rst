.. _changelog:

Changelog
=========

Version 0.5
-----------

**Release date**: yyyy-mm-dd

**Changes**:

* Started an example gallery (`matplotlib style
  <http://matplotlib.org/gallery.html>`__) using the Sphinx plug-in
  `sphinx-gallery <http://sphinx-gallery.readthedocs.io/>`__.
  (`PR 282 <https://github.com/fatiando/fatiando/pull/282>`__)
* Added several functions for padding arrays of arbitrary dimension.
  ``fatiando.gridder.pad_array`` pads an array with a variety of padding and
  taper options.  ``fatiando.gridder.unpad_array`` returns the original,
  unpadded array.  ``fatiando.gridder.pad_coords`` pads the coordinate vectors
  associated with the arrays padded above. Added Kass in the contributors.
  (`PR 239 <https://github.com/fatiando/fatiando/pull/239>`__)
* Better navigation for long pages in the docs by adding a sidebar with links
  to subsections.
  (`PR 275 <https://github.com/fatiando/fatiando/pull/275>`__)



Version 0.4
-----------

**Release date**: 2016-04-05

**Changes**:

* **New** obtain a synthetic convolutional seismogram in
  ``fatiando.seismic.conv``. It can be given a depth model that will be
  converted to a time model before generating the synthetic seismogram.
  (`PR 190 <https://github.com/fatiando/fatiando/pull/190>`__)
* **Refactor** ``fatiando.inversion``. Completely redesigned classes make
  implementing new inversions simpler. Subclassing ``Misfit`` is simpler, with
  fewer parameters necessary. The usage of existing inversions has changed
  little. A **new dependency** ``future`` was added to ease the transition to
  support Python 3.
  (`PR 127 <https://github.com/fatiando/fatiando/pull/127>`__)
* Fix the broken software carpentry links in ``develop.rst``.
  (`PR 245 <https://github.com/fatiando/fatiando/pull/245>`__)
* Fix the doctest for ``fatiando.gravmag.tensor.center_of_mass``.
  (`PR 242 <https://github.com/fatiando/fatiando/pull/242>`__)
* **BUG FIX**: Tesseroid computations failed (silently) when tesseroids were
  smaller than 1e-6 degrees on a side (~ 10 cm). Code now ignores these
  tesseroids on input and warns the user about it. If a tesseroid becomes
  smaller than this during adaptive discretization, the tesseroid effect will
  be computed without division.  The user will be warned when this happens.
  (`PR 228 <https://github.com/fatiando/fatiando/pull/228>`__)
* **New** reduction to the pole and upward continuation with FFT in
  ``fatiando.gravmag.transform``. The pole reduction allows both remanent and
  induced magnetization. Upward continuation is more stable and faster than the
  old space domain approach that was implemented.
  (`PR 156 <https://github.com/fatiando/fatiando/pull/156>`__)
* **IMPORTANT BUG FIX**: Fixed wrong ordering of nodes in
  ``fatiando.mesher.PointGrid``. The order of nodes had the same problem as the
  regular grids (fixed in
  `196 <https://github.com/fatiando/fatiando/pull/196>`__). This was not caught
  before because ``PointGrid`` didn't use ``gridder.regular`` to generate its
  internal regular grid. This is an example of why reuse is a good thing! Tests
  now should catch any future problems.
  (`PR 209 <https://github.com/fatiando/fatiando/pull/209>`__)
* **IMPORTANT BUG FIX**: ``fatiando.gridder.regular`` and many other places in
  Fatiando where using the wrong convention for x, y dimensions.
  x should point North and y East. Thus, a data matrix (regular grid) should
  have x varying in the lines and y varying in the columns. This is **oposite**
  what we had. This fix also changes the ``shape`` argument to be ``(nx, ny)``
  instead of ``(ny, nx)``. **Users should be aware of this and double check
  their code.**
  (`PR 196 <https://github.com/fatiando/fatiando/pull/196>`__)
* More stable derivatives in ``fatiando.gravamag.transform``. The horizontal
  derivatives default to central finite-differences for greater stability. The
  FFT based derivatives use a grid padding to avoid edge effects.
  Thanks to `Matteo Niccoli <https://mycarta.wordpress.com/>`__ for suggesting
  this fix.
  (`PR 196 <https://github.com/fatiando/fatiando/pull/196>`__)
* **Renamed** ``fatiando.gravmag.fourier.ansig`` to
  ``fatiando.gravmag.transform.tga``
  (`PR 186 <https://github.com/fatiando/fatiando/pull/186>`__)
* **Remove** ``fatiando.gravmag.fourier`` by moving relevant functions into
  ``fatiando.gravmag.transform``.
  (`PR 186 <https://github.com/fatiando/fatiando/pull/186>`__)
* **New** ``seismic_wiggle`` and ``seismic_image`` plotting functions for
  seismic data in :ref:`fatiando.vis.mpl <fatiando_vis_mpl>` (`PR 192
  <https://github.com/fatiando/fatiando/pull/192>`__) plus cookbook
* **Remove** OpenMP parallelism from the ``fatiando.gravmag`` Cython coded
  forward modeling. Caused the majority of our install problems and didn't
  offer a great speed up anyway (< 2x). Can be replaced by ``multiprocessing``
  parallelism without the install problems
  (`PR 177 <https://github.com/fatiando/fatiando/pull/177>`__)
* Tesseroid forward modeling functions in ``fatiando.gravmag.tesseroid`` take
  an optional ``pool`` argument. Use it to pass an open
  ``multiprocessing.Pool`` for the function to use. Useful to avoid processes
  spawning overhead when calling the forward modeling many times
  (`PR 183 <https://github.com/fatiando/fatiando/pull/183>`__)
* **BUG FIX**: Avoid weird numba error when tesseroid has zero volume. Let to
  better sanitizing the input model. Tesseroids with dimensions < 1cm are
  ignored because they have almost zero gravitational effect
  (`PR 179 <https://github.com/fatiando/fatiando/pull/179>`__)
* Ported the tesseroid forward modeling code from Cython to numba. This is
  following the discussion on issue
  `#169 <https://github.com/fatiando/fatiando/issues/169>`__ to make installing
  less of burden by removing the compilation step. The numba code runs just as
  fast. New functions support multiprocessing parallelism.
  Thanks to new contributor Graham Markall for help with numba.
  (`PR 175 <https://github.com/fatiando/fatiando/pull/175>`__)
* Better documentation and faster implementation of
  ``fatiando.gravmag.tesseroid``
  (`PR 118 <https://github.com/fatiando/fatiando/pull/118>`__)
* **BUG FIX**: Replace ``matplotlib.mlab.griddata`` with
  ``scipy.interpolate.griddata`` in ``fatiando.gridder.interp`` to avoid
  incompatibilities when using ``matplotlib > 1.3``
  (at least in MacOS). Nearest neighbor interpolation method flagged as ``nn``
  was removed. Now it becomes only ``nearest``. Also replace ``matplotlib``
  with ``scipy`` in ``fatiando.mesher.PrismMesh.carvetopo``
  (`PR 148 <https://github.com/fatiando/fatiando/pull/148>`_)
* **New class** ``fatiando.gravmag.basin2d.PolygonalBasinGravity`` for 2D
  gravity inversion for the relief of a basin.
  (`PR 149 <https://github.com/fatiando/fatiando/pull/149>`__)
* Significant progress on the :ref:`Developer Guide <develop>`. From getting
  started to making a release on PyPI.
  (`PR 144 <https://github.com/fatiando/fatiando/pull/144>`__)
* **Removed** package ``fatiando.gui``. This was an experimental and temporary
  package to explore interactivity. Given new developments, like the
  `IPython HTML widgets
  <http://nbviewer.ipython.org/github/ipython/ipython/blob/master/examples/Interactive%20Widgets/Index.ipynb>`__,
  it is no longer relevant. The package will be replaced by package specific
  ``interactive`` modules.
  From the original classes implemented in this package, only ``Moulder`` has
  been saved.
  (`PR 143 <https://github.com/fatiando/fatiando/pull/143>`__)
* Moved ``Moulder`` to the **new module** ``fatiando.gravmag.interactive``.
  Completely rewrote the application. It now allows editing, moving, and
  deleting polygons, persisting the application to a pickle file and reloading,
  etc.
  (`PR 143 <https://github.com/fatiando/fatiando/pull/143>`__)

Version 0.3
-----------

**Release date**: 2014-10-28

**Changes**:

* **New module** :ref:`fatiando.gravmag.normal_gravity
  <fatiando_gravmag_normal_gravity>` to calculate normal gravity (the gravity
  of reference ellipsoids).
  (`PR 133 <https://github.com/fatiando/fatiando/pull/133>`_)
* Using `versioneer <https://github.com/warner/python-versioneer>`__ to manage
  version numbers. Access the version number + git commit hash from
  ``fatiando.__version__``.
  (`PR 117 <https://github.com/fatiando/fatiando/pull/117>`_)
* **BUG FIX**: :ref:`fatiando.gravmag.prism <fatiando_gravmag_prism>`
  gravitational field functions give correct results in all sides of the prism.
  There were singularities due to log(0) and weird results because of arctan2.
  (`PR 113 <https://github.com/fatiando/fatiando/pull/113>`_)
* `PEP8 <http://www.python.org/dev/peps/pep-0008/>`__ compliance (started by
  @SamuelMarks).
  (`PR 115 <https://github.com/fatiando/fatiando/pull/115>`_)
* Multithreaded parallelism with OpenMP in
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>`,
  :ref:`fatiando.gravmag.polyprism <fatiando_gravmag_polyprism>` and
  :ref:`fatiando.gravmag.prism <fatiando_gravmag_prism>`.
  Speedups are range from practically none to over 3x.
  Works automatically.
  **Windows users will have to install an extra dependency!**
  See the :ref:`install instructions <install>`.
  (`PR 106 <https://github.com/fatiando/fatiando/pull/106>`_)
* Faster Cython implementations of
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>` and
  :ref:`fatiando.gravmag.polyprism <fatiando_gravmag_polyprism>`.
  Also separated gravmag forward modeling functions into "kernels" for gravity
  tensor components. This allows them to be reused in the magnetic field
  computations.
  (`PR 105 <https://github.com/fatiando/fatiando/pull/105>`_)
* Added ``xy2ne`` flag for ``square`` and ``points`` functions in
  :ref:`fatiando.vis.mpl <fatiando_vis_mpl>`.
  (`PR 94 <https://github.com/fatiando/fatiando/pull/94>`_)
* **New** class ``LCurve`` in :ref:`fatiando.inversion.regularization
  <fatiando_inversion_regularization>` for estimating the regularization
  parameter using an L-curve criterion.
  (`PR 90 <https://github.com/fatiando/fatiando/pull/90>`_)
* Added support for ``vmin`` and ``vmax`` arguments in
  :ref:`fatiando.vis.mpl.contourf <fatiando_vis_mpl>`.
  (`PR 89 <https://github.com/fatiando/fatiando/pull/89>`_)
* **New** module :ref:`fatiando.gravmag.magdir <fatiando_gravmag_magdir>` for
  estimating the total magnetization vector of multiple sources.
  (`PR 87 <https://github.com/fatiando/fatiando/pull/87>`_)

Version 0.2
-----------

**Release date**: 2014-01-15

**Changes**:

* Complete re-implementation of :ref:`fatiando.inversion <fatiando_inversion>`
  and all modules that depended on it. Inversion routines now have a standard
  interface. (`PR 72 <https://github.com/fatiando/fatiando/pull/72>`_)
* Added moving window solution for Euler deconvolution in
  :ref:`fatiando.gravmag.euler <fatiando_gravmag_euler>`.
  (`PR 85 <https://github.com/fatiando/fatiando/pull/85>`_)
* Renamed the ``fatiando.io`` module to
  :ref:`fatiando.datasets <fatiando_datasets>`
  (`PR 82 <https://github.com/fatiando/fatiando/pull/82>`_)
* :ref:`fatiando.utils.contaminate <fatiando_utils>` can now take multiple data
  vectors and stddevs
* 2x speed-up of :ref:`fatiando.gravmag.talwani <fatiando_gravmag_talwani>`
  with smarter numpy array usage. (`PR 57
  <https://github.com/fatiando/fatiando/pull/57>`_)
* 300x speed-up of :ref:`fatiando.seismic.ttime2d <fatiando_seismic_ttime2d>`
  with new Cython code. (`PR 62
  <https://github.com/fatiando/fatiando/pull/62>`_)
* Speed-up of :ref:`fatiando.gravmag.tesseroid <fatiando_gravmag_tesseroid>`
  with better Cython code. (`PR 58
  <https://github.com/fatiando/fatiando/pull/58>`_)
* Various tweaks to :ref:`fatiando.vis.myv <fatiando_vis_myv>`. (`PR 56
  <https://github.com/fatiando/fatiando/pull/56>`_ and `PR 60
  <https://github.com/fatiando/fatiando/pull/60>`_)
* **New** gravity gradient tensor modeling with spheres in
  :ref:`fatiando.gravmag.sphere <fatiando_gravmag_sphere>`. (`PR 55
  <https://github.com/fatiando/fatiando/pull/55>`_ and `PR 24
  <https://github.com/fatiando/fatiando/pull/24>`_, the first one by
  `Vanderlei <http://fatiando.org/people/oliveira-jr/>`__)
* **New** function :ref:`fatiando.gridder.profile <fatiando_gridder>` to
  extract a profile (cross-section) from map data. (`PR 46
  <https://github.com/fatiando/fatiando/pull/46>`_)
* Better support for random numbers. ``contaminate`` function now guaranteed to
  use errors with zero mean. Can now control the random seed used in all
  functions relying on random numbers. (`PR 41
  <https://github.com/fatiando/fatiando/pull/41>`_)
* **New** scalar wave 2D finite differences modeling in
  :ref:`fatiando.seismic.wavefd <fatiando_seismic_wavefd>`. (`PR 38
  <https://github.com/fatiando/fatiando/pull/38>`_ the first by `Andre
  <http://www.fatiando.org/people/ferreira/>`__!)
* **New** algorithms in :ref:`fatiando.seismic.wavefd
  <fatiando_seismic_wavefd>` for elastic waves and a new scalar wave solver!
  Using staggered grid finite
  differences makes elastic wave methods are more stable. (`PR 52
  <https://github.com/fatiando/fatiando/pull/52>`_)
* **New** ``extrapolate_nans`` function in
  :ref:`fatiando.gridder <fatiando_gridder>` to fill NaNs and masked
  values in arrays using the nearest data point.
* ``interp`` function of :ref:`fatiando.gridder <fatiando_gridder>` has option
  to extrapolate values outside the convex hull of the data (enabled by
  default). Uses better cubic interpolation by default and returns
  1D arrays like the rest of fatiando, instead of 2D. (`PR 44
  <https://github.com/fatiando/fatiando/pull/44>`_ and `PR 42
  <https://github.com/fatiando/fatiando/pull/42>`_)
* **New** function to load a grid in Surfer format. (`PR
  <https://github.com/fatiando/fatiando/pull/33>`_ the first by `Henrique
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
  <https://github.com/fatiando/fatiando/pull/30>`_)

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
