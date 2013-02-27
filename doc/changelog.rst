.. _changelog:

Changelog
=========

Version 0.1
-----------

**Release date**: NOT RELEASED

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
  changeset hash in a file. Then :ref:`fatiando.logger.header <fatiando_logger>`
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

**Bug fixes**:


