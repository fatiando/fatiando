.. _changelog:

Changelog
=========

Version 0.1
-----------

**Release date**: NOT RELEASED

**New features**:

* Change license to BSD (see the :ref:`license text <license>`).
* Added a :ref:`Cookbook <cookbook>` section to the documentation with all the
  sample scripts from the cookbook folder.
* Implemented "Robust 3D gravity gradient inversion by planting anomalous
  densities" by
  `Uieda and Barbosa, 2012 <http://fatiando.org/people/uieda/publications/>`_
* Added harvester command line program that runs this new inversion
* Added magnetic total field anomaly function to
  :ref:`fatiando.pot.prism <fatiando_pot_prism>`
* Added :ref:`fatiando.vis.vtk.savefig3d <fatiando_vis_vtk>` to save a Mayavi
  scene
* Added :ref:`fatiando.vis.vtk.polyprisms <fatiando_vis_vtk>` 3D plotter
  function for PolygonalPrism
* Added :ref:`fatiando.vis.vtk.points3d <fatiando_vis_vtk>` 3D plotter
  function for points
* Added gravity gradient tensor components and magnetic total field anomaly to
  :ref:`fatiando.pot.polyprism <fatiando_pot_polyprism>`
* Added option to control the line width to `prisms` and `polyprisms` in
  :ref:`fatiando.vis.vtk <fatiando_vis_vtk>`
* Added module :ref:`fatiando.pot.tensor <fatiando_pot_tensor>` for
  processing gradient tensor data. Includes eigenvalues and eigenvectors,
  tensor invariants, center of mass estimation, etc.
* Added :ref:`tutorials <tutorials>` to the documentation
* Added module :ref:`fatiando.pot.imaging <fatiando_pot_imaging>` with imaging
  methods for potential fields
* Added module :ref:`fatiando.pot.euler <fatiando_pot_euler>` with Euler
  deconvolution methods for potential field data
* Added module :ref:`fatiando.seis.wavefd <fatiando_seis_wavefd>` with 2D Finite
  Difference simulations of elastic seismic waves
* Added unit conversion functions to :ref:`fatiando.utils <fatiando_utils>`

**Improved features**:

* The API is now fully accessible by only importing ``fatiando``
* Modules and packages have shorter for easier access (e.g., pot instead of
  potential).
* Now all plotting functions are available from vis directly.
  Import all from map and vtk in vis/__init__.py, plus some functions from
  matplotlib.
* If building inplace or packaging, the setup script puts the Mercurial
  changeset hash in a file. Then :ref:`fatiando.log.header <fatiando_log>` loads
  the hash from file and put
  a "Unknown" if it can't read. This way importing fatiando won't fail if the
  there is no changeset information available.
* :ref:`fatiando.msh.ddd.PrismMesh.dump <fatiando_msh_ddd>`: takes a mesh
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
  :ref:`fatiando.pot.basin2d <fatiando_pot_basin2d>`
* Functions in :ref:`fatiando.pot.basin2d <fatiando_pot_basin2d>` spit out
  Polygons instead of the vertices estimated. Now you don't have to build the
  polygons by hand.

**Bug fixes**:


