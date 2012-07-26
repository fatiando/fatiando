.. _changelog:

Changelog
=========

Changes in v0.1
---------------

* Change license to BSD (see the :ref:`license text <license>`).
* The API is now fully accessible by only importing ``fatiando``. Modules and
  packages have short nicknames for easier access (e.g., pot for potential).
* Import all from map and vtk in vis/__init__.py, plus some functions from
  matplotlib. Now all plotting functions are available from vis directly.
* If building inplace or packaging, the setup script puts the Mercurial
  changeset hash in a file. Then logger.header loads the hash from file and put
  a "Unknown" if it can't read. This way importing fatiando won't fail if the
  there is no changeset information available.
* In fatiando.mesher.ddd.PrismMesh.dump: takes a mesh file, a property file and
  a property name. Saves the output to these files.
* Implemented "Robust 3D gravity gradient inversion by planting anomalous
  densities" by Uieda and Barbosa, 2011
* Added harvester command line program that runs this new inversion
* Added tutorial to the documentation
* Transformed all geometric elements (like Prism, Polygon, etc) into classes
* Ported all C extensions to Python + Numpy. This way compiling is not a
  prerequisite to installing
* Using `Cython <http://www.cython.org>`_ for optional extension modules. If
  they exist, they are loaded to replace the Python + Numpy versions. This all
  happens at runtime.
* Move all physical constants used in :ref:`fatiando <fatiando>` to module
  :ref:`fatiando.constants <constants>`
* Added magnetic total field anomaly function to
  :ref:`potential.prism <potential.prism>`
* Added vis.vtk.savefig3d to save a Mayavi scene
* Added a 3D plotter function for PolygonalPrism
* Added gravity gradient tensor components and magnetic total field anomaly to
  :ref:`potential.polyprism <potential.polyprism>`
* Added option to control the line width to `prisms` and `polyprisms` in
  :ref:`vis.vtk <vis.vtk>`
* Added module :ref:`fatiando.potential.tensor <potential.tensor>` for
  processing gradient tensor data. Includes eigenvalues and eigenvectors,
  tensor invariants, center of mass estimation, etc.
