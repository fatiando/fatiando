.. _changelog:

Changelog
=========

Changes in 0.1
--------------

* Change license to BSD (see the :ref:`license text <license>`).
* The API is now fully accessible by only import fatiando. Modules and packages
  have short nicknames for easier access (e.g., pot for potential).
* Import all from map and vtk in vis/__init__.py, plus some functions from
  matplotlib. Now all plotting functions are available from vis directly.
* If building inplace or packaging, the setup script puts the Mercurial
  changeset hash in a file. Then logger.header loads the hash from file and put
  a "Unknown" if it can't read. This way importing fatiando won't fail if the
  there is no changeset information available.
