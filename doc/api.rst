.. _api:

=======================================
API reference: The ``fatiando`` package
=======================================

.. automodule:: fatiando
    :no-members:
    :no-inherited-members:


``fatiando.gridder``: Grids and irregularly spaced data
=======================================================

.. automodule:: fatiando.gridder
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.gridder

.. autosummary::
    :toctree: api/
    :template: function.rst

    regular
    scatter
    circular_scatter
    cut
    inside
    profile
    interp
    interp_at
    spacing
    pad_array
    unpad_array
    pad_coords


``fatiando.mesher``: Geometric objects and meshes
=================================================

.. automodule:: fatiando.mesher
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.mesher

Geometric objects
-----------------

.. autosummary::
    :toctree: api/
    :template: class.rst

    Polygon
    Square
    Prism
    PolygonalPrism
    Sphere
    Tesseroid

Meshes and collections
----------------------

.. autosummary::
    :toctree: api/
    :template: class.rst

    SquareMesh
    PrismMesh
    TesseroidMesh
    PrismRelief
    PointGrid


``fatiando.utils``: Utility functions
=====================================

.. automodule:: fatiando.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.utils

.. autosummary::
    :toctree: api/
    :template: function.rst

    contaminate
    gaussian
    gaussian2d
    safe_solve
    safe_dot
    safe_diagonal
    safe_inverse
    si2mgal
    mgal2si
    si2eotvos
    eotvos2si
    si2nt
    nt2si
    sph2cart
    dircos
    ang2vec


``fatiando.vis``: Visualization
===============================

.. automodule:: fatiando.vis
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.vis


``fatiando.datasets``: Load and fetch standard datasets
=======================================================

.. automodule:: fatiando.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.datasets

Functions
---------

.. autosummary::
    :toctree: api/
    :template: function.rst

    load_surfer
    check_hash
    fetch_hawaii_gravity
    from_image


``fatiando.seismic``: Seismology and seismics
=============================================

.. automodule:: fatiando.seismic
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.seismic

Classes
-------

.. autosummary::
    :toctree: api/
    :template: class.rst

    RickerWavelet

Functions
---------

.. autosummary::
    :toctree: api/
    :template: function.rst

    lame_mu
    lame_lambda


``fatiando.gravmag``: Gravity and magnetics
===========================================

.. automodule:: fatiando.gravmag
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.gravmag

Forward modeling
----------------

The ``fatiando.gravmag`` package defines a few modules that calculate the
potential fields of some geometric primitives.

Spheres
+++++++

.. autosummary::
    :toctree: api/
    :template: function.rst

    sphere.gz
    sphere.gxx
    sphere.gxy
    sphere.gxz
    sphere.gyy
    sphere.gyz
    sphere.gzz
    sphere.tf
    sphere.bx
    sphere.by
    sphere.bz
    sphere.kernelxx
    sphere.kernelxy
    sphere.kernelxz
    sphere.kernelyy
    sphere.kernelyz
    sphere.kernelzz

Polygonal prisms
++++++++++++++++

.. autosummary::
    :toctree: api/
    :template: function.rst

    polyprism.gz
    polyprism.gxx
    polyprism.gxy
    polyprism.gxz
    polyprism.gyy
    polyprism.gyz
    polyprism.gzz
    polyprism.tf
    polyprism.bx
    polyprism.by
    polyprism.bz
    polyprism.kernelxx
    polyprism.kernelxy
    polyprism.kernelxz
    polyprism.kernelyy
    polyprism.kernelyz
    polyprism.kernelzz

Right-rectangular prisms
++++++++++++++++++++++++

.. autosummary::
    :toctree: api/
    :template: function.rst

    prism.potential
    prism.gx
    prism.gy
    prism.gz
    prism.gxx
    prism.gxy
    prism.gxz
    prism.gyy
    prism.gyz
    prism.gzz
    prism.tf
    prism.bx
    prism.by
    prism.bz

Polygons (2D)
+++++++++++++

.. autosummary::
    :toctree: api/
    :template: function.rst

    talwani.gz

Tesseroids (spherical prisms)
+++++++++++++++++++++++++++++

.. autosummary::
    :toctree: api/
    :template: function.rst

    tesseroid.potential
    tesseroid.gx
    tesseroid.gy
    tesseroid.gz
    tesseroid.gxx
    tesseroid.gxy
    tesseroid.gxz
    tesseroid.gyy
    tesseroid.gyz
    tesseroid.gzz


``fatiando.geothermal``: Geothermal methods
===========================================

.. automodule:: fatiando.geothermal
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.geothermal

