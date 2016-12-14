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
    PrismRelief
    TesseroidMesh
    PointGrid

Utilities
---------

.. autosummary::
    :toctree: api/
    :template: function.rst

    extract
    vfilter
    vremove


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
    normal
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
    fromimage
    dircos
    ang2vec
    vecnorm
    vecmean
    vecstd
    sec2hms
    sec2year
    year2sec


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
    fetch_crust2
    crust2_to_tesseroids
    fetch_bouguer_alps_egm


``fatiando.gravmag``: Gravity and magnetics
===========================================

.. automodule:: fatiando.gravmag
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.gravmag


``fatiando.seismic``: Seismology and Seismics
=============================================

.. automodule:: fatiando.seismic
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.seismic


``fatiando.geothermal``: Geothermal methods
===========================================

.. automodule:: fatiando.geothermal
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.geothermal

