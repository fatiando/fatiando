.. _api:

=======================================
API reference: The ``fatiando`` package
=======================================

.. automodule:: fatiando
    :no-members:
    :no-inherited-members:


``fatiando.mesher``: Geometric objects and meshes
=================================================

.. automodule:: fatiando.mesher
    :no-members:
    :no-inherited-members:

**More details**: See :ref:`userguide_mesher`.

.. currentmodule:: fatiando.mesher

Objects
-------

.. autosummary::
    :toctree: api/
    :template: class.rst

    Polygon
    Square
    Prism
    PolygonalPrism
    Sphere
    Tesseroid

Meshes
------

.. autosummary::
    :toctree: api/
    :template: class.rst

    SquareMesh
    PrismMesh
    PrismRelief
    TesseroidMesh
    PointGrid

Misc
------

.. autosummary::
    :toctree: api/
    :template: function.rst

    extract
    vfilter
    vremove


``fatiando.gridder``: Grids and irregularly spaced data
=======================================================

.. automodule:: fatiando.gridder
    :no-members:
    :no-inherited-members:

**More details**: See :ref:`userguide_gridder`.

.. currentmodule:: fatiando.gridder

.. autosummary::
    :toctree: api/
    :template: function.rst

    regular
    scatter
    cut
    profile
    interp
    interp_at
    extrapolate_nans
    load_surfer
    spacing



``fatiando.gravmag``: Gravity and magnetics
===========================================

.. automodule:: fatiando.gravmag
    :no-members:
    :no-inherited-members:

**More details**: See :ref:`userguide_gravmag`.


.. currentmodule:: fatiando.gravmag


Forward modeling
----------------

The forward modeling modules contain functions that calculate the potential
fields caused by geometric objects.

.. autosummary::
    :toctree: api/

    prism
    polyprism
    sphere
    tesseroid
    talwani


Inversion
---------

.. autosummary::
    :toctree: api/

    basin2d
    harvester
    euler
    magdir
    imaging

Processing
----------

.. autosummary::
    :toctree: api/

    normal_gravity
    eqlayer
    transform
    tensor

Misc
----

.. autosummary::
    :toctree: api/

    interactive



``fatiando.utils``: Utility functions
=====================================

.. automodule:: fatiando.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: fatiando.utils

.. autosummary::
    :toctree: api/
    :template: function.rst

    normal
    gaussian
    gaussian2d
    safe_solve
    safe_dot
    safe_diagonal
    safe_inverse
    random_points
    circular_points
    connect_points
    si2mgal
    mgal2si
    si2eotvos
    eotvos2si
    si2nt
    nt2si
    sph2cart
    fromimage
    contaminate
    dircos
    ang2vec
    vecnorm
    vecmean
    vecstd
    sec2hms
    sec2year
    year2sec

.. autosummary::
    :toctree: api/
    :template: class.rst

    SparseList
