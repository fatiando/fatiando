"""
Plotting utilities

Wrappers to facilitate common plotting tasks using powerful third-party
libraries.

These functions here are separated into modules, if you have a thing for
namespaces, but importing :mod:`~fatiando.vis` will load everything you need
(including some :mod:`matplotlib` functions, like ``plot`` and ``show``).

* :mod:`~fatiando.vis.map`
    2D plotting using matplotlib_
* :mod:`~fatiando.vis.vtk`
    3D plotting using Mayavi_

.. _matplotlib: http://matplotlib.sourceforge.net/
.. _Mayavi: http://code.enthought.com/projects/mayavi/

----
   
"""

from fatiando.vis.map import *
from fatiando.vis.vtk import *

# Get some useful things from matplotlib
from matplotlib.pyplot import (plot, show, figure, xlim, ylim, xlabel, ylabel,
    gca, axis, subplot, savefig, legend, colorbar, grid, hist, title, twinx,
    twiny)
    
