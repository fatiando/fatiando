"""
Direct modeling of seismic wave travel times.

**Straight rays**

* :func:`~fatiando.seismic.traveltime.straight_ray_2d`

----

"""

import numpy

from fatiando import logger
#from fatiando.seismic import _traveltime


log = logger.dummy('fatiando.seismic.traveltime')

def straight_ray_2d(cells, prop, srcs, recs):
    """
    Calculate the travel times inside a list of 2D square cells between
    source and receiver pairs assuming the rays are straight lines
    (no refraction or reflection).

    .. note:: Don't care about the units as long they are compatible.

    For a homogeneous model, *cells* can be a list with only one big cell.

    Parameters:

    * cells : list of :func:`~fatiando.mesher.dd.Square`
        The velocity model to use to trace the straight rays. Cells must have
        the physical property given in parameter *prop*. This will be used
        as the velocity of each cell. (*cells* can also be a
        :class:`~fatiando.mesher.dd.SquareMesh`)
    * prop : str
        Which physical property of the cells to use as velocity.
        Normaly one would choose ``'vp'`` or ``'vs'``
    * srcs : list fo lists
        List with [x, y] coordinate pairs of the wave sources.
    * recs : list fo lists
        List with [x, y] coordinate pairs of the receivers sources
    
    *srcs* and *recs* are lists of source-receiver pairs. Each source in *srcs*
    is associated with the corresponding receiver in *recs* for a given travel
    time.

    For example::

        >>> # One source was recorded at 3 receivers.
        >>> # The medium is homogeneous and can be
        >>> # represented by a single Square
        >>> from fatiando.mesher.dd import Square
        >>> cells = [Square([0, 10, 0, 10], {'vp':2})]
        >>> src = (5, 0)
        >>> srcs = [src, src, src]
        >>> recs = [(0, 0), (5, 10), (10, 0)]
        >>> print straight_ray_2d(cells, 'vp', srcs, recs)
        [ 2.5  5.   2.5]

    Returns:

    * times : array
        The total times each ray took to get from a source to a receiver (in
        compatible units with *prop*)

    """
    if len(srcs) != len(recs):
        raise ValueError("srcs and recs must have the same length")
    x_src, y_src = numpy.array(srcs, dtype='f').T
    x_rec, y_rec = numpy.array(recs, dtype='f').T
    times = numpy.zeros_like(x_src)
    for c in cells:
        if c is not None:
            times += _traveltime.straight_ray_2d(float(c[prop]),
                float(c['x1']), float(c['y1']), float(c['x2']), float(c['y2']),
                x_src, y_src, x_rec, y_rec)
    return times
