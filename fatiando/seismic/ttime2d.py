"""
Calculate travel-times of seismic waves in 2D.

* :func:`~fatiando.seismic.ttime2d.straight`: Calculate the travel-time of a
  straight ray through a mesh of square cells

----

"""
import math
import numpy

from fatiando import logger

log = logger.dummy('fatiando.seismic.ttime2d')


def straight(cells, prop, srcs, recs, velocity=None):
    """
    Calculate the travel times inside a mesh of square cells between source and
    receiver pairs assuming the rays are straight lines (no refraction or
    reflection).

    .. note:: Don't care about the units as long they are compatible.

    For a homogeneous model, *cells* can be a list with only one big cell.

    Parameters:

    * cells : list of :func:`fatiando.mesher.dd.Square`
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
    * velocity : float or None
        If not None, will use this value instead of the prop of cells as the
        velocity. Useful when building sensitivity matrices (use velocity = 1).
    
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
        >>> print straight(cells, 'vp', srcs, recs)
        [ 2.5  5.   2.5]

    Returns:

    * times : array
        The total times each ray took to get from a source to a receiver (in
        compatible units with *prop*)

    """
    if len(srcs) != len(recs):
        raise ValueError("Must have the same number of sources and receivers")
    times = numpy.zeros(len(srcs), dtype='f')
    for l in xrange(len(times)):
        x_src, y_src = srcs[l]
        x_rec, y_rec = recs[l]
        maxx = max(x_src, x_rec)
        maxy = max(y_src, y_rec)
        minx = min(x_src, x_rec)
        miny = min(y_src, y_rec)
        for cell in cells:
            if cell is None or (prop not in cell and velocity is None):
                continue
            x1, x2, y1, y2 = cell['x1'], cell['x2'], cell['y1'], cell['y2']
            if velocity is None:
                vel = cell[prop]
            else:
                vel = velocity
            # Check if the cell is in the rectangle with the ray path as a
            # diagonal. If not, then the ray doesn't go through the cell.
            if x2 < minx or x1 > maxx or y2 < miny or y1 > maxy:
                continue
            # Now need to find the places where the ray intersects the cell
            # If the ray is vertical
            if (x_rec - x_src) == 0:
                xps = [x_rec]*4
                yps = [y_rec, y_src, y1, y2]
            # If the ray is horizontal
            elif (y_rec - y_src) == 0:
                xps = [x_rec, x_src, x1, x2]
                yps = [y_rec]*4
            else:
                # Angular and linear coefficients of the ray
                a_ray = float(y_rec - y_src)/(x_rec - x_src)
                b_ray = y_src - a_ray*(x_src)                
                # Add the src and rec locations so that the travel time of a src
                # or rec inside a cell is accounted for
                xps = [x1,  x2, (y1 - b_ray)/a_ray, (y2 - b_ray)/a_ray, x_src,
                       x_rec]                
                yps = [a_ray*x1 + b_ray, a_ray*x2 + b_ray, y1, y2, y_src, y_rec]
            # Find out how many points are inside both the cell and the
            # rectangle with the ray path as a diagonal
            incell = lambda x, y: x <= x2 and x >= x1 and y <= y2 and y >= y1
            inray = \
                lambda x, y: x <= maxx and x >= minx and y <= maxy and y >= miny
            cross = [[x, y] for x, y in zip(xps, yps)
                     if incell(x, y) and inray(x, y)]
            # Remove the duplicates
            cross = [p for i, p in enumerate(cross) if p not in cross[0:i]]
            if len(cross) > 2:
                raise ValueError('More than 2 crossings ' +
                    'for cell %s and ray src:%s rec:%s'
                    % (str(cell), str(srcs[l]), str(recs[l])))
            if len(cross) == 2:
                p1, p2 = cross
                distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                times[l] += distance/float(vel)
    return times
