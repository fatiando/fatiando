# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Direct modeling of seismic wave travel times.

**Straight rays**

* :func:`fatiando.seismic.traveltime.straight_ray_2d`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import numpy

from fatiando.seismic import _traveltime
from fatiando import logger

log = logger.dummy()


def straight_ray_2d(cells, prop, srcs, recs):
    """
    Calculate the travel times inside a list of 2D square cells between
    source and receiver pairs assuming the rays are straight lines
    (no refraction or reflection).

    **NOTE**: Don't care about the units as long they are compatible.

    For a homogeneous model, *cells* can be a list with only one big cell.

    Parameters:

    * cells
        List of square cells (:func:`fatiando.mesher.dd.Square` or
        :class:`fatiando.mesher.dd.SquareMesh`)
    * prop
        String with which physical property of the cells to use as velocity.
        Normaly one would choose ``'vp'`` or ``'vs'``
    * srcs
        List with [x, y] coordinate pairs of the wave sources.
    * recs
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

    * times
        Array with the total times each ray took to get from a source to a
        receiver (in compatible units with *prop*)

    """
    if len(srcs) != len(recs):
        raise ValueError, "srcs and recs must have the same length"
    x_src, y_src = numpy.array(srcs, dtype='f').T
    x_rec, y_rec = numpy.array(recs, dtype='f').T
    times = numpy.zeros_like(x_src)
    for c in cells:
        if c is not None:
            times += _traveltime.straight_ray_2d(float(1./c[prop]),
                float(c['x1']), float(c['y1']), float(c['x2']), float(c['y2']),
                x_src, y_src, x_rec, y_rec)
    return times
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()

