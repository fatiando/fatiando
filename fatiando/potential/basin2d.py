# Copyright 2012 The Fatiando a Terra Development Team
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
Estimate the basement relief of two-dimensional basins from potential field
data.

**POLYGONAL PARAMETRIZATION**

* :func:`fatiando.potential.basin2d.triangular`
* :func:`fatiando.potential.basin2d.trapezoidal`

Uses 2D bodies with a polygonal cross-section to parameterize the basin relief.
Potential fields are calculated using the :mod:`fatiando.potential.talwani`
module. **WARNING**: Vertices of polygons must always be in clockwise order!

**Triangular basin**

Use when the basin can be approximated by a 2D body with **triangular** vertical
cross-section. The triangle is assumed to have 2 known vertices at the surface
(the edges of the basin) and one unknown vertice in the subsurface. The
inversion will then estimate the (x, z) coordinates of the unknown vertice.

Example using synthetic data::

    >>> import numpy
    >>> from fatiando.mesher.dd import Polygon
    >>> from fatiando.potential import talwani
    >>> from fatiando.inversion.gradient import levmarq
    >>> # Make a triangular basin model (will estimate the last point)
    >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
    >>> left, middle, right = verts
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = TriangularGzDM(xs, zs, gz, prop=500, verts=[left, middle])
    >>> # Estimate the coordinates of the last point using Levenberg-Marquardt
    >>> solver = levmarq(initial=(10000, 1000))
    >>> p, residuals = triangular([dm], solver)
    >>> print '%.1f, %.1f' % (p[0], p[1])
    50000.0, 5000.0

Same example but this time using ``iterate=True`` to view the steps of the
algorithm::

    >>> import numpy
    >>> from fatiando.mesher.dd import Polygon
    >>> from fatiando.potential import talwani
    >>> from fatiando.inversion.gradient import levmarq
    >>> # Make a triangular basin model (will estimate the last point)
    >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
    >>> left, middle, right = verts
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = TriangularGzDM(xs, zs, gz, prop=500, verts=[left, middle])
    >>> # Estimate the coordinates of the last point using Levenberg-Marquardt
    >>> solver = levmarq(initial=(70000, 2000))
    >>> for p, residuals in triangular([dm], solver, iterate=True):
    ...     print '%.4f, %.4f' % (p[0], p[1])
    70000.0000, 2000.0000
    69999.8789, 2005.4746
    69998.6811, 2059.0979
    69986.4657, 2502.6963
    69843.9888, 3960.5020
    67972.7649, 4728.4963
    59022.3155, 4820.1361
    50714.4193, 4952.5626
    50001.0132, 4999.4348
    50000.0020, 5000.0002
    49999.9981, 5000.0002

**Trapezoidal basin**

Use when the basin can be approximated by a 2D body with **trapezoidal**
vertical cross-section.
The trapezoid is assumed to have 2 known vertices at the surface
(the edges of the basin) and two unknown vertice in the subsurface.
We assume that the x coordinates of the unknown vertices are the same as the x
coordinates of the known vertices (i.e., the unknown vertices are directly under
the known vertices). The inversion will then estimate the z coordinates of the
unknown vertices.

Example of inverting for the z coordinates of the unknown vertices::

    >>> import numpy
    >>> from fatiando.mesher.dd import Polygon
    >>> from fatiando.potential import talwani
    >>> from fatiando.inversion.gradient import levmarq
    >>> # Make a trapezoidal basin model (will estimate the last two point)
    >>> verts = [(10000, 1), (90000, 1), (90000, 5000), (10000, 3000)]
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = TrapezoidalGzDM(xs, zs, gz, prop=500, verts=verts[0:2])
    >>> # Estimate the coordinates of the two z coords using Levenberg-Marquardt
    >>> solver = levmarq(initial=(1000, 500))
    >>> p, residuals = trapezoidal([dm], solver)
    >>> print '%.1f, %.1f' % (p[0], p[1])
    5000.0, 3000.0

Same example but this time using ``iterate=True`` to view the steps of the
algorithm::

    >>> import numpy
    >>> from fatiando.mesher.dd import Polygon
    >>> from fatiando.potential import talwani
    >>> from fatiando.inversion.gradient import levmarq
    >>> # Make a trapezoidal basin model (will estimate the last two point)
    >>> verts = [(10000, 5), (90000, 10), (90000, 5000), (10000, 3000)]
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = TrapezoidalGzDM(xs, zs, gz, prop=500, verts=verts[0:2])
    >>> # Estimate the coordinates of the two z coords using Levenberg-Marquardt
    >>> solver = levmarq(initial=(1000, 500))
    >>> for p, residuals in trapezoidal([dm], solver, iterate=True):
    ...     print '%.4f, %.4f' % (p[0], p[1])
    1000.0000, 500.0000
    1010.4376, 509.4190
    1111.6982, 600.5537
    1888.0891, 1281.9118
    3926.6116, 2780.5289
    4903.8182, 3040.3440
    4998.6975, 3001.0088
    4999.9983, 3000.0017
    4999.9998, 3000.0000

**PRISMATIC PARAMETRIZATION**

* :func:`fatiando.potential.basin2d.prism`
 
Uses juxtaposed 2D right rectangular prisms to parameterize the basin relief.
Potential fields are calculated using the :mod:`fatiando.potential.prism2d`
module. 

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Jan-2012'

import time
import itertools
import numpy

from fatiando.potential import _talwani
from fatiando import inversion, utils, logger

log = logger.dummy()


class TriangularGzDM(inversion.datamodule.DataModule):
    """
    Data module for the inversion to estimate the relief of a triangular basin.

    Packs the necessary data and interpretative model information.

    The forward modeling is done using :mod:`fatiando.potential.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.
    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * xp, zp
        Arrays with the x and z coordinates of the profile data points
    * data
        Array with the profile data
    * verts
        List with the (x, z) coordinates of the two know vertices. Very
        important that the vertices in the list be ordered from left to right!
        Otherwise the forward model will give results with an inverted sign and
        terrible things may happen!
    * prop
        Value of the physical property of the basin. The physical property must
        be compatible with the potential field used! I.e., gravitational fields
        require a value of density contrast.
    * delta
        Interval used to calculate the approximate derivatives
        
    """

    def __init__(self, xp, zp, data, verts, prop, delta=1.):
        inversion.datamodule.DataModule.__init__(self, data)
        log.info("Initializing TriangularDM data module:")
        if len(xp) != len(zp) != len(data):
            raise ValueError, "xp, zp, and data must be of same length"
        if len(verts) != 2:
            raise ValueError, "Need exactly 2 vertices. %d given" % (len(verts))
        self.xp = numpy.array(xp, dtype=numpy.float64)
        self.zp = numpy.array(zp, dtype=numpy.float64)
        self.prop = float(prop)
        self.verts = verts
        self.delta = numpy.array([0., 0., delta], dtype='f')
        log.info("  number of data: %d" % (len(data)))
        log.info("  physical property: %s" % (str(prop)))
        
    def get_predicted(self, p):
        tmp = [v for v in self.verts]
        tmp.append(p)
        xs, zs = numpy.array(tmp, dtype='f').T
        return _talwani.talwani_gz(self.prop, xs, zs, self.xp, self.zp)

    def sum_gradient(self, gradient, p, residuals):
        xp, zp = self.xp, self.zp
        delta = self.delta
        tmp = [v for v in self.verts]
        tmp.append(p)
        xs, zs = numpy.array(tmp, dtype='f').T
        at_p = _talwani.talwani_gz(self.prop, xs, zs, xp, zp)
        jacx = ((_talwani.talwani_gz(self.prop, xs + delta, zs, xp, zp) - at_p)/
                delta[-1])
        jacz = ((_talwani.talwani_gz(self.prop, xs, zs + delta, xp, zp) - at_p)/
                delta[-1])
        self.jac_T = numpy.array([jacx, jacz])
        return gradient - 2.*numpy.dot(self.jac_T, residuals)

    def sum_hessian(self, hessian, p):
        return hessian + 2*numpy.dot(self.jac_T, self.jac_T.T)

def triangular(dms, solver, iterate=False):
    """
    Estimate basement relief of a triangular basin. The basin is modeled as a
    triangle with two known vertices at the surface. The parameters estimated
    are the x and z coordinates of the third vertice.

    Parameters:

    * dms
        List of data modules, like
        :class:`fatiando.potential.basin2d.TriangularGzDM`
    * solver
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module.
    * iterate
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.
        
    Returns:

    * [estimate, residuals]
        The estimated (x, z) coordinates of the missing vertice and a list of
        the residuals (difference between measured and predicted data) for each
        data module in *dms*
    
    """
    log.info("Estimating relief of a triangular basin:")
    log.info("  iterate: %s" % (str(iterate)))
    if iterate:
        return _iterator(dms, solver)
    else:
        return _solver(dms, solver)

class TrapezoidalGzDM(inversion.datamodule.DataModule):
    """
    Data module for the inversion to estimate the relief of a trapezoidal basin.

    Packs the necessary data and interpretative model information.

    The forward modeling is done using :mod:`fatiando.potential.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.
    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * xp, zp
        Arrays with the x and z coordinates of the profile data points
    * data
        Array with the profile data
    * verts
        List with the (x, z) coordinates of the two know vertices. 
    * prop
        Value of the physical property of the basin. The physical property must
        be compatible with the potential field used! I.e., gravitational fields
        require a value of density contrast.
    * delta
        Interval used to calculate the approximate derivatives

    **WARNING**: It is very important that the vertices in the list be ordered
    clockwise! Otherwise the forward model will give results with an inverted
    sign and terrible things may happen!
        
    """

    field = "gz"

    def __init__(self, xp, zp, data, verts, prop, delta=1.):
        inversion.datamodule.DataModule.__init__(self, data)
        log.info("Initializing %s data module for trapezoidal basin:" %
                (self.field))
        if len(xp) != len(zp) != len(data):
            raise ValueError, "xp, zp, and data must be of same length"
        if len(verts) != 2:
            raise ValueError, "Need exactly 2 vertices. %d given" % (len(verts))
        self.xp = numpy.array(xp, dtype=numpy.float64)
        self.zp = numpy.array(zp, dtype=numpy.float64)
        self.prop = float(prop)
        self.verts = verts
        log.info("  number of data: %d" % (len(data)))
        log.info("  physical property: %s" % (str(prop)))
        self.delta = delta
        self.d1 = numpy.array([0., 0., delta, 0.], dtype='f')
        self.d2 = numpy.array([0., 0., 0., delta], dtype='f')
        self.xs = [x for x in reversed(numpy.array(verts).T[0])]
                    
    def get_predicted(self, p):
        tmp = [[x, z] for x, z in zip(self.xs, p)]
        vertices = itertools.chain(self.verts, tmp)
        xs, zs = numpy.array([v for v in vertices], dtype='f').T
        return _talwani.talwani_gz(self.prop, xs, zs, self.xp, self.zp)

    def sum_gradient(self, gradient, p, residuals):
        xp, zp = self.xp, self.zp
        tmp = [[x, z] for x, z in zip(self.xs, p)]
        vertices = itertools.chain(self.verts, tmp)
        xs, zs = numpy.array([v for v in vertices], dtype='f').T        
        at_p = _talwani.talwani_gz(self.prop, xs, zs, xp, zp)
        jacz1 = ((_talwani.talwani_gz(self.prop, xs, zs + self.d1, xp, zp) -
                  at_p)/self.delta)
        jacz2 = ((_talwani.talwani_gz(self.prop, xs, zs + self.d2, xp, zp) -
                  at_p)/self.delta)
        self.jac_T = numpy.array([jacz1, jacz2])
        return gradient - 2.*numpy.dot(self.jac_T, residuals)

    def sum_hessian(self, hessian, p):
        return hessian + 2*numpy.dot(self.jac_T, self.jac_T.T)

def trapezoidal(dms, solver, iterate=False):
    """
    Estimate basement relief of a triangular basin. The basin is modeled as a
    triangle with two known vertices at the surface. The parameters estimated
    are the x and z coordinates of the third vertice.

    Parameters:

    * dms
        List of data modules, like
        :class:`fatiando.potential.basin2d.TriangularGzDM`
    * solver
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module.
    * iterate
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.
        
    Returns:

    * [estimate, residuals]
        The estimated (x, z) coordinates of the missing vertice and a list of
        the residuals (difference between measured and predicted data) for each
        data module in *dms*
    
    """
    log.info("Estimating relief of a trapezoidal basin:")
    log.info("  iterate: %s" % (str(iterate)))
    if iterate:
        return _iterator(dms, solver)
    else:
        return _solver(dms, solver)

def _solver(dms, solver):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, [])):
            continue
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
    return chset['estimate'], chset['residuals']

def _iterator(dms, solver):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, [])):
            yield chset['estimate'], chset['residuals']
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))    
    
def prism():
    pass
            
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
    
