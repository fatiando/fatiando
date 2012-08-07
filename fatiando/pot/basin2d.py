"""
Estimate the basement relief of two-dimensional basins from potential field
data.

**POLYGONAL PARAMETRIZATION**

* :func:`~fatiando.pot.basin2d.triangular`
* :func:`~fatiando.pot.basin2d.trapezoidal`

Uses 2D bodies with a polygonal cross-section to parameterize the basin relief.
Potential fields are calculated using the :mod:`~fatiando.pot.talwani`
module.

.. warning:: Vertices of polygons must always be in clockwise order!

**Triangular basin**

Use when the basin can be approximated by a 2D body with **triangular** vertical
cross-section. The triangle is assumed to have 2 known vertices at the surface
(the edges of the basin) and one unknown vertice in the subsurface. The
inversion will then estimate the (x, z) coordinates of the unknown vertice.

Example using synthetic data::

    >>> import numpy
    >>> import fatiando as ft
    >>> # Make a triangular basin model (will estimate the last point)
    >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
    >>> left, middle, right = verts
    >>> model = ft.msh.dd.Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = ft.pot.talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = ft.pot.basin2d.TriangularGzDM(xs, zs, gz, prop=500,
    ...                                    verts=[left, middle])
    >>> # Estimate the coordinates of the last point using Levenberg-Marquardt
    >>> solver = ft.inversion.gradient.levmarq(initial=(10000, 1000))
    >>> p, residuals = ft.pot.basin2d.triangular([dm], solver)
    >>> print '%.1f, %.1f' % (p[0], p[1])
    50000.0, 5000.0

Same example but this time using ``iterate=True`` to view the steps of the
algorithm::

    >>> import numpy
    >>> import fatiando as ft
    >>> # Make a triangular basin model (will estimate the last point)
    >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
    >>> left, middle, right = verts
    >>> model = ft.msh.dd.Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = ft.pot.talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = ft.pot.basin2d.TriangularGzDM(xs, zs, gz, prop=500,
    ...                                    verts=[left, middle])
    >>> # Estimate the coordinates of the last point using Levenberg-Marquardt
    >>> solver = ft.inversion.gradient.levmarq(initial=(70000, 2000))
    >>> iterator = ft.pot.basin2d.triangular([dm], solver, iterate=True)
    >>> for p, residuals in iterator:
    ...     print '%.4f, %.4f' % (p[0], p[1])
    70000.0000, 2000.0000
    69999.8803, 2005.4746
    69998.6825, 2059.0979
    69986.4671, 2502.6963
    69843.9902, 3960.5022
    67972.7679, 4728.4970
    59022.3186, 4820.1359
    50714.4178, 4952.5628
    50001.0118, 4999.4345
    50000.0006, 4999.9999

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
    >>> import fatiando as ft
    >>> # Make a trapezoidal basin model (will estimate the last two point)
    >>> verts = [(10000, 1), (90000, 1), (90000, 5000), (10000, 3000)]
    >>> model = ft.msh.dd.Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = ft.pot.talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = ft.pot.basin2d.TrapezoidalGzDM(xs, zs, gz, prop=500,
    ...                                     verts=verts[0:2])
    >>> # Estimate the coordinates of the two z coords using Levenberg-Marquardt
    >>> solver = ft.inversion.gradient.levmarq(initial=(1000, 500))
    >>> p, residuals = ft.pot.basin2d.trapezoidal([dm], solver)
    >>> print '%.1f, %.1f' % (p[0], p[1])
    5000.0, 3000.0

Same example but this time using ``iterate=True`` to view the steps of the
algorithm::

    >>> import numpy
    >>> import fatiando as ft
    >>> # Make a trapezoidal basin model (will estimate the last two point)
    >>> verts = [(10000, 5), (90000, 10), (90000, 5000), (10000, 3000)]
    >>> model = ft.msh.dd.Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> xs = numpy.arange(0, 100000, 10000)
    >>> zs = numpy.zeros_like(xs)
    >>> gz = ft.pot.talwani.gz(xs, zs, [model])
    >>> # Pack the data nicely in a DataModule
    >>> dm = ft.pot.basin2d.TrapezoidalGzDM(xs, zs, gz, prop=500,
    ...                                     verts=verts[0:2])
    >>> # Estimate the coordinates of the two z coords using Levenberg-Marquardt
    >>> solver = ft.inversion.gradient.levmarq(initial=(1000, 500))
    >>> iterator = ft.pot.basin2d.trapezoidal([dm], solver, iterate=True)
    >>> for p, residuals in iterator:
    ...     print '%.4f, %.4f' % (p[0], p[1])
    1000.0000, 500.0000
    1010.4375, 509.4191
    1111.6975, 600.5546
    1888.0846, 1281.9163
    3926.6071, 2780.5317
    4903.8174, 3040.3444
    4998.6977, 3001.0087
    4999.9980, 3000.0017
    5000.0000, 2999.9999

----

"""

import time
import itertools
import numpy

from fatiando.pot import talwani
from fatiando.msh.dd import Polygon
from fatiando import inversion, utils
import fatiando.log


log = fatiando.log.dummy('fatiando.pot.basin2d')

class TriangularGzDM(inversion.datamodule.DataModule):
    """
    Data module for the inversion to estimate the relief of a triangular basin.

    Packs the necessary gravity anomaly data and interpretative model
    information.

    The forward modeling is done using :mod:`~fatiando.pot.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.
    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * xp, zp : array
        Arrays with the x and z coordinates of the profile data points
    * data : array
        The profile gravity anomaly data
    * verts : list of lists
        List of the [x, z] coordinates of the two know vertices. Very
        important that the vertices in the list be ordered from left to right!
        Otherwise the forward model will give results with an inverted sign and
        terrible things may happen!
    * prop : float
        Density contrast of the basin
    * delta : float
        Interval used to calculate the approximate derivatives

    .. warning:: It is very important that the vertices in the list be ordered
        clockwise! Otherwise the forward model will give results with an
        inverted sign and terrible things may happen!

    """

    def __init__(self, xp, zp, data, verts, prop, delta=1.):
        inversion.datamodule.DataModule.__init__(self, data)
        if len(xp) != len(zp) != len(data):
            raise ValueError, "xp, zp, and data must be of same length"
        if len(verts) != 2:
            raise ValueError, "Need exactly 2 vertices. %d given" % (len(verts))
        self.xp = numpy.array(xp, dtype=numpy.float64)
        self.zp = numpy.array(zp, dtype=numpy.float64)
        self.prop = {'density':prop}
        self.verts = list(verts)
        self.delta = delta

    def get_predicted(self, p):
        polygon = Polygon(self.verts + [p], self.prop)
        return talwani.gz(self.xp, self.zp, [polygon])

    def sum_gradient(self, gradient, p, residuals):
        polygon = Polygon(self.verts + [p], self.prop)
        at_p = talwani.gz(self.xp, self.zp, [polygon])
        polygon = Polygon(self.verts + [[p[0] + self.delta, p[1]]], self.prop)
        jacx = (talwani.gz(self.xp, self.zp, [polygon]) - at_p)/self.delta
        polygon = Polygon(self.verts + [[p[0], p[1] + self.delta]], self.prop)
        jacz = (talwani.gz(self.xp, self.zp, [polygon]) - at_p)/self.delta
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

    * dms : list
        The data modules, like
        :class:`~fatiando.pot.basin2d.TriangularGzDM`
    * solver : function
        A non-linear inverse problem solver generated by a factory function
        from a module of the :mod:`~fatiando.inversion` package.
    * iterate : True or False
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    Returns:

    * results : list = [estimate, residuals]:

        * estimate : array
            The estimated [x, z] coordinates of the missing vertice
        * residuals : list of arrays
            The residuals of the inversion (difference between measured and
            predicted data) for each data module in *dms*

    """
    log.info("Estimating relief of a triangular basin:")
    log.info("  iterate: %s" % (str(iterate)))
    if iterate:
        return _iterator(dms, solver, log)
    else:
        return _solver(dms, solver, log)

class TrapezoidalGzDM(inversion.datamodule.DataModule):
    """
    Data module for the inversion to estimate the relief of a trapezoidal basin.

    Packs the necessary data and interpretative model information.

    The forward modeling is done using :mod:`~fatiando.pot.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.
    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * xp, zp : array
        Arrays with the x and z coordinates of the profile data points
    * data : array
        The profile data
    * verts : list of lists
        List with the [x, z] coordinates of the two know vertices.
    * prop : float
        Value of the physical property of the basin. The physical property must
        be compatible with the potential field used! I.e., gravitational fields
        require a value of density contrast.
    * delta : float
        Interval used to calculate the approximate derivatives

    .. warning:: It is very important that the vertices in the list be ordered
        clockwise! Otherwise the forward model will give results with an
        inverted sign and terrible things may happen!

    """

    field = "gz"

    def __init__(self, xp, zp, data, verts, prop, delta=1.):
        inversion.datamodule.DataModule.__init__(self, data)
        if len(xp) != len(zp) != len(data):
            raise ValueError, "xp, zp, and data must be of same length"
        if len(verts) != 2:
            raise ValueError, "Need exactly 2 vertices. %d given" % (len(verts))
        self.xp = numpy.array(xp, dtype=numpy.float64)
        self.zp = numpy.array(zp, dtype=numpy.float64)
        self.prop = {'density':prop}
        self.verts = list(verts)
        self.delta = delta
        self.xs = [x for x in reversed(numpy.array(verts).T[0])]

    def get_predicted(self, p):
        x1, x2 = self.verts[1][0], self.verts[0][0]
        z1, z2 = p
        polygon = Polygon(self.verts + [[x1, z1], [x2, z2]], self.prop)
        return talwani.gz(self.xp, self.zp, [polygon])

    def sum_gradient(self, gradient, p, residuals):
        x1, x2 = self.verts[1][0], self.verts[0][0]
        z1, z2 = p
        delta = self.delta
        polygon = Polygon(self.verts + [[x1, z1], [x2, z2]], self.prop)
        at_p = talwani.gz(self.xp, self.zp, [polygon])
        polygon = Polygon(self.verts + [[x1, z1 + delta], [x2, z2]], self.prop)
        jacz1 = (talwani.gz(self.xp, self.zp, [polygon]) - at_p)/delta
        polygon = Polygon(self.verts + [[x1, z1], [x2, z2 + delta]], self.prop)
        jacz2 = (talwani.gz(self.xp, self.zp, [polygon]) - at_p)/delta
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

    * dms : list
        List of data modules, like
        :class:`~fatiando.pot.basin2d.TriangularGzDM`
    * solver : function
        A non-linear inverse problem solver generated by a factory function
        from a module of the :mod:`~fatiando.inversion` package.
    * iterate : True or False
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    Returns:

    * results : list = [estimate, residuals]:

        * estimate : array
            The estimated [z1, z2] coordinates of the bottom vertices.
        * residuals : list of arrays
            The residuals of the inversion (difference between measured and
            predicted data) for each data module in *dms*

    """
    log.info("Estimating relief of a trapezoidal basin:")
    log.info("  iterate: %s" % (str(iterate)))
    if iterate:
        return _iterator(dms, solver, log)
    else:
        return _solver(dms, solver, log)

def _solver(dms, solver, log):
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

def _iterator(dms, solver, log):
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
