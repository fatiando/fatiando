"""
Estimate the basement relief of two-dimensional basins from gravity data.

There are several parametrizations available. The simplest are:

* :func:`~fatiando.gravmag.basin2d.Triangular`
* :func:`~fatiando.gravmag.basin2d.Trapezoidal`

More complex and realistic parametrizations are:

* :func:`~fatiando.gravmag.basin2d.Polynomial`

----

"""
from __future__ import division
import numpy

from ..inversion.base import Misfit
from . import talwani
from ..mesher import Polygon
from .. import utils


class Triangular(Misfit):
    """
    Estimate the relief of a triangular basin.

    Use when the basin can be approximated by a 2D body with **triangular**
    vertical cross-section, like foreland basins.

    The triangle is assumed to have 2 known vertices at the surface (the edges
    of the basin) and one unknown vertice in the subsurface.
    The inversion will estimate the (x, z) coordinates of the unknown vertice.

    The forward modeling is done using :mod:`~fatiando.gravmag.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.
    The Hessian matrix is calculated using a Gauss-Newton approximation.

    .. tip::

        Use :meth:`~fatiando.gravmag.basin2d.Triangular.topolygon` to
        produce a polygon from the estimate returned by ``fit``.


    Parameters:

    * x, z : array
        Arrays with the x and z coordinates of the profile data points
    * gz : array
        The profile gravity anomaly data
    * verts : list of lists
        ``[[x1, z1], [x2, z2]]`` List of the [x, z] coordinates of the left and
        right know vertices, respectively.

        .. warning::

            Very important that the vertices in the list be ordered from left
            to right! Otherwise the forward model will give results with an
            inverted sign and terrible things may happen!

    * density : float
        Density contrast of the basin
    * delta : float
        Interval used to calculate the approximate derivatives

    Example using synthetic data::

        >>> import numpy
        >>> from fatiando.mesher import Polygon
        >>> from fatiando.gravmag import talwani
        >>> # Make a triangular basin model (will estimate the last point)
        >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
        >>> left, middle, right = verts
        >>> model = Polygon(verts, {'density':500})
        >>> # Generate the synthetic gz profile
        >>> x = numpy.arange(0, 100000, 10000)
        >>> z = numpy.zeros_like(x)
        >>> gz = talwani.gz(x, z, [model])
        >>> # Make a solver and fit it to the data
        >>> solver = Triangular(x, z, gz, [left, middle], 500)
        >>> x, z = solver.fit(initial=[10000, 1000])
        >>> print '%.1f, %.1f' % (x, z)
        50000.0, 5000.0

    """

    def __init__(self, x, z, gz, verts, density):
        if len(x) != len(z) != len(gz):
            raise ValueError("x, z, and data must be of same length")
        if len(verts) != 2:
            raise ValueError("Need exactly 2 vertices. %d given" %
                    (len(verts)))
        super(Triangular, self).__init__(
            data=gz,
            positional=dict(x=numpy.array(x, dtype=numpy.float),
                            z=numpy.array(z, dtype=numpy.float)),
            model=dict(density=density, verts=list(verts)),
            nparams=2, islinear=False)

    def _get_predicted(self, p):
        polygon = Polygon(self.model['verts'] + [p],
                          {'density':self.model['density']})
        x, z = self.positional['x'], self.positional['z']
        return talwani.gz(x, z, [polygon])

    def _get_jacobian(self, p):
        delta = 1.
        props = {'density':self.model['density']}
        verts = self.model['verts']
        xp, zp = self.positional['x'], self.positional['z']
        x, z = p
        jac = numpy.transpose([
            (talwani.gz(xp, zp, [Polygon(verts + [[x + delta, z]], props)])
             - talwani.gz(xp, zp, [Polygon(verts + [[x - delta, z]], props)])
            )/(2.*delta),
            (talwani.gz(xp, zp, [Polygon(verts + [[x, z + delta]], props)])
             - talwani.gz(xp, zp, [Polygon(verts + [[x, z - delta]], props)])
            )/(2.*delta)])
        return jac

    def topolygon(self, p):
        """
        Convert the estimated parameter vector *p* to a Polygon.

        This way, the :class:`fatiando.mesher.Polygon` can be passed to
        plotting functions and forward modeling functions.

        Parameters:

        * p : 1d-array
            The estimate parameter vector. Produced by fitting the data (using
            :meth:`~fatiando.gravmag.basin2d.Triangular.fit` for example).

        Returns:

        * polygon : :class:`fatiando.mesher.Polygon`
            A polygon representation of the estimate.

        """
        left, right = self.model['verts']
        props = {'density':self.model['density']}
        return Polygon([left, right, p], props=props)

class Trapezoidal(Misfit):
    """
    Estimate the relief of a trapezoidal basin.

    Use when the basin can be approximated by a 2D body with **trapezoidal**
    vertical cross-section, like in rifts.

    The trapezoid is assumed to have 2 known vertices at the surface
    (the edges of the basin) and two unknown vertices in the subsurface.
    We assume that the x coordinates of the unknown vertices are the same as
    the x coordinates of the known vertices (i.e., the unknown vertices are
    directly under the known vertices).
    The inversion will then estimate the z coordinates of the unknown vertices.

    The forward modeling is done using :mod:`~fatiando.gravmag.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.

    .. tip::

        Use :meth:`~fatiando.gravmag.basin2d.Trapezoidal.topolygon` to
        produce a polygon from the estimate returned by ``fit``.


    Parameters:

    * x, z : array
        Arrays with the x and z coordinates of the profile data points
    * gz : array
        The profile gravity anomaly data
    * verts : list of lists
        ``[[x1, z1], [x2, z2]]`` List of the [x, z] coordinates of the left and
        right know vertices, respectively.

        .. warning::

            Very important that the vertices in the list be ordered from left
            to right! Otherwise the forward model will give results with an
            inverted sign and terrible things may happen!

    * density : float
        Density contrast of the basin
    * delta : float
        Interval used to calculate the approximate derivatives

    Example with synthetic data:

        >>> import numpy
        >>> from fatiando.mesher import Polygon
        >>> from fatiando.gravmag import talwani
        >>> # Make a trapezoidal basin model (will estimate the z coordinates
        >>> # of the last two points)
        >>> verts = [(10000, 1), (90000, 1), (90000, 5000), (10000, 3000)]
        >>> model = Polygon(verts, {'density':500})
        >>> # Generate the synthetic gz profile
        >>> x = numpy.arange(0, 100000, 10000)
        >>> z = numpy.zeros_like(x)
        >>> gz = talwani.gz(x, z, [model])
        >>> # Make a solver and fit it to the data
        >>> solver = Trapezoidal(x, z, gz, verts[0:2], 500)
        >>> z1, z2 = solver.fit(initial=[1000, 500])
        >>> print '%.1f, %.1f' % (z1, z2)
        5000.0, 3000.0

    """

    def __init__(self, x, z, gz, verts, density):
        if len(x) != len(z) != len(gz):
            raise ValueError, "x, z, and data must be of same length"
        if len(verts) != 2:
            raise ValueError, "Need exactly 2 vertices. %d given" % (len(verts))
        super(Trapezoidal, self).__init__(
            data=gz,
            positional=dict(x=numpy.array(x, dtype=numpy.float),
                            z=numpy.array(z, dtype=numpy.float)),
            model=dict(density=density, verts=list(verts)),
            nparams=2, islinear=False)

    def _get_predicted(self, p):
        z1, z2 = p
        x1, x2 = self.model['verts'][1][0], self.model['verts'][0][0]
        x, z = self.positional['x'], self.positional['z']
        props = {'density':self.model['density']}
        pred = talwani.gz(x, z,
            [Polygon(self.model['verts'] + [[x1, z1], [x2, z2]], props)])
        return pred

    def _get_jacobian(self, p):
        z1, z2 = p
        x1, x2 = self.model['verts'][1][0], self.model['verts'][0][0]
        props = {'density':self.model['density']}
        x, z = self.positional['x'], self.positional['z']
        verts = self.model['verts']
        delta = 1.
        jac = numpy.transpose([
            (talwani.gz(x, z,
                [Polygon(verts + [[x1, z1 + delta], [x2, z2]], props)])
             - talwani.gz(x, z,
                [Polygon(verts + [[x1, z1 - delta], [x2, z2]], props)])
            )/(2.*delta),
            (talwani.gz(x, z,
                [Polygon(verts + [[x1, z1], [x2, z2 + delta]], props)])
             - talwani.gz(x, z,
                [Polygon(verts + [[x1, z1], [x2, z2 - delta]], props)])
            )/(2.*delta)])
        return jac

    def topolygon(self, p):
        """
        Convert the estimated parameter vector *p* to a Polygon.

        This way, the :class:`fatiando.mesher.Polygon` can be passed to
        plotting functions and forward modeling functions.

        Parameters:

        * p : 1d-array
            The estimate parameter vector. Produced by fitting the data (using
            :meth:`~fatiando.gravmag.basin2d.Trapezoidal.fit` for example).

        Returns:

        * polygon : :class:`fatiando.mesher.Polygon`
            A polygon representation of the estimate.

        """
        z1, z2 = p
        x1, x2 = self.model['verts'][1][0], self.model['verts'][0][0]
        props = {'density':self.model['density']}
        left, right = self.model['verts']
        return Polygon([left, right, (x1, z1), (x2, z2)], props)
