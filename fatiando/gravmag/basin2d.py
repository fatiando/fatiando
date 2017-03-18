"""
2D inversion for the basement relief of sedimentary basins.

There are different parametrizations available.

The simplest are meant more as an exercise and initiation in inverse problems:

* :func:`~fatiando.gravmag.basin2d.Triangular`: assumes a basin with a
  triangular cross-section (think "foreland").
* :func:`~fatiando.gravmag.basin2d.Trapezoidal`: assumes a basin with a
  trapezoidal cross-section (think "grabben").

More complex parametrizations are:

* :func:`~fatiando.gravmag.basin2d.PolygonalBasinGravity`: approximate the
  basin by a polygon.

----

"""
from __future__ import division, absolute_import
from future.builtins import super, range
import numpy as np

from ..inversion.misfit import Misfit
from . import talwani
from ..mesher import Polygon
from .. import utils


class PolygonalBasinGravity(Misfit):
    """
    Estimate the relief of a sedimentary basin approximating by a polygon.

    Currently only works with gravity data.

    The top of the basin is straight and fixed at a given height. Polygon
    vertices are distributed evenly in the x-direction.  The inversion
    estimates the depths of each vertex.

    This is a non-linear inversion. Therefore you must configure it before
    running to choose a solver method and set the initial estimate.
    Use the ``config`` method for this.

    Recommended configuration: Levemberg-Marquardt algorithm (``'levmarq'``)
    with initial estimate to the average expected depth of the basin.

    Typical regularization to use with this class are:
    :class:`~fatiando.inversion.regularization.Damping`,
    :class:`~fatiando.inversion.regularization.Smoothness1D`,
    :class:`~fatiando.inversion.regularization.TotalVariation1D`.

    The forward modeling is done using :mod:`~fatiando.gravmag.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.

    .. tip::

        Use the ``estimate_`` attribute to get a
        :class:`~fatiando.mesher.Polygon` version of the estimated parameters
        (attribute ``p_``).

    Parameters:

    * x, z : 1d-arrays
        The x and z coordinates of the observations. In meters.
    * data : 1d-array
        The observed data.
    * npoints : int
        Number of points to use
    * props : dict
        The physical properties dictionary that will be assigned to the
        basin :class:`~fatiando.mesher.Polygon`. Ex: to give the basin a
        density contrast of 500 kg/m3 ``props={'density': 500}``.
    * top : float
        The value of the z-coordinate where the top of the basin will be fixed.
        In meters. Default: 0.
    * xlim : None or list = [xmin, xmax]
        The horizontal limits of the model. If not given, will use the limits
        of the data (i.e., ``[x.min(), x.max()]``).

    Examples:

    Lets run an inversion on synthetic data from a simple model of a trapezoid
    basin (a polygon with 4 vertices). We'll assume that the horizontal limits
    of the basin are the same as the limits of the data:

    >>> from fatiando.mesher import Polygon
    >>> from fatiando.gravmag import talwani
    >>> import numpy as np
    >>> # Make some synthetic data from a simple basin
    >>> props = {'density': -500}
    >>> model = [Polygon([[3000, 0], [2000, 800], [1000, 500], [0, 0]],
    ...                  props)]
    >>> x = np.linspace(0, 3000, 50)
    >>> z = -np.ones_like(x)  # Put data at 1m height
    >>> data = talwani.gz(x, z, model)
    >>> # Make the solver, configure, and invert.
    >>> # Will use only 2 points because the two in the corners are
    >>> # considered fixed in the inversion (at 'top').
    >>> misfit = PolygonalBasinGravity(x, z, data, 2, props, top=0)
    >>> _ = misfit.config('levmarq', initial=100*np.ones(misfit.nparams)).fit()
    >>> misfit.p_
    array([ 800.,  500.])
    >>> type(misfit.estimate_)
    <class 'fatiando.mesher.geometry.Polygon'>
    >>> misfit.estimate_.vertices
    array([[ 3000.,     0.],
           [ 2000.,   800.],
           [ 1000.,   500.],
           [    0.,     0.]])

    If the x range of the data points is larger than the basin, you can specify
    a horizontal range for the basin model. When this is not specified, it is
    deduced from the data:

    >>> x = np.linspace(-500, 3500, 80)
    >>> z = -np.ones_like(x)
    >>> data = talwani.gz(x, z, model)
    >>> # Specify that the model used for inversion should be within
    >>> # x => [0, 3000]
    >>> misfit = PolygonalBasinGravity(x, z, data, 2, props, top=0,
    ...                                xlim=[0, 3000])
    >>> _ = misfit.config('levmarq', initial=100*np.ones(misfit.nparams)).fit()
    >>> misfit.p_
    array([ 800.,  500.])
    >>> misfit.estimate_.vertices
    array([[ 3000.,     0.],
           [ 2000.,   800.],
           [ 1000.,   500.],
           [    0.,     0.]])


    """
    def __init__(self, x, z, data, npoints, props, top=0, xlim=None):
        super().__init__(data=data.ravel(), nparams=npoints, islinear=False)
        self.npoints = npoints
        self.x = x
        self.z = z
        self.props = props
        self.top = top
        if xlim is None:
            xlim = [x.min(), x.max()]
        self.xlim = xlim
        self._modelx = np.linspace(xlim[0], xlim[1], npoints + 2)[::-1]

    def p2vertices(self, p):
        """
        Convert a parameter vector into vertices a Polygon.

        Parameters:

        * p : 1d-array
            The parameter vector with the depth of the polygon vertices

        Returns:

        * vertices : 2d-array
            Like a list of [x, z] coordinates of each vertex

        Examples:

        >>> import numpy as np
        >>> # Make some arrays to create the estimator clas
        >>> x = np.linspace(-100, 300, 50)
        >>> z = np.zeros_like(x)
        >>> data = z
        >>> misfit = PolygonalBasinGravity(x, z, data, 3, {}, top=-100)
        >>> misfit.p2vertices([1, 2, 3])
        array([[ 300., -100.],
               [ 200.,    1.],
               [ 100.,    2.],
               [   0.,    3.],
               [-100., -100.]])

        """
        h = self.top
        verts = np.empty((self.nparams + 2, 2))
        verts[:, 0] = self._modelx
        verts[:, 1] = np.concatenate([[h], p, [h]])
        return verts

    def predicted(self, p):
        """
        Calculate the predicted data for a parameter vector.
        """
        verts = self.p2vertices(p)
        poly = Polygon(verts, self.props)
        return talwani.gz(self.x, self.z, [poly])

    def jacobian(self, p):
        """
        Calculate the Jacobian (sensitivity) matrix for a parameter vector.
        """
        verts = self.p2vertices(p)
        delta = np.array([0, 1])
        jac = np.empty((self.ndata, self.nparams))
        for i in range(self.nparams):
            diff = Polygon([verts[i + 2], verts[i + 1] - delta,
                            verts[i], verts[i + 1] + delta], self.props)
            jac[:, i] = talwani.gz(self.x, self.z, [diff])/(2*delta[1])
        return jac

    def fmt_estimate(self, p):
        """
        Convert the parameter vector to a  :class:`fatiando.mesher.Polygon` so
        that it can be used for plotting and forward modeling.

        Examples:

        >>> import numpy as np
        >>> # Make some arrays to create the estimator clas
        >>> x = np.linspace(-100, 300, 50)
        >>> z = np.zeros_like(x)
        >>> data = z
        >>> misfit = PolygonalBasinGravity(x, z, data, 3, {}, top=-100)
        >>> poly = misfit.fmt_estimate([1, 2, 3])
        >>> type(poly)
        <class 'fatiando.mesher.geometry.Polygon'>
        >>> poly.vertices
        array([[ 300., -100.],
               [ 200.,    1.],
               [ 100.,    2.],
               [   0.,    3.],
               [-100., -100.]])

        """
        poly = Polygon(self.p2vertices(p), self.props)
        return poly


class Triangular(Misfit):
    """
    Estimate the relief of a triangular basin.

    Use when the basin can be approximated by a 2D body with **triangular**
    vertical cross-section, like foreland basins.

    The triangle is assumed to have 2 known vertices at the surface (the edges
    of the basin) and one unknown vertex in the subsurface.
    The inversion will estimate the (x, z) coordinates of the unknown vertex.

    The forward modeling is done using :mod:`~fatiando.gravmag.talwani`.
    Derivatives are calculated using a 2-point finite difference approximation.

    .. tip::

        Use the ``estimate_`` attribute to produce a polygon from the
        estimated parameter vector (``p_``).

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

    .. note::

        The recommended solver for this inverse problem is the
        Levemberg-Marquardt method. Since this is a non-linear problem, set the
        desired method and initial solution using the ``config`` method of
        this class. See the example bellow.

    Example using synthetic data:

    >>> import numpy as np
    >>> from fatiando.mesher import Polygon
    >>> from fatiando.gravmag import talwani
    >>> # Make a triangular basin model (will estimate the last point)
    >>> verts = [(10000, 1), (90000, 1), (50000, 5000)]
    >>> left, middle, right = verts
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> x = np.linspace(0, 100000, 50)
    >>> z = np.zeros_like(x)
    >>> gz = talwani.gz(x, z, [model])
    >>> # Make a solver and fit it to the data
    >>> solver = Triangular(x, z, gz, [left, middle], 500).config(
    ...     'levmarq', initial=[10000, 1000]).fit()
    >>> # p_ is the estimated parameter vector (x and z in this case)
    >>> x, z = solver.p_
    >>> print('{:.1f}, {:.1f}'.format(x, z))
    50000.0, 5000.0
    >>> # The parameter vector is not that useful so use estimate_ to get a
    >>> # Polygon object
    >>> poly = solver.estimate_
    >>> poly.vertices
    array([[  1.00000000e+04,   1.00000000e+00],
           [  9.00000000e+04,   1.00000000e+00],
           [  5.00000000e+04,   5.00000000e+03]])
    >>> poly.props
    {'density': 500}
    >>> # Check is the residuals are all small
    >>> np.all(np.abs(solver.residuals()) < 10**-10)
    True

    """

    def __init__(self, x, z, gz, verts, density):
        assert x.shape == z.shape == gz.shape, \
            "x, z, and data must be of same length"
        assert len(verts) == 2, \
            "Need exactly 2 vertices. {} given".format(len(verts))
        super().__init__(data=gz, nparams=2, islinear=False)
        self.x = np.array(x, dtype=np.float)
        self.z = np.array(z, dtype=np.float)
        self.density = density
        self.verts = list(verts)

    def predicted(self, p):
        """
        Calculate predicted data for a given parameter vector.
        """
        polygon = Polygon(self.verts + [p], {'density': self.density})
        return talwani.gz(self.x, self.z, [polygon])

    def jacobian(self, p):
        """
        Calculate the Jacobian (sensitivity) matrix for a given parameter
        vector.
        """
        delta = 1.
        props = {'density': self.density}
        xp, zp = self.x, self.z
        verts = self.verts
        x, z = p
        jac = np.transpose([
            (talwani.gz(xp, zp, [Polygon(verts + [[x + delta, z]], props)]) -
             talwani.gz(xp, zp, [Polygon(verts + [[x - delta, z]], props)])
             ) / (2. * delta),
            (talwani.gz(xp, zp, [Polygon(verts + [[x, z + delta]], props)]) -
             talwani.gz(xp, zp, [Polygon(verts + [[x, z - delta]], props)])
             ) / (2. * delta)])
        return jac

    def fmt_estimate(self, p):
        """
        Convert the parameter vector to a  :class:`~fatiando.mesher.Polygon` so
        that it can be used for plotting and forward modeling.
        """
        left, right = self.verts
        props = {'density': self.density}
        poly = Polygon(np.array([left, right, p]), props=props)
        return poly


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

        Use the ``estimate_`` attribute to produce a polygon from the
        estimated parameter vector (``p_``).


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

    .. note::

        The recommended solver for this inverse problem is the
        Levemberg-Marquardt method. Since this is a non-linear problem, set the
        desired method and initial solution using the ``config`` method.
        See the example bellow.

    Example with synthetic data:

    >>> import numpy as np
    >>> from fatiando.mesher import Polygon
    >>> from fatiando.gravmag import talwani
    >>> # Make a trapezoidal basin model (will estimate the z coordinates
    >>> # of the last two points)
    >>> verts = [[10000, 1], [90000, 1], [90000, 5000], [10000, 3000]]
    >>> model = Polygon(verts, {'density':500})
    >>> # Generate the synthetic gz profile
    >>> x = np.linspace(0, 100000, 50)
    >>> z = np.zeros_like(x)
    >>> gz = talwani.gz(x, z, [model])
    >>> # Make a solver and fit it to the data
    >>> solver = Trapezoidal(x, z, gz, verts[0:2], 500).config(
    ...     'levmarq', initial=[1000, 500]).fit()
    >>> # p_ is the estimated parameter vector (z1 and z2 in this case)
    >>> z1, z2 = solver.p_
    >>> print('{:.1f}, {:.1f}'.format(z1, z2))
    5000.0, 3000.0
    >>> # The parameter vector is not that useful so use estimate_ to get a
    >>> # Polygon object
    >>> poly = solver.estimate_
    >>> poly.vertices
    array([[  1.00000000e+04,   1.00000000e+00],
           [  9.00000000e+04,   1.00000000e+00],
           [  9.00000000e+04,   5.00000000e+03],
           [  1.00000000e+04,   3.00000000e+03]])
    >>> poly.props
    {'density': 500}
    >>> # Check is the residuals are all small
    >>> np.all(np.abs(solver.residuals()) < 10**-10)
    True

    """

    def __init__(self, x, z, gz, verts, density):
        assert x.shape == z.shape == gz.shape, \
            "x, z, and data must be of same length"
        assert len(verts) == 2, \
            "Need exactly 2 vertices. {} given".format(len(verts))
        super().__init__(data=gz, nparams=2, islinear=False)
        self.x = np.array(x, dtype=np.float)
        self.z = np.array(z, dtype=np.float)
        self.density = density
        self.props = {'density': self.density}
        self.verts = list(verts)
        self.x1, self.x2 = verts[1][0], verts[0][0]

    def predicted(self, p):
        z1, z2 = p
        model = [Polygon(self.verts + [[self.x1, z1], [self.x2, z2]],
                         self.props)]
        pred = talwani.gz(self.x, self.z, model)
        return pred

    def jacobian(self, p):
        z1, z2 = p
        x1, x2 = self.x1, self.x2
        x, z = self.x, self.z
        props = self.props
        verts = self.verts
        delta = 1.
        jac = np.empty((self.ndata, self.nparams), dtype=np.float)
        z1p = [Polygon(verts + [[x1, z1 + delta], [x2, z2]], props)]
        z1m = [Polygon(verts + [[x1, z1 - delta], [x2, z2]], props)]
        jac[:, 0] = (talwani.gz(x, z, z1p) - talwani.gz(x, z, z1m))/(2*delta)
        z2p = [Polygon(verts + [[x1, z1], [x2, z2 + delta]], props)]
        z2m = [Polygon(verts + [[x1, z1], [x2, z2 - delta]], props)]
        jac[:, 1] = (talwani.gz(x, z, z2p) - talwani.gz(x, z, z2m))/(2*delta)
        return jac

    def fmt_estimate(self, p):
        """
        Convert the parameter vector to a  :class:`fatiando.mesher.Polygon` so
        that it can be used for plotting and forward modeling.
        """
        z1, z2 = p
        left, right = self.verts
        poly = Polygon(np.array([left, right, [self.x1, z1], [self.x2, z2]]),
                       self.props)
        return poly
