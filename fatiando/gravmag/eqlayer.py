"""
Equivalent layer processing.

Use the classes here to estimate an equivalent layer from potential field data.
Then you can use the estimated layer to perform tranformations (gridding,
continuation, derivation, reduction to the pole, etc.) by forward modeling
the layer. Use :mod:`fatiando.gravmag.sphere` for forward modeling.

**Algorithms**

* :class:`~fatiando.gravmag.eqlayer.EQLGravity` and
  :class:`~fatiando.gravmag.eqlayer.EQLTotalField`: The classic (space domain)
  equivalent layer as formulated in Li and Oldenburg (2010) or
  Oliveira Jr. et al (2012).
  Doesn't have wavelet compression or other tweaks.
* :class:`~fatiando.gravmag.eqlayer.PELGravity` and
  :class:`~fatiando.gravmag.eqlayer.PELTotalField`: The polynomial equivalent
  layer of Oliveira Jr. et al (2012). A fast and memory efficient algorithm.
  Both of these require special regularization
  (:class:`~fatiando.gravmag.eqlayer.PELSmoothness`).

**References**

Li, Y., and D. W. Oldenburg (2010), Rapid construction of equivalent sources
using wavelets, Geophysics, 75(3), L51-L59, doi:10.1190/1.3378764.

Oliveira Jr., V. C., V. C. F. Barbosa, and L. Uieda (2012), Polynomial
equivalent layer, Geophysics, 78(1), G1-G13, doi:10.1190/geo2012-0196.1.

----

"""
from __future__ import division
from future.builtins import super
import numpy
import scipy.sparse

from . import sphere as kernel
from ..utils import dircos, safe_dot
from ..inversion import Misfit, Smoothness


class EQLBase(Misfit):
    """
    Base class for the classic equivalent layer.
    """

    def __init__(self, x, y, z, data, grid):
        super().__init__(data=data, nparams=grid.size, islinear=True)
        self.x = x
        self.y = y
        self.z = z
        self.grid = grid

    def predicted(self, p):
        """
        Calculate the data predicted by a given parameter vector.

        Parameters:

        * p : 1d-array (optional)
            The parameter vector with the estimated physical properties of the
            layer. If not given, will use the value calculated by ``.fit()``.

        Returns:

        * result : 1d-array
            The predicted data vector.

        """
        return safe_dot(self.jacobian(p), p)


class EQLGravity(EQLBase):
    """
    Estimate an equivalent layer from gravity data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The gravity data at each point.
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the density of
        each point in the grid.
    * field : string
        Which gravitational field is the data. Options are: ``'gz'`` (gravity
        anomaly), ``'gxx'``, ``'gxy'``, ..., ``'gzz'`` (gravity gradient
        tensor). Defaults to ``'gz'``.

    Examples:

    Use the layer to fit some gravity data and check if our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue)

    .. plot::
        :include-source:
        :context:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from fatiando import gridder
        >>> from fatiando.gravmag import sphere, prism
        >>> from fatiando.gravmag.eqlayer import EQLGravity
        >>> from fatiando.mesher import Prism, PointGrid
        >>> from fatiando.inversion.regularization import Damping
        >>> # Produce some gravity data
        >>> area = (0, 10000, 0, 10000)
        >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
        >>> model = [Prism(4500, 5500, 4500, 5500, 200, 5000,
        ...                {'density': 1000})]
        >>> gz = prism.gz(x, y, z, model)
        >>> # Plot the data
        >>> fig = plt.figure(figsize=(6, 5))
        >>> _ = plt.tricontourf(y, x, gz, 30, cmap='Reds')
        >>> plt.colorbar(pad=0, aspect=30).set_label('mGal')
        >>> _ = plt.plot(y, x, '.k')

    .. plot::
        :include-source:
        :context:

        >>> # Setup a layer
        >>> layer = PointGrid(area, 500, (25, 25))
        >>> solver = (EQLGravity(x, y, z, gz, layer) +
        ...           10**-24*Damping(layer.size)).fit()
        >>> # Check that the predicted data fits the observations
        >>> np.allclose(gz, solver[0].predicted(), rtol=0.01, atol=0.5)
        True
        >>> # Add the densities to the layer
        >>> layer.addprop('density', solver.estimate_)
        >>> # Make a regular grid
        >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
        >>> # Interpolate and check against the model
        >>> gz_layer = sphere.gz(x, y, z, layer)
        >>> gz_model = prism.gz(x, y, z, model)
        >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
        True
        >>> # Upward continue and check against model data
        >>> zup = z - 500
        >>> gz_layer_up = sphere.gz(x, y, zup, layer)
        >>> gz_model_up = prism.gz(x, y, zup, model)
        >>> np.allclose(gz_layer_up, gz_model_up, rtol=0.01, atol=0.1)
        True
        >>> # Plot the interpolated and upward continued data
        >>> plt.close()
        >>> fig = plt.figure(figsize=(6, 5))
        >>> _ = plt.tricontourf(y, x, gz_layer_up, 30, cmap='Reds')
        >>> plt.colorbar(pad=0, aspect=30).set_label('mGal')

    If you have multiple types of gravity data (like gravity anomaly and
    gradient tensor components), you can add ``EQLGravity`` instances together
    for a joint inversion:

    .. plot::
        :include-source:
        :context:

        >>> x1, y1, z1 = gridder.scatter(area, 200, z=-400, seed=0)
        >>> gz = prism.gz(x1, y1, z1, model)
        >>> x2, y2, z2 = gridder.scatter(area, 400, z=-150, seed=2)
        >>> gxy = prism.gxy(x2, y2, z2, model)
        >>> # Plot the gz and gxy data
        >>> plt.close()
        >>> fig = plt.figure(figsize=(12, 5))
        >>> ax = plt.subplot(121, aspect='equal')
        >>> _ = plt.title('gz')
        >>> _ = plt.tricontourf(y1, x1, gz, 30, cmap='Reds')
        >>> plt.colorbar(pad=0, aspect=30).set_label('mGal')
        >>> _ = plt.plot(y1, x1, '.k')
        >>> ax = plt.subplot(122)
        >>> _ = plt.title('gxy')
        >>> _ = plt.tricontourf(y2, x2, gxy, 30, cmap='RdBu_r')
        >>> plt.colorbar(pad=0, aspect=30).set_label('Eotvos')
        >>> _ = plt.plot(y2, x2, '.k')
        >>> plt.tight_layout()

    .. plot::
        :include-source:
        :context:
        :nofigs:

        >>> # Setup a layer
        >>> layer = PointGrid(area, 500, (25, 25))
        >>> solver = (EQLGravity(x1, y1, z1, gz, layer, field='gz') +
        ...           EQLGravity(x2, y2, z2, gxy, layer, field='gxy') +
        ...           10**-24*Damping(layer.size)).fit()
        >>> # Check the fit
        >>> gz_pred = solver[0].predicted()
        >>> gxy_pred = solver[1].predicted()
        >>> np.allclose(gz, gz_pred, rtol=0.01, atol=0.5)
        True
        >>> np.allclose(gxy, gxy_pred, rtol=0.01, atol=0.5)
        True
        >>> # Add the densities to the layer
        >>> layer.addprop('density', solver.estimate_)

    Now that we have the layer, we can do any operation by forward modeling the
    layer. For example, lets just upward continue gxy (without interpolation).

    .. plot::
        :include-source:
        :context:

        >>> # Upward continue gxy only without interpolation
        >>> zup = z2 - 500
        >>> gxy_layer = sphere.gxy(x2, y2, zup, layer)
        >>> # Check against model data
        >>> gxy_model = prism.gxy(x2, y2, zup, model)
        >>> np.allclose(gxy_layer, gxy_model, rtol=0.01, atol=0.5)
        True
        >>> # Plot the upward continued gxy
        >>> plt.close()
        >>> fig = plt.figure(figsize=(6, 5))
        >>> _ = plt.title('Upward continued gxy')
        >>> _ = plt.tricontourf(y2, x2, gxy_layer, 30, cmap='RdBu_r')
        >>> plt.colorbar(pad=0, aspect=30).set_label('Eotvos')
        >>> _ = plt.plot(y2, x2, '.k')

    """

    def __init__(self, x, y, z, data, grid, field='gz'):
        super().__init__(x, y, z, data, grid)
        self.field = field

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix for a given parameter vector.
        """
        x = self.x
        y = self.y
        z = self.z
        func = getattr(kernel, self.field)
        jac = numpy.empty((self.ndata, self.nparams), dtype=numpy.float)
        for i, c in enumerate(self.grid):
            jac[:, i] = func(x, y, z, [c], dens=1.)
        return jac


class EQLTotalField(EQLBase):
    """
    Estimate an equivalent layer from total field magnetic anomaly data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field anomaly data at each point.
    * inc, dec : floats
        The inclination and declination of the inducing field
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the magnetization
        intensity of each point in the grid.
    * sinc, sdec : None or floats
        The inclination and declination of the equivalent layer. Use these if
        there is remanent magnetization and the total magnetization of the
        layer if different from the induced magnetization.
        If there is only induced magnetization, use None

    Examples:

    Use the layer to fit some synthetic data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue,
    reduce to the pole)


    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> from fatiando.inversion import Damping
    >>> # Produce some synthetic data
    >>> area = (0, 1000, 0, 1000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(450, 550, 450, 550, 100, 500, {'magnetization':5})]
    >>> inc, dec = 10, 23
    >>> tf = prism.tf(x, y, z, model, inc, dec)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 200, (25, 25))
    >>> solver = (EQLTotalField(x, y, z, tf, inc, dec, layer) +
    ...           10**-17*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> np.allclose(tf, solver[0].predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the magnetization to the layer
    >>> layer.addprop('magnetization', solver.estimate_)
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> # Interpolate and check agains the model
    >>> tf_layer = sphere.tf(x, y, z, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, z, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=0.5)
    True
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> tf_layer = sphere.tf(x, y, zup, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, zup, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=0.5)
    True
    >>> # Reduce to the pole and check agains model data
    >>> tf_layer = sphere.tf(x, y, zup, layer, 90, 0)
    >>> tf_model = prism.tf(x, y, zup, model, 90, 0)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=2)
    True

    """

    def __init__(self, x, y, z, data, inc, dec, grid, sinc=None, sdec=None):
        super().__init__(x, y, z, data, grid)
        self.inc, self.dec = inc, dec
        self.sinc = sinc if sinc is not None else inc
        self.sdec = sdec if sdec is not None else dec

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix for a given parameter vector.
        """
        x = self.x
        y = self.y
        z = self.z
        inc, dec = self.inc, self.dec
        mag = dircos(self.sinc, self.sdec)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.grid):
            jac[:, i] = kernel.tf(x, y, z, [c], inc, dec, pmag=mag)
        return jac


class PELBase(EQLBase):
    """
    Base class for the Polynomial Equivalent Layer.

    .. note::

        Overloads *fit* to convert the estimated coefficients to physical
        properties. The coefficients are stored in the ``coeffs_`` attribute.

    """

    def __init__(self, x, y, z, data, grid, windows, degree):
        super().__init__(x, y, z, data, grid)
        self.nparams = windows[0]*windows[1]*ncoeffs(degree)
        self.windows = windows
        self.degree = degree

    def fmt_estimate(self, coefs):
        """
        Convert the estimated polynomial coefficients to physical property
        values along the layer.

        Parameters:

        * coefs : 1d-array
            The estimated parameter vector with the polynomial coefficients

        Returns:

        * estimate : 1d-array
            The converted physical property values along the layer.

        """
        ny, nx = self.windows
        pergrid = ncoeffs(self.degree)
        estimate = numpy.empty(self.grid.shape, dtype=float)
        grids = self.grid.split(self.windows)
        k = 0
        ystart = 0
        gny, gnx = grids[0].shape
        for i in xrange(ny):
            yend = ystart + gny
            xstart = 0
            for j in xrange(nx):
                xend = xstart + gnx
                g = grids[k]
                bk = _bkmatrix(g, self.degree)
                window_coefs = coefs[k * pergrid:(k + 1) * pergrid]
                window_props = safe_dot(bk, window_coefs).reshape(g.shape)
                estimate[ystart:yend, xstart:xend] = window_props
                xstart = xend
                k += 1
            ystart = yend
        self.coeffs_ = coefs
        return estimate.ravel()


def _bkmatrix(grid, degree):
    """
    Make the Bk polynomial coefficient matrix for a given PointGrid.

    This matrix converts the coefficients into physical property values.

    Parameters:

    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer
    * degree : int
        The degree of the bivariate polynomial

    Returns:

    * bk : 2d-array
        The matrix


    Examples:

    >>> from fatiando.mesher import PointGrid
    >>> grid = PointGrid((0, 1, 0, 2), 10, (2, 2))
    >>> print _bkmatrix(grid, 2)
    [[ 1.  0.  0.  0.  0.  0.]
     [ 1.  2.  0.  4.  0.  0.]
     [ 1.  0.  1.  0.  0.  1.]
     [ 1.  2.  1.  4.  2.  1.]]
    >>> print _bkmatrix(grid, 1)
    [[ 1.  0.  0.]
     [ 1.  2.  0.]
     [ 1.  0.  1.]
     [ 1.  2.  1.]]
    >>> print _bkmatrix(grid, 3)
    [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  2.  0.  4.  0.  0.  8.  0.  0.  0.]
     [ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
     [ 1.  2.  1.  4.  2.  1.  8.  4.  2.  1.]]

    """
    bmatrix = numpy.transpose(
        [(grid.x**i)*(grid.y**j)
         for l in xrange(1, degree + 2)
         for i, j in zip(xrange(l), xrange(l - 1, -1, -1))])
    return bmatrix


def ncoeffs(degree):
    """
    Calculate the number of coefficients in a bivarite polynomail.

    Parameters:

    * degree : int
        The degree of the polynomial

    Returns:

    * n : int
        The number of coefficients

    Examples:

    >>> ncoeffs(1)
    3
    >>> ncoeffs(2)
    6
    >>> ncoeffs(3)
    10
    >>> ncoeffs(4)
    15

    """
    return sum(xrange(1, degree + 2))


class PELGravity(PELBase):
    """
    Estimate a polynomial equivalent layer from gravity data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The gravity data at each point.
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the density of
        each point in the grid.
    * windows : tuple = (ny, nx)
        The number of windows that the layer will be divided in the y and x
        directions, respectively
    * degree : int
        The degree of the bivariate polynomials used in each window of the PEL
    * field : string
        Which gravitational field is the data. Options are: ``'gz'`` (gravity
        anomaly), ``'gxx'``, ``'gxy'``, ..., ``'gzz'`` (gravity gradient
        tensor). Defaults to ``'gz'``.

    Examples:

    Use the layer to fit some gravity data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue)

    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> # Produce some gravity data
    >>> area = (0, 10000, 0, 10000)
    >>> x, y, z = gridder.scatter(area, 500, z=-150, seed=0)
    >>> model = [Prism(4500, 5500, 4500, 5500, 200, 5000, {'density':1000})]
    >>> gz = prism.gz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (48, 48))
    >>> windows = (12, 12)
    >>> degree = 1
    >>> solver = (PELGravity(x, y, z, gz, layer, windows, degree) +
    ...           10**-24*PELSmoothness(layer, windows, degree)).fit()
    >>> # Check the fit
    >>> np.allclose(gz, solver[0].predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> gz_layer = sphere.gz(x, y, zup, layer)
    >>> gz_model = prism.gz(x, y, zup, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-150)
    >>> # Interpolate and check agains the model
    >>> gz_layer = sphere.gz(x, y, z, layer)
    >>> gz_model = prism.gz(x, y, z, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True

    """

    def __init__(self, x, y, z, data, grid, windows, degree, field='gz'):
        super().__init__(x, y, z, data, grid, windows, degree)
        self.field = field

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix for a given parameter vector.
        """
        x = self.x
        y = self.y
        z = self.z
        func = getattr(kernel, self.field)
        grids = self.grid.split(self.windows)
        pergrid = ncoeffs(self.degree)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        gk = numpy.empty((self.ndata, grids[0].size), dtype=float)
        for i, grid in enumerate(grids):
            bk = _bkmatrix(grid, self.degree)
            for k, c in enumerate(grid):
                gk[:, k] = func(x, y, z, [c], dens=1.)
            jac[:, i*pergrid:(i + 1)*pergrid] = safe_dot(gk, bk)
        return jac


class PELTotalField(PELBase):
    """
    Estimate a polynomial equivalent layer from magnetic total field anomaly.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field magnetic anomaly data at each point.
    * inc, dec : floats
        The inclination and declination of the inducing field
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the magnetization
        intensity of each point in the grid.
    * windows : tuple = (ny, nx)
        The number of windows that the layer will be divided in the y and x
        directions, respectively
    * degree : int
        The degree of the bivariate polynomials used in each window of the PEL
    * sinc, sdec : None or floats
        The inclination and declination of the equivalent layer. Use these if
        there is remanent magnetization and the total magnetization of the
        layer if different from the induced magnetization.
        If there is only induced magnetization, use None

    Examples:

    Use the layer to fit some synthetic data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue,
    reduce to the pole)

    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> # Produce some synthetic data
    >>> area = (0, 1000, 0, 1000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(450, 550, 450, 550, 100, 500, {'magnetization':5})]
    >>> inc, dec = 10, 23
    >>> tf = prism.tf(x, y, z, model, inc, dec)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 200, (60, 60))
    >>> windows = (12, 12)
    >>> degree = 1
    >>> solver = (PELTotalField(x, y, z, tf, inc, dec, layer, windows, degree)
    ...           + 10**-15*PELSmoothness(layer, windows, degree)).fit()
    >>> # Check the fit
    >>> np.allclose(tf, solver[0].predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the magnetization to the layer
    >>> layer.addprop('magnetization', solver.estimate_)
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> tf_layer = sphere.tf(x, y, zup, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, zup, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=0.5)
    True
    >>> # Reduce to the pole and check agains model data
    >>> tf_layer = sphere.tf(x, y, zup, layer, 90, 0)
    >>> tf_model = prism.tf(x, y, zup, model, 90, 0)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=5)
    True
    >>> # Interpolate and check agains the model
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> tf_layer = sphere.tf(x, y, z, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, z, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=5)
    True


    """

    def __init__(self, x, y, z, data, inc, dec, grid, windows, degree,
                 sinc=None, sdec=None):
        super().__init__(x, y, z, data, grid, windows, degree)
        self.inc, self.dec = inc, dec
        self.sinc = sinc if sinc is not None else inc
        self.sdec = sdec if sdec is not None else dec

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix for a given parameter vector.
        """
        x = self.x
        y = self.y
        z = self.z
        inc, dec = self.inc, self.dec
        mag = dircos(self.sinc, self.sdec)
        grids = self.grid.split(self.windows)
        pergrid = ncoeffs(self.degree)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        gk = numpy.empty((self.ndata, grids[0].size), dtype=float)
        for i, grid in enumerate(grids):
            bk = _bkmatrix(grid, self.degree)
            for k, c in enumerate(grid):
                gk[:, k] = kernel.tf(x, y, z, [c], inc, dec, pmag=mag)
            jac[:, i*pergrid:(i + 1)*pergrid] = safe_dot(gk, bk)
        return jac


class PELSmoothness(Smoothness):
    """
    Regularization to "join" neighboring windows in the PEL.

    Use this with :class:`~fatiando.gravmag.eqlayer.PELGravity` and
    :class:`~fatiando.gravmag.eqlayer.PELTotalField`.

    Parameters passed to PELSmoothness must be the same as passed to the PEL
    solvers.

    Parameters:

    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer.
    * windows : tuple = (ny, nx)
        The number of windows that the layer will be divided in the y and x
        directions, respectively.
    * degree : int
        The degree of the bivariate polynomials used in each window of the PEL

    See the docstring of :class:`~fatiando.gravmag.eqlayer.PELGravity` for an
    example usage.

    """

    def __init__(self, grid, windows, degree):
        super().__init__(_pel_fdmatrix(windows, grid, degree))


def _pel_fdmatrix(windows, grid, degree):
    """
    Makes the finite difference matrix for PEL smoothness.
    """
    ny, nx = windows
    grids = grid.split(windows)
    gsize = grids[0].size
    gny, gnx = grids[0].shape
    nderivs = (nx - 1) * grid.shape[0] + (ny - 1) * grid.shape[1]
    rmatrix = scipy.sparse.lil_matrix((nderivs, grid.size))
    deriv = 0
    # derivatives in x
    for k in xrange(0, len(grids) - ny):
        bottom = k * gsize + gny * (gnx - 1)
        top = (k + ny) * gsize
        for i in xrange(gny):
            rmatrix[deriv, bottom + i] = -1.
            rmatrix[deriv, top + 1] = 1.
            deriv += 1
    # derivatives in y
    for k in xrange(0, len(grids)):
        if (k + 1) % ny == 0:
            continue
        right = k * gsize + gny - 1
        left = (k + 1) * gsize
        for i in xrange(gnx):
            rmatrix[deriv, right + i * gny] = -1.
            rmatrix[deriv, left + i * gny] = 1.
            deriv += 1
    rmatrix = rmatrix.tocsr()
    # Make the RB matrix because R is for the sources, B converts it to
    # coefficients.
    pergrid = ncoeffs(degree)
    ncoefs = len(grids) * pergrid
    fdmatrix = numpy.empty((nderivs, ncoefs), dtype=float)
    st = 0
    for i, g in enumerate(grids):
        bk = _bkmatrix(g, degree)
        en = st + g.size
        fdmatrix[:, i*pergrid:(i + 1)*pergrid] = safe_dot(rmatrix[:, st:en],
                                                          bk)
        st = en
    return fdmatrix
