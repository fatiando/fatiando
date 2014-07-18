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

**References**

Li, Y., and D. W. Oldenburg (2010), Rapid construction of equivalent sources
using wavelets, Geophysics, 75(3), L51-L59, doi:10.1190/1.3378764.

Oliveira Jr., V. C., V. C. F. Barbosa, and L. Uieda (2012), Polynomial
equivalent layer, Geophysics, 78(1), G1-G13, doi:10.1190/geo2012-0196.1.

----

"""
from __future__ import division
import numpy
import scipy.sparse

from . import sphere as kernel
from ..utils import dircos, safe_dot
from ..inversion.base import Misfit
from ..inversion.regularization import Smoothness


class EQLBase(Misfit):

    """
    Base class for the classic equivalent layer.
    """

    def __init__(self, x, y, z, data, grid):
        super(EQLBase, self).__init__(data=data,
                                      positional={'x': x, 'y': y, 'z': z},
                                      model={'grid': grid},
                                      nparams=grid.size, islinear=True)

    def _get_predicted(self, p):
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

    Use the layer to fit some gravity data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue)

    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> from fatiando.inversion.regularization import Damping
    >>> # Produce some gravity data
    >>> area = (0, 10000, 0, 10000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(4500, 5500, 4500, 5500, 200, 5000, {'density':1000})]
    >>> gz = prism.gz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (25, 25))
    >>> solver = (EQLGravity(x, y, z, gz, layer) +
    ...           10**-26*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> np.allclose(gz, solver.predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> # Interpolate and check agains the model
    >>> gz_layer = sphere.gz(x, y, z, layer)
    >>> gz_model = prism.gz(x, y, z, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> gz_layer = sphere.gz(x, y, zup, layer)
    >>> gz_model = prism.gz(x, y, zup, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True

    If you have multiple types of gravity data (like gravity anomaly and
    gradient tensor components), you can add EQLGravity instances together for
    a joint inversion:

    >>> x, y, z = gridder.scatter(area, 500, z=-150, seed=0)
    >>> gz = prism.gz(x, y, z, model)
    >>> gzz = prism.gzz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (25, 25))
    >>> solver = (EQLGravity(x, y, z, gz, layer, field='gz') +
    ...           EQLGravity(x, y, z, gzz, layer, field='gzz') +
    ...           10**-24*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> gz_pred, gzz_pred = solver.predicted()
    >>> np.allclose(gz, gz_pred, rtol=0.01, atol=0.5)
    True
    >>> np.allclose(gzz, gzz_pred, rtol=0.01, atol=1)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Upward continue gzz only and check agains model data
    >>> zup = z - 50
    >>> gzz_layer = sphere.gzz(x, y, zup, layer)
    >>> gzz_model = prism.gzz(x, y, zup, model)
    >>> np.allclose(gzz_layer, gzz_model, rtol=0.01, atol=1)
    True

    """

    def __init__(self, x, y, z, data, grid, field='gz'):
        super(EQLGravity, self).__init__(x, y, z, data, grid)
        self.field = field

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        func = getattr(kernel, self.field)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.model['grid']):
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
    >>> from fatiando.inversion.regularization import Damping
    >>> # Produce some synthetic data
    >>> area = (0, 1000, 0, 1000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(450, 550, 450, 550, 100, 500, {'magnetization':5})]
    >>> inc, dec = 10, 23
    >>> tf = prism.tf(x, y, z, model, inc, dec)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 200, (25, 25))
    >>> solver = (EQLTotalField(x, y, z, tf, inc, dec, layer) +
    ...           10**-19*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> np.allclose(tf, solver.predicted(), rtol=0.01, atol=0.5)
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
        super(EQLTotalField, self).__init__(x, y, z, data, grid)
        self.inc, self.dec = inc, dec
        self.model['inc'] = sinc if sinc is not None else inc
        self.model['dec'] = sdec if sdec is not None else dec

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        inc, dec = self.inc, self.dec
        mag = dircos(self.model['inc'], self.model['dec'])
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.model['grid']):
            jac[:, i] = kernel.tf(x, y, z, [c], inc, dec, pmag=mag)
        return jac


class PELBase(Misfit):

    """
    Base class for the Polynomial Equivalent Layer.

    .. note::

        Overloads *fit* to convert the estimated coefficients to physical
        properties. The coefficients are stored in the ``coeffs_`` attribute.

    """

    def __init__(self, x, y, z, data, grid, windows, degree):
        super(PELBase, self).__init__(
            data=data,
            positional={'x': x, 'y': y, 'z': z},
            model={'grid': grid, 'windows': windows, 'degree': degree},
            nparams=windows[0]*windows[1]*ncoeffs(degree),
            islinear=True)

    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def fit(self):
        """
        Solve for the physical property distribution that fits the data.

        Uses the optimization method and parameters defined using the
        :meth:`~fatiando.inversion.base.FitMixin.config` method.

        The estimated physical properties can be accessed through
        :meth:`~fatiando.inversion.base.FitMixin.estimate_`.
        The estimate polynomial coefficients are stored in the ``coeffs_``
        attribute.

        """
        super(PELBase, self).fit()
        coefs = self.p_
        ny, nx = self.model['windows']
        pergrid = ncoeffs(self.model['degree'])
        estimate = numpy.empty(self.model['grid'].shape, dtype=float)
        grids = self.model['grid'].split(self.model['windows'])
        k = 0
        ystart = 0
        gny, gnx = grids[0].shape
        for i in xrange(ny):
            yend = ystart + gny
            xstart = 0
            for j in xrange(nx):
                xend = xstart + gnx
                g = grids[k]
                estimate[ystart:yend, xstart:xend] = safe_dot(
                    _bkmatrix(g, self.model['degree']),
                    coefs[k * pergrid:(k + 1) * pergrid]
                ).reshape(g.shape)
                xstart = xend
                k += 1
            ystart = yend
        self.coeffs_ = self.p_
        self._estimate = estimate.ravel()
        return self


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
     [ 1.  0.  1.  0.  0.  1.]
     [ 1.  2.  0.  4.  0.  0.]
     [ 1.  2.  1.  4.  2.  1.]]
    >>> print _bkmatrix(grid, 1)
    [[ 1.  0.  0.]
     [ 1.  0.  1.]
     [ 1.  2.  0.]
     [ 1.  2.  1.]]
    >>> print _bkmatrix(grid, 3)
    [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
     [ 1.  2.  0.  4.  0.  0.  8.  0.  0.  0.]
     [ 1.  2.  1.  4.  2.  1.  8.  4.  2.  1.]]

    """
    bmatrix = numpy.transpose(
        [(grid.x ** i) * (grid.y ** j)
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
    ...           10**-27*PELSmoothness(layer, windows, degree)).fit()
    >>> # Check the fit
    >>> np.allclose(gz, solver.predicted(), rtol=0.01, atol=0.5)
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
        super(PELGravity, self).__init__(x, y, z, data, grid, windows, degree)
        self.field = field

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        func = getattr(kernel, self.field)
        grids = self.model['grid'].split(self.model['windows'])
        pergrid = ncoeffs(self.model['degree'])
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        gk = numpy.empty((self.ndata, grids[0].size), dtype=float)
        for i, grid in enumerate(grids):
            bk = _bkmatrix(grid, self.model['degree'])
            for k, c in enumerate(grid):
                gk[:, k] = func(x, y, z, [c], dens=1.)
            jac[:, i * pergrid:(i + 1) * pergrid] = safe_dot(gk, bk)
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
    >>> np.allclose(tf, solver.predicted(), rtol=0.01, atol=0.5)
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
        super(PELTotalField, self).__init__(x, y, z, data, grid, windows,
                                            degree)
        self.inc, self.dec = inc, dec
        self.model['inc'] = sinc if sinc is not None else inc
        self.model['dec'] = sdec if sdec is not None else dec

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        inc, dec = self.inc, self.dec
        mag = dircos(self.model['inc'], self.model['dec'])
        grids = self.model['grid'].split(self.model['windows'])
        pergrid = ncoeffs(self.model['degree'])
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        gk = numpy.empty((self.ndata, grids[0].size), dtype=float)
        for i, grid in enumerate(grids):
            bk = _bkmatrix(grid, self.model['degree'])
            for k, c in enumerate(grid):
                gk[:, k] = kernel.tf(x, y, z, [c], inc, dec, pmag=mag)
            jac[:, i * pergrid:(i + 1) * pergrid] = safe_dot(gk, bk)
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
        super(PELSmoothness, self).__init__(
            _pel_fdmatrix(windows, grid, degree))


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
        fdmatrix[
            :, i * pergrid:(i + 1) * pergrid] = safe_dot(rmatrix[:, st:en], bk)
        st = en
    return fdmatrix
