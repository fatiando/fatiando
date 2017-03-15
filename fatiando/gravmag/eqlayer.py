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
from __future__ import division, absolute_import
from future.builtins import super, range
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
        super().__init__(data=data, nparams=len(grid), islinear=True)
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
        for i in range(ny):
            yend = ystart + gny
            xstart = 0
            for j in range(nx):
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
         for l in range(1, degree + 2)
         for i, j in zip(range(l), range(l - 1, -1, -1))])
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
    return sum(range(1, degree + 2))


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
    for k in range(0, len(grids) - ny):
        bottom = k * gsize + gny * (gnx - 1)
        top = (k + ny) * gsize
        for i in range(gny):
            rmatrix[deriv, bottom + i] = -1.
            rmatrix[deriv, top + 1] = 1.
            deriv += 1
    # derivatives in y
    for k in range(0, len(grids)):
        if (k + 1) % ny == 0:
            continue
        right = k * gsize + gny - 1
        left = (k + 1) * gsize
        for i in range(gnx):
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
