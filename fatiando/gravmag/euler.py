"""
Euler deconvolution methods for potential fields.

**Implementations**

* :class:`~fatiando.gravmag.euler.Classic`: The classic 3D solution to Euler's
  equation for potential fields (Reid et al., 1990). Runs on the whole dataset.

**Solution selection procedures**

* :class:`~fatiando.gravmag.euler.ExpandingWindow`: Run a given Euler
  deconvolution on an expanding window and keep the best estimate.
* :class:`~fatiando.gravmag.euler.MovingWindow`: Run a given Euler
  deconvolution on a moving window to produce a set of estimates.

**References**

Reid, A. B., J. M. Allsop, H. Granser, A. J. Millett, and I. W. Somerton
(1990), Magnetic interpretation in three dimensions using Euler deconvolution,
Geophysics, 55(1), 80-91, doi:10.1190/1.1442774.


----

"""
from __future__ import division
from future.builtins import super
import numpy as np

from .. import gridder
from ..inversion import Misfit
from ..utils import safe_inverse, safe_dot, safe_diagonal


class Classic(Misfit):
    """
    Classic 3D Euler deconvolution of potential field data.

    Follows the formulation of Reid et al. (1990). Performs the deconvolution
    on the whole data set. For windowed approaches, use
    :class:`~fatiando.gravmag.euler.ExpandingWindow`.

    Works on any potential field that satisfies Euler's homogeneity equation.

    Estimated coordinates are stored in ``estimate_`` attribute and the
    estimated base level is stored in ``baselevel_``.

    .. note::

        The data does **not** need to be gridded for this! So long as you
        can calculate the derivatives of non-gridded data (using an Equivalent
        Layer, for example).

    .. note:: x is North, y is East, and z is down.

    .. warning::

        Units of the input data (x, y, z, field, derivatives) must be in SI
        units! Otherwise, the results will be in strange units. Use functions
        in :mod:`fatiando.utils` to convert between units.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, and z coordinates of the observation points
    * field : 1d-array
        The potential field measured at the observation points
    * xderiv, yderiv, zderiv : 1d-arrays
        The x-, y-, and z-derivatives of the potential field (measured or
        calculated) at the observation points
    * index : float
        The structural index of the source


    """

    def __init__(self, x, y, z, field, xderiv, yderiv, zderiv,
                 structural_index):
        same_shape = all(i.shape == x.shape
                         for i in [y, z, field, xderiv, yderiv, zderiv])
        assert same_shape, 'All input arrays should have the same shape.'
        assert structural_index >= 0, \
            "Invalid structural index '{}'. Should be >= 0".format(
                structural_index)
        super().__init__(
            data=-x*xderiv - y*yderiv - z*zderiv - structural_index*field,
            nparams=4, islinear=True)
        self.x = x
        self.y = y
        self.z = z
        self.field = field
        self.xderiv = xderiv
        self.yderiv = yderiv
        self.zderiv = zderiv
        self.structural_index = structural_index

    def jacobian(self, p):
        jac = np.empty((self.ndata, self.nparams), dtype=np.float)
        jac[:, 0] = -self.xderiv
        jac[:, 1] = -self.yderiv
        jac[:, 2] = -self.zderiv
        jac[:, 3] = -self.structural_index*np.ones(self.ndata)
        return jac

    def predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    @property
    def baselevel_(self):
        assert self.p_ is not None, "No estimates found. Run 'fit' first."
        return self.p_[3]

    def fmt_estimate(self, p):
        """
        Separate the (x, y, z) point coordinates from the baselevel.

        Coordinates are stored in ``estimate_`` and a base level is stored in
        ``baselevel_``.
        """
        return p[:3]

    def cut_window(self, area):
        """
        Return a copy of self with only data that falls inside the given area

        Parameters:

        * area : list = (x1, x2, y1, y2)
            The limiting coordinates of the area

        Returns:

        * subset
            An instance of this class.

        """
        x, y = self.x, self.y
        x1, x2, y1, y2 = area
        indices = ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))
        slices = [i[indices] for i in [self.x, self.y, self.z, self.field,
                                       self.xderiv, self.yderiv, self.zderiv]]
        slices.append(self.structural_index)
        return self.__class__(*slices)


class ExpandingWindow(object):
    """
    Solve an Euler deconvolution problem using an expanding window scheme.

    Uses data inside a window of growing size to perform the Euler
    deconvolution. Keeps the best result, judged by the estimated error.

    Like any other Euler solver, use the
    :meth:`~fatiando.gravmag.euler.ExpandingWindow.fit` method to produce an
    estimate. The estimated point is stored in ``estimate_``, the base level in
    ``baselevel_``.

    Parameters:

    * euler : Euler solver
        An instance of an Euler deconvolution solver, like
        :class:`~fatiando.gravmag.euler.Classic`.
    * center : [x, y]
        The x, y coordinates of the center of the expanding windows.
    * sizes : list or 1d-array
        The sizes of the windows.

    """

    def __init__(self, euler, center, sizes):
        self.euler = euler
        self.center = center
        self.sizes = sizes
        self.estimate_ = None
        self.p_ = None

    def fit(self):
        """
        Perform the Euler deconvolution with expanding windows.

        The estimated point is stored in ``estimate_``, the base level in
        ``baselevel_``.

        """
        xc, yc = self.center
        results = []
        errors = []
        for size in self.sizes:
            ds = 0.5*size
            window = [xc - ds, xc + ds, yc - ds, yc + ds]
            solver = self.euler.cut_window(window).fit()
            # Don't really know why dividing by ndata makes this better but it
            # does.
            cov = safe_inverse(solver.hessian(solver.p_)/solver.ndata)
            uncertainty = np.sqrt(safe_diagonal(cov)[0:3])
            mean_error = np.linalg.norm(uncertainty)
            errors.append(mean_error)
            results.append(solver.p_)
        self.p_ = results[np.argmin(errors)]
        self.estimate_ = self.p_[:3]
        self.baselevel_ = self.p_[3]
        return self


class MovingWindow(object):
    """
    Solve an Euler deconvolution problem using a moving window scheme.

    Uses data inside a window moving to perform the Euler deconvolution. Keeps
    the estimate from all windows.

    Like any other Euler solver, use the
    :meth:`~fatiando.gravmag.euler.MovingWindow.fit` method to produce an
    estimate. The estimated points are stored in ``estimate_``, the base levels
    in ``baselevel_``.

    Parameters:

    * euler : Euler solver
        An instance of an Euler deconvolution solver, like
        :class:`~fatiando.gravmag.euler.Classic`.
    * windows : (ny, nx)
        The number of windows in the y and x directions
    * size : (dy, dx)
        The size of the windows in the y and x directions
    * keep : float
        Decimal percentage of solutions to keep. Will rank the solutions by
        increasing error and keep only the first *keep* percent.

    """

    def __init__(self, euler, windows, size, keep=0.2):
        self.euler = euler
        self.windows = windows
        self.size = size
        self.keep = keep
        self.window_centers = self.get_window_centers()
        self.estimate_ = None
        self.p_ = None

    def get_window_centers(self):
        """
        Calculate the center coordinates of the windows.

        Based on the data stored in the given Euler Deconvolution solver.

        Returns:

        * centers : list
            List of [x, y] coordinate pairs for the center of each window.

        """
        ny, nx = self.windows
        dy, dx = self.size
        x, y = self.euler.x, self.euler.y
        x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()
        centers = []
        xmidpoints = np.linspace(x1 + 0.5 * dx, x2 - 0.5 * dx, nx)
        ymidpoints = np.linspace(y1 + 0.5 * dy, y2 - 0.5 * dy, ny)
        for yc in ymidpoints:
            for xc in xmidpoints:
                centers.append([xc, yc])
        return centers

    def fit(self):
        """
        Perform the Euler deconvolution on a moving window.

        The estimated points are stored in ``estimate_``, the base levels in
        ``baselevel_``.

        """
        dy, dx = self.size
        euler = self.euler
        paramvecs = []
        estimates = []
        baselevels = []
        errors = []
        # Thank you Saulinho for the solution!
        # Calculate the mid-points of the windows
        for xc, yc in self.window_centers:
                window = [xc - 0.5 * dx, xc + 0.5 * dx,
                          yc - 0.5 * dy, yc + 0.5 * dy]
                solver = euler.cut_window(window).fit()
                cov = safe_inverse(solver.hessian(solver.p_))
                uncertainty = np.sqrt(safe_diagonal(cov)[0:3])
                mean_error = np.linalg.norm(uncertainty)
                errors.append(mean_error)
                paramvecs.append(solver.p_)
                estimates.append(solver.estimate_)
                baselevels.append(solver.baselevel_)
        best = np.argsort(errors)[:int(self.keep * len(errors))]
        self.p_ = np.array(paramvecs)[best]
        self.estimate_ = np.array(estimates)[best]
        self.baselevel_ = np.array(baselevels)[best]
        return self
