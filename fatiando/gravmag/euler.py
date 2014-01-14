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
import numpy

from .. import gridder
from ..inversion.base import Misfit
from ..utils import safe_inverse, safe_dot, safe_diagonal


class Classic(Misfit):
    """
    Classic 3D Euler deconvolution of potential field data.

    Follows the formulation of Reid et al. (1990). Performs the deconvolution
    on the whole data set. For windowed approaches, use
    :class:`~fatiando.gravmag.euler.ExpandingWindow`.

    Works on any potential field that satisfies Euler's homogeneity equation.

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
        if (len(x) != len(y) != len(z) != len(field) != len(xderiv)
            != len(yderiv) != len(zderiv)):
            raise ValueError("x, y, z, field, xderiv, yderiv, zderiv must " +
                "have the same number of elements")
        if structural_index < 0:
            raise ValueError("Invalid structural index '%g'. Should be >= 0"
                % (structural_index))
        super(Classic, self).__init__(
            data=-x*xderiv - y*yderiv - z*zderiv - structural_index*field,
            positional=dict(x=x, y=y, z=z, field=field, xderiv=xderiv,
                yderiv=yderiv, zderiv=zderiv),
            model=dict(structural_index=structural_index),
            nparams=4, islinear=True)

    def _get_jacobian(self, p):
        jac = numpy.transpose(
            [-self.positional['xderiv'], -self.positional['yderiv'],
             -self.positional['zderiv'],
             -self.model['structural_index']*numpy.ones(self.ndata)])
        return jac

    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def fit(self):
        """
        Solve the deconvolution on the whole data set.

        Estimates an (x, y, z) point (stored in ``estimate_``) and a base level
        (stored in ``baselevel_``).
        """
        super(Classic, self).fit()
        self._estimate = self.p_[:3]
        self.baselevel_ = self.p_[3]
        return self

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
        euler = self.euler
        x, y = euler.positional['x'], euler.positional['y']
        results = []
        errors = []
        for size in self.sizes:
            ds = 0.5*size
            xmin, xmax, ymin, ymax = xc - ds, xc + ds, yc - ds, yc + ds
            indices = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            if not numpy.any(indices):
                continue
            solver = euler.subset(indices).fit()
            cov = safe_inverse(solver.hessian(solver.p_))
            uncertainty = numpy.sqrt(safe_diagonal(cov)[0:3])
            mean_error = numpy.linalg.norm(uncertainty)
            errors.append(mean_error)
            results.append(solver.p_)
        self.p_ = results[numpy.argmin(errors)]
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
        self.window_centers = None
        self.estimate_ = None
        self.p_ = None

    def fit(self):
        """
        Perform the Euler deconvolution on a moving window.

        The estimated points are stored in ``estimate_``, the base levels in
        ``baselevel_``.

        """
        ny, nx = self.windows
        dy, dx = self.size
        euler = self.euler
        x, y = euler.positional['x'], euler.positional['y']
        x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()
        paramvecs = []
        estimates = []
        baselevels = []
        errors = []
        # Thank you Saulinho for the solution!
        # Calculate the mid-points of the windows
        self.window_centers = []
        xmidpoints = numpy.linspace(x1 + 0.5*dx, x2 -  0.5*dx, nx)
        ymidpoints = numpy.linspace(y1 + 0.5*dy, y2 -  0.5*dy, ny)
        for yc in ymidpoints:
            for xc in xmidpoints:
                self.window_centers.append([xc, yc])
                # Separate the indices that fall inside the window with center
                # (xc, yc)
                indices = ((x >= xc - 0.5*dx) & (x <= xc + 0.5*dx) &
                           (y >= yc - 0.5*dy) & (y <= yc + 0.5*dy))
                if not numpy.any(indices):
                    continue
                solver = euler.subset(indices).fit()
                cov = safe_inverse(solver.hessian(solver.p_))
                uncertainty = numpy.sqrt(safe_diagonal(cov)[0:3])
                mean_error = numpy.linalg.norm(uncertainty)
                errors.append(mean_error)
                paramvecs.append(solver.p_)
                estimates.append(solver.estimate_)
                baselevels.append(solver.baselevel_)
        best = numpy.argsort(errors)[:int(self.keep*len(errors))]
        self.p_ = numpy.array(paramvecs)[best]
        self.estimate_ = numpy.array(estimates)[best]
        self.baselevel_ = numpy.array(baselevels)[best]
        return self

