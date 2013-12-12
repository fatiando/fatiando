"""
Euler deconvolution methods for potential fields.

* :class:`~fatiando.gravmag.euler.Classic`: The classic 3D solution to Euler's
  equation for potential fields (Reid et al., 1990).
* :class:`~fatiando.gravmag.euler.ExpandingWindow`: Run an Euler deconvolution
  on an expanding window and return the best estimate.

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

    Works on any potential field that satisfies Euler's homogeneity equation.

    Follows the formulation of Reid et al. (1990).

    .. note::

        The data does **not** need to be gridded for this! So long as you
        can calculate the derivatives of non-gridded data (using an Equivalent
        Layer, for example).

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


class ExpandingWindow(object):
    """
    Solve an Euler deconvolution problem using an expanding window scheme.

    Uses data inside a window of growing size to perform the Euler
    deconvolution. Keeps the best result, judged by the estimated error.

    Like any other Euler solver, use the
    :meth:`~fatiando.gravmag.euler.ExpandingWindow.fit` method to produce an
    estimate.

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

    def fit(self, **kwargs):
        """
        Perform the Euler deconvolution with expanding windows.

        Keyword arguments given will be passed to the ``fit`` method of the
        Euler solver.

        Returns:

        * estimate : 1d-array
            The best estimate out of all windows.

        """
        xc, yc = self.center
        euler = self.euler
        x, y = euler.positional['x'], euler.positional['y']
        results = []
        errors = []
        for size in self.sizes:
            ds = 0.5*size
            xmin, xmax, ymin, ymax = xc - ds, xc + ds, yc - ds, yc + ds
            indices = [i for i in xrange(euler.ndata)
                if x[i] >= xmin and x[i] <= xmax
                and y[i] >= ymin and y[i] <= ymax]
            if not indices:
                continue
            euler.use_subset(indices)
            p = euler.fit(**kwargs)
            euler.use_all()
            results.append(p)
            cov = safe_inverse(euler.hessian(p))
            uncertainty = numpy.sqrt(safe_diagonal(cov)[0:3])
            mean_error = numpy.linalg.norm(uncertainty)
            errors.append(mean_error)
        best = results[numpy.argmin(errors)]
        return best

