"""
Euler deconvolution methods for potential fields.

* :class:`~fatiando.gravmag.euler.Classic`: The classic 3D solution to Euler's
  equation for potential fields (Reid et al., 1990).


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

    def __init__(self, x, y, z, field, xderiv, yderiv, zderiv, index):
        if (len(x) != len(y) != len(z) != len(field) != len(xderiv)
            != len(yderiv) != len(zderiv)):
            raise ValueError("x, y, z, field, xderiv, yderiv, zderiv must " +
                "have the same number of elements")
        if index < 0:
            raise ValueError("Invalid structural index '%g'. Should be >= 0"
                % (index))
        super(Classic, self).__init__(
            data=-x*xderiv - y*yderiv - z*zderiv - index*field,
            nparams=4, islinear=True)
        self.x, self.y, self.z = x, y, z
        self.structural_index = index
        self.field = field
        self.xderiv, self.yderiv, self.zderiv = xderiv, yderiv, zderiv

    def _get_jacobian(self, p):
        jac = numpy.transpose(
            [-self.xderiv, -self.yderiv, -self.zderiv,
             -self.structural_index*numpy.ones_like(self.field)])
        return jac

    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def covariance(self, p):
        """
        """
        variance = numpy.linalg.norm(self.residuals(p))**2/(self.ndata - 4)
        covar = variance*safe_inverse(self.hessian(p))
        return covar

    def expanding_window(self, center, ranges, nwin=20, covariance=False):
        """
        Run Euler deconvolution on windows of growing size and return the best
        Parameters:

        * center : list = [x, y]
            The coordinates of the center of the expanding window
        * ranges : list = [min, max]
            The minimum and maximum size of the expanding window
        * nwin : int
            Number of windows to use between the minimum and maximum sizes
        * covariance : True or False
            If True, will also return the covariance matrix of the estimate

        Returns:

        If ``covariance==True``

        * p, cov : 1d-array and 2d-array
            The best estimate and it's covariance matrix

        Else

        * p : 1d-array
            The best estimate

        """
        xc, yc = center
        results = []
        errors = []
        minsize, maxsize = ranges
        for size in numpy.linspace(minsize, maxsize, nwin):
            ds = 0.5*size
            area = [xc - ds, xc + ds, yc - ds, yc + ds]
            wx, wy, wscalars = gridder.cut(self.x, self.y,
                [self.z, self.field, self.xderiv, self.yderiv, self.zderiv],
                area)
            wz, wfield, wxderiv, wyderiv, wzderiv  = wscalars
            euler = self.__class__(wx, wy, wz, wfield, wxderiv, wyderiv,
                wzderiv, self.structural_index)
            p = euler.fit()
            cov = euler.covariance(p)
            if covariance:
                results.append([p, cov])
            else:
                results.append(p)
            uncertainty = numpy.sqrt(safe_diagonal(cov)[0:3])
            mean_error = numpy.linalg.norm(uncertainty)
            errors.append(mean_error)
        best = results[numpy.argmin(errors)]
        return best

