"""
Estimation of the total magnetization vector of homogeneous bodies.

It estimates parameters related to the magnetization vector of homogeneous
bodies.

**Algorithms**

* :class:`~fatiando.gravmag.magdir.DipoleMagDir`: This class estimates
  the Cartesian components of the magnetization vector of homogeneous
  dipolar bodies with known center. The estimated magnetization vector
  is converted to dipole moment, inclination (positive down) and declination
  (with respect to x, North).

----

"""

from __future__ import division, absolute_import
from future.builtins import super
import numpy as np
from ..inversion import Misfit
from .. import mesher
from ..utils import ang2vec, vec2ang, safe_dot
from . import sphere
from ..constants import G, CM, T2NT, SI2EOTVOS
from .._our_duecredit import due, Doi, BibTeX


@due.dcite(Doi("10.5194/npg-22-215-2015"),
           description='Estimates total mag. dir. of approx. spherical bodies')
class DipoleMagDir(Misfit):
    """
    Estimate the magnetization vector of a set of dipoles from magnetic
    total field anomaly using the method of Oliveira Jr. et al. (2015).

    By using the well-known first-order approximation of the total field
    anomaly (Blakely, 1996, p. 179) produced by a set of dipoles, the
    estimation of the Cartesian components of the magnetization vectors is
    formulated as linear inverse problem. After estimating the magnetization
    vectors, they are converted to dipole moment, inclination (positive down)
    and declination (with respect to x, North).

    After solving, use the ``estimate_`` attribute to get the estimated
    magnetization vectors in dipole moment, inclination and declination.  The
    estimated magnetization vectors in Cartesian coordinates can be accessed
    through the ``p_`` attribute.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field magnetic anomaly data at each point.
    * inc, dec : floats
        The inclination and declination of the inducing field
    * points : list of points [x, y, z]
        Each point [x, y, z] is the center of a dipole. Will invert for
        the Cartesian components of the magnetization vector of each
        dipole. Subsequently, the estimated magnetization vectors are
        converted to dipole moment, inclination and declination.

    .. note:: Inclination is positive down and declination is measured with
        respect to x (North).

    References:

    Blakely, R. (1996), Potential theory in gravity and magnetic applications:
    CUP

    Oliveira Jr., V. C., D. P. Sales, V. C. F. Barbosa, and L. Uieda (2015),
    Estimation of the total magnetization direction of approximately spherical
    bodies, Nonlin. Processes Geophys., 22(2), 215-232,
    doi:10.5194/npg-22-215-2015.

    Examples:

    Estimation of the total magnetization vector of dipoles with known centers

    >>> import numpy as np
    >>> from fatiando import gridder, utils
    >>> from fatiando.gravmag import sphere
    >>> from fatiando.mesher import Sphere, Prism
    >>> # Produce some synthetic data
    >>> area = (0, 10000, 0, 10000)
    >>> x, y, z = gridder.scatter(area, 500, z=-150, seed=0)
    >>> model = [Sphere(3000, 3000, 1000, 1000,
    ...              {'magnetization': utils.ang2vec(6.0, -20.0, -10.0)}),
    ...          Sphere(7000, 7000, 1000, 1000,
    ...              {'magnetization': utils.ang2vec(6.0, 30.0, -40.0)})]
    >>> inc, dec = -9.5, -13
    >>> tf = sphere.tf(x, y, z, model, inc, dec)
    >>> # Give the coordinates of the dipoles
    >>> points = [[3000.0, 3000.0, 1000.0], [7000.0, 7000.0, 1000.0]]
    >>> p_true = np.hstack((ang2vec(CM*(4.*np.pi/3.)*6.0*1000**3,
    ...                                             -20.0, -10.0),
    ...                        ang2vec(CM*(4.*np.pi/3.)*6.0*1000**3,
    ...                                              30.0, -40.0)))
    >>> estimate_true = [utils.vec2ang(p_true[3*i : 3*i + 3]) for i
    ...                                in range(len(points))]
    >>> # Make a solver and fit it to the data
    >>> solver = DipoleMagDir(x, y, z, tf, inc, dec, points).fit()
    >>> # Check the fit
    >>> np.allclose(tf, solver.predicted(), rtol=0.001, atol=0.001)
    True
    >>> # solver.p_ returns the Cartesian components of the
    >>> # estimated magnetization vectors
    >>> for p in solver.p_: print "%.10f" % p
    2325.8255393651
    -410.1057950109
    -859.5903757213
    1667.3411086852
    -1399.0653093445
    1256.6370614359
    >>> # Check the estimated parameter vector
    >>> np.allclose(p_true, solver.p_, rtol=0.001, atol=0.001)
    True
    >>> # The parameter vector is not that useful so use solver.estimate_
    >>> # to convert the estimated magnetization vectors in dipole moment,
    >>> # inclination and declination.
    >>> for e in solver.estimate_:
    ...    print "%.10f %.10f %.10f" % (e[0], e[1], e[2])
    2513.2741228718 -20.0000000000 -10.0000000000
    2513.2741228718 30.0000000000 -40.0000000000
    >>> # Check the converted estimate
    >>> np.allclose(estimate_true, solver.estimate_, rtol=0.001,
    ...                                                 atol=0.001)
    True

    """

    def __init__(self, x, y, z, data, inc, dec, points):
        super().__init__(data=data, nparams=3*len(points), islinear=True)
        self.x, self.y, self.z = x, y, z
        self.inc, self.dec = inc, dec
        self.points = points
        # Constants
        self.ndipoles = len(points)
        self.cte = 1/((4*np.pi/3)*G*SI2EOTVOS)
        # Geomagnetic Field versor
        self.F_versor = ang2vec(1.0, inc, dec)

    def predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def jacobian(self, p):
        x = self.x
        y = self.y
        z = self.z
        dipoles = [mesher.Sphere(xp, yp, zp, 1.) for xp, yp, zp in
                   self.points]
        jac = np.empty((self.ndata, self.nparams), dtype=np.float)
        for i, dipole in enumerate(dipoles):
            k = 3*i
            derivative_gxx = sphere.gxx(x, y, z, [dipole], dens=self.cte)
            derivative_gxy = sphere.gxy(x, y, z, [dipole], dens=self.cte)
            derivative_gxz = sphere.gxz(x, y, z, [dipole], dens=self.cte)
            derivative_gyy = sphere.gyy(x, y, z, [dipole], dens=self.cte)
            derivative_gyz = sphere.gyz(x, y, z, [dipole], dens=self.cte)
            derivative_gzz = sphere.gzz(x, y, z, [dipole], dens=self.cte)
            jac[:, k] = T2NT * ((self.F_versor[0] * derivative_gxx) +
                                (self.F_versor[1] * derivative_gxy) +
                                (self.F_versor[2] * derivative_gxz))
            jac[:, k + 1] = T2NT * ((self.F_versor[0] * derivative_gxy) +
                                    (self.F_versor[1] * derivative_gyy) +
                                    (self.F_versor[2] * derivative_gyz))
            jac[:, k + 2] = T2NT * ((self.F_versor[0] * derivative_gxz) +
                                    (self.F_versor[1] * derivative_gyz) +
                                    (self.F_versor[2] * derivative_gzz))
        return jac

    def fmt_estimate(self, p):
        """
        Convert the estimate parameters from Cartesian to inclination,
        declication, and intensity.
        """
        angles = [vec2ang(p[3*i: 3*i + 3]) for i in range(len(self.points))]
        return angles
