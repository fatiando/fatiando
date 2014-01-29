"""
Estimation of the total magnetization vector of homogeneous bodies.

This class estimates the Cartesian components of the magnetization
vector of homogeneous dipolar bodies with known center. The estimated
magnetization vector is converted to dipole moment, inclination
(positive down) and declination (with respect to x, North).

**Algorithm**

* :class:`~fatiando.gravmag.magdir.DipoleMagDir`: By using the well- 
known first-order approximation of the total field anomaly (Blakely, 
1996, p. 179) produced by a set of dipoles, the estimation of the 
Cartesian components of the magnetization vectors is formulated as 
linear inverse problem. After estimating the magnetization vectors, 
they are converted to to dipole moment, inclination (positive down) 
and declination (with respect to x, North).

**References**

Blakely, R. (1996), Potential theory in gravity and magnetic applications: CUP


----

"""

from __future__ import division

import numpy
from ..inversion.base import Misfit
from .. import mesher
from ..utils import ang2vec, vec2ang, safe_dot
from . import sphere
from ..constants import G, CM, T2NT, SI2EOTVOS

class DipoleMagDir(Misfit):
    """
    Estimate the magnetization vector of a set of dipoles from magnetic
    total field anomaly.

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
        dipole. Subsequently, the estimated magnetization vectors are converted 
        to dipole moment, inclination and declination.
        
    .. note:: Inclination is positive down and declination is measured with respect
        to x (North).

    Examples:

    Estimate the magnetization vector of dipolar synthetic bodies
    with know centers.
    
    >>> import numpy
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
    >>> # Make a solver and fit it to the data
    >>> solver = DipoleMagDir(x, y, z, tf, inc, dec, points).fit()
    >>> # Check the fit
    >>> numpy.allclose(tf, solver.predicted(), rtol=0.001, atol=0.001)
    True
    >>> # p_ is the estimated parameter vector (Cartesian components of the
    >>> # estimated magnetization vectors)
    >>> solver.p_
	>>> # Check the estimated parameter vector
	>>> p_true = numpy.hstack((ang2vec(CM*(4.*numpy.pi/3.)*6.0*1000**3, -20.0, -10.0), 
	...						   ang2vec(CM*(4.*numpy.pi/3.)*6.0*1000**3, 30.0, -40.0)))
	>>> numpy.allclose(p_true, solver.p_, rtol=0.001, atol=0.001)
	True
    >>> # The parameter vector is not that useful so use estimate_ to convert
    >>> # the estimated magnetization vectors in dipole moment, inclination 
    >>> # and declination.
    >>> solver.estimate_
	>>> # Check the converted estimate
	>>> estimate_true = [utils.vec2ang(p_true[3*i : 3*i + 3]) for i in range(len(points))]
	>>> numpy.allclose(estimate_true, solver.estimate_, rtol=0.001, atol=0.001)
	True
    
    """
	
    def __init__(self, x, y, z, data, inc, dec, points):
        super(DipoleMagDir, self).__init__(
            data=data,
            positional={'x':x, 'y':y, 'z':z},
            model={'inc':inc, 'dec':dec, 'points':points},
            nparams=3*len(points),
            islinear=True)
        #Constants
        self.rad2degree = 180.0/numpy.pi
        self.degree2rad = numpy.pi/180.0
        self.ndipoles = len(points)
        self.cte = 1.0/((4.0*numpy.pi/3.0)*G*SI2EOTVOS)
        #Geomagnetic Field versor
        self.F_versor = ang2vec(1.0, self.model['inc'], self.model['dec'])
        
    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        dipoles = []
        for i in range(self.ndipoles):
            dipoles.append(mesher.Sphere(self.model['points'][i][0], 
                                         self.model['points'][i][1], 
                                         self.model['points'][i][2], 
                                         1.0))
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, dipole in enumerate(dipoles):
            k = 3*i
            derivative_gxx = sphere.gxx(x, y, z, [dipole], dens=self.cte)
            derivative_gxy = sphere.gxy(x, y, z, [dipole], dens=self.cte)
            derivative_gxz = sphere.gxz(x, y, z, [dipole], dens=self.cte)
            derivative_gyy = sphere.gyy(x, y, z, [dipole], dens=self.cte)
            derivative_gyz = sphere.gyz(x, y, z, [dipole], dens=self.cte)
            derivative_gzz = sphere.gzz(x, y, z, [dipole], dens=self.cte)
            jac[:,k]   = T2NT*((self.F_versor[0]*derivative_gxx) + 
                               (self.F_versor[1]*derivative_gxy) + 
                               (self.F_versor[2]*derivative_gxz))
            jac[:,k+1] = T2NT*((self.F_versor[0]*derivative_gxy) + 
                               (self.F_versor[1]*derivative_gyy) + 
                               (self.F_versor[2]*derivative_gyz))
            jac[:,k+2] = T2NT*((self.F_versor[0]*derivative_gxz) + 
                               (self.F_versor[1]*derivative_gyz) + 
                               (self.F_versor[2]*derivative_gzz))
        return jac
    
    def fit(self):
        """
        Solve for the magnetization direction of a set of dipoles.

        After solving, use the ``estimate_`` attribute to get the
        estimated magnetization vectors in dipole moment, inclination 
        and declination.

        The estimated magnetization vectors in Cartesian coordinates can
        be accessed through the ``p_`` attribute.

        See the the docstring of :class:`~fatiando.gravmag.magdir.DipoleMagDir`
        for examples.

        """
        super(DipoleMagDir, self).fit()
        self._estimate = [vec2ang(self.p_[3*i : 3*i + 3]) for i in range(len(self.model['points']))]
        return self

