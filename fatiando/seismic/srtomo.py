# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
SrTomo: Straight-ray 2D travel-time tomography (i.e., does not consider
reflection or refraction)

**Solvers**

* :func:`fatiando.seismic.srtomo.smooth`
* :func:`fatiando.seismic.srtomo.sharp`

**Data Modules**

* :class:`fatiando.seismic.srtomo.TravelTimeStraightRay2D`

**Examples**

Using simple synthetic data::

    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
    >>> from fatiando.mesher.dd import Square, SquareMesh
    >>> model = [Square([0, 10, 0, 5], {'vp':2}),
    ...          Square([0, 10, 5, 10], {'vp':5})]
    >>> src = (5, 0)
    >>> srcs = [src, src, src]
    >>> recs = [(0, 0), (5, 10), (10, 0)]
    >>> # Calculate the synthetic travel-times
    >>> from fatiando.seismic.traveltime import straight_ray_2d
    >>> ttimes = straight_ray_2d(model, 'vp', srcs, recs)
    >>> print ttimes
    [ 2.5  3.5  2.5]
    >>> # Run the tomography to calculate the 2 velocities
    >>> mesh = SquareMesh((0, 10, 0, 10), shape=(2, 1))
    >>> estimate, residuals = smooth(ttimes, srcs, recs, mesh)
    >>> for v in estimate:
    ...     print '%.2f' % (v),
    2.00 5.00
    >>> for v in residuals:
    ...     print '%.2f' % (v),
    0.00 -0.00 0.00

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'

import time
import numpy
import scipy.sparse
import scipy.sparse.linalg

from fatiando import logger, inversion, utils
from fatiando.seismic import _traveltime
log = logger.dummy()


class TravelTimeStraightRay2D(inversion.datamodule.DataModule):
    """
    Data module for the 2D travel-time straight-ray tomography problem.

    Bundles together the travel-time data, source and receiver positions
    and uses this to calculate gradients, Hessians and other inverse problem
    related quantities.

    Parameters:

    * ttimes
        Array with the travel-times of the straight seismic rays.
    * srcs
        Array with the (x, y) positions of the sources.
    * recs
        Array with the (x, y) positions of the receivers.
    * mesh
        The mesh where the inversion (tomography) will take place.
        Typically a :class:`fatiando.mesher.dd.SquareMesh`

    The ith travel-time is the time between the ith element in *srcs* and the
    ith element in *recs*.

    For example::

        >>> # One source
        >>> src = (5, 0)
        >>> # was recorded at 3 receivers
        >>> recs = [(0, 0), (5, 10), (10, 0)]
        >>> # Resulting in Vp travel times
        >>> ttimes = [2.5, 5., 2.5]
        >>> # If running the tomography on a mesh like
        >>> from fatiando.mesher.dd import SquareMesh
        >>> mesh = SquareMesh(bounds=(0, 10, 0, 10), shape=(10, 10))
        >>> # what TravelTimeStraightRay2D expects is
        >>> srcs = [src, src, src]
        >>> dm = TravelTimeStraightRay2D(ttimes, srcs, recs, mesh)

    
    """

    def __init__(self, ttimes, srcs, recs, mesh):
        inversion.datamodule.DataModule.__init__(self, ttimes)
        self.x_src, self.y_src = numpy.array(srcs, dtype='f').T
        self.x_rec, self.y_rec = numpy.array(recs, dtype='f').T
        self.mesh = mesh
        log.info("Initializing TravelTimeStraightRay2D data module:")
        log.info("  Calculating Jacobian (sensitivity matrix)...")
        start = time.time()
        self.jacobian = self._get_jacobian()
        log.info("  time: %s" % (utils.sec2hms(time.time() - start)))
        self.jacobian_T = self.jacobian.T
        self.hessian = numpy.dot(self.jacobian_T, self.jacobian)

    def _get_jacobian(self):
        """
        Build the Jacobian (sensitivity) matrix using the travel-time data
        stored.
        """
        jac = numpy.array(
            [_traveltime.straight_ray_2d(1., float(c['x1']), float(c['y1']),
                float(c['x2']), float(c['y2']), self.x_src, self.y_src,
                self.x_rec, self.y_rec)
            for c in self.mesh]).T
        return jac

    def get_misfit(self, residuals):
        return numpy.linalg.norm(residuals)

    def get_predicted(self, p):
        return numpy.dot(self.jacobian, p)

    def sum_gradient(self, gradient, p=None, residuals=None):
        if p is None:
            return gradient - 2.*numpy.dot(self.jacobian_T, self.data)
        else:
            return gradient - 2.*numpy.dot(self.jacobian_T, residuals)

    def sum_hessian(self, hessian, p=None):
        return hessian + 2.*self.hessian

def _slow2vel(slowness, tol=10**(-5)):
    """
    Safely convert slowness to velocity. 0 slowness is mapped to 0 velocity.
    """
    if slowness <= tol and slowness >= -tol:
        return 0.
    else:
        return 1./float(slowness)

def smooth(ttimes, srcs, recs, mesh, damping=0.):
    """
    Perform a tomography with smoothing regularization.

    Parameters:

    * ttimes
        Array with the travel-times of the straight seismic rays.
    * srcs
        Array with the (x, y) positions of the sources.
    * recs
        Array with the (x, y) positions of the receivers.
    * mesh
        The mesh where the inversion (tomography) will take place.
        Typically a :class:`fatiando.mesher.dd.SquareMesh`
    * damping
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.

    Returns:

    * [estimate, residuals]
        Arrays with the estimated velocity and residual vector, respectively
        
    """
    if len(ttimes) != len(srcs) != len(recs):
        msg = "Must have same number of travel-times, sources and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping must be positive"        
    regs = [inversion.regularizer.Damping(damping)]
    dms = [TravelTimeStraightRay2D(ttimes, srcs, recs, mesh)]
    initial = numpy.zeros(mesh.size, dtype='f')
    log.info("Running smooth straight-ray 2D travel-time tomography (SrTomo):")
    log.info("  damping: %g" % (damping))
    iterator = inversion.gradient.newton(dms, initial, regs, tol=0.001)
    start = time.time()
    #for i, chset in enumerate(iterator):
        #continue
    try:
        for i, chset in enumerate(iterator):
            continue
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")        
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
    velocity = [_slow2vel(s) for s in chset['estimate']]
    residuals = chset['residuals'][0]
    return velocity, residuals
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
