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

**SOLVERS**

* :func:`fatiando.seismic.srtomo.smooth`
* :func:`fatiando.seismic.srtomo.sharp`

**Examples**

Using simple synthetic data and a linear solver::

    >>> from fatiando.mesher.dd import Square, SquareMesh
    >>> from fatiando.inversion import linear
    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
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
    >>> # Will use a linear overdetermined solver
    >>> solver = linear.overdet(mesh.size)
    >>> estimate, residuals = smooth(ttimes, srcs, recs, mesh, solver)
    >>> # Actually returns slowness instead of velocity
    >>> for slowness in estimate:
    ...     print '%.4f' % (1./slowness),
    2.0000 5.0000
    >>> for v in residuals:
    ...     print '%.4f' % (v),
    0.0000 0.0000 0.0000

Again, using simple synthetic data but this time use Newton's method to solve::

    >>> from fatiando.mesher.dd import Square, SquareMesh
    >>> from fatiando.inversion.gradient import newton
    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
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
    >>> # Will use Newton's method to solve this
    >>> solver = newton(initial=[0, 0], maxit=5)
    >>> estimate, residuals = smooth(ttimes, srcs, recs, mesh, solver)
    >>> # Actually returns slowness instead of velocity
    >>> for v in estimate:
    ...     print '%.4f' % (1./v),
    2.0000 5.0000
    >>> for v in residuals:
    ...     print '%.4f' % (v),
    0.0000 0.0000 0.0000

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'

import time
import numpy
import scipy.sparse

from fatiando import logger, inversion, utils
from fatiando.seismic import _traveltime
log = logger.dummy()


class TravelTime(inversion.datamodule.DataModule):
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
        >>> # what TravelTime expects is
        >>> srcs = [src, src, src]
        >>> dm = TravelTime(ttimes, srcs, recs, mesh)
    
    """

    def __init__(self, ttimes, srcs, recs, mesh):
        inversion.datamodule.DataModule.__init__(self, ttimes)
        self.xsrc, self.ysrc = numpy.array(srcs, dtype='f').T
        self.xrec, self.yrec = numpy.array(recs, dtype='f').T
        self.mesh = mesh
        self.nparams = mesh.size
        self.ndata = len(ttimes)
        log.info("Initializing travel time data module:")
        log.info("  number of parameters: %d" % (self.nparams))
        log.info("  number of data: %d" % (self.ndata))
        log.info("  calculating Jacobian (sensitivity matrix)...")
        start = time.time()
        self.jacobian = self._get_jacobian()
        log.info("  time: %s" % (utils.sec2hms(time.time() - start)))
        self.jacobian_T = self.jacobian.T
        log.info("  calculating Hessian matrix...")
        start = time.time()
        self.hessian = self._get_hessian()
        log.info("  time: %s" % (utils.sec2hms(time.time() - start)))

    def _get_jacobian(self):
        """
        Build the Jacobian (sensitivity) matrix using the travel-time data
        stored.
        """
        xsrc, ysrc, xrec, yrec = self.xsrc, self.ysrc, self.xrec, self.yrec
        jac = numpy.array(
            [_traveltime.straight_ray_2d(1., float(c['x1']), float(c['y1']),
                float(c['x2']), float(c['y2']), xsrc, ysrc, xrec, yrec)
            for c in self.mesh]).T
        return jac

    def _get_hessian(self):
        """
        Build the Hessian matrix by Gauss-Newton approximation.
        """
        return numpy.dot(self.jacobian_T, self.jacobian)

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

class TravelTimeSparse(TravelTime):
    """
    Version of the :class:`fatiando.seismic.srtomo.TravelTime` class that uses
    sparse matrices.
    """

    def __init__(self, ttimes, srcs, recs, mesh):
        TravelTime.__init__(self, ttimes, srcs, recs, mesh)

    def _get_jacobian(self):
        """
        Build the sparse Jacobian (sensitivity) matrix using the travel-time
        data stored.
        """
        shoot = _traveltime.straight_ray_2d
        xsrc, ysrc, xrec, yrec = self.xsrc, self.ysrc, self.xrec, self.yrec
        nonzero = []
        for j, c in enumerate(self.mesh):
            nonzero.extend((i, j, tt) for i, tt in enumerate(shoot(1.,
                float(c['x1']), float(c['y1']), float(c['x2']), float(c['y2']),
                xsrc, ysrc, xrec, yrec)) if tt != 0)
        row, col, val = numpy.array(nonzero).T
        shape = (self.ndata, self.nparams)
        jac = scipy.sparse.csr_matrix((val, (row, col)), shape)
        return jac

    def _get_hessian(self):
        """
        Build the Hessian matrix by Gauss-Newton approximation.
        """
        return self.jacobian_T*self.jacobian

    def get_predicted(self, p):
        return self.jacobian*p

    def sum_gradient(self, gradient, p=None, residuals=None):
        if p is None:
            return gradient - 2.*self.jacobian_T*self.data
        else:
            return gradient - 2.*self.jacobian_T*residuals   

def _slow2vel(slowness, tol=10**(-5)):
    """
    Safely convert slowness to velocity. 0 slowness is mapped to 0 velocity.
    """
    if slowness <= tol and slowness >= -tol:
        return 0.
    else:
        return 1./float(slowness)

def smooth(ttimes, srcs, recs, mesh, solver, sparse=False, damping=0.):
    """
    Perform a tomography with smoothing regularization. Estimates the slowness
    (1/velocity) of cells in mesh (because slowness is linear and easier)

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
    * sparse
        If True, will use sparse matrices from scipy.sparse
        Don't forget to turn on sparcity in the inversion solver module.
        The usual way of doing this is by calling the ``use_sparse`` function.
        Ex: ``fatiando.inversion.gradient.use_sparse()``
        WARNING: Jacobian matrix building using sparse matrices isn't very
        optimized. It will be slow but won't overflow the memory.
    * damping
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.

    Returns:

    * [estimate, residuals]
        Arrays with the estimated slowness and residual vector, respectively
        
    """
    if len(ttimes) != len(srcs) != len(recs):
        msg = "Must have same number of travel-times, sources and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping must be positive"
    if sparse:
        dms=[TravelTimeSparse(ttimes, srcs, recs, mesh)]   
        regs = [inversion.regularizer.DampingSparse(damping)]     
    else:
        dms = [TravelTime(ttimes, srcs, recs, mesh)]
        regs = [inversion.regularizer.Damping(damping)]
    log.info("Running smooth straight-ray 2D travel-time tomography (SrTomo):")
    log.info("  damping: %g" % (damping))
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, regs)):
            continue
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")        
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
    slowness = chset['estimate']
    residuals = chset['residuals'][0]
    return slowness, residuals
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
