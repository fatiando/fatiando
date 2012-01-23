# Copyright 2012 The Fatiando a Terra Development Team
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
Estimate the epicenter of a seismic event considering various approximations for
the Earth.

Flat and homogeneous Earth
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :func:`fatiando.seismic.epicenter.flat_homogeneous`

Estimates the (x, y) cartesian coordinates of the epicenter based on travel-time
residuals between S and P waves.

Example::

    >>> # Generate synthetic travel-time residuals
    >>> from fatiando.mesher.dd import Square
    >>> from fatiando.seismic.traveltime import straight_ray_2d
    >>> area = (0, 10, 0, 10)
    >>> vp = 2
    >>> vs = 1
    >>> model = [Square(area, props={'vp':vp, 'vs':vs})]
    >>> # The true source (epicenter)
    >>> src = (5, 5)
    >>> recs = [(5, 0), (5, 10), (10, 0)]
    >>> srcs = [src, src, src]
    >>> ptime = straight_ray_2d(model, 'vp', srcs, recs)
    >>> stime = straight_ray_2d(model, 'vs', srcs, recs)
    >>> ttres = stime - ptime
    >>> # Estimate the epicenter
    >>> initial = (1, 1)
    >>> estimate, residuals = flat_homogeneous(ttres, recs, vp, vs, initial)
    >>> print estimate
    [ 5.  5.]

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'

import time

import numpy

from fatiando import logger, inversion, utils

log = logger.dummy()


class TTResidualsFlatHomogeneous(inversion.datamodule.DataModule):
    """
    Data module for epicenter estimation using travel-time residuals and
    assuming a flat and homogeneous Earth.

    Parameters:

    * ttresiduals
        Array with the travel-time residuals between S and P waves
    * receivers
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    
    """

    def __init__(self, ttresiduals, receivers, vp, vs):
        inversion.datamodule.DataModule.__init__(self, ttresiduals)
        self.x_rec, self.y_rec = numpy.array(receivers, dtype='f').T
        self.vp = vp
        self.vs = vs
        self.alpha = 1./float(vs) - 1./float(vp)

    def get_misfit(self, residuals):
        return numpy.linalg.norm(residuals)

    def get_predicted(self, p):
        x, y = p
        return self.alpha*numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        
    def sum_gradient(self, gradient, p=None, residuals=None):
        x, y = p
        sqrt = numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        jac_T = [-self.alpha*(self.x_rec - x)/sqrt,
                 -self.alpha*(self.y_rec - y)/sqrt]
        return gradient - 2.*numpy.dot(jac_T, residuals)

    def sum_hessian(self, hessian, p=None):
        x, y = p
        sqrt = numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        jac_T = numpy.array([-self.alpha*(self.x_rec - x)/sqrt,
                             -self.alpha*(self.y_rec - y)/sqrt])
        return hessian + 2.*numpy.dot(jac_T, jac_T.T)        

def flat_homogeneous(ttresiduals, receivers, vp, vs, initial, damping=0.):
    """
    Estimate the (x, y) coordinates of the epicenter of an event using
    travel-time residuals between P and S waves and assuming a flat and
    homogeneous Earth.

    The travel-time residual measured by the ith receiver is a function of the
    (x, y) coordinates of the epicenter:

    .. math::

        t_{S_i} - t_{P_i} = \\Delta t_i (x, y) =
        \\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\sqrt{(x_i - x)^2 + (y_i - y)^2}

    Parameters:

    * ttresiduals
        Array with the travel-time residuals between S and P waves
    * receivers
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    * initial
        (x, y) coordinates of the initial estimate

    Returns:

    * [estimate, residuals]
        The estimated (x, y) coordinates and the residuals (difference between
        measured and predicted travel-time residuals)
    
    """    
    if len(ttresiduals) != len(receivers):
        msg = "Must have same number of travel-time residuals and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping must be positive"        
    regs = [inversion.regularizer.Damping(damping)]
    dms = [TTResidualsFlatHomogeneous(ttresiduals, receivers, vp, vs)]
    log.info("Estimating epicenter assuming flat and homogeneous Earth:")
    log.info("  damping: %g" % (damping))
    iterator = inversion.gradient.newton(dms, numpy.array(initial, dtype='f'),
        regs, tol=10**(-3), maxit=1000)
    start = time.time()
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
    return chset['estimate'], chset['residuals'][0]

def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
