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

**FLAT AND HOMOGENEOUS EARTH**

* :func:`fatiando.seismic.epicenter.solve_flat`
* :func:`fatiando.seismic.epicenter.iterate_flat`

Estimates the (x, y) cartesian coordinates of the epicenter based on travel-time
residuals between S and P waves.

Use the :func:`fatiando.seismic.epicenter.solve_flat` function to
obtain an estimate, or the
:func:`fatiando.seismic.epicenter.iterate_flat` function to see each
step of the solver algorithm.

Example using the solve function::

    >>> from fatiando.mesher.dd import Square
    >>> from fatiando.seismic.traveltime import straight_ray_2d
    >>> from fatiando.inversion import gradient
    >>> # Generate synthetic travel-time residuals
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
    >>> # Solve using Newton's
    >>> # Generate synthetic travel-time residuals method
    >>> solver = gradient.newton(initial=(1, 1), tol=10**(-3), maxit=1000)
    >>> # Estimate the epicenter
    >>> p, residuals = solve_flat(ttres, recs, vp, vs, solver)
    >>> print "(%.4f, %.4f)" % (p[0], p[1])
    (5.0000, 5.0000)

Example using the iterate function::

    >>> from fatiando.mesher.dd import Square
    >>> from fatiando.seismic.traveltime import straight_ray_2d
    >>> from fatiando.inversion import gradient
    >>> # Generate synthetic travel-time residuals
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
    >>> # Solve using Newton's
    >>> # Generate synthetic travel-time residuals method
    >>> solver = gradient.newton(initial=(1, 1), tol=10**(-3), maxit=5)    
    >>> # Show the steps to estimate the epicenter
    >>> for p, r in iterate_flat(ttres, recs, vp, vs, solver):
    ...     print "(%.4f, %.4f)" % (p[0], p[1])
    (2.4157, 5.8424)
    (4.3279, 4.7485)
    (4.9465, 4.9998)
    (4.9998, 5.0000)
    (5.0000, 5.0000)

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
    Data module for epicenter estimation using travel-time residuals between
    S and P waves, assuming a flat and homogeneous Earth.

    The travel-time residual measured by the ith receiver is a function of the
    (x, y) coordinates of the epicenter:

    .. math::

        t_{S_i} - t_{P_i} = \\Delta t_i (x, y) =
        \\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\sqrt{(x_i - x)^2 + (y_i - y)^2}

    The elements :math:`G_{i1}` and :math:`G_{i2}` of the Jacobian matrix for
    this data type are

    .. math::

        G_{i1}(x, y) = -\\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\frac{x_i - x}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    .. math::

        G_{i2}(x, y) = -\\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\frac{y_i - y}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}
        
    Parameters:

    * ttres
        Array with the travel-time residuals between S and P waves
    * recs
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    
    """

    def __init__(self, ttres, recs, vp, vs):
        inversion.datamodule.DataModule.__init__(self, ttres)
        self.x_rec, self.y_rec = numpy.array(recs, dtype='f').T
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

class MinimumDistance(inversion.regularizer.Regularizer):
    """
    A regularizing function that imposes that the solution (estimated epicenter)
    be at the smallest distance from the receivers as possible.

    .. math::

        \\theta(\\bar{p}) = \\bar{d}^T\\bar{d}

    where :math:`\\bar{d}` is a vector with the distance from the current
    estimate :math:`\\bar{p} = \\begin{bmatrix}x && y\end{bmatrix}^T` to the
    receivers.

    .. math::

        d_i = \\sqrt{(x_i - x)^2 + (y_i - y)^2}

    The gradient vector of :math:`\\theta(\\bar{p})` is given by

    .. math::

        \\bar{g}(\\bar{p}) = \\begin{bmatrix}
        -2\\sum\\limits_{i=1}^N x_i - x \\\\[0.5cm]
        -2\\sum\\limits_{i=1}^N y_i - y \end{bmatrix}
        
    The elements :math:`G_{i1}` and :math:`G_{i2}` of the Jacobian matrix of
    :math:`\\theta(\\bar{p})` are

    .. math::

        G_{i1}(x, y) = -\\frac{x_i - x}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    .. math::

        G_{i2}(x, y) = -\\frac{y_i - y}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    And the Hessian matrix can be approximated by
    :math:`\\bar{\\bar{G}}^T\\bar{\\bar{G}}`.
        
    """

    def __init__(self, mu, recs):
        inversion.regularizer.Regularizer.__init__(self, mu)
        self.xrec, self.yrec = numpy.array(recs, dtype='f').T   

    def value(self, p):
        x, y = p
        distance = numpy.sqrt((self.xrec - x)**2 + (self.yrec - y)**2)
        return self.mu*numpy.linalg.norm(distance)**2

    def sum_gradient(self, gradient, p):
        x, y = p
        grad = [numpy.sum(self.xrec - x), numpy.sum(self.yrec - y)]
        return gradient + self.mu*-2.*numpy.array(grad)
        
    def sum_hessian(self, hessian, p):
        x, y = p
        sqrt = numpy.sqrt((self.xrec - x)**2 + (self.yrec - y)**2)
        jac_T = numpy.array([(x - self.xrec)/sqrt,
                             (y - self.yrec)/sqrt])
        return hessian + self.mu*2.*numpy.dot(jac_T, jac_T.T)        
        
def solve_flat(ttres, recs, vp, vs, solver, damping=0., mindist=0.):
    """
    Estimate the (x, y) coordinates of the epicenter of an event using
    travel-time residuals between P and S waves and assuming a flat and
    homogeneous Earth.

    Parameters:

    * ttres
        Array with the travel-time residuals between S and P waves
    * recs
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    * solver
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module.
    * damping
        Positive scalar regularizing parameter for Damping regularization.
        (:class:`fatiando.inversion.regularizer.Damping`).
    * mindist
        Positive scalar regularizing parameter for the Minimum Distance to
        Receivers regularization
        (:class:`fatiando.seismic.epicenter.MinimumDistance`).

    Returns:

    * [estimate, residuals]
        The estimated (x, y) coordinates and the residuals (difference between
        measured and predicted travel-time residuals)
    
    """    
    if len(ttres) != len(recs):
        msg = "Must have same number of travel-time residuals and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping must be positive"
    if mindist < 0:
        raise ValueError, "Mindist regularization parameter must be positive"
    dms = [TTResidualsFlatHomogeneous(ttres, recs, vp, vs)]
    regs = [MinimumDistance(mindist, recs),
            inversion.regularizer.Damping(damping)]
    log.info("Estimating epicenter assuming flat and homogeneous Earth:")
    log.info("  damping: %g" % (damping))
    log.info("  minimum distance from receivers: %g" % (mindist))
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
    return chset['estimate'], chset['residuals'][0]

def iterate_flat(ttres, recs, vp, vs, solver, damping=0., mindist=0.):
    """
    Yields the steps taken to estimate the (x, y) coordinates of the epicenter
    of an event using travel-time residuals between P and S waves and assuming
    a flat and homogeneous Earth.

    Parameters:

    * ttres
        Array with the travel-time residuals between S and P waves
    * recs
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    * solver
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module.
    * damping
        Positive scalar regularizing parameter for Damping regularization.
        (:class:`fatiando.inversion.regularizer.Damping`).
    * mindist
        Positive scalar regularizing parameter for the Minimum Distance to
        Receivers regularization
        (:class:`fatiando.seismic.epicenter.MinimumDistance`).

    Yields:

    * [estimate, residuals]
        The estimated (x, y) coordinates and the residuals (difference between
        measured and predicted travel-time residuals) for each iteration of the
        non-linear solver
    
    """
    if len(ttres) != len(recs):
        msg = "Must have same number of travel-time residuals and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping regularization parameter must be positive"
    if mindist < 0:
        raise ValueError, "Mindist regularization parameter must be positive"
    dms = [TTResidualsFlatHomogeneous(ttres, recs, vp, vs)]
    regs = [MinimumDistance(mindist, recs),
            inversion.regularizer.Damping(damping)]
    log.info("Estimating epicenter assuming flat and homogeneous Earth:")
    log.info("  damping: %g" % (damping))
    log.info("  minimum distance from receivers: %g" % (mindist))
    try:
        for i, chset in enumerate(solver(dms, regs)):
            yield chset['estimate'], chset['residuals'][0]       
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))

def mapgoal(xs, ys, ttres, recs, vp, vs, damping=0., mindist=0.):
    """
    Make a map of the goal function for a given set of inversion parameters.

    The goal function is define as:

    .. math::

        \\Gamma(\\bar{p}) = \\phi(\\bar{p}) + \\sum\\limits_{r=1}^{R}
        \\mu_r\\theta_r(\\bar{p})

    where :math:`\\phi(\\bar{p})` is the data-misfit function, 
    :math:`\\theta_r(\\bar{p})` is the rth of the R regularizing function used,
    and :math:`\\mu_r` is the rth regularizing parameter.

    Parameters:

    * xs, ys
        Lists of x and y values where the goal function will be calculated
    * ttres
        Array with the travel-time residuals between S and P waves
    * recs
        List with the (x, y) coordinates of the receivers
    * vp
        Assumed velocity of P waves
    * vs
        Assumed velocity of S waves
    * damping
        Positive scalar regularizing parameter for Damping regularization.
        (:class:`fatiando.inversion.regularizer.Damping`).
    * mindist
        Positive scalar regularizing parameter for the Minimum Distance to
        Receivers regularization
        (:class:`fatiando.seismic.epicenter.MinimumDistance`).

    Returns:

    * goals
        Array with the goal function values
    
    """    
    dm = TTResidualsFlatHomogeneous(ttres, recs, vp, vs)
    reg1 = MinimumDistance(mindist, recs)
    reg2 = inversion.regularizer.Damping(damping)
    return numpy.array([dm.get_misfit(ttres - dm.get_predicted(p)) +
                        reg1.value(p) + reg2.value(p) for p in zip(xs, ys)])
        
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
