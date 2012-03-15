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
Direct modeling and inversion of seismic profiles.

**VERTICAL SEISMIC PROFILING**

* :func:`fatiando.seismic.profile.vertical`
* :func:`fatiando.seismic.profile.invert_vertical`

Model and invert vertical seismic profiling data. In this kind of profiling, the
wave source is located at the surface on top of the well. The travel-times of
first arrivals is then measured at different depths along the well. The
ith travel-time :math:`t_i` measured at depth :math:`z_i` is a function of the
wave velocity :math:`v_j` and distance :math:`d_{ij}` that it traveled in each
layer

.. math::

    t_i(z_i) = \\sum\\limits_{j=1}^M \\frac{d_{ij}}{v_j}

The distance :math:`d_{ij}` is smaller or equal to the thickness of the layer
:math:`s_j`. Notice that :math:`d_{ij} = 0` if the jth layer is bellow
:math:`z_i`, :math:`d_{ij} = s_j` if the jth layer is above :math:`z_i`, and
:math:`d_{ij} < s_j` if :math:`z_i` is inside the jth layer.

To generate synthetic seismic profiling data, use
:func:`fatiando.seismic.profile.vertical` like so::

    >>> from fatiando.seismic import profile
    >>> # Make the synthetic 4 layer model
    >>> thicks = [10, 20, 10, 30]
    >>> vels = [2, 4, 10, 5]
    >>> # Make an array with the z_i
    >>> zs = [10, 30, 40, 70]
    >>> # Calculate the travel-times
    >>> for t in profile.vertical(thicks, vels, zs):    
    ...     print '%.1f' % (t), 
    5.0 10.0 11.0 17.0


To make :math:`t_i` linear with respect to :math:`v_j`, we can use
*slowness* :math:`w_j` instead of velocity

.. math::

    t_i(z_i) = \\sum\\limits_{j=1}^M d_{ij} w_j

This allows us to easily invert for the slowness of each layer, given their
thickness. Here's an example of using
:func:`fatiando.seismic.profile.invert_vertical` to do this on some synthetic
data::

    >>> import numpy
    >>> from fatiando.seismic import profile
    >>> from fatiando.inversion import linear
    >>> # Make the synthetic 4 layer model
    >>> thicks = [10, 20, 10, 30]
    >>> vels = [2, 4, 10, 8]
    >>> # Make an array with the z_i
    >>> zs = numpy.arange(5, sum(thicks), 1)
    >>> # Calculate the travel-times
    >>> tts = profile.vertical(thicks, vels, zs)
    >>> # Make a linear solver and solve for the slowness
    >>> solver = linear.overdet(nparams=len(thicks))
    >>> p, residuals = profile.invert_vertical(tts, zs, thicks, solver)
    >>> for slow in p:
    ...     print '%.1f' % (1./slow), 
    2.0 4.0 10.0 8.0

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Jan-2012'

import time

import numpy
import numpy.linalg.linalg

from fatiando.seismic import traveltime
from fatiando.mesher.dd import Square
from fatiando import logger, inversion, utils


class VerticalSlownessDM(inversion.datamodule.DataModule):
    """
    Data module for a vertical seismic profile first-arrival travel-time data.
    Assumes that only the slowness of the layers are parameters in the
    inversion.

    In this case, the inverse problem in linear. The element :math:`G_{ij}` of
    the Jacobian (sensitivity) matrix is given by

    .. math::

        G_{ij} = d_{ij}

    where :math:`d_{ij}` is the distance that the ith first-arrival traveled
    inside the jth layer.

    Uses :func:`fatiando.seismic.traveltime.straight_ray_2d` for direct modeling
    to build the Jacobian matrix.

    Parameters:

    * traveltimes
        List with the first-arrival travel-times calculated at the measurement
        stations
    * zp
        List with the depths of the measurement stations (seismometers)
    * thickness
        List with the thickness of each layer in order of increasing depth
    
    """

    def __init__(self, traveltimes, zp, thickness):
        log = logger.dummy('fatiando.seismic.profile.VerticalSlownessDM')
        inversion.datamodule.DataModule.__init__(self, traveltimes)
        self.zp = zp
        self.thickness = thickness
        log.info("  calculating Jacobian (sensitivity) matrix...")
        self.jac_T = self._get_jacobian()

    def _get_jacobian(self):    
        nlayers = len(self.thickness)   
        zmax = sum(self.thickness) 
        z = [sum(self.thickness[:i]) for i in xrange(nlayers + 1)]
        layers = [Square((0, zmax, z[i], z[i + 1]), props={'vp':1.})
                  for i in xrange(nlayers)]
        srcs = [(0, 0)]*len(self.zp)
        recs = [(0, z) for z in self.zp]        
        jac_T = numpy.array([traveltime.straight_ray_2d([l], 'vp', srcs, recs)
                             for l in layers])
        return jac_T

    def get_predicted(self, p):
        return vertical(self.thickness, 1./numpy.array(p), self.zp)

    def sum_gradient(self, gradient, p=None, residuals=None):
        return gradient - 2.*numpy.dot(self.jac_T, self.data)

    def sum_hessian(self, hessian, p=None):
        return hessian + 2.*numpy.dot(self.jac_T, self.jac_T.T)       
    
def vertical(thickness, velocity, zp):
    """
    Calculates the first-arrival travel-times for given a layered model.
    Simulates a vertical seismic profile.

    The source is assumed to be at z = 0. The z-axis is positive downward.

    Parameters:

    * thickness
        List with the thickness of each layer in order of increasing depth
    * velocity
        List with the velocity of each layer in order of increasing depth
    * zp
        List with the depths of the measurement stations (seismometers)
        
    Returns:

    * travel_times
        List with the first-arrival travel-times calculated at the measurement
        stations.
    
    """
    if len(thickness) != len(velocity):
        raise ValueError, "thickness and velocity must have same length"
    nlayers = len(thickness)
    zmax = sum(thickness)
    z = [sum(thickness[:i]) for i in xrange(nlayers + 1)]
    layers = [Square((0, zmax, z[i], z[i + 1]), props={'vp':velocity[i]})
              for i in xrange(nlayers)]
    srcs = [(0, 0)]*len(zp)
    recs = [(0, z) for z in zp]
    return traveltime.straight_ray_2d(layers, 'vp', srcs, recs)

def invert_vertical(traveltimes, zp, thickness, solver, damping=0., smooth=0.,
    sharp=0., beta=10.**(-10), iterate=False):
    """
    Invert first-arrival travel-time data for the slowness of each layer.

    Parameters:

    * traveltimes
        List with the first-arrival travel-times calculated at the measurement
        stations
    * zp
        List with the depths of the measurement stations (seismometers)
    * thickness
        List with the thickness of each layer in order of increasing depth
    * solver
        A linear or non-linear inverse problem solver generated by a factory
        function from a :mod:`fatiando.inversion` inverse problem solver module.
    * damping
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.
    * smooth
        Smoothness regularizing parameter (i.e., how much smoothness to apply).
        Must be a positive scalar.    
    * sharp
        Sharpness (total variation) regularizing parameter (i.e., how much
        sharpness to apply). Must be a positive scalar.
    * beta
        Total variation parameter. See
        :class:`fatiando.inversion.regularizer.TotalVariation` for details
    * iterate
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    Returns:

    * [slowness, residuals]

        * slowness: Array with the slowness of each layer
        * residuals: Array with the inversion residuals (observed travel-times
            minus predicted travel-times by the slowness estimate)
        
    """
    log = logger.dummy('fatiando.seismic.profile.invert_vertical')
    if damping < 0:
        raise ValueError, "Damping parameter must be positive"
    if smooth < 0:
        raise ValueError, "Smoothness parameter must be positive"
    if len(traveltimes) != len(zp):
        raise ValueError, "traveltimes and zp must have same length"
    log.info("Invert a vertical seismic profile for slowness:")
    log.info("  number of layers: %d" % (len(thickness)))
    log.info("  iterate: %s" % (str(iterate)))
    log.info("  damping: %g" % (damping))
    log.info("  smoothness: %g" % (smooth))
    log.info("  sharpness: %g" % (sharp))
    log.info("  beta (total variation parameter): %g" % (beta))
    nparams = len(thickness)
    dms = [VerticalSlownessDM(traveltimes, zp, thickness)]
    regs = []
    if damping != 0.:
        regs.append(inversion.regularizer.Damping(damping, nparams))
    if smooth != 0.:
        regs.append(inversion.regularizer.Smoothness1D(smooth, nparams))
    if sharp != 0.:
        regs.append(inversion.regularizer.TotalVariation1D(sharp, nparams,
                                                           beta))
    if iterate:
        return _iterator(dms, regs, solver)
    else:
        return _solver(dms, regs, solver)
        
def _solver(dms, regs, solver):
    log = logger.dummy('fatiando.seismic.profile.invert_vertical')
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

def _iterator(dms, regs, solver):
    log = logger.dummy('fatiando.seismic.profile.invert_vertical')
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, regs)):
            yield chset['estimate'], chset['residuals'][0]
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
