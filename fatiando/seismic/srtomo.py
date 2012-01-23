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
Straight-ray 2D travel time tomography (i.e. does not consider reflection or
refraction)

Examples::

    some code
    

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



    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
