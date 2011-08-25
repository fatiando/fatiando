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
Functions for adding various kinds of regularization to an inverse problem
solver.

Implemented regularizations:

* Tikhonov orders 0, 1 and 2
    Imposes minimum norm (damping), smoothness and minimum curvature,
    respectively, on the solution.

* Total Variation
    Imposes minimum l1 norm of the model derivatives (ie, discontinuities)

* Compact
    Imposes minimum area (or volume) of the solution (as in Last and Kubic
    (1983) without parameter freezing)

Functions:

* :func:`fatiando.inv.reg.damp_norm`
    Calculate the norm of the damping regularizing function

* :func:`fatiando.inv.reg.damp_grad`
    Sum the gradient vector of the damping regularizing function

* :func:`fatiando.inv.reg.damp_hess`
    Sum the Hessian matrix of the damping regularizing function

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 25-Apr-2011'


import numpy


def damp_norm(damping, estimate):
    """
    Calculate the value of the damping regularizing function.

    Parameters:

    * damping:
        Regularization parameter. Must be >= 0.

    * estimate:
        1D array with the current estimate (not used)

    Returns:

    * float:
        value of the damping regularizing function

    """
    return damping*(numpy.linalg.norm(estimate)**2)


def damp_grad(damping, estimate, grad):
    return grad + damping*estimate


def damp_hess(damping, hess):
    return hess + damping*numpy.eye(hess.shape[0])


def fdmatrix2d(nx, ny):
    """
    Builds a finite-differences matrix for a 2D grid with the given shape.
    """
    deriv_num = (nx - 1)*ny + (ny - 1)*nx
    fdmatrix = numpy.zeros((deriv_num, nx*ny))
    deriv_i = 0
    # Derivatives in the x direction
    param_i = 0
    for i in xrange(ny):
        for j in xrange(nx - 1):
            fdmatrix[deriv_i][param_i] = 1
            fdmatrix[deriv_i][param_i + 1] = -1
            deriv_i += 1
            param_i += 1
        param_i += 1
    # Derivatives in the y direction
    param_i = 0
    for i in xrange(ny - 1):
        for j in xrange(nx):
            fdmatrix[deriv_i][param_i] = 1
            fdmatrix[deriv_i][param_i + nx] = -1
            deriv_i += 1
            param_i += 1
    return fdmatrix


def smooth2d_norm(smoothness, weights, estimate):
    return smoothness*numpy.dot(numpy.dot(estimate, weights), estimate)


def smooth2d_grad(smoothness, weights, estimate, grad):
    return grad + smoothness*numpy.dot(weights, estimate)


def smooth2d_hess(smoothness, weights, hess):
    return hess + smoothness*weights
