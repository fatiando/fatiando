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
    return damping*(numpy.linalg.norm(estimate)**2)

def damp_grad(damping, estimate, grad):
    grad += damping*estimate

def damp_hess(damping, hess):
    for i in xrange(len(hess)):
        # Use the comma notation for accessing elements to be compatible
        # with Scipy's sparse matrices
        hess[i,i] += damping
