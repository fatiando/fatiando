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
Calculate an equivalent layer of sources and make gravity field transformations
with it.

Functions:


"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 30-Nov-2010'


import time
import logging

import numpy

import fatiando
import fatiando.grav.sphere
import fatiando.grid

log = logging.getLogger('fatiando.grav.eqlayer')
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)



def _build_jacobian(data, layer):
    """
    Build the Jacobian matrix for the equivalent layer
    """

    jacobian = []

    append_row = jacobian.append

    for x, y, z in zip(data['x'], data['y'], data['z']):

        row = [fatiando.grav.sphere.gz(1., 1., xc, yc, zc, x, y, z)
                for xc, yc, zc in zip(layer['x'], layer['y'], layer['z'])
                ]

        append_row(row)

    jacobian = numpy.array(jacobian)

    return jacobian


def _build_first_deriv(layer):
    """
    Build the first derivative finite differences matrix for the layer
    """

    ny, nx = layer['ny'], layer['nx']

    deriv_num = (nx - 1)*ny + (ny - 1)*nx

    first_deriv = numpy.zeros((deriv_num, nx*ny))

    deriv_i = 0

    # Derivatives in the x direction
    param_i = 0
    for i in xrange(ny):

        for j in xrange(nx - 1):

            first_deriv[deriv_i][param_i] = 1

            first_deriv[deriv_i][param_i + 1] = -1

            deriv_i += 1

            param_i += 1

        param_i += 1

    # Derivatives in the y direction
    param_i = 0
    for i in xrange(ny - 1):

        for j in xrange(nx):

            first_deriv[deriv_i][param_i] = 1

            first_deriv[deriv_i][param_i + nx] = -1

            deriv_i += 1

            param_i += 1

    return first_deriv


def generate(layer, data, damping=10**(-10), smoothness=10**(-10)):
    """
    """

    log.info("Solving for the mass distribution:")

    estimate = numpy.zeros_like(layer['x'])

    data_vector = data['value']

    start = time.time()

    jacobian = _build_jacobian(data, layer)

    end = time.time()

    log.info("Build Jacobian matrix (%g s)" % (end - start))

    # Make a Jacobian builder to overload the corresponding function in solvers
    def dummy_jacobian(estimate):

        return jacobian

    def first_deriv():

        return _build_first_deriv(layer)

    # Also need a function to calculate the adjusted (predicted) data
    def calc_adjusted(estimate):

        tmp = fatiando.grid.copy(layer)

        adjusted = fatiando.grid.copy(data)

        density = numpy.reshape(estimate, (layer['ny'], layer['nx']))

        fatiando.grid.fill(density, tmp)

        adjusted = adjustment(tmp, data)

        return ajudsted['value']

    # Import a local copy of solvers so that it won't interfere with other
    # functions
    from fatiando.inv import solvers

    solvers._build_jacobian = dummy_jacobian
    solvers._build_first_deriv_matrix = first_deriv
    solvers._calc_adjustment = calc_adjusted
    solvers.damping = damping
    solvers.smoothness = smoothness

    results = solvers.linear_underdet(data_vector)

    estimate, residuals, goal = results

    density = numpy.reshape(estimate, (layer['ny'], layer['nx']))

    fatiando.grid.fill(density, layer)

    return residuals


def calculate(layer, grid, field='gz'):
    """
    Calculate a gravity field cause by *layer* on the grid **IN PLACE**

    Use this to upward-continue, interpolate and calculate other components as
    well as calculate the adjustment to the original data set.

    Parameters:

    * layer
        An equivalent layer stored in a grid dictionary, as output by
        :func:`fatiando.grav.eqlayer.generate`. (see :mod:`fatiando.grid`)

    * grid
        Grid where *field* will be calculated. (see :mod:`fatiando.grid`)

    * field
        What component of the gravity field to calculate. Can be any one of
        ``'gz'``, ``'gxx'``, ``'gxy'``, ``'gxz'``, ``'gyy'``, ``'gyz'``,
        ``'gzz'``

    """

    layer_it = zip(layer['x'], layer['y'], layer['z'], layer['value'])

    grid['value'] = [
        sum([fatiando.grav.sphere.gz(dens, 1., xc, yc, zc, x, y, z)
            for xc, yc, zc, dens in layer_it])
        for x, y, z in zip(grid['x'], grid['y'], grid['z'])]