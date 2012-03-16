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
Base DataModule class with the format expected by all inverse problem solvers.

See the docs for the :mod:`~fatiando.inversion` package for more information on
the role of the data modules.

:author: Leonardo Uieda (leouieda@gmail.com)
:date: Created 19-Jan-2012
:license: GNU Lesser General Public License v3 (http://www.gnu.org/licenses/)

----

"""

import numpy

from fatiando import logger


log = logger.dummy('fatiando.inversion.datamodule')


class DataModule(object):
    """
    A generic data module.

    Use this class as a skeleton for building custom data modules for a specific
    geophysical data set and interpretative model, like gravity anomaly for
    right rectangular prism models, travel time residuals for epicenter
    calculation, etc.

    Data modules are how each inverse problem solver knows how to calculate
    things like:

    * Predicted data
    * Data-misfit function
    * Gradient of the data-misfit function
    * Hessian of the data-misfit function

    Not all solvers use all of the above. For examples, heuristic solvers don't
    require gradient and hessian calculations.

    This class has templates for all of these methods so that all solvers know
    what to expect.

    Normally, all data modules should store the value of the latest residual
    vector calculated.

    Constructor parameters common to all methods:

    * data : array
        The observed data.
                
    """

    def __init__(self, data):
        self.data = data

    def get_misfit(self, residuals):
        """
        Returns the value of the data-misfit function for a given residual
        vector

        Parameters:

        * residuals : array
            The residual vector

        Returns:

        * misfit : float
            Scalar value of the data-misfit
            
        """
        return numpy.linalg.norm(residuals)

    def get_predicted(self, p):
        """
        Returns the predicted data vector for a given parameter vector.

        Parameters:

        * p : array
            The parameter vector

        Returns:

        * pred : array
            The calculated predicted data vector
            
        """
        raise NotImplementedError("get_predicted method not implemented")

    def sum_gradient(self, gradient, p=None, residuals=None):
        """
        Sums the gradient vector of this data module to *gradient* and returns
        the result.

        Parameters:

        * gradient : array
            The old gradient vector
        * p : array
            The parameter vector
        * residuals : array
            The residuals evaluated for parameter vector *p*
            
        .. note:: Solvers for linear problems will use ``p = None`` and
            ``residuals = None`` so that the class knows how to calculate
            gradients more efficiently for these cases.

        Returns:

        * new_gradient : array
            The new gradient vector
        
        """
        raise NotImplementedError("sum_gradient method not implemented")

    def sum_hessian(self, hessian, p=None):
        """
        Sums the Hessian matrix of this data module to *hessian* and returns
        the result.

        Parameters:

        * hessian : array
            2D array with the old Hessian matrix
        * p : array
            The parameter vector
            
        .. note:: Solvers for linear problems will use ``p = None`` so that the
            class knows how to calculate gradients more efficiently for these
            cases.

        Returns:

        * new_hessian : array
            2D array with the new Hessian matrix
        
        """
        raise NotImplementedError("sum_hessian method not implemented")
