"""
Base DataModule class with the format expected by all inverse problem solvers.

See the docs for the :mod:`~fatiando.inversion` package for more information on
the role of the data modules.

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
