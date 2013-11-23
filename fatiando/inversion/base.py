"""
The base classes for inverse problem solving.

----

"""
from __future__ import division

import hashlib

import numpy
import scipy.sparse

from ..utils import safe_solve, safe_diagonal, safe_dot

class Objective(object):
    """
    An objective function for an inverse problem.

    Objective functions have methods to find the parameter vector *p* that
    minimizes them. The :meth:`~fatiando.inversion.base.Objective.fit` method
    defaults to a linear solver for linear problems and the Levemberg-Marquardt
    algorithm for non-linear problems.

    Objective functions also know how to calculate their value, gradient and/or
    Hessian matrix for a given parameter vector *p*. These functions are
    problem specific and need to be implemented when subclassing *Objective*.

    Parameters:

    * nparams : int
        The number of parameters the objective function takes.
    * islinear : True or False
        Wether the functions is linear with respect to the parameters.

    """

    def __init__(self, nparams, islinear=False):
        pass

    def _init_stats(self):
        "Initialize the *stats* attribute with default values"
        pass

    def report(self, summary=True):
        """
        Produce a report of the last optimization method used.

        Uses the information in the *stats* attribute of the class to produce
        the output.

        Parameters:

        * summary : True or False
            If True, will make a summary report.

        Returns:

        * report : string
            The report.

        """
        pass

    def value(self, p):
        """
        The value of the objective function for a given parameter vector.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of the objective function

        """
        raise NotImplementedError("Misfit value not implemented")

    def gradient(self, p):
        """
        The gradient of the objective function with respect to the parameters.

        Parameters:

        * p : 1d-array
            The parameter vector where the gradient is evaluated.

        Returns:

        * gradient : 1d-array
            The gradient vector

        """
        raise NotImplementedError("Gradient vector not implemented")

    def hessian(self, p):
        """
        The Hessian of the objective function with respect to the parameters

        Parameters:

        * p : 1d-array
            The parameter vector where the Hessian is evaluated

        Returns:

        * hessian : 2d-array
            The Hessian matrix

        """
        raise NotImplementedError("Hessian matrix not implemented")

    def linear(self, precondition=True):
        """
        Solve for the parameter vector assuming that the problem is linear.

        Parameters:

        * precondition : True or False
            If True, will use Jacobi preconditioning.

        """
        pass
