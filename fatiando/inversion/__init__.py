"""
Everything you need to solve inverse problems!

Examples
--------

Linear Regression
=================

Here is an example of how to implement a simple linear regression using the
:class:`~fatiando.inversion.base.Misfit` class.

>>> import numpy as np
>>> from fatiando.inversion.base import Misfit
>>> class Regression(Misfit):
...     "Perform a linear regression"
...     def __init__(self, x, y):
...         super(Regression, self).__init__(data=y, positional={'x':x},
...             model={}, nparams=2, islinear=True)
...     def _get_predicted(self, p):
...         a, b = p
...         return a*self.positional['x'] + b
...     def _get_jacobian(self, p):
...         return np.transpose([self.positional['x'], np.ones(self.ndata)])
>>> x = np.linspace(0, 5, 6)
>>> y = 2*x + 5
>>> y
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> solver = Regression(x, y)
>>> solver.fit().estimate_
array([ 2.,  5.])
>>> solver.predicted()
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)

Polynomial fit
==============

A more complicated example would be to implement a generic polynomial fit.

>>> class PolynomialRegression(Misfit):
...     "Perform a polynomial regression"
...     def __init__(self, x, y, degree):
...         super(PolynomialRegression, self).__init__(
...             data=y, positional={'x':x},
...             model={'degree':degree}, nparams=degree + 1, islinear=True)
...     def _get_predicted(self, p):
...         return sum(p[i]*self.positional['x']**i
...                    for i in xrange(self.model['degree'] + 1))
...     def _get_jacobian(self, p):
...         return np.transpose([self.positional['x']**i
...                             for i in xrange(self.model['degree'] + 1)])
>>> solver = PolynomialRegression(x, y, 1)
>>> solver.fit().estimate_
array([ 5.,  2.])
>>> # Use a second order polynomial
>>> y = 0.1*x**2 + 3*x + 6
>>> solver = PolynomialRegression(x, y, 2)
>>> solver.fit().estimate_
array([ 6. ,  3. ,  0.1])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)

You can also configure the solver to use a different optimization method:

>>> # Configure solver to use the Levemberg-Marquardt method
>>> solver.config('levmarq', initial=[1, 1, 1]).fit().estimate_
array([ 6. ,  3. ,  0.1])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)


Non-linear Gaussian fit
=======================

In this example, I want to fit an equation of the form

.. math::

    f(x) = a\exp(-b(x + c)^{2})

Function *f* is non-linear with respect to inversion parameters *a, b, c*.

>>> class GaussianFit(Misfit):
...     def __init__(self, x, y):
...         super(GaussianFit, self).__init__(data=y,
...             positional={'x':x},
...             model={},
...             nparams=3, islinear=False)
...     def _get_predicted(self, p):
...         a, b, c = p
...         x = self.positional['x']
...         return a*np.exp(-b*(x + c)**2)
...     def _get_jacobian(self, p):
...         a, b, c = p
...         x = self.positional['x']
...         jac = np.transpose([
...             np.exp(-b*(x + c)**2),
...             # Do a numerical derivative for these two because I'm lazy
...             (a*np.exp(-(b + 0.005)*(x + c)**2) -
...              a*np.exp(-(b - 0.005)*(x + c)**2))/0.01,
...             (a*np.exp(-(b)*(x + c + 0.005)**2) -
...              a*np.exp(-(b)*(x + c - 0.005)**2))/0.01])
...         return jac
>>> x = np.linspace(0, 10, 1000)
>>> a, b, c = 100, 0.1, -2
>>> y = a*np.exp(-b*(x + c)**2)
>>> # Non-linear solvers have to be configured. Lets use Levemberg-Marquardt.
>>> solver = GaussianFit(x, y).config('levmarq', initial=[1, 1, 1])
>>> solver.fit().estimate_
array([ 100. ,    0.1,   -2. ])
>>> np.all(np.abs(solver.residuals()) < 10**-10)
True

Joint inversion
===============

Sometimes multiple data types depend on the same parameters (e.g., gravity
and gravity gradients depend of density). In these cases, the inversion of both
datasets can be performed simultaneously:

>>> x1 = np.linspace(0, 5, 6)
>>> y1 = 2*x1 + 5
>>> y1
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> x2 = np.linspace(100, 1000, 4)
>>> y2 = 2*x2 + 5
>>> y2
array([  205.,   805.,  1405.,  2005.])
>>> solver = Regression(x1, y1) + Regression(x2, y2)
>>> solver.fit().estimate_
array([ 2.,  5.])
>>> y1pred, y2pred = solver.predicted()
>>> y1pred
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> y2pred
array([  205.,   805.,  1405.,  2005.])
>>> res1, res2 = solver.residuals()
>>> np.abs(res1) < 10**-10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> np.abs(res2) < 10**-10
array([ True,  True,  True,  True], dtype=bool)



----

"""
