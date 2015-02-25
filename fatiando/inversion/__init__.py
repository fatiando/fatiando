"""
Everything you need to solve inverse problems!

This package provides the basic building blocks to implement inverse problem
solvers. The main class for this is :class:`~fatiando.inversion.base.Misfit`.
It represents a data-misfit function and contains various tools to fit a
model to some data. All you have to do is implement methods to calculate the
predicted (or modeled) data and (optionally) the Jacobian (or sensitivity)
matrix. With only that, you have access to a range of optimization methods,
regularization, joint inversion, etc.

Modules
-------

* :mod:`~fatiando.inversion.base`: Base classes for building inverse problem
  solvers
* :mod:`~fatiando.inversion.regularization`: Classes for common regularizing
  functions and base classes for building new ones
* :mod:`~fatiando.inversion.solvers`: Functions for several optimization
  methods (used by :class:`~fatiando.inversion.base.Misfit`)

Have a look at the examples bellow on how to use the package. More geophysical
examples include :mod:`fatiando.seismic.srtomo`,
:mod:`fatiando.seismic.profile`,
:mod:`fatiando.gravmag.basin2d`,
:mod:`fatiando.gravmag.eqlayer`,
and
:mod:`fatiando.gravmag.euler`.


Examples
--------

Linear Regression
+++++++++++++++++

Here is an example of how to implement a simple linear regression using the
:class:`~fatiando.inversion.base.Misfit` class.

What we want to do is fit :math:`f(a, b, x) = y = ax + b` to find a and b.
Putting a and b in a parameter vector `p = [a, b]` we get:

.. math::

    \mathbf{d} = \mathbf{A} \mathbf{p}

where :math:`\mathbf{d}` is the data vector containing all the values of y
and :math:`\mathbf{A}` is the Jacobian matrix with the values of x in the first
column and 1 in the second column.

All we have to do to implement a solver for this problem is write the
`predicted` (to calculate y from values of a and b) and `jacobian` (to
calculate the Jacobian matrix):


>>> import numpy as np
>>> from fatiando.inversion.base import Misfit
>>> class Regression(Misfit):
...     "Perform a linear regression"
...     def __init__(self, x, y):
...         super(Regression, self).__init__(data=y, nparams=2, islinear=True)
...         self.x = x
...     def predicted(self, p):
...         a, b = p
...         return a*self.x + b
...     def jacobian(self, p):
...         jac = np.ones((self.ndata, self.nparams))
...         jac[:, 0] = self.x
...         return jac
>>> x = np.linspace(0, 5, 6)
>>> x
array([ 0.,  1.,  2.,  3.,  4.,  5.])
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
>>> solver.jacobian(solver.p_)
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.]])

Polynomial fit
++++++++++++++

A more complicated example would be to implement a generic polynomial fit.

>>> class PolynomialRegression(Misfit):
...     "Perform a polynomial regression"
...     def __init__(self, x, y, degree):
...         super(PolynomialRegression, self).__init__(
...             data=y, nparams=degree + 1, islinear=True)
...         self.x = x
...         self.degree = degree
...     def predicted(self, p):
...         return sum(p[i]*self.x**i  for i in xrange(self.nparams))
...     def jacobian(self, p):
...         jac = np.ones((self.ndata, self.nparams))
...         for i in xrange(self.nparams):
...             jac[:, i] = self.x**i
...         return jac
>>> solver = PolynomialRegression(x, y, 1)
>>> solver.fit().estimate_
array([ 5.,  2.])
>>> solver.jacobian(solver.p_)
array([[ 1.,  0.],
       [ 1.,  1.],
       [ 1.,  2.],
       [ 1.,  3.],
       [ 1.,  4.],
       [ 1.,  5.]])
>>> # Use a second order polynomial
>>> y = 0.1*x**2 + 3*x + 6
>>> solver = PolynomialRegression(x, y, 2).fit()
>>> solver.estimate_
array([ 6. ,  3. ,  0.1])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> solver.jacobian(solver.p_)
array([[  1.,   0.,   0.],
       [  1.,   1.,   1.],
       [  1.,   2.,   4.],
       [  1.,   3.,   9.],
       [  1.,   4.,  16.],
       [  1.,   5.,  25.]])

You can also configure the solver to use a different (non-linear) optimization
method:

>>> # Configure solver to use the Levemberg-Marquardt method
>>> solver.config('levmarq', initial=[1, 1, 1]).fit().estimate_
array([ 6. ,  3. ,  0.1])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> # Configure solver to use Ant Colony Optimization
>>> solver.config('acor', bounds=[0, 10, 0, 10, 0, 1], seed=0).fit().estimate_
array([ 6.03600966,  2.95744428,  0.10781796])
>>> np.abs(solver.residuals()) < 10**10
array([ True,  True,  True,  True,  True,  True], dtype=bool)


Non-linear Gaussian fit
+++++++++++++++++++++++

In this example, I want to fit an equation of the form

.. math::

    f(x) = a\exp(-b(x + c)^{2})

Function *f* is non-linear with respect to inversion parameters *a, b, c*.
Thus, we need to configure the solver and choose an optimization method before
we can call ``fit()``.

>>> class GaussianFit(Misfit):
...     def __init__(self, x, y):
...         super(GaussianFit, self).__init__(
...             data=y, nparams=3, islinear=False)
...         self.x = x
...     def predicted(self, p):
...         a, b, c = p
...         return a*np.exp(-b*(self.x + c)**2)
...     def jacobian(self, p):
...         a, b, c = p
...         jac = np.ones((self.ndata, self.nparams))
...         var = self.x + c
...         exponential = np.exp(-b*var**2)
...         jac[:, 0] = exponential
...         jac[:, 1] = -a*exponential*(var**2)
...         jac[:, 2] = -a*exponential*2*b*var
...         return jac
>>> x = np.linspace(0, 10, 1000)
>>> a, b, c = 100, 0.1, -2
>>> y = a*np.exp(-b*(x + c)**2)
>>> # Non-linear solvers have to be configured. Lets use Levemberg-Marquardt.
>>> solver = GaussianFit(x, y).config('levmarq', initial=[1, 1, 1]).fit()
>>> solver.estimate_
array([ 100. ,    0.1,   -2. ])
>>> np.all(np.abs(solver.residuals()) < 10**-10)
True

Multiple data sets (joint inversion)
++++++++++++++++++++++++++++++++++++

Sometimes multiple data types depend on the same parameters (e.g., gravity
and gravity gradients depend of density). In these cases, the inversion of both
datasets can be performed simultaneously by simply adding the Misfits together.

Lets go back to the regression example and pretend that the same linear
equation can represent 2 types of data, ``y1`` and ``y2``:

>>> x1 = np.linspace(0, 5, 6)
>>> y1 = 2*x1 + 5
>>> y1
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> x2 = np.linspace(100, 1000, 4)
>>> y2 = 2*x2 + 5
>>> y2
array([  205.,   805.,  1405.,  2005.])
>>> # Simply sum the 2 classes
>>> solver = Regression(x1, y1) + Regression(x2, y2)
>>> solver.fit().estimate_
array([ 2.,  5.])
>>> # Index the solver to get each Regression
>>> solver[0].predicted()
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> solver[1].predicted()
array([  205.,   805.,  1405.,  2005.])
>>> np.abs(solver[0].residuals()) < 10**-10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> np.abs(solver[1].residuals()) < 10**-10
array([ True,  True,  True,  True], dtype=bool)
>>> # We can configure the joint solver just like any other
>>> solver.config('levmarq', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])

This can also be expanded to 3 or more data types:

>>> x3 = np.linspace(10, 11, 6)
>>> y3 = 2*x3 + 5
>>> y3
array([ 25. ,  25.4,  25.8,  26.2,  26.6,  27. ])
>>> solver = Regression(x1, y1) + Regression(x2, y2) + Regression(x3, y3)
>>> solver.fit().estimate_
array([ 2.,  5.])
>>> y1pred, y2pred, y3pred = [s.predicted() for s in solver]
>>> y1pred
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> y2pred
array([  205.,   805.,  1405.,  2005.])
>>> y3pred
array([ 25. ,  25.4,  25.8,  26.2,  26.6,  27. ])
>>> res1, res2, res3 = [s.residuals() for s in solver]
>>> np.abs(res1) < 10**-10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> np.abs(res2) < 10**-10
array([ True,  True,  True,  True], dtype=bool)
>>> np.abs(res3) < 10**-10
array([ True,  True,  True,  True,  True,  True], dtype=bool)
>>> solver.config('levmarq', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])


----

"""
