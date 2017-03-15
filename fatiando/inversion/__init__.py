"""
Everything you need to solve inverse problems!

This package provides the basic building blocks to implement inverse problem
solvers. The main class for this is :class:`~fatiando.inversion.misfit.Misfit`.
It represents a data-misfit function and contains various tools to fit a
model to some data. All you have to do is implement methods to calculate the
predicted (or modeled) data and (optionally) the Jacobian (or sensitivity)
matrix. With only that, you have access to a range of optimization methods,
regularization, joint inversion, etc.

Modules
-------

* :mod:`~fatiando.inversion.misfit`: Defines the data-misfit classes. Used to
  implement new inverse problems. The main class is ``Misfit``. It offers a
  template for you to create standard least-squares inversion methods.
* :mod:`~fatiando.inversion.regularization`: Classes for common regularizing
  functions and base classes for building new ones.
* :mod:`~fatiando.inversion.hyper_param`: Classes hyper parameter optimization
  (estimating the regularization parameter), like L-curve analysis and (in the
  future) cross-validation.
* :mod:`~fatiando.inversion.optimization`: Functions for several optimization
  methods (used internally by :class:`~fatiando.inversion.misfit.Misfit`).
  In most cases you won't need to touch this.
* :mod:`~fatiando.inversion.base`: Base classes used internally to define
  common operations and method.
  In most cases you won't need to touch this.

You can import the ``Misfit``, regularization, and hyper parameter optimization
classes directly from the ``fatiando.inversion`` namespace:

>>> from fatiando.inversion import Misfit, LCurve, Damping, Smoothness

The :class:`~fatiando.inversion.misfit.Misfit` class is used internally in
Fatiando to implement all of our inversion algorithms. Take a look at the
modules below for some examples:

* :mod:`fatiando.seismic.srtomo`
* :mod:`fatiando.gravmag.basin2d`
* :mod:`fatiando.gravmag.eqlayer`
* :mod:`fatiando.gravmag.euler`
* :mod:`fatiando.gravmag.magdir`
* :mod:`fatiando.seismic.profile`

Examples
--------

The ``Misfit`` class works by subclassing it. Doing this gives you access to
all optimization functions and possible combinations of regularization. When
subclassing ``Misfit``, you'll need to implement the ``predicted`` method that
calculates a predicted data vector based on an input parameter vector. This is
virtually all that is problem-specific in an inverse problem. If you want to
use gradient-based optimization (or linear problems) you'll need to implement
the ``jacobian``  method as well that calculates the Jacobian (or sensitivity)
matrix.

Linear Regression
-----------------

Here is an example of how to implement a simple linear regression using the
:class:`~fatiando.inversion.misfit.Misfit` class.

What we want to do is fit :math:`f(a, b, x) = y = ax + b` to find a and b.
Putting a and b in a parameter vector ``p = [a, b]`` we get:

.. math::

    \mathbf{d} = \mathbf{A} \mathbf{p}

where :math:`\mathbf{d}` is the data vector containing all the values of y
and :math:`\mathbf{A}` is the Jacobian matrix with the values of x in the first
column and 1 in the second column.

All we have to do to implement a solver for this problem is write the
``predicted`` (to calculate y from values of a and b) and ``jacobian`` (to
calculate the Jacobian matrix):

First, I'll load numpy and the ``Misfit`` class.

>>> import numpy as np
>>> from fatiando.inversion import Misfit

We'll also need some compatibility code to support Python 2 and 3 at the same
time.

>>> from __future__ import division
>>> from future.builtins import super

Now, I'll make my regression class that *inherits* from ``Misfit``.
All of the least-squares solving code and much more we get for free just by
using ``Misfit`` as template for our regression class. Note ``Misfit`` wants a
1D data vector, no matter what your data is (line, grid, volume).

>>> class Regression(Misfit):
...     "Perform a linear regression"
...     def __init__(self, x, y):
...         # Call the initialization of Misfit
...         super().__init__(data=y, nparams=2, islinear=True)
...         self.x = x  # Store this to use in the other methods
...     def predicted(self, p):
...         "Calculate the predicted data for a given parameter vector"
...         a, b = p
...         return a*self.x + b
...     def jacobian(self, p):
...         "Calculate the Jacobian (ignores p because the problem is linear)"
...         jac = np.empty((self.ndata, self.nparams))
...         jac[:, 0] = self.x
...         jac[:, 1] = 1
...         return jac

To test our regression, let's generate some data based on known values of a and
b.

>>> x = np.linspace(0, 5, 6)
>>> x
array([ 0.,  1.,  2.,  3.,  4.,  5.])
>>> y = 2*x + 5
>>> y
array([  5.,   7.,   9.,  11.,  13.,  15.])

We must create a solver object to perform our regression. Fitting the data
(running the optimization) is done by calling the ``fit`` method.
``fit`` returns the regression class itself we can chain calls to it like so:

>>> solver = Regression(x, y).fit()

The estimated parameter vector is stored in the ``p_`` attribute.
``Misfit`` also provides a ``estimate_`` attribute that can be a custom (user
defined) formatted version of ``p_``. It's better to use ``estimate_`` if
you're not interested in the parameter vector as it is. Since we didn't
implement this custom formatting, both should give the same value.

>>> solver.p_
array([ 2.,  5.])
>>> solver.estimate_
array([ 2.,  5.])

The methods ``predicted`` and ``residuals`` will use the cached values based
on ``p_`` if  the parameter vector is omitted as an argument.

>>> solver.predicted()
array([  5.,   7.,   9.,  11.,  13.,  15.])
>>> np.all(np.abs(solver.residuals()) < 10**10)
True


Caching
-------

The ``Misfit`` class caches some of the matrices that it computes, like the
Jacobian matrix. This is needed because separate methods call ``jacobian`` with
the same input ``p``, so recomputing the matrix would be a waste.

For linear problems, the Jacobian matrix is cached permanently. It is only
calculated once, no matter what ``p`` is passed to it.

>>> A = solver.jacobian(solver.p_)
>>> A
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.]])
>>> B = solver.jacobian(np.array([20, 30]))
>>> B
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.]])
>>> A is B
True

For non-linear problems, the Jacobian is also cached but it will be
recalculated if passed a different value of ``p`` (see the
:ref:`non-linear example below <inversion_non_lin_problems>`).

The Hessian matrix (:math:`2\mathbf{A}^T\mathbf{A}`) is also cached permanently
for linear problems.

>>> H = solver.hessian(solver.p_)
>>> H
array([[ 110.,   30.],
       [  30.,   12.]])
>>> H2 = solver.hessian(np.array([20, 30]))
>>> H2
array([[ 110.,   30.],
       [  30.,   12.]])
>>> H is H2
True


Non-linear optimization
-----------------------

You can configure the solver using the ``config`` method. This allows you to
choose the optimization method you want to use and it's corresponding
parameters. The ``config`` method also returns the solver class itself so it
can be chained with ``fit``:

>>> # Configure solver to use the Levemberg-Marquardt method
>>> solver.config('levmarq', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])
>>> np.all(np.abs(solver.residuals()) < 10**10)
True
>>> # or the Steepest Descent method
>>> solver.config('steepest', initial=[1, 1]).fit().estimate_
array([ 2.,  5.])
>>> # or the Gauss-Newton method
>>> solver.config('newton', initial=[1, 1], maxit=5).fit().estimate_
array([ 2.,  5.])

The ``Misfit`` class keeps track of the optimization process and makes that
data available through the ``stats_`` attribute (a dictionary).

>>> list(sorted(solver.stats_))
['iterations', 'method', 'objective']
>>> solver.stats_['method']
"Newton's method"
>>> solver.stats_['iterations']
5

The ``'objective'`` key holds a list of the objective function value per
iteration of the optimization process.


Re-weighted least squares
-------------------------

``Misfit`` allows you to set weights to the data in the form of a weight
matrix or vector (the vector is assumed to be the diagonal of the weight
matrix). We can use this to perform a re-weighted least-squares fit to remove
outliers from our data.

>>> y_outlier = y.copy()
>>> y_outlier[3] += 20
>>> y_outlier
array([  5.,   7.,   9.,  31.,  13.,  15.])

First, we run the regression without any weights.

>>> solver = Regression(x, y_outlier).fit()
>>> print(np.array_repr(solver.estimate_, precision=3))
array([ 2.571,  6.905])

Now, we can use the inverse of the residuals to set the weights for our data.
We repeat this for a few iterations and should have our robust estimate by the
end of it.

>>> for i in range(20):
...     r = np.abs(solver.residuals())
...     # Avoid small residuals because of zero-division errors
...     r[r < 1e-10] = 1
...     _ = solver.set_weights(1/r).fit()
>>> solver.estimate_
array([ 2.,  5.])


.. _inversion_non_lin_problems:

Non-linear problems
-------------------

In this example, I want to fit a Gaussian to my data:

.. math::

    f(x) = a\exp(-b(x + c)^{2})

Function *f* is non-linear with respect to inversion parameters *a, b, c*.
Thus, we need to configure the solver and choose an optimization method before
we can call ``fit()``.

First, lets create our solver class based on ``Misfit`` and implement the
``predicted`` and ``jacobian`` methods.

>>> class GaussianFit(Misfit):
...     def __init__(self, x, y):
...         super().__init__(
...             data=y, nparams=3, islinear=False)
...         self.x = x
...     def predicted(self, p):
...         a, b, c = p
...         return a*np.exp(-b*(self.x + c)**2)
...     def jacobian(self, p):
...         a, b, c = p
...         jac = np.empty((self.ndata, self.nparams))
...         var = self.x + c
...         exponential = np.exp(-b*var**2)
...         jac[:, 0] = exponential
...         jac[:, 1] = -a*exponential*(var**2)
...         jac[:, 2] = -a*exponential*2*b*var
...         return jac

Let's create some data to test this.

>>> x = np.linspace(0, 10, 1000)
>>> a, b, c = 100, 0.1, -2
>>> y = a*np.exp(-b*(x + c)**2)

For non-linear problems, we **have** to configure the optimization method.
Lets use Levemberg-Marquardt because it generally offers good convergence.

>>> solver = GaussianFit(x, y).config('levmarq', initial=[1, 1, 1]).fit()
>>> # Print the estimated coefficients
>>> print(', '.join(['{:.1f}'.format(i) for i in solver.estimate_]))
100.0, 0.1, -2.0
>>> np.all(np.abs(solver.residuals()) < 10**-10)
True

We can use other optimization methods without having to re-implement our
solution. For example, let's see how well the Ant Colony Optimization for
Continuous Domains (ACO-R) does for this problem:

>>> # bounds are the min, max values of the search domain for each parameter
>>> _ = solver.config('acor', bounds=[50, 500, 0, 1, -20, 0], seed=0).fit()
>>> print(', '.join(['{:.1f}'.format(i) for i in solver.estimate_]))
100.0, 0.1, -2.0

For non-linear problems, the Jacobian and Hessian are cached but not
permanently. Calling ``jacobian`` twice in a row with the same parameter vector
will not trigger a computation and will return the cached value instead.

>>> A = solver.jacobian(np.array([1, 1, 1]))
>>> B = solver.jacobian(np.array([1, 1, 1]))
>>> A is B
True

But passing a different ``p`` will trigger a computation and the cache will be
replaced by the new value.

>>> C = solver.jacobian(np.array([1, 1, 1.1]))
>>> A is C
False
>>> np.all(A == C)
False
>>> D = solver.jacobian(np.array([1, 1, 1.1]))
>>> D is C
True

----

"""
from __future__ import absolute_import
from .misfit import Misfit
from .regularization import Damping, Smoothness, Smoothness1D, Smoothness2D, \
    TotalVariation, TotalVariation1D, TotalVariation2D
from .hyper_param import LCurve
