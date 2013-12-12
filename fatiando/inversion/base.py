"""
The base classes for inverse problem solving.


Examples:

Here is an example of how to implement a simple linear regression using the
:class:`~fatiando.inversion.base.Misfit` class.

>>> import numpy as np
>>> class Regression(Misfit):
...     "Perform a linear regression"
...     def __init__(self, x, y):
...         super(Regression, self).__init__(data=y, positional={'x':x},
...             model={}, nparams=2, islinear=True)
...     def _get_predicted(self, p):
...         a, b = p
...         return a*self.postional['x'] + b
...     def _get_jacobian(self, p):
...         return np.transpose([self.positional['x'], np.ones(self.ndata)])
>>> x = np.linspace(0, 5, 10)
>>> y = 2*x + 5
>>> solver = Regression(x, y)
>>> solver.fit()
array([ 2.,  5.])

A more complicated example would be to implement a generic polynomial fit.

>>> class PolynomialRegression(Misfit):
...     "Perform a polynomial regression"
...     def __init__(self, x, y, degree):
...         super(PolynomialRegression, self).__init__(
...             data=y, positional={'x':x},
...             model={'degree':degree}, nparams=degree + 1, islinear=True)
...     def _get_predicted(self, p):
...         return sum(p[i]*self.postional['x']**i
...                    for i in xrange(self.model['degree'] + 1))
...     def _get_jacobian(self, p):
...         return np.transpose([self.positional['x']**i
...                             for i in xrange(self.model['degree'] + 1)])
>>> solver = PolynomialRegression(x, y, 1)
>>> solver.fit()
array([ 5.,  2.])
>>> y = 0.1*x**2 + 3*x + 6
>>> solver = PolynomialRegression(x, y, 2)
>>> solver.fit()
array([ 6. ,  3. ,  0.1])


----

"""
from __future__ import division
import hashlib
import numpy
import scipy.sparse

from .solvers import linear, levmarq, steepest, newton, acor
from ..utils import safe_dot


class Objective(object):
    """
    An objective function for an inverse problem.

    Objective functions have a :methd:`~fatiando.inversion.base.Objective.fit``
    method that finds the parameter vector *p* that minimizes them. You can
    specify a range of optimization methods through the ``method`` argument.
    Alternatively, you can call the optimization methods directly:

    * :mesh:`~fatiando.inversion.base.Objective.linear`
    * :mesh:`~fatiando.inversion.base.Objective.levmarq`
    * :mesh:`~fatiando.inversion.base.Objective.newton`
    * :mesh:`~fatiando.inversion.base.Objective.steepest`
    * :mesh:`~fatiando.inversion.base.Objective.acor`

    Keep in mind that using ``fit`` is the **preferred way**.

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
        self._cache = {}
        self.hasher = lambda x: hashlib.sha1(x).hexdigest()
        self.islinear = islinear
        self.nparams = nparams
        self.ndata = 0
        self.default_solver = 'linear' if islinear else 'levmarq'

    def __repr__(self):
        return 'Objective(nparams=%d, islinear=%s)' % (self.nparams,
                str(self.islinear))

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
        The gradient of the objective function with respect to the parameter

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

    # Overload some operators. Adding and multiplying by a scalar transform the
    # objective function into a multiobjetive function (weighted sum of
    # objective functions)
    ###########################################################################
    def __add__(self, other):
        if not isinstance(other, Objective):
            raise TypeError('Can only add derivatives of the Objective class')
        multiobj = MultiObjective()
        if isinstance(self, MultiObjective):
            multiobj.merge(self)
        else:
            multiobj.add_objective(self)
        if isinstance(other, MultiObjective):
            multiobj.merge(other)
        else:
            multiobj.add_objective(other)
        return multiobj

    def __mul__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Can only multiply a Objective by a float or int')
        return MultiObjective([(other, self)])

    def __rmul__(self, other):
        return self.__mul__(other)
    ###########################################################################

    def fit(self, method='default', **kwargs):
        """
        Solve for the parameter vector that minimizes this objective function.

        Parameters:

        * method : string
            The optimization method to use. If *method* is ``'default'``, will
            use the solver defined in ``default_solver`` class attribute.

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        Optimization methods:

        * 'linear': directly solve a linear problem
        * 'levmarq': gradient descent with the Levemberg-Marquardt algorithm
        * 'newton': gradient descent with Newton's method
        * 'steepest': gradient descent with the Steepest Descent method
        * 'acor': heuristic Ant Colony Optimization for Continuous Domains

        For other arguments that can be passed to each specific optimization
        method, see the docstrings for respectively named class methods. These
        options should be passed as keyword arguments.

        """
        if method == 'default':
            method = self.default_solver
        solvers = {'linear':self.linear, 'steepest':self.steepest,
                   'newton':self.newton, 'levmarq':self.levmarq,
                   'acor':self.acor}
        if method not in solvers:
            raise ValueError("Invalid optimization method '%s'" % (method))
        return solvers[method](**kwargs)

    def linear(self, precondition=True):
        """
        Solve for the parameter vector assuming that the problem is linear.

        See :func:`fatiando.inversion.solvers.linear` for more details.

        Parameters:

        * precondition : True or False
            If True, will use Jacobi preconditioning.

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        hessian = self.hessian(None)
        gradient = self.gradient(None)
        p = linear(hessian, gradient, precondition=precondition)
        return p

    def levmarq(self, initial, maxit=30, maxsteps=10, lamb=1, dlamb=2,
                tol=10**-5, precondition=True, iterate=False):
        """
        Solve using the Levemberg-Marquardt algorithm.

        See :func:`fatiando.inversion.solvers.levmarq` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * maxsteps : int
            The maximum number of times to try to take a step before giving
            up
        * lamb : float
            Initial amount of step regularization. The larger this is, the
            more the algorithm will resemble Steepest Descent in the initial
            iterations.
        * dlamb : float
            Factor by which *lamb* is divided or multiplied when taking steps
         * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted
        * precondition : True or False
            If True, will use Jacobi preconditioning
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = levmarq(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, maxsteps=maxsteps, lamb=lamb, dlamb=dlamb,
                tol=tol, precondition=precondition)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def newton(self, initial, maxit=30, tol=10**-5, precondition=True,
               iterate=False):
        """
        Minimize an objective function using Newton's method.

        See :func:`fatiando.inversion.solvers.newton` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted.
        * precondition : True or False
            If True, will use Jacobi preconditioning.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = newton(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, tol=tol, precondition=precondition)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def steepest(self, initial, stepsize=0.1, maxsteps=30, maxit=1000,
                 tol=10**-5, iterate=False):
        """
        Minimize an objective function using the Steepest Descent method.

        See :func:`fatiando.inversion.solvers.steepest` for more details.

        Parameters:

        * initial : 1d-array
            The initial estimate for the gradient descent.
        * maxit : int
            The maximum number of iterations allowed.
        * maxsteps : int
            The maximum number of times to try to take a step before giving
            up
        * stepsize : float
            Initial amount of step step size.
        * tol : float
            The convergence criterion. The lower it is, the more steps are
            permitted.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        solver = steepest(initial, self.gradient, self.value, maxit=maxit,
                maxsteps=maxsteps, stepsize=stepsize, tol=tol)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

    def acor(self, bounds, nants=None, archive_size=None, maxit=1000,
             diverse=0.5, evap=0.85, seed=None, iterate=True):
        """
        Minimize the objective function using ACO-R.

        See :func:`fatiando.inversion.solvers.acor` for more details.

        Parameters:

        * bounds : list
            The bounds of the search space. If only two values are given,
            will interpret as the minimum and maximum, respectively, for all
            parameters.
            Alternatively, you can given a minimum and maximum for each
            parameter, e.g., for a problem with 3 parameters you could give
            `bounds = [min1, max1, min2, max2, min3, max3]`.
        * nants : int
            The number of ants to use in the search. Defaults to the number
            of parameters.
        * archive_size : int
            The number of solutions to keep in the solution archive.
            Defaults to 10 x nants
        * maxit : int
            The number of iterations to run.
        * diverse : float
            Scalar from 0 to 1, non-inclusive, that controls how much better
            solutions are favored when constructing new ones.
        * evap : float
            The pheromone evaporation rate (evap > 0). Controls how spread
            out the search is.
        * seed : None or int
            Seed for the random number generator.
        * iterate : True or False
            If True, will return an iterator that yields one estimated
            parameter vector at a time for each iteration of the algorithm

        Returns:

        * estimate : 1d-array
            The best estimate

        """
        solver = acor(self.value, bounds, self.nparams, nants=nants,
                archive_size=archive_size, maxit=maxit, diverse=diverse,
                evap=evap, seed=seed)
        if iterate:
            return solver
        for p in solver:
            continue
        return p

class Misfit(Objective):
    r"""
    An l2-norm data-misfit function.

    This is a kind of objective function that measures the misfit between
    observed data :math:`\bar{d}^o` and data predicted by a set of model
    parameters :math:`\bar{d} = \bar{f}(\bar{p})`.

    The l2-norm data-misfit is defined as:

    .. math::

        \phi(\bar{p}) = \dfrac{\bar{r}^T\bar{r}}{N}

    where :math:`\bar{r} = \bar{d}^o - \bar{d}` is the residual vector and
    :math:`N` is the number of data.

    This class inherits the solvers from
    :class:`~fatiando.inversion.base.Objective` that estimate a parameter
    vector :math:`\bar{p}` that minimizes it.
    See :class:`~fatiando.inversion.base.Objective` for more details.

    When subclassing this class, you must implement methods two methods:

    * ``_get_predicted(self, p)``: calculates the predicted data
      :math:`\bar{d}` for a given parameter vector ``p``
    * ``_get_jacobian(self, p)``: calculates the Jacobian matrix of
      :math:`\bar{f}(\bar{p})` evaluated at ``p``

    If :math:`\bar{f}` is linear, then the Jacobian will be cached in memory so
    that it is only calculated once when using the class multiple times. So
    solving the same problem with different methods or using an iterative
    method doesn't have the penalty of recalculating the Jacobian.


    Parameters:

    * data : 1d-array
        The observed data vector :math:`\bar{d}^o`
    * positional : dict
        A dictionary with the positional arguments of the data, for example, x,
        y coordinates, depths, etc. Keys should the string name of the argument
        and values should be 1d-arrays with the same size as *data*.
    * model : dict
        A dictionary with the model parameters, like the mesh, physical
        properties, etc.
    * nparams : int
        The number of parameters in parameter vector :math:`\bar{p}`
    * weights : 1d-array
        Weights to be applied to the each element in *data* when computing the
        l2-norm. Effectively the diagonal of a matrix :math:`\bar{\bar{W}}`
        such that :math:`\phi = \bar{r}^T\bar{\bar{W}}\bar{r}`
    * islinear : True or False
        Whether :math:`\bar{f}` is linear or not.

    Examples:

        >>> import numpy
        >>> solver = Misfit(numpy.array([1, 2, 3]),
        ...                 positional={'x':numpy.array([4, 5, 6])},
        ...                 model={},
        ...                 nparams=2)
        >>> solver
        Misfit(
            data=array([1, 2, 3]),
            positional={
                'x':array([4, 5, 6]),
                },
            model={
                },
            nparams=2,
            islinear=False,
            weights=None)
        >>> solver.use_tmp_data(numpy.array([4, 5, 6]))
        Misfit(
            data=array([4, 5, 6]),
            positional={
                'x':array([4, 5, 6]),
                },
            model={
                },
            nparams=2,
            islinear=False,
            weights=None)
        >>> solver.reset_data()
        Misfit(
            data=array([1, 2, 3]),
            positional={
                'x':array([4, 5, 6]),
                },
            model={
                },
            nparams=2,
            islinear=False,
            weights=None)

    """

    def __init__(self, data, positional, model, nparams, weights=None,
                 islinear=False):
        super(Misfit, self).__init__(nparams, islinear=islinear)
        self.data = data
        self._backup_data = data
        self.ndata = len(data)
        self.subset = None
        self.positional = positional
        self._backup_positional = positional.copy()
        self._backup_jacobian = None
        self.model = model
        self._cache['predicted'] = {'hash':'', 'array':None}
        self._cache['jacobian'] = {'hash':'', 'array':None}
        self._cache['hessian'] = {'hash':'', 'array':None}
        self.weights = None
        if weights is not None:
            self.set_weights(weights)

    def _clear_cache(self):
        "Reset the cached matrices"
        self._cache['predicted'] = {'hash':'', 'array':None}
        self._cache['jacobian'] = {'hash':'', 'array':None}
        self._cache['hessian'] = {'hash':'', 'array':None}

    def __repr__(self):
        lw = 60
        prec = 3
        text = [
            'Misfit(',
            '    data=%s,' % (numpy.array_repr(
                self.data, max_line_width=lw, precision=prec)),
            '    positional={',]
        if self.positional:
            text.append(
            '%s' % ('\n'.join([
                "        '%s':%s," % (k, numpy.array_repr(self.positional[k],
                    max_line_width=lw, precision=prec))
                for k in self.positional])))
        text.extend([
            '        },',
            '    model={'])
        if self.model:
            text.append(
            '%s' % ('\n'.join([
                "        '%s':%s," % (k, str(self.model[k]))
                for k in self.model])))
        text.extend([
            '        },',
            '    nparams=%d,'  % (self.nparams),
            '    islinear=%s,' % (repr(self.islinear)),
            '    weights=%s)' % (
                repr(self.weights) if self.weights is None
                else numpy.array_repr(
                    self.weights, max_line_width=lw, precision=prec))])
        return '\n'.join(text)

    def _get_predicted(self, p):
        raise NotImplementedError("Predicted data not implemented")

    def _get_jacobian(self, p):
        raise NotImplementedError("Jacobian matrix not implemented")

    def use_tmp_data(self, data):
        """
        Temporarily use the given data vector instead.

        To reset the original data, use
        :meth:`~fatiando.inversion.base.Misfit.reset_data`.

        Parameters:

        * data : 1d-array
            The observed data vector.

        """
        self.data = data
        return self

    def reset_data(self):
        """
        Reset the original data vector.

        See :meth:`~fatiando.inversion.base.Misfit.use_tmp_data`.
        """
        self.data = self._backup_data
        return self

    def use_all(self):
        """
        Use all the data in the original data vector.

        See :meth:`~fatiando.inversion.base.Misfit.use_subset`.
        """
        self.data = self._backup_data
        self.positional = self._backup_positional
        self.ndata = len(self.data)
        self._clear_cache()
        return self

    def use_subset(self, indices):
        """
        Use only a subset of the observed data.

        Parameters:

        * indices : list
            The indeces of the elements in the data array that will be used.

        """
        self.data = self.data[indices]
        self.positional = dict((k, self.positional[k][indices])
                              for k in self.positional)
        self.ndata = len(indices)
        self._clear_cache()
        return self

    def set_weights(self, weights):
        """
        Set the data weights array.

        See :class:`~fatiando.inversion.base.Misfit` for more information.

        Parameters:

        * weights : 1d-array
            A vector with the data weights.

        """
        self.weights = scipy.sparse.diags(weights, 0)
        # Weights change the Hessian
        self._cache['hessian'] = {'hash':'', 'array':None}
        return self

    def residuals(self, p):
        """
        Calculate the residuals vector (observed - predicted data).

        Parameters:

        * p : 1d-array
            The parameter vector used to calculate the predicted data.

        Returns:

        * residuals : 1d-array
            The residual vector

        """
        return self.data - self.predicted(p)

    def predicted(self, p):
        """
        Calculate the predicted data for a given parameter vector.

        Parameters:

        * p : 1d-array
            The parameter vector used to calculate the predicted data.

        Returns:

        * predicted : 1d-array
            The predicted data

        """
        if p is None:
            pred = 0
        else:
            hash = self.hasher(p)
            if hash != self._cache['predicted']['hash']:
                self._cache['predicted']['array'] = self._get_predicted(p)
                self._cache['predicted']['hash'] = hash
            pred = self._cache['predicted']['array']
        return pred

    def jacobian(self, p):
        """
        Calculate the Jacobian matrix evaluated at a given parameter vector.

        The Jacobian matrix is cached in memory, so passing the same
        parameter vector again will not trigger a re-calculation. However, only
        one matrix is cached at a time.

        Parameters:

        * p : 1d-array or None
            The parameter vector. If the problem is linear, pass ``None``

        Returns:

        * jacobian : 2d-array
            The Jacobian matrix

        """
        if self.islinear:
            hash = ''
        else:
            hash = self.hasher(p)
        if (hash != self._cache['jacobian']['hash'] or
                self._cache['jacobian']['array'] is None):
            self._cache['jacobian']['array'] = self._get_jacobian(p)
            self._cache['jacobian']['hash'] = hash
        return self._cache['jacobian']['array']

    def value(self, p):
        r"""
        Calculate the value of the misfit for a given parameter vector.

        The value is given by:

        .. math::

            \phi(\bar{p}) = \dfrac{\bar{r}^T\bar{\bar{W}}\bar{r}}{N}


        where :math:`\bar{r}` is the residual vector and :math:`bar{\bar{W}}`
        are optional data weights.

        Parameters:

        * p : 1d-array or None
            The parameter vector. If the problem is linear, pass ``None``

        Returns:

        * value : float
            The value of the misfit function.

        """
        if self.weights is None:
            return numpy.linalg.norm(
                self.data - self.predicted(p)
                )**2/self.ndata
        else:
            return numpy.sum(self.weights*(
                        (self.data - self.predicted(p))**2)
                        )/self.ndata

    def hessian(self, p):
        r"""
        The Hessian of the misfit function with respect to the parameters

        Calculated using the Gauss approximation:

        .. math::

            \bar{\bar{H}} \approx 2\bar{\bar{J}}^T\bar{\bar{J}}

        where :math:`\bar{\bar{J}}` is the Jacobian matrix.

        For linear problems, the Hessian matrix is cached in memory, so calling
        this method again will not trigger a re-calculation.

        Parameters:

        * p : 1d-array
            The parameter vector where the Hessian is evaluated

        Returns:

        * hessian : 2d-array
            The Hessian matrix

        """
        if self.islinear and self._cache['hessian']['array'] is not None:
            hessian = self._cache['hessian']['array']
        else:
            jacobian = self.jacobian(p)
            if self.weights is None:
                hessian = (2/self.ndata)*safe_dot(jacobian.T, jacobian)
            else:
                hessian = (2/self.ndata)*safe_dot(
                    jacobian.T, self.weights*jacobian)
            if self.islinear:
                self._cache['hessian']['array'] = hessian
        return hessian

    def gradient(self, p):
        r"""
        The gradient vector of the misfit function.

        .. math::

            \bar{g} = -2\bar{\bar{J}}^T\bar{r}

        where :math:`\bar{\bar{J}}` is the Jacobian matrix and :math:`\bar{r}`
        is the residual vector.

        Parameters:

        * p : 1d-array
            The parameter vector where the Hessian is evaluated

        Returns:

        * gradient : 1d-array
            The gradient vector.

        """
        jacobian = self.jacobian(p)
        if self.weights is None:
            grad = (-2/self.ndata)*safe_dot(
                jacobian.T, self.data - self.predicted(p))
        else:
            grad = (-2/self.ndata)*safe_dot(
                jacobian.T, self.weights*(self.data - self.predicted(p)))
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break loose
            grad = numpy.array(grad).ravel()
        return grad

class MultiObjective(Objective):
    r"""
    A multi-objective function.

    It is a weighted sum of objective functions:

    .. math::

        \Gamma(\bar{p}) = \sum\limits_{k=1}^{N} \mu_k \phi_k(\bar{p})

    :math:`\mu_k` are regularization parameters that control the trade-off
    between each objective function.

    MultiObjective have the same methods that Objective has and can be
    optimized in the same way to produce an estimated parameter vector.

    There are several ways of creating MultiObjective from
    :class:`~fatiando.inversion.base.Objective` instances and its derivatives
    (like :class:`~fatiando.inversion.base.Misfit` and
    :class:`~fatiando.inversion.regularization.Damping`):

        >>> import numpy
        >>> obj1 = Misfit(data=numpy.array([1, 2, 3, 4]), positional={},
        ...               model={}, nparams=3)
        >>> obj1
        Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)
        >>> obj2 = Objective(nparams=3, islinear=True)
        >>> obj2
        Objective(nparams=3, islinear=True)

    1. Pass a list of lists to the constructor like so:

        >>> mu1, mu2 = 1, 0.01
        >>> multiobj = MultiObjective([[mu1, obj1], [mu2, obj2]])
        >>> multiobj
        MultiObjective(objs=[
            [1, Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    2. Sum objective functions::

        >>> multiobj = mu1*obj1 + mu2*obj2
        >>> multiobj
        MultiObjective(objs=[
            [1, Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])
        >>> # Since mu1 == 1, the following is the equivalent
        >>> multiobj = obj1 + mu2*obj2
        >>> multiobj
        MultiObjective(objs=[
            [1, Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    3. Use the ``add_objective`` method::

        >>> multiobj = MultiObjective()
        >>> multiobj
        MultiObjective(objs=[
        ])
        >>> multiobj.add_objective(obj1)
        MultiObjective(objs=[
            [1, Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)],
        ])
        >>> multiobj.add_objective(obj2, regul_param=mu2)
        MultiObjective(objs=[
            [1, Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    You can access the different objective functions in a MultiObjective like
    lists::

        >>> mu1, obj1 = multiobj[0]
        >>> print mu1, obj1
        1 Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)
        >>> mu2, obj2 = multiobj[1]
        >>> print mu2, obj2
        0.01 Objective(nparams=3, islinear=True)

    and like lists, you can iterate over them as well::

        >>> for mu, obj in multiobj:
        ...     print mu, obj
        1 Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)
        0.01 Objective(nparams=3, islinear=True)

    You can check which of the objective functions has data associated with it
    (i.e., is a data-misfit function)::

        >>> multiobj.havedata()
        [Misfit(
            data=array([1, 2, 3, 4]),
            positional={
                },
            model={
                },
            nparams=3,
            islinear=False,
            weights=None)]


    """

    def __init__(self, objs=None):
        super(MultiObjective, self).__init__(nparams=None, islinear=False)
        self.objs = []
        if objs is not None:
            for mu, obj in objs:
                self.add_objective(obj, regul_param=mu)

    def __repr__(self):
        text = '\n'.join(['MultiObjective(objs=['] +
            ['    [%g, %s],' % (mu, repr(obj)) for mu, obj in self.objs] +
            ['])'])
        return text

    def add_objective(self, obj, regul_param=1):
        """
        Add an objective function to the multi-objective.

        Parameters:

        * obj : Objective
            A derivative of the Objective class (like data-misfit,
            regularization, etc.)
        * regul_param : float
            A positive scalar that controls the weight of this objective on the
            multi-objective (like the regularization parameters).

        """
        nparams = obj.nparams
        if self.nparams is not None:
            if numpy.any([nparams != o.nparams for _, o in self.objs]):
                raise ValueError(
                    'Objective function must have %d parameters, not %d'
                    % (self.nparams, nparams))
        else:
            self.nparams = nparams
        self.objs.append([regul_param, obj])
        if numpy.all([o.islinear for _, o in self.objs]):
            self.islinear = True
            self.default_solver = 'linear'
        else:
            self.islinear = False
            self.default_solver = 'levmarq'
        return self

    def merge(self, multiobj):
        """
        Merge an multi-objective function to this one.

        Will append it's objective functions to this one.

        Parameters:

        * multiobj : MultiObjective
            The multi-objective

        """
        for mu, obj in multiobj:
            self.add_objective(obj, regul_param=mu)
        return self

    def havedata(self):
        """
        Return a list of objectives that have data in this multi-objective.
        """
        return [o for  _, o in self.objs if o.ndata > 0]

    # Can increment instead of add_objective or merge
    def __iadd__(self, other):
        if not isinstance(other, Objective):
            raise TypeError('Can only add derivatives of the Objective class')
        if isinstance(other, MultiObjective):
            self.merge(other)
        else:
            self.add_objective(other)
        return self

    # Allow iterating over the multi-objective, returning pairs [mu, obj]
    def __len__(self):
        return len(self.objs)

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, index):
        return self.objs[index]

    def next(self):
        if self.index >= len(self.objs):
            raise StopIteration
        mu, obj = self.__getitem__(self.index)
        self.index += 1
        return mu, obj

    def value(self, p):
        """
        The value of the multi-objective function for a given parameter vector.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of the objective function

        """
        return sum(mu*obj.value(p) for mu, obj in self.objs)

    def gradient(self, p):
        """
        The gradient of the multi-objective function for a parameter vector

        Parameters:

        * p : 1d-array
            The parameter vector where the gradient is evaluated.

        Returns:

        * gradient : 1d-array
            The gradient vector

        """
        return sum(mu*obj.gradient(p) for mu, obj in self.objs)

    def hessian(self, p):
        """
        The Hessian matrix of the multi-objective function

        Parameters:

        * p : 1d-array
            The parameter vector where the Hessian is evaluated

        Returns:

        * hessian : 2d-array
            The Hessian matrix

        """
        return sum(mu*obj.hessian(p) for mu, obj in self.objs)

    def predicted(self, p):
        """
        The predicted data for all data-misfit functions in the multi-objective

        Will compute the predicted data for each data-misfit at the given
        parameter vector.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * pred : list or 1d-array
            A list with 1d-arrays of predicted data for each data-misfit
            function that makes up the multi-objective. They will be in the
            order in which the data-misfits were added to the multi-objective.
            If there is only one data-misfit, will return the 1d-array, not a
            list.

        """
        pred = []
        for mu, obj in self.objs:
            if callable(getattr(obj, 'predicted', None)):
                pred.append(obj.predicted(p))
        if len(pred) == 1:
            pred = pred[0]
        return pred

    def residuals(self, p):
        """
        The residuals vector for data-misfit functions in the multi-objective

        Will compute the residual vector for each data-misfit at the given
        parameter vector.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * res : list or 1d-array
            A list with 1d-arrays of residual vectors for each data-misfit
            function that makes up the multi-objective. They will be in the
            order in which the data-misfits were added to the multi-objective.
            If there is only one data-misfit, will return the 1d-array, not a
            list.

        """
        res = []
        for mu, obj in self.objs:
            if callable(getattr(obj, 'residuals', None)):
                res.append(obj.residuals(p))
        if len(res) == 1:
            res = res[0]
        return res
