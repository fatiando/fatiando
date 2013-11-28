"""
The base classes for inverse problem solving.

----

"""
from __future__ import division

import hashlib

import numpy

from .solvers import linear, levmarq, steepest, newton, acor


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
        self._init_stats()
        self._cache = {}
        self.hasher = lambda x: hashlib.sha1(x).hexdigest()
        if islinear:
            self.fit = self.linear
            self.islinear = True
        else:
            self.fit = self.levmarq
            self.islinear = False
        self.nparams = nparams

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

    def _init_stats(self):
        "Initialize the *stats* attribute with default values"
        self.stats = {
            'method':'',
            'iterations':'N/A',
            'misfit/iteration':'N/A',
            'step search stagnation':'N/A',
            'trial steps/iteration':'N/A',
            'preconditioned':'N/A'}

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
        text = '\n'.join([
            'method: %s' % (self.stats['method']),
            'iterations: %s' % (str(self.stats['iterations'])),
            'preconditioned: %s' % (str(self.stats['preconditioned'])),
            'step search stagnation: %s' % (
                str(self.stats['step search stagnation']))
            ])
        if not summary:
            text = '\n'.join([text,
                'misfit/iteration: %s' % (str(self.stats['misfit/iteration'])),
                'step size/iteration: %s' % (
                    str(self.stats['step size/iteration'])),
                'trial steps/iteration: %s' % (
                    str(self.stats['trial steps/iteration']))
                ])
        return text

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
                tol=10**-5, precondition=True):
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

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        self._init_stats()
        solver = levmarq(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, maxsteps=maxsteps, lamb=lamb, dlamb=dlamb,
                tol=tol, precondition=precondition, stats=self.stats)
        for p in solver:
            continue
        return p

    def ilevmarq(self, initial, maxit=30, maxsteps=10, lamb=1, dlamb=2,
                 tol=10**-5, precondition=True):
        """
        Solve using the Levemberg-Marquardt algorithm.

        This is the iterator version of the solver. Instead of running until
        the end, will yield the solution one iteration at a time.

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

        Yields:

        * estimate : 1d-array
            The estimated parameter vector at each iteration

        """
        self._init_stats()
        solver = levmarq(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, maxsteps=maxsteps, lamb=lamb, dlamb=dlamb,
                tol=tol, precondition=precondition, stats=self.stats)
        return solver

    def newton(self, initial, maxit=30, tol=10**-5, precondition=True):
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

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        self._init_stats()
        solver = newton(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, tol=tol, precondition=precondition,
                stats=self.stats)
        for p in solver:
            continue
        return p

    def inewton(self, initial, maxit=30, tol=10**-5, precondition=True):
        """
        Minimize an objective function using Newton's method.

        This is the iterator version of the solver. Instead of running until
        the end, will yield the solution one iteration at a time.

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

        Yields:

        * estimate : 1d-array
            The estimated parameter vector at each iteration

        """
        self._init_stats()
        solver = newton(initial, self.hessian, self.gradient, self.value,
                maxit=maxit, tol=tol, precondition=precondition,
                stats=self.stats)
        return solver

    def steepest(self, initial, stepsize=0.1, maxsteps=30, maxit=1000,
                 tol=10**-5):
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

        Returns:

        * estimate : 1d-array
            The estimated parameter vector

        """
        self._init_stats()
        solver = steepest(initial, self.gradient, self.value, maxit=maxit,
                maxsteps=maxsteps, stepsize=stepsize, tol=tol,
                stats=self.stats)
        for p in solver:
            continue
        return p

    def isteepest(self, initial, stepsize=0.1, maxsteps=30, maxit=1000,
                 tol=10**-5):
        """
        Minimize an objective function using the Steepest Descent method.

        This is the iterator version of the solver. Instead of running until
        the end, will yield the solution one iteration at a time.

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

        Yields:

        * estimate : 1d-array
            The estimated parameter vector at each iteration

        """
        self._init_stats()
        solver = steepest(initial, self.gradient, self.value, maxit=maxit,
                maxsteps=maxsteps, stepsize=stepsize, tol=tol,
                stats=self.stats)
        return solver

    def acor(self, bounds, nants=None, archive_size=None, maxit=1000,
             diverse=0.5, evap=0.85, seed=None):
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

        Returns:

        * estimate : 1d-array
            The best estimate

        """
        self._init_stats()
        solver = acor(self.value, bounds, self.nparams, nants=nants,
                archive_size=archive_size, maxit=maxit, diverse=diverse,
                evap=evap, seed=seed)
        for p in solver:
            continue
        return p

    def iacor(self, bounds, nants=None, archive_size=None, maxit=1000,
             diverse=0.5, evap=0.85, seed=None):
        """
        Minimize the objective function using ACO-R.

        This is the iterator version of the solver. Instead of running until
        the end, will yield the solution one iteration at a time.

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

        Yields:

        * estimate : 1d-array
            The best estimate at each iteration

        """
        self._init_stats()
        solver = acor(self.value, bounds, self.nparams, nants=nants,
                archive_size=archive_size, maxit=maxit, diverse=diverse,
                evap=evap, seed=seed)
        return solver

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
    (like :class:`~fatiando.inversion.misfit.L2Norm` and
    :class:`~fatiando.inversion.regularization.Damping`):

        >>> obj1 = Objective(nparams=3)
        >>> obj1
        Objective(nparams=3, islinear=False)
        >>> obj2 = Objective(nparams=3, islinear=True)
        >>> obj2
        Objective(nparams=3, islinear=True)

    1. Pass a list of lists to the constructor like so:

        >>> mu1, mu2 = 1, 0.01
        >>> multiobj = MultiObjective([[mu1, obj1], [mu2, obj2]])
        >>> multiobj
        MultiObjective(objs=[
            [1, Objective(nparams=3, islinear=False)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    2. Sum objective functions::

        >>> multiobj = mu1*obj1 + mu2*obj2
        >>> multiobj
        MultiObjective(objs=[
            [1, Objective(nparams=3, islinear=False)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])
        >>> # Since mu1 == 1, the following is the equivalent
        >>> multiobj = obj1 + mu2*obj2
        >>> multiobj
        MultiObjective(objs=[
            [1, Objective(nparams=3, islinear=False)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    3. Use the ``add_objective`` method::

        >>> multiobj = MultiObjective()
        >>> multiobj
        MultiObjective(objs=[
        ])
        >>> multiobj.add_objective(obj1)
        MultiObjective(objs=[
            [1, Objective(nparams=3, islinear=False)],
        ])
        >>> multiobj.add_objective(obj2, regul_param=mu2)
        MultiObjective(objs=[
            [1, Objective(nparams=3, islinear=False)],
            [0.01, Objective(nparams=3, islinear=True)],
        ])

    You can access the different objective functions in a MultiObjective like
    lists::

        >>> mu1, obj1 = multiobj[0]
        >>> print mu1, obj1
        1 Objective(nparams=3, islinear=False)
        >>> mu2, obj2 = multiobj[1]
        >>> print mu2, obj2
        0.01 Objective(nparams=3, islinear=True)

    and like lists, you can iterate over them as well::

        >>> for mu, obj in multiobj:
        ...     print mu, obj
        1 Objective(nparams=3, islinear=False)
        0.01 Objective(nparams=3, islinear=True)

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
            self.fit = self.linear
        else:
            self.islinear = False
            self.fit = self.levmarq
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
