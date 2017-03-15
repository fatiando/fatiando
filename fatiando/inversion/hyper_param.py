r"""
Classes for hyper parameter estimation (like the regularizing parameter).

These classes copy the interface of the standard inversion classes based on
:class:`~fatiando.inversion.misfit.Misfit` (i.e.,
``solver.config(...).fit().estimate_``). When their ``fit`` method is called,
they perform many runs of the inversion and try to select the optimal values
for the hyper parameters. The class will then behave as the solver that yields
the best estimate (e.g., ``solver[0].predicted()``).

Available classes:

* :class:`~fatiando.inversion.hyper_param.LCurve`: Estimate the regularizing
  parameter using an L-curve analysis.

----

"""
from __future__ import division, absolute_import
from future.builtins import range
import multiprocessing
import numpy

from ..vis import mpl
from .base import OptimizerMixin


class LCurve(OptimizerMixin):
    """
    Use the L-curve criterion to estimate the regularization parameter.

    Runs the inversion using several specified regularization parameters.
    The best value is the one that falls on the corner of the log-log plot of
    the data misfit vs regularizing function.
    This point is automatically found using the triangle method of
    Castellanos et al. (2002).

    This class behaves as :class:`~fatiando.inversion.base.Misfit`.
    To use it, simply call ``fit`` and optionally ``config``.
    The estimate will be stored in ``estimate_`` and ``p_``.
    The estimated regularization parameter will be stored in ``regul_param_``.

    Parameters:

    * datamisfit : :class:`~fatiando.inversion.base.Misfit`
        The data misfit instance for the inverse problem. Can be a sum of other
        misfits.
    * regul : A class from :mod:`fatiando.inversion.regularization`
        The regularizing function.
    * regul_params : list
        The values of the regularization parameter that will be tested.
    * loglog : True or False
        If True, will use a log-log scale for the L-curve (recommended).
    * jobs : None or int
        If not None, will use *jobs* processes to calculate the L-curve.

    References:

    Castellanos, J. L., S. Gomez, and V. Guerra (2002), The triangle method for
    finding the corner of the L-curve, Applied Numerical Mathematics, 43(4),
    359-373, doi:10.1016/S0168-9274(01)00179-9.

    Examples:

    We'll use the L-curve to estimate the best regularization parameter for a
    smooth inversion using :mod:`fatiando.seismic.srtomo`.

    First, we'll setup some synthetic data:

    >>> import numpy
    >>> from fatiando.mesher import SquareMesh
    >>> from fatiando.seismic import ttime2d, srtomo
    >>> from fatiando.inversion import Smoothness2D, LCurve
    >>> from fatiando import utils, gridder
    >>> area = (0, 2, 0, 2)
    >>> shape = (10, 10)
    >>> model = SquareMesh(area, shape)
    >>> vp = 4*numpy.ones(shape)
    >>> vp[3:7,3:7] = 10
    >>> vp
    array([[  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.]])
    >>> model.addprop('vp', vp.ravel())
    >>> src_loc_x, src_loc_y = gridder.scatter(area, 30, seed=0)
    >>> src_loc = numpy.transpose([src_loc_x, src_loc_y])
    >>> rec_loc_x, rec_loc_y = gridder.circular_scatter(area, 20,
    ...                                                 random=True, seed=0)
    >>> rec_loc = numpy.transpose([rec_loc_x, rec_loc_y])
    >>> srcs = [src for src in src_loc for _ in rec_loc]
    >>> recs = [rec for _ in src_loc for rec in rec_loc]
    >>> tts = ttime2d.straight(model, 'vp', srcs, recs)
    >>> tts = utils.contaminate(tts, 0.01, percent=True, seed=0)

    Now we can setup a tomography by creating the necessary data misfit
    (``SRTomo``) and regularization (``Smoothness2D``) objects. We'll normalize
    the data misfit by the number of data points to make the scale of the
    regularization parameter more tractable.

    >>> mesh = SquareMesh(area, shape)
    >>> datamisfit = (1./tts.size)*srtomo.SRTomo(tts, srcs, recs, mesh)
    >>> regul = Smoothness2D(mesh.shape)

    The tomography solver will be the ``LCurve`` solver. It works by calling
    ``fit()`` and accessing ``estimate_``, exactly like any other solver:

    >>> regul_params = [10**i for i in range(-10, -2, 1)]
    >>> tomo = LCurve(datamisfit, regul, regul_params)
    >>> _ = tomo.fit()
    >>> print(numpy.array_repr(tomo.estimate_.reshape(shape), precision=0))
    array([[  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,   9.,  11.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  11.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,  10.,  11.,   9.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.]])

    When ``fit()`` is called, the ``LCurve``  will run the inversion for each
    value of the regularization parameter, build an l-curve, and find the
    best solution (i.e., the corner value of the l-curve).

    The ``LCurve`` object behaves like a normal multi-objective function.
    In fact, it will try to mirror the objective function that resulted in the
    best solution.
    You can index it to access the data misfit and regularization parts.
    For example, to get the residuals vector or the predicted data:

    >>> predicted = tomo[0].predicted()
    >>> residuals = tomo[0].residuals()
    >>> print '%.4f %.4f' % (residuals.mean(), residuals.std())
    -0.0000 0.0047

    The estimated regularization parameter is stored in ``regul_param_``:

    >>> tomo.regul_param_
    1e-05

    You can run the l-curve analysis in parallel by specifying the ``njobs``
    argument. This will spread the computations over ``njobs`` number of
    processes and give some speedup over running sequentially. Note that you
    should **not** enable any kind of multi-processes parallelism
    on the data misfit class. It is often better to run each inversion
    sequentially and run many of them in parallel. Note that you'll enough
    memory to run multiple inversions at the same time, so this is not suited
    for large, memory hungry inversions.

    >>> par_tomo = LCurve(datamisfit, regul, regul_params, njobs=2)
    >>> _ = par_tomo.fit()  # Will you 2 processes to run inversions
    >>> par_tomo.regul_param_
    1e-05
    >>> print(numpy.array_repr(par_tomo.estimate_.reshape(shape), precision=0))
    array([[  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,   9.,  11.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  11.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,  10.,  11.,   9.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.]])

    ``LCurve`` also has a ``config`` method to configure the optimization
    process for non-linear problems, for example:

    >>> initial = numpy.ones(mesh.size)
    >>> _ = tomo.config('newton', initial=initial, tol=0.2).fit()
    >>> tomo.regul_param_
    1e-05
    >>> print(numpy.array_repr(tomo.estimate_.reshape(shape), precision=0))
    array([[  4.,   4.,   3.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  12.,   9.,  11.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,  11.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  10.,  10.,  10.,  10.,   4.,   4.,   4.],
           [  4.,   4.,   4.,  11.,  10.,  11.,   9.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   5.,   4.,   4.,   4.],
           [  4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.,   4.]])

    You can view the optimization information for the run corresponding to the
    best estimate using the ``stats_`` attribute:

    >>> list(sorted(tomo.stats_))
    ['iterations', 'method', 'objective']
    >>> tomo.stats_['method']
    "Newton's method"
    >>> tomo.stats_['iterations']
    2

    """

    def __init__(self, datamisfit, regul, regul_params, loglog=True,
                 njobs=1):
        assert njobs >= 1, "njobs should be >= 1. {} given.".format(njobs)
        self.regul_params = regul_params
        self.datamisfit = datamisfit
        self.regul = regul
        self.objectives = None
        self.dnorm = None
        self.mnorm = None
        self.fit_method = None
        self.fit_args = None
        self.njobs = njobs
        self.loglog = loglog
        # Estimated parameters from the L curve
        self.corner_ = None

    def _run_fit_first(self):
        """
        Check if a solution was found by running fit.
        Will raise an ``AssertionError`` if not.
        """
        assert self.corner_ is not None, \
            'No optimal solution found. Run "fit" to run the L-curve analysis.'

    @property
    def regul_param_(self):
        """
        The regularization parameter corresponding to the best estimate.
        """
        self._run_fit_first()
        return self.regul_params[self.corner_]

    @property
    def objective_(self):
        """
        The objective function corresponding to the best estimate.
        """
        self._run_fit_first()
        return self.objectives[self.corner_]

    @property
    def stats_(self):
        """
        The optimization information for the best solution found.
        """
        return self.objective_.stats_

    @property
    def p_(self):
        """
        The estimated parameter vector obtained from the best regularization
        parameter.
        """
        return self.objective_.p_

    def fmt_estimate(self, p):
        """
        Return the ``estimate_`` attribute of the optimal solution.
        """
        return self.objective_.estimate_

    def __getitem__(self, i):
        return self.objective_[i]

    def fit(self):
        """
        Solve for the parameter vector and optimum regularization parameter.

        Combines the data-misfit and regularization solvers using the range of
        regularization parameters provided and calls ``fit`` and ``config`` on
        each.

        The ``p_`` and ``estimate_`` attributes correspond to the combination
        that falls in the corner of the L-curve.

        The regularization parameter for this corner point if stored in the
        ``regul_param_`` attribute.

        Returns:

        * self

        """
        if self.datamisfit.islinear:
            self.datamisfit.jacobian('null')
        solvers = [
            self.datamisfit + mu * self.regul for mu in self.regul_params]
        if self.fit_method is not None:
            for solver in solvers:
                solver.config(self.fit_method, **self.fit_args)
        if self.njobs > 1:
            pool = multiprocessing.Pool(self.njobs)
            results = pool.map(_fit_solver, solvers)
            pool.close()
            pool.join()
        else:
            results = [s.fit() for s in solvers]
        self.objectives = results
        self.dnorm = numpy.array(
            [self.datamisfit.value(s.p_) for s in results])
        self.mnorm = numpy.array([self.regul.value(s.p_) for s in results])
        self.select_corner()
        return self

    def _scale_curve(self):
        """
        Puts the data-misfit and regularizing function values in the range
        [-10, 10].
        """
        if self.loglog:
            x, y = numpy.log(self.dnorm), numpy.log(self.mnorm)
        else:
            x, y = self.dnorm, self.mnorm

        def scale(a):
            vmin, vmax = a.min(), a.max()
            l, u = -10, 10
            return (((u - l) / (vmax - vmin)) *
                    (a - (u * vmin - l * vmax) / (u - l)))
        return scale(x), scale(y)

    def select_corner(self):
        """
        Select the corner value of the L-curve formed inversion results.

        This is performed automatically after calling the
        :meth:`~fatiando.inversion.hyper_param.LCurve.fit` method.
        You can run this method separately after
        :meth:`~fatiando.inversion.hyper_param.LCurve.fit` has been called to
        tweak the results.

        You can access the estimated values by:

        * The ``p_`` and ``estimate_`` attributes will hold the estimated
          parameter vector and formatted estimate, respective, corresponding
          to the corner value.
        * The ``regul_param_`` attribute holds the value of the regularization
          parameter corresponding to the corner value.
        * The ``corner_`` attribute will hold the index of the corner value
          in the list of computed solutions.

        Uses the Triangle method of Castellanos et al. (2002).

        References:

        Castellanos, J. L., S. Gomez, and V. Guerra (2002), The triangle method
        for finding the corner of the L-curve, Applied Numerical Mathematics,
        43(4), 359-373, doi:10.1016/S0168-9274(01)00179-9.

        """
        x, y = self._scale_curve()
        n = len(self.regul_params)
        corner = n - 1

        def dist(p1, p2):
            "Return the geometric distance between p1 and p2"
            return numpy.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        cte = 7. * numpy.pi / 8.
        angmin = None
        c = [x[-1], y[-1]]
        for k in range(0, n - 2):
            b = [x[k], y[k]]
            for j in range(k + 1, n - 1):
                a = [x[j], y[j]]
                ab = dist(a, b)
                ac = dist(a, c)
                bc = dist(b, c)
                cosa = (ab ** 2 + ac ** 2 - bc ** 2) / (2. * ab * ac)
                ang = numpy.arccos(cosa)
                area = 0.5 * ((b[0] - a[0]) * (a[1] - c[1]) -
                              (a[0] - c[0]) * (b[1] - a[1]))
                # area is > 0 because in the paper C is index 0
                if area > 0 and (ang < cte and
                                 (angmin is None or ang < angmin)):
                    corner = j
                    angmin = ang
        self.corner_ = corner

    def plot_lcurve(self, ax=None, guides=True):
        """
        Make a plot of the data-misfit x regularization values.

        The estimated corner value is shown as a blue triangle.

        Parameters:

        * ax : matplotlib Axes
            If not ``None``, will plot the curve on this Axes instance.
        * guides : True or False
            Plot vertical and horizontal lines across the corner value.

        """
        if ax is None:
            ax = mpl.gca()
        else:
            mpl.sca(ax)
        x, y = self.dnorm, self.mnorm
        if self.loglog:
            mpl.loglog(x, y, '.-k')
        else:
            mpl.plot(x, y, '.-k')
        if guides:
            vmin, vmax = ax.get_ybound()
            mpl.vlines(x[self.corner_], vmin, vmax)
            vmin, vmax = ax.get_xbound()
            mpl.hlines(y[self.corner_], vmin, vmax)
        mpl.plot(x[self.corner_], y[self.corner_], '^b', markersize=10)
        mpl.xlabel('Data misfit')
        mpl.ylabel('Regularization')


def _fit_solver(solver):
    """
    Call ``fit`` on the solver. Needed for multiprocessing.
    """
    return solver.fit()
