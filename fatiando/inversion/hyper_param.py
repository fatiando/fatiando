r"""
Classes for dealing with hyper parameters.

----

"""
from __future__ import division
import multiprocessing
import numpy

from ..vis import mpl

__all__ = ['LCurve']


class LCurve(object):
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
    >>> from fatiando import utils
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
    >>> src_loc = utils.random_points(area, 30, seed=0)
    >>> rec_loc = utils.circular_points(area, 20, random=True, seed=0)
    >>> srcs, recs = utils.connect_points(src_loc, rec_loc)
    >>> tts = ttime2d.straight(model, 'vp', srcs, recs)
    >>> tts = utils.contaminate(tts, 0.01, percent=True, seed=0)

    Now we can setup a tomography by creating the necessary data misfit
    (`SRTomo`) and regularization (`Smoothness2D`) objects:

    >>> mesh = SquareMesh(area, shape)
    >>> datamisfit = srtomo.SRTomo(tts, srcs, recs, mesh)
    >>> regul = Smoothness2D(mesh.shape)

    The tomography solver will be the `LCurve` solver. It works by calling
    `fit()` and accessing `estimate_`, exactly like any other solver:

    >>> tomo = LCurve(datamisfit, regul, [10**i for i in range(-10, -2, 1)])
    >>> e = tomo.fit().estimate_
    >>> print numpy.array_repr(e.reshape(shape), precision=0)
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

    The estimated regularization parameter is stored in `regul_param_`:

    >>> tomo.regul_param_
    1e-05

    The `LCurve` object also exposes the `residuals()` and `predicted()`
    methods of the data misfit class:

    >>> residuals = tomo.residuals()
    >>> print '%.4f %.4f' % (residuals.mean(), residuals.std())
    -0.0000 0.0047

    `LCurve` also has a `config` method to configure the optimization process
    for non-linear problems, for example:

    >>> initial = 1./4.*numpy.ones(mesh.size)
    >>> tomo = LCurve(datamisfit, regul, [10**i for i in range(-10, -2, 1)])
    >>> e = tomo.config('levmarq', initial=initial).fit().estimate_
    >>> tomo.regul_param_
    1e-05
    >>> print numpy.array_repr(e.reshape(shape), precision=0)
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

    """

    def __init__(self, datamisfit, regul, regul_params, loglog=True,
                 jobs=None):
        self.regul_params = regul_params
        self.datamisfit = datamisfit
        self.regul = regul
        self.regul_param_ = None
        self.objectives = None
        self.corner_ = None
        self.estimate_ = None
        self.p_ = None
        self.dnorm = None
        self.mnorm = None
        self.fit_method = None
        self.fit_args = None
        self.jobs = jobs
        self.loglog = loglog

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
        if self.jobs is None:
            results = [_run_lcurve(s) for s in solvers]
        else:
            pool = multiprocessing.Pool(self.jobs)
            results = pool.map(_run_lcurve, solvers)
            pool.close()
            pool.join()
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
        Selects the corner value of the L-curve and sets the estimate to it.

        Uses the Triangle method of Castellanos et al. (2002).

        The index of the corner value is stored in the ``corner_`` attribute.

        Returns:

        * self

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
        for k in xrange(0, n - 2):
            b = [x[k], y[k]]
            for j in xrange(k + 1, n - 1):
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
        self.regul_param_ = self.regul_params[corner]
        self.p_ = self.objectives[corner].p_
        self.estimate_ = self.objectives[corner].estimate_
        return self

    def config(self, method, **kwargs):
        """
        Configure the optimization method and its parameters.

        This sets the method used by
        :meth:`~fatiando.inversion.regularization.LCurve.fit` and the keyword
        arguments that are passed to it.

        Parameters:

        * method : string
            The optimization method. One of: ``'linear'``, ``'newton'``,
            ``'levmarq'``, ``'steepest'``, ``'acor'``

        Other keyword arguments that can be passed are the ones allowed by each
        method.

        See :meth:`fatiando.inversion.base.Misfit.config`.

        Returns:

        * self

        """
        self.fit_method = method
        self.fit_args = kwargs
        return self

    def plot_lcurve(self, guides=True):
        """
        Make a plot of the data-misfit x regularization values.

        The estimated corner value is shown as a blue triangle.

        Parameters:

        * guides : True or False
            Plot vertical and horizontal lines across the corner value.

        """
        x, y = self.dnorm, self.mnorm
        if self.loglog:
            mpl.loglog(x, y, '.-k')
        else:
            mpl.plot(x, y, '.-k')
        if guides:
            ax = mpl.gca()
            vmin, vmax = ax.get_ybound()
            mpl.vlines(x[self.corner_], vmin, vmax)
            vmin, vmax = ax.get_xbound()
            mpl.hlines(y[self.corner_], vmin, vmax)
        mpl.plot(x[self.corner_], y[self.corner_], '^b', markersize=10)
        mpl.xlabel('Data misfit')
        mpl.ylabel('Regularization')


def _run_lcurve(solver):
    """
    Call ``fit`` on the solver. Needed for multiprocessing.
    """
    result = solver.fit()
    return result
