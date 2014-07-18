"""
Ready made and base classes for regularization.

All classes are derived from :class:`~fatiando.inversion.base.Objective` and
should be used by summing them to a :class:`~fatiando.inversion.base.Misfit`
derivative.

The regularization parameter is set by multiplying the regularization instance
by a scalar, e.g., ``solver = misfit + 0.1*regularization``.

See :class:`fatiando.gravmag.eqlayer.EQLGravity` for an example.

**List of classes**

* :class:`~fatiando.inversion.regularization.Damping`: Damping regularization
  (or 0th order Tikhonov regularization)
* :class:`~fatiando.inversion.regularization.Smoothness`: Generic smoothness
  regularization (or 1st order Tikhonov regularization). Requires a finite
  difference matrix to specify the parameter derivatives to minimize.
* :class:`~fatiando.inversion.regularization.Smoothness1D`: Smoothness for 1D
  problems. Automatically builds a finite difference matrix based on the number
  of parameters
* :class:`~fatiando.inversion.regularization.Smoothness2D`: Smoothness for 2D
  grid based problems. Automatically builds a finite difference matrix of
  derivatives in the two spacial dimensions based on the shape of the parameter
  grid
* :class:`~fatiando.inversion.regularization.TotalVariation`: Generic total
  variation regularization (enforces sharpness of the solution). Requires a
  finite difference matrix to specify the parameter derivatives.
* :class:`~fatiando.inversion.regularization.TotalVariation1D`: Total variation
  for 1D problems. Similar to Smoothness1D
* :class:`~fatiando.inversion.regularization.TotalVariation2D`: Total variation
  for 2D grid based problems. Similar to Smoothness2D

**Regularization parameter estimation**

Bellow are classes that estimate an optimal value for the regularization
parameter. They work exactly like an Objective function, i.e., run the
inversion by calling their `fit()` method and accessing
the estimates by `p_`, `estimate_`, `residuals()` and `predicted()`.

* :class:`~fatiando.inversion.regularization.LCurve`: Use an L-curve criterion.
  Runs the inversion using several regularization parameters. The best value
  is the one that falls on the corner of the log-log plot of the data
  misfit vs regularizing function. Only works for a single regularization.


----

"""
from __future__ import division
import multiprocessing

import numpy
import scipy.sparse

from .base import Objective
from ..utils import safe_dot
from ..vis import mpl


class Damping(Objective):

    r"""
    Damping (0th order Tikhonov) regularization.

    Imposes the minimum norm of the parameter vector.

    The regularizing function if of the form

    .. math::

        \theta^{NM}(\bar{p}) = \bar{p}^T\bar{p}

    Its gradient and Hessian matrices are, respectively,

    .. math::

        \bar{\nabla}\theta^{NM}(\bar{p}) = 2\bar{\bar{I}}\bar{p}

    .. math::

        \bar{\bar{\nabla}}\theta^{NM}(\bar{p}) = 2\bar{\bar{I}}

    Parameters:

    * nparams : int
        The number of parameter

    Examples:

    >>> import numpy
    >>> damp = Damping(3)
    >>> p = numpy.array([0, 0, 0])
    >>> damp.value(p)
    0.0
    >>> damp.hessian(p).todense()
    matrix([[ 2.,  0.,  0.],
            [ 0.,  2.,  0.],
            [ 0.,  0.,  2.]])
    >>> damp.gradient(p)
    array([ 0.,  0.,  0.])
    >>> p = numpy.array([1, 0, 0])
    >>> damp.value(p)
    1.0
    >>> damp.hessian(p).todense()
    matrix([[ 2.,  0.,  0.],
            [ 0.,  2.,  0.],
            [ 0.,  0.,  2.]])
    >>> damp.gradient(p)
    array([ 2.,  0.,  0.])

    """

    def __init__(self, nparams):
        super(Damping, self).__init__(nparams, islinear=True)

    def _get_hessian(self, p):
        """
        Calculate the Hessian matrix.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * hessian : 2d-array
            The Hessian

        """
        # This is cheap so there is no need to cache it
        return 2 * scipy.sparse.identity(self.nparams).tocsr()

    def _get_gradient(self, p):
        """
        Calculate the gradient vector.

        Parameters:

        * p : 1d-array or ``'null'``
            The parameter vector. If ``'null'``, will return 0.

        Returns:

        * gradient : 1d-array
            The gradient

        """
        if p is 'null':
            grad = 0
        else:
            grad = 2. * p
        return grad

    def _get_value(self, p):
        """
        Calculate the value of this function.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of this function evaluated at *p*

        """
        return numpy.linalg.norm(p) ** 2


class Smoothness(Objective):

    r"""
    Smoothness (1st order Tikhonov) regularization.

    Imposes that adjacent parameters have values close to each other.

    The regularizing function if of the form

    .. math::

        \theta^{SV}(\bar{p}) =
        \bar{p}^T \bar{\bar{R}}^T \bar{\bar{R}}\bar{p}

    Its gradient and Hessian matrices are, respectively,

    .. math::

        \bar{\nabla}\theta^{SV}(\bar{p}) =
        2\bar{\bar{R}}^T \bar{\bar{R}}\bar{p}

    .. math::

        \bar{\bar{\nabla}}\theta^{SV}(\bar{p}) =
        2\bar{\bar{R}}^T \bar{\bar{R}}

    in which matrix :math:`\bar{\bar{R}}` is a finite difference matrix. It
    represents the differences between one parameter and another and is what
    indicates what *adjacent* means.

    Parameters:

    * fdmat : 2d-array or sparse matrix
        The finite difference matrix

    Examples:

    >>> import numpy as np
    >>> fd = np.array([[1, -1, 0],
    ...                [0, 1, -1]])
    >>> s = Smoothness(fd)
    >>> p = np.array([0, 0, 0])
    >>> s.value(p)
    0.0
    >>> s.gradient(p)
    array([0, 0, 0])
    >>> s.hessian(p)
    array([[ 2, -2,  0],
           [-2,  4, -2],
           [ 0, -2,  2]])
    >>> p = np.array([1, 0, 1])
    >>> s.value(p)
    2.0
    >>> s.gradient(p)
    array([ 2, -4,  2])
    >>> s.hessian(p)
    array([[ 2, -2,  0],
           [-2,  4, -2],
           [ 0, -2,  2]])

    """

    def __init__(self, fdmat):
        super(Smoothness, self).__init__(fdmat.shape[1], islinear=True)
        self._cache = {}
        self._cache['hessian'] = {'hash': '',
                                  'array': 2 * safe_dot(fdmat.T, fdmat)}

    def _get_hessian(self, p):
        """
        Calculate the Hessian matrix.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * hessian : 2d-array
            The Hessian

        """
        return self._cache['hessian']['array']

    def _get_gradient(self, p):
        """
        Calculate the gradient vector.

        Parameters:

        * p : 1d-array or ``'null'``
            The parameter vector. If ``'null'``, will return 0.

        Returns:

        * gradient : 1d-array
            The gradient

        """
        if p is 'null':
            grad = 0
        else:
            grad = safe_dot(self.hessian(p), p)
        return grad

    def _get_value(self, p):
        """
        Calculate the value of this function.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of this function evaluated at *p*

        """
        # Need to divide by 2 because the hessian is 2*R.T*R
        return numpy.sum(p * safe_dot(self.hessian(p), p)) / 2.


class Smoothness1D(Smoothness):

    """
    Smoothness regularization for 1D problems.

    Extends the generic :class:`~fatiando.inversion.regularization.Smoothness`
    class by automatically building the finite difference matrix.

    Parameters:

    * npoints : int
        The number of parameters

    Examples:

    >>> import numpy as np
    >>> s = Smoothness1D(3)
    >>> p = np.array([0, 0, 0])
    >>> s.value(p)
    0.0
    >>> s.gradient(p)
    array([0, 0, 0])
    >>> s.hessian(p).todense()
    matrix([[ 2, -2,  0],
            [-2,  4, -2],
            [ 0, -2,  2]])
    >>> p = np.array([1, 0, 1])
    >>> s.value(p)
    2.0
    >>> s.gradient(p)
    array([ 2, -4,  2])
    >>> s.hessian(p).todense()
    matrix([[ 2, -2,  0],
            [-2,  4, -2],
            [ 0, -2,  2]])

    """

    def __init__(self, npoints):
        super(Smoothness1D, self).__init__(fd1d(npoints))


class Smoothness2D(Smoothness):

    """
    Smoothness regularization for 2D problems.

    Extends the generic :class:`~fatiando.inversion.regularization.Smoothness`
    class by automatically building the finite difference matrix.

    Parameters:

    * shape : tuple = (ny, nx)
        The shape of the parameter grid. Number of parameters in the y and x
        (or z and x, time and offset, etc) dimensions.

    Examples:

    >>> import numpy as np
    >>> s = Smoothness2D((2, 2))
    >>> p = np.array([[0, 0],
    ...               [0, 0]]).ravel()
    >>> s.value(p)
    0.0
    >>> s.gradient(p)
    array([0, 0, 0, 0])
    >>> s.hessian(p).todense()
    matrix([[ 4, -2, -2,  0],
            [-2,  4,  0, -2],
            [-2,  0,  4, -2],
            [ 0, -2, -2,  4]])
    >>> p = np.array([[1, 0],
    ...               [2, 3]]).ravel()
    >>> s.value(p)
    12.0
    >>> s.gradient(p)
    array([ 0, -8,  0,  8])
    >>> s.hessian(p).todense()
    matrix([[ 4, -2, -2,  0],
            [-2,  4,  0, -2],
            [-2,  0,  4, -2],
            [ 0, -2, -2,  4]])

    """

    def __init__(self, shape):
        super(Smoothness2D, self).__init__(fd2d(shape))


class TotalVariation(Objective):

    r"""
    Total variation regularization.

    Imposes that adjacent parameters have a few sharp transitions.

    The regularizing function if of the form

    .. math::

        \theta^{VT}(\bar{p}) = \sum\limits_{k=1}^L |v_k|

    where vector :math:`\bar{v} = \bar{\bar{R}}\bar{p}`. See
    :class:`~fatiando.inversion.regularization.Smoothness` for the definition
    of the :math:`\bar{\bar{R}}` matrix.

    This functions is not differentiable at the null vector, so the following
    form is used to calculate the gradient and Hessian

    .. math::

        \theta^{VT}(\bar{p}) \approx \theta^{VT}_\beta(\bar{p}) =
        \sum\limits_{k=1}^L \sqrt{v_k^2 + \beta}

    Its gradient and Hessian matrices are, respectively,

    .. math::

        \bar{\nabla}\theta^{VT}_\beta(\bar{p}) =
        \bar{\bar{R}}^T \bar{q}(\bar{p})

    .. math::

        \bar{\bar{\nabla}}\theta^{VT}_\beta(\bar{p}) =
        \bar{\bar{R}}^T \bar{\bar{Q}}(\bar{p})\bar{\bar{R}}

    and

    .. math::

        \bar{q}(\bar{p}) =
        \begin{bmatrix}
            \dfrac{v_1}{\sqrt{v_1^2 + \beta}} \\
            \dfrac{v_2}{\sqrt{v_2^2 + \beta}} \\
            \vdots \\ \dfrac{v_L}{\sqrt{v_L^2 + \beta}}
        \end{bmatrix}

    .. math::

        \bar{\bar{Q}}(\bar{p}) =
        \begin{bmatrix}
            \dfrac{\beta}{(v_1^2 + \beta)^{\frac{3}{2}}} & 0 & \ldots & 0 \\
            0 & \dfrac{\beta}{(v_2^2 + \beta)^{\frac{3}{2}}} & \ldots & 0 \\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & \ldots & \dfrac{\beta}{(v_L^2 + \beta)^{\frac{3}{2}}}
        \end{bmatrix}

    Parameters:

    * beta : float
        The beta parameter for the differentiable approximation. The larger it
        is, the closer total variation is to
        :class:`~fatiando.inversion.regularization.Smoothness`. Should be a
        small, positive value.
    * fdmat : 2d-array or sparse matrix
        The finite difference matrix

    """

    def __init__(self, beta, fdmat):
        if beta <= 0:
            raise ValueError("Invalid beta=%g. Must be > 0" % (beta))
        super(TotalVariation, self).__init__(
            nparams=fdmat.shape[1], islinear=False)
        self.beta = beta
        self._fdmat = fdmat

    def _get_value(self, p):
        """
        Calculate the value of this function.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of this function evaluated at *p*

        """
        return numpy.linalg.norm(safe_dot(self._fdmat, p), 1)

    def _get_hessian(self, p):
        """
        Calculate the Hessian matrix.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * hessian : 2d-array
            The Hessian

        """
        derivs = safe_dot(self._fdmat, p)
        q = self.beta / ((derivs ** 2 + self.beta) ** 1.5)
        q_matrix = scipy.sparse.diags(q, 0).tocsr()
        return safe_dot(self._fdmat.T, q_matrix * self._fdmat)

    def _get_gradient(self, p):
        """
        Calculate the gradient vector.

        Parameters:

        * p : 1d-array
            The parameter vector.

        Returns:

        * gradient : 1d-array
            The gradient

        """
        derivs = safe_dot(self._fdmat, p)
        q = derivs / numpy.sqrt(derivs ** 2 + self.beta)
        grad = safe_dot(self._fdmat.T, q)
        if len(grad.shape) > 1:
            grad = numpy.array(grad.T).ravel()
        return grad


class TotalVariation1D(TotalVariation):

    """
    Total variation regularization for 1D problems.

    Extends the generic
    :class:`~fatiando.inversion.regularization.TotalVariation`
    class by automatically building the finite difference matrix.

    Parameters:

    * beta : float
        The beta parameter for the differentiable approximation. The larger it
        is, the closer total variation is to
        :class:`~fatiando.inversion.regularization.Smoothness`. Should be a
        small, positive value.
    * npoints : int
        The number of parameters

    """

    def __init__(self, beta, npoints):
        super(TotalVariation1D, self).__init__(beta, fd1d(npoints))


class TotalVariation2D(TotalVariation):

    """
    Total variation regularization for 2D problems.

    Extends the generic
    :class:`~fatiando.inversion.regularization.TotalVariation`
    class by automatically building the finite difference matrix.

    Parameters:

    * beta : float
        The beta parameter for the differentiable approximation. The larger it
        is, the closer total variation is to
        :class:`~fatiando.inversion.regularization.Smoothness`. Should be a
        small, positive value.
    * shape : tuple = (ny, nx)
        The shape of the parameter grid. Number of parameters in the y and x
        (or z and x, time and offset, etc) dimensions.

    """

    def __init__(self, beta, shape):
        super(TotalVariation2D, self).__init__(beta, fd2d(shape))


def fd1d(size):
    """
    Produce a 1D finite difference matrix.

    Parameters:

    * size : int
        The number of points

    Returns:

    * fd : sparse CSR matrix
        The finite difference matrix

    Examples:

    >>> fd1d(2).todense()
    matrix([[ 1, -1]])
    >>> fd1d(3).todense()
    matrix([[ 1, -1,  0],
            [ 0,  1, -1]])
    >>> fd1d(4).todense()
    matrix([[ 1, -1,  0,  0],
            [ 0,  1, -1,  0],
            [ 0,  0,  1, -1]])

    """
    i = range(size - 1) + range(size - 1)
    j = range(size - 1) + range(1, size)
    v = [1] * (size - 1) + [-1] * (size - 1)
    return scipy.sparse.coo_matrix((v, (i, j)), (size - 1, size)).tocsr()


def fd2d(shape):
    """
    Produce a 2D finite difference matrix.

    Parameters:

    * shape : tuple = (ny, nx)
        The shape of the parameter grid. Number of parameters in the y and x
        (or z and x, time and offset, etc) dimensions.

    Returns:

    * fd : sparse CSR matrix
        The finite difference matrix

    Examples:

    >>> fd2d((2, 2)).todense()
    matrix([[ 1, -1,  0,  0],
            [ 0,  0,  1, -1],
            [ 1,  0, -1,  0],
            [ 0,  1,  0, -1]])
    >>> fd2d((2, 3)).todense()
    matrix([[ 1, -1,  0,  0,  0,  0],
            [ 0,  1, -1,  0,  0,  0],
            [ 0,  0,  0,  1, -1,  0],
            [ 0,  0,  0,  0,  1, -1],
            [ 1,  0,  0, -1,  0,  0],
            [ 0,  1,  0,  0, -1,  0],
            [ 0,  0,  1,  0,  0, -1]])

    """
    ny, nx = shape
    nderivs = (nx - 1) * ny + (ny - 1) * nx
    I, J, V = [], [], []
    deriv = 0
    param = 0
    for i in xrange(ny):
        for j in xrange(nx - 1):
            I.extend([deriv, deriv])
            J.extend([param, param + 1])
            V.extend([1, -1])
            deriv += 1
            param += 1
        param += 1
    param = 0
    for i in xrange(ny - 1):
        for j in xrange(nx):
            I.extend([deriv, deriv])
            J.extend([param, param + nx])
            V.extend([1, -1])
            deriv += 1
            param += 1
    return scipy.sparse.coo_matrix((V, (I, J)), (nderivs, nx * ny)).tocsr()


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
    >>> from fatiando.inversion.regularization import Smoothness2D, LCurve
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
        dist = lambda p1, p2: numpy.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
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

    def predicted(self, p=None):
        """
        Returns the predicted data for a given parameter vector.

        Uses the solver that falls on the corner of the L-curve.

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the predicted data. If None,
            will use the current estimate stored in ``estimate_``.

        Returns:

        * predicted : 1d-array or list of 1d-arrays
            The predicted data. If this is the sum of 1 or more Misfit
            instances, will return the predicted data from each of the summed
            misfits in the order of the sum.

        """
        return self.objectives[self.corner_].predicted(p)

    def residuals(self, p=None):
        """
        Returns the residuals vector (observed - predicted data).

        Uses the solver that falls on the corner of the L-curve.

        Parameters:

        * p : 1d-array or None
            The parameter vector used to calculate the residuals. If None, will
            use the current estimate stored in ``estimate_``.

        Returns:

        * residuals : 1d-array or list of 1d-arrays
            The residual vector. If this is the sum of 1 or more Misfit
            instances, will return the residual vector from each of the summed
            misfits in the order of the sum.

        """
        return self.objectives[self.corner_].residuals(p)

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
