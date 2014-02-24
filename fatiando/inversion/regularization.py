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

* :class:`~fatiando.inversion.regularization.LCurve`: Use an L-curve criteon.
  Runs the inversion using several regularization parameters. The best value
  is the one that falls on the corner of the log-log plot of the data
  misfit vs regulazing function. Only works for a single regularization.


----

"""
from __future__ import division
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

    def hessian(self, p):
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
        return 2*scipy.sparse.identity(self.nparams).tocsr()

    def gradient(self, p):
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
            grad = 2.*p
        return grad

    def value(self, p):
        """
        Calculate the value of this function.

        Parameters:

        * p : 1d-array
            The parameter vector

        Returns:

        * value : float
            The value of this function evaluated at *p*

        """
        return numpy.linalg.norm(p)**2

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
        self._cache['hessian'] = {'hash':'',
                                  'array':2*safe_dot(fdmat.T, fdmat)}

    def hessian(self, p):
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

    def gradient(self, p):
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

    def value(self, p):
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
        return numpy.sum(p*safe_dot(self.hessian(p), p))/2.

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

    def value(self, p):
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

    def hessian(self, p):
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
        q_matrix = scipy.sparse.diags(self.beta/((derivs**2 + self.beta)**1.5),
                                      0).tocsr()
        return safe_dot(self._fdmat.T, q_matrix*self._fdmat)

    def gradient(self, p):
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
        q = derivs/numpy.sqrt(derivs**2 + self.beta)
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
    v = [1]*(size - 1) + [-1]*(size - 1)
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
    nderivs = (nx - 1)*ny + (ny - 1)*nx
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
    return scipy.sparse.coo_matrix((V, (I, J)), (nderivs, nx*ny)).tocsr()

class LCurve(object):
    """
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
    >>> print '%.9f %.5f' % (residuals.mean(), residuals.std())
    -0.000008199 0.00471

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

    def __init__(self, datamisfit, regul, regul_params):
        self.regul_params = regul_params
        self.datamisfit = datamisfit
        self.regul = regul
        self.regul_param_ = None
        self.objectives = None
        self.corner_ = None
        self.estimate_ = None
        self.p_ = None
        self.dnorm = numpy.empty(len(regul_params))
        self.mnorm = numpy.empty(len(regul_params))
        self.fit_method = None
        self.fit_args = None

    def fit(self):
        self.objectives = []
        for i, mu in enumerate(self.regul_params):
            solver = self.datamisfit + mu*self.regul
            if self.fit_method is not None:
                solver.config(self.fit_method, **self.fit_args)
            solver.fit()
            p = solver.p_
            self.dnorm[i] = self.datamisfit.value(p)
            self.mnorm[i] = self.regul.value(p)
            self.objectives.append(solver)
        corner = self.select_corner(self.dnorm, self.mnorm)
        self.corner_ = corner
        self.regul_param_ = self.regul_params[corner]
        self.p_ = self.objectives[corner].p_
        self.estimate_ = self.objectives[corner].estimate_
        return self

    def select_corner(self, dnorm, mnorm):
        # Uses http://www.sciencedirect.com/science/article/pii/S0168927401001799
        x, y = numpy.log(dnorm), numpy.log(mnorm)
        n = len(dnorm)
        corner = n - 1
        dist = lambda p1, p2: numpy.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        cte = 7.*numpy.pi/8.
        angmin = None
        c = [x[-1], y[-1]]
        for k in xrange(0, n - 2):
            b = [x[k], y[k]]
            for j in xrange(k + 1, n - 1):
                a = [x[j], y[j]]
                ab = dist(a, b)
                ac = dist(a, c)
                bc = dist(b, c)
                cosa = (ab**2 + ac**2 - bc**2)/(2.*ab*ac)
                ang = numpy.arccos(cosa)
                area = 0.5*((b[0] - a[0])*(a[1] - c[1]) - (a[0] - c[0])*(b[1] - a[1]))
                # area is > 0 because in the paper C is index 0
                if area > 0 and (ang < cte and (angmin is None or ang < angmin)):
                    corner = j
                    angmin = ang
        return corner

    def config(self, method, **kwargs):
        self.fit_method = method
        self.fit_args = kwargs
        return self

    def predicted(self, p=None):
        return self.objectives[self.corner_].predicted(p)

    def residuals(self, p=None):
        return self.objectives[self.corner_].residuals(p)

    def plot_lcurve(self):
        mpl.loglog(self.dnorm, self.mnorm, '.-k')
        ax = mpl.gca()
        vmin, vmax = ax.get_ybound()
        mpl.vlines(self.dnorm[self.corner_], vmin, vmax)
        vmin, vmax = ax.get_xbound()
        mpl.hlines(self.mnorm[self.corner_], vmin, vmax)
        mpl.plot(self.dnorm[self.corner_], self.mnorm[self.corner_], '^b', markersize=10)
        mpl.xlabel('Data misfit')
        mpl.ylabel('Regularization')
